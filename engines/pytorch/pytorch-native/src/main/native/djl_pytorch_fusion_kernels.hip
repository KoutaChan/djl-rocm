/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
#include "djl_pytorch_fusion_kernels.h"

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace djl::pytorch::fusion {
namespace {

constexpr int kThreadsPerBlock = 256;

int64_t CheckedMultiply(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " factors must not be negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
      name, " exceeds the supported range");
  return left * right;
}

int64_t RequiredBlockCount(int64_t element_count) {
  return element_count / kThreadsPerBlock +
      (element_count % kThreadsPerBlock == 0 ? 0 : 1);
}

struct OutputPackArguments {
  const void* sources[kMaximumOutputPackSources];
  int64_t widths[kMaximumOutputPackSources];
  int64_t destination_offsets[kMaximumOutputPackSources];
  int32_t data_types[kMaximumOutputPackSources];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct SegmentedOutputPackArguments {
  const void* sources[kMaximumOutputPackSources];
  int32_t data_types[kMaximumOutputPackSources];
  int64_t source_prefix_counts[kMaximumOutputPackSources];
  int64_t source_token_counts[kMaximumOutputPackSources];
  int64_t token_offsets[kMaximumOutputPackSources];
  int64_t token_counts[kMaximumOutputPackSources];
  int64_t hidden_widths[kMaximumOutputPackSources];
  int64_t destination_offsets[kMaximumOutputPackSources];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct BinaryBranchBlendArguments {
  const void* baseline_context;
  const void* selected_context;
  const void* selected_logit;
  const void* baseline_presence;
  const void* selected_presence;
  float* output;
  int64_t row_count;
  int64_t width;
  int32_t selected_logit_data_type;
  int32_t baseline_presence_data_type;
  int32_t selected_presence_data_type;
};

struct AffineInputPackArguments {
  const void* sources[kMaximumAffineTerms];
  int64_t widths[kMaximumAffineTerms];
  int64_t destination_offsets[kMaximumAffineTerms];
  int32_t data_types[kMaximumAffineTerms];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct AffineFinalizeArguments {
  const void* sources[kMaximumAffineGroups];
  int64_t source_rows_per_batch[kMaximumAffineGroups];
  int64_t source_strides[kMaximumAffineGroups][kMaximumAffinePrefixRank];
  const void* bias;
  void* output;
  int64_t batch_count;
  int64_t output_prefix[kMaximumAffinePrefixRank];
  int64_t output_prefix_rows;
  int64_t output_width;
  int32_t source_count;
  int32_t output_prefix_rank;
  int32_t activation;
};

struct IndexedAffineInputPackArguments {
  const void* sources[kMaximumIndexedAffineSources];
  int64_t widths[kMaximumIndexedAffineSources];
  int64_t destination_offsets[kMaximumIndexedAffineSources];
  int64_t index_divisors[kMaximumIndexedAffineSources];
  int64_t source_rows[kMaximumIndexedAffineSources];
  int32_t data_types[kMaximumIndexedAffineSources];
  const void* indices;
  void* output;
  int64_t active_rows;
  int64_t destination_rows;
  int64_t output_width;
};

struct IndexedAffineFinalizeArguments {
  const void* indices;
  const void* hidden;
  const void* hidden_bias;
  const void* output_weight;
  const void* output_bias;
  void* output;
  int64_t active_rows;
  int64_t destination_rows;
  int64_t hidden_width;
  int64_t output_width;
  int32_t activation;
};

template <typename output_t, typename scalar_t>
__device__ void pack_source(const OutputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const scalar_t*>(arguments.sources[source_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    output[row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(source[index]);
  }
}

template <typename output_t>
__global__ void output_pack_kernel(OutputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_source<output_t, c10::Half>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_source<output_t, c10::BFloat16>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_source<output_t, float>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename output_t, typename source_t>
__device__ void pack_segmented_source(
    const SegmentedOutputPackArguments& arguments, int32_t segment_index) {
  const auto* source =
      static_cast<const source_t*>(arguments.sources[segment_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t prefix_count = arguments.source_prefix_counts[segment_index];
  const int64_t source_token_count =
      arguments.source_token_counts[segment_index];
  const int64_t token_offset = arguments.token_offsets[segment_index];
  const int64_t token_count = arguments.token_counts[segment_index];
  const int64_t hidden_width = arguments.hidden_widths[segment_index];
  const int64_t width = prefix_count * token_count * hidden_width;
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    const int64_t destination_token = column / hidden_width;
    const int64_t feature = column - destination_token * hidden_width;
    const int64_t prefix = destination_token / token_count;
    const int64_t token = destination_token - prefix * token_count + token_offset;
    const int64_t source_element =
        ((row * prefix_count + prefix) * source_token_count + token) *
            hidden_width +
        feature;
    output[row * arguments.output_width +
        arguments.destination_offsets[segment_index] + column] =
        static_cast<output_t>(source[source_element]);
  }
}

template <typename output_t>
__global__ void segmented_output_pack_kernel(
    SegmentedOutputPackArguments arguments) {
  const int32_t segment_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[segment_index])) {
    case torch::kFloat16:
      pack_segmented_source<output_t, c10::Half>(arguments, segment_index);
      break;
    case torch::kBFloat16:
      pack_segmented_source<output_t, c10::BFloat16>(arguments, segment_index);
      break;
    case torch::kFloat32:
      pack_segmented_source<output_t, float>(arguments, segment_index);
      break;
    default:
      break;
  }
}

__device__ float load_fusion_float(
    const void* data, int64_t index, int32_t data_type) {
  switch (static_cast<torch::ScalarType>(data_type)) {
    case torch::kFloat16:
      return static_cast<float>(static_cast<const c10::Half*>(data)[index]);
    case torch::kBFloat16:
      return static_cast<float>(static_cast<const c10::BFloat16*>(data)[index]);
    case torch::kFloat32:
      return static_cast<const float*>(data)[index];
    default:
      return 0.0f;
  }
}

template <typename score_t, typename mask_t, typename value_t>
__global__ void mapped_grouped_masked_softmax_pool_kernel(
    const score_t* scores, const mask_t* masks, const value_t* values,
    const int32_t* destination_metadata, float* contexts,
    mask_t* presence, int64_t batch_count, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count) {
  __shared__ float probabilities[256];
  __shared__ int32_t has_values;
  const int64_t row_count = batch_count * destination_count;
  for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const int64_t batch = row / destination_count;
    const int64_t destination = row - batch * destination_count;
    const int32_t* metadata = destination_metadata + destination * 4;
    const int32_t source_group = metadata[0];
    const int64_t output_row =
        static_cast<int64_t>(metadata[2]) +
        batch * static_cast<int64_t>(metadata[1]) + metadata[3];
    if (source_group < 0) {
      if (threadIdx.x == 0) {
        presence[output_row] = static_cast<mask_t>(0.0f);
      }
      for (int64_t column = threadIdx.x; column < width;
           column += blockDim.x) {
        contexts[output_row * width + column] = 0.0f;
      }
      continue;
    }

    if (threadIdx.x < warpSize) {
      const int lane = threadIdx.x;
      float maximum = -INFINITY;
      int selected_count = 0;
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        const int64_t mask_index =
            (batch * candidate_count + candidate) * group_count +
            source_group;
        if (static_cast<float>(masks[mask_index]) != 0.0f) {
          const float score = static_cast<float>(
              scores[batch * candidate_count + candidate]);
          maximum = isnan(maximum) || isnan(score)
              ? NAN
              : fmaxf(maximum, score);
          ++selected_count;
        }
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        const float other_maximum = __shfl_down(maximum, offset);
        maximum = isnan(maximum) || isnan(other_maximum)
            ? NAN
            : fmaxf(maximum, other_maximum);
        selected_count += __shfl_down(selected_count, offset);
      }
      maximum = __shfl(maximum, 0);
      selected_count = __shfl(selected_count, 0);

      float local_denominator = 0.0f;
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        const int64_t mask_index =
            (batch * candidate_count + candidate) * group_count +
            source_group;
        const float probability =
            selected_count > 0 &&
                    static_cast<float>(masks[mask_index]) != 0.0f
                ? expf(static_cast<float>(
                           scores[batch * candidate_count + candidate]) -
                      maximum)
                : 0.0f;
        probabilities[candidate] = probability;
        local_denominator += probability;
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_denominator += __shfl_down(local_denominator, offset);
      }
      const float denominator = __shfl(local_denominator, 0);
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        if (probabilities[candidate] != 0.0f) {
          probabilities[candidate] /= denominator;
        }
      }
      if (lane == 0) {
        has_values = selected_count > 0;
        presence[output_row] = static_cast<mask_t>(
            selected_count > 0 ? 1.0f : 0.0f);
      }
    }
    __syncthreads();
    for (int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      float sum = 0.0f;
      for (int64_t candidate = 0; candidate < candidate_count;
           ++candidate) {
        const float probability = probabilities[candidate];
        if (probability != 0.0f) {
          sum += probability *
              static_cast<float>(
                  values[(batch * candidate_count + candidate) * width +
                      column]);
        }
      }
      contexts[output_row * width + column] = sum;
    }
    __syncthreads();
  }
}

template <typename baseline_t, typename selected_t>
__global__ void binary_branch_blend_kernel(
    BinaryBranchBlendArguments arguments) {
  const auto* baseline_context =
      static_cast<const baseline_t*>(arguments.baseline_context);
  const auto* selected_context =
      static_cast<const selected_t*>(arguments.selected_context);
  __shared__ float baseline_weight;
  __shared__ float selected_weight;
  for (int64_t row = blockIdx.x; row < arguments.row_count;
       row += gridDim.x) {
    if (threadIdx.x == 0) {
      const float selected_logit = load_fusion_float(arguments.selected_logit,
          row, arguments.selected_logit_data_type);
      const float selected_probability =
          1.0f / (1.0f + expf(-selected_logit));
      const float baseline_presence = load_fusion_float(
          arguments.baseline_presence, row,
          arguments.baseline_presence_data_type);
      const float selected_presence = load_fusion_float(
          arguments.selected_presence, row,
          arguments.selected_presence_data_type);
      const float joint_presence = baseline_presence * selected_presence;
      baseline_weight =
          baseline_presence - joint_presence * selected_probability;
      selected_weight = selected_presence - joint_presence +
          joint_presence * selected_probability;
    }
    __syncthreads();
    const int64_t row_offset = row * arguments.width;
    for (int64_t column = threadIdx.x; column < arguments.width;
         column += blockDim.x) {
      const int64_t index = row_offset + column;
      arguments.output[index] =
          baseline_weight * static_cast<float>(baseline_context[index]) +
          selected_weight * static_cast<float>(selected_context[index]);
    }
    __syncthreads();
  }
}

template <typename output_t, typename input_t>
__device__ void pack_affine_source(
    const AffineInputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const input_t*>(arguments.sources[source_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    output[row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(static_cast<float>(source[index]));
  }
}

template <typename output_t>
__global__ void affine_input_pack_kernel(AffineInputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_affine_source<output_t, c10::Half>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_affine_source<output_t, c10::BFloat16>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_affine_source<output_t, float>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename scalar_t>
__global__ void affine_finalize_kernel(AffineFinalizeArguments arguments) {
  auto* output = static_cast<scalar_t*>(arguments.output);
  const auto* bias = static_cast<const scalar_t*>(arguments.bias);
  const int64_t output_row_count = arguments.batch_count * arguments.output_prefix_rows;
  const int64_t element_count = output_row_count * arguments.output_width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t output_row = index / arguments.output_width;
    const int64_t column = index - output_row * arguments.output_width;
    const int64_t batch = output_row / arguments.output_prefix_rows;
    int64_t prefix_index = output_row - batch * arguments.output_prefix_rows;
    int64_t coordinates[kMaximumAffinePrefixRank];
    for (int32_t axis = arguments.output_prefix_rank - 1; axis >= 0; --axis) {
      coordinates[axis] = prefix_index % arguments.output_prefix[axis];
      prefix_index /= arguments.output_prefix[axis];
    }

    float sum = bias == nullptr ? 0.0f : static_cast<float>(bias[column]);
    for (int32_t source_index = 0; source_index < arguments.source_count; ++source_index) {
      int64_t source_row = batch * arguments.source_rows_per_batch[source_index];
      for (int32_t axis = 0; axis < arguments.output_prefix_rank; ++axis) {
        source_row += coordinates[axis] * arguments.source_strides[source_index][axis];
      }
      const auto* source = static_cast<const scalar_t*>(arguments.sources[source_index]);
      sum += static_cast<float>(
          source[source_row * arguments.output_width + column]);
    }
    if (arguments.activation == static_cast<int32_t>(AffineActivation::kSilu)) {
      sum = sum / (1.0f + expf(-sum));
    }
    output[index] = static_cast<scalar_t>(sum);
  }
}

template <typename output_t, typename input_t, typename index_t>
__device__ void pack_indexed_affine_source(
    const IndexedAffineInputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const input_t*>(arguments.sources[source_index]);
  const auto* indices = static_cast<const index_t*>(arguments.indices);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.active_rows * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_row = index / width;
    const int64_t column = index - active_row * width;
    const int64_t destination_row = static_cast<int64_t>(indices[active_row]);
    if (destination_row < 0 || destination_row >= arguments.destination_rows) {
      continue;
    }
    const int64_t source_row = destination_row / arguments.index_divisors[source_index];
    if (source_row >= arguments.source_rows[source_index]) {
      continue;
    }
    output[active_row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(static_cast<float>(source[source_row * width + column]));
  }
}

template <typename output_t, typename index_t>
__global__ void indexed_affine_input_pack_kernel(
    IndexedAffineInputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_indexed_affine_source<output_t, c10::Half, index_t>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_indexed_affine_source<output_t, c10::BFloat16, index_t>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_indexed_affine_source<output_t, float, index_t>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename scalar_t, typename index_t>
__global__ void indexed_affine_finalize_kernel(
    IndexedAffineFinalizeArguments arguments) {
  const auto* indices = static_cast<const index_t*>(arguments.indices);
  const auto* hidden = static_cast<const scalar_t*>(arguments.hidden);
  const auto* hidden_bias = static_cast<const scalar_t*>(arguments.hidden_bias);
  const auto* output_weight = static_cast<const scalar_t*>(arguments.output_weight);
  const auto* output_bias = static_cast<const scalar_t*>(arguments.output_bias);
  auto* output = static_cast<scalar_t*>(arguments.output);
  const int64_t element_count = arguments.active_rows * arguments.output_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t active_row = index / arguments.output_width;
    const int64_t output_column = index - active_row * arguments.output_width;
    float sum = 0.0f;
    for (int64_t hidden_column = lane;
         hidden_column < arguments.hidden_width; hidden_column += warpSize) {
      float value = static_cast<float>(
          hidden[active_row * arguments.hidden_width + hidden_column]);
      if (hidden_bias != nullptr) {
        value += static_cast<float>(hidden_bias[hidden_column]);
      }
      if (arguments.activation == static_cast<int32_t>(AffineActivation::kSilu)) {
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<scalar_t>(value));
      }
      sum += value * static_cast<float>(
          output_weight[output_column * arguments.hidden_width + hidden_column]);
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
      if (output_bias != nullptr) {
        sum += static_cast<float>(output_bias[output_column]);
      }
      const int64_t destination_row = static_cast<int64_t>(indices[active_row]);
      if (destination_row >= 0 && destination_row < arguments.destination_rows) {
        output[destination_row * arguments.output_width + output_column] =
            static_cast<scalar_t>(sum);
      }
    }
  }
}

constexpr int kTransformerHiddenWidth = 256;
constexpr int kTransformerMaximumTokens = 8;
constexpr int kTransformerAttentionHeads = 4;
constexpr int kTransformerAttentionWidth = 128;
constexpr int kTransformerHeadWidth = 32;
constexpr int kTransformerCachedValuesPerLane =
    kTransformerHiddenWidth / 32;

template <typename scalar_t>
__global__ void indexed_relation_attention_bias_kernel(
    const scalar_t* relation_logits, const int16_t* relation_ids,
    const scalar_t* pair_bias, scalar_t* attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale) {
  const int64_t element_count = batch_count * attention_heads * token_count *
      padded_token_count;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t coordinate = index;
    const int64_t key = coordinate % padded_token_count;
    coordinate /= padded_token_count;
    const int64_t query = coordinate % token_count;
    coordinate /= token_count;
    const int64_t head = coordinate % attention_heads;
    const int64_t batch = coordinate / attention_heads;
    if (key >= token_count) {
      attention_bias[index] = static_cast<scalar_t>(0.0f);
      continue;
    }
    const int64_t relation = static_cast<int64_t>(
        relation_ids[query * token_count + key]);
    const int64_t query_row = batch * token_count + query;
    const int64_t relation_logit_index =
        (head * batch_count * token_count + query_row) * relation_count + relation;
    const int64_t pair_bias_index =
        (head * token_count + query) * token_count + key;
    attention_bias[index] = static_cast<scalar_t>(
        static_cast<float>(relation_logits[relation_logit_index]) * scale +
        static_cast<float>(pair_bias[pair_bias_index]));
  }
}

__device__ float TransformerWaveSum(float value) {
  for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return value;
}

template <typename scalar_t, typename norm_t>
__global__ void transformer_copy_layer_norm_kernel(
    const scalar_t* input, scalar_t* state, scalar_t* normalized,
    const norm_t* weight, const norm_t* bias, int64_t active_rows,
    float epsilon, bool copy_input) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * kTransformerHiddenWidth;
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane;
         ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kTransformerHiddenWidth) {
        const int64_t offset = row_offset + feature;
        const float value = static_cast<float>(input[offset]);
        values[index] = value;
        if (copy_input) {
          state[offset] = static_cast<scalar_t>(value);
        }
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation =
        rsqrtf(__shfl(moment, 0) / kTransformerHiddenWidth + epsilon);
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane;
         ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kTransformerHiddenWidth) {
        normalized[row_offset + feature] = static_cast<scalar_t>(
            (values[index] - mean) * inverse_standard_deviation *
                static_cast<float>(weight[feature]) +
            static_cast<float>(bias[feature]));
      }
    }
  }
}

template <typename scalar_t>
__global__ void transformer_attention_kernel(const scalar_t* query_key_value,
    scalar_t* context, int64_t batch_count, int64_t token_count) {
  const int64_t batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }
  const int32_t feature = threadIdx.x;
  const int32_t lane = feature % warpSize;
  const int32_t wave = feature / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  __shared__ float probabilities
      [kTransformerAttentionHeads * kTransformerMaximumTokens *
       kTransformerMaximumTokens];

  const int64_t qkv_batch_offset =
      batch * token_count * (3 * kTransformerAttentionWidth);
  constexpr float scale = 0.1767766952966369f;
  const int32_t attention_row_count =
      kTransformerAttentionHeads * static_cast<int32_t>(token_count);
  for (int32_t attention_row = wave; attention_row < attention_row_count;
       attention_row += waves_per_block) {
    const int32_t head = attention_row / token_count;
    const int32_t query = attention_row - head * token_count;
    float logits[kTransformerMaximumTokens];
    for (int32_t key = 0; key < token_count; ++key) {
      float product = 0.0f;
      if (lane < kTransformerHeadWidth) {
        const int64_t query_offset =
            qkv_batch_offset + query * (3 * kTransformerAttentionWidth) +
            head * kTransformerHeadWidth + lane;
        const int64_t key_offset =
            qkv_batch_offset + key * (3 * kTransformerAttentionWidth) +
            kTransformerAttentionWidth + head * kTransformerHeadWidth + lane;
        product = static_cast<float>(query_key_value[query_offset]) *
            static_cast<float>(query_key_value[key_offset]);
      }
      product = TransformerWaveSum(product);
      if (lane == 0) {
        logits[key] = product * scale;
      }
    }
    if (lane == 0) {
      float maximum = logits[0];
      for (int32_t key = 1; key < token_count; ++key) {
        maximum = fmaxf(maximum, logits[key]);
      }
      float denominator = 0.0f;
      for (int32_t key = 0; key < token_count; ++key) {
        logits[key] = expf(logits[key] - maximum);
        denominator += logits[key];
      }
      const int32_t probability_offset =
          (head * kTransformerMaximumTokens + query) *
          kTransformerMaximumTokens;
      for (int32_t key = 0; key < token_count; ++key) {
        probabilities[probability_offset + key] = logits[key] / denominator;
      }
    }
  }
  __syncthreads();

  const int32_t context_elements =
      static_cast<int32_t>(token_count) * kTransformerAttentionWidth;
  for (int32_t index = feature; index < context_elements;
       index += kThreadsPerBlock) {
    const int32_t query = index / kTransformerAttentionWidth;
    const int32_t attention_feature =
        index - query * kTransformerAttentionWidth;
    const int32_t head = attention_feature / kTransformerHeadWidth;
    float value = 0.0f;
    const int32_t probability_offset =
        (head * kTransformerMaximumTokens + query) *
        kTransformerMaximumTokens;
    for (int32_t key = 0; key < token_count; ++key) {
      const int64_t value_offset =
          qkv_batch_offset + key * (3 * kTransformerAttentionWidth) +
          2 * kTransformerAttentionWidth + attention_feature;
      value += probabilities[probability_offset + key] *
          static_cast<float>(query_key_value[value_offset]);
    }
    context[(batch * token_count + query) * kTransformerAttentionWidth +
        attention_feature] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void transformer_bias_silu_kernel(
    scalar_t* values, const scalar_t* bias, int64_t element_count,
    int64_t width, bool round_bias_before_activation) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float value = static_cast<float>(values[index]) +
        static_cast<float>(bias[index % width]);
    if (round_bias_before_activation) {
      value = static_cast<float>(static_cast<scalar_t>(value));
    }
    values[index] = static_cast<scalar_t>(value / (1.0f + expf(-value)));
  }
}

template <typename scalar_t, typename norm_t>
__global__ void transformer_residual_layer_norm_kernel(
    scalar_t* state, const scalar_t* update, const scalar_t* update_bias,
    const norm_t* norm_weight, const norm_t* norm_bias, scalar_t* normalized,
    int64_t active_rows, float epsilon, bool round_update_before_residual) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * kTransformerHiddenWidth;
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane;
         ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kTransformerHiddenWidth) {
        const int64_t offset = row_offset + feature;
        float update_value = static_cast<float>(update[offset]) +
            static_cast<float>(update_bias[feature]);
        if (round_update_before_residual) {
          update_value =
              static_cast<float>(static_cast<scalar_t>(update_value));
        }
        const float value =
            static_cast<float>(state[offset]) + update_value;
        const scalar_t rounded = static_cast<scalar_t>(value);
        state[offset] = rounded;
        values[index] = static_cast<float>(rounded);
        ++count;
        const float statistics_value =
            round_update_before_residual ? value : values[index];
        const float delta = statistics_value - mean;
        mean += delta / count;
        moment += delta * (statistics_value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation =
        rsqrtf(__shfl(moment, 0) / kTransformerHiddenWidth + epsilon);
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane;
         ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kTransformerHiddenWidth) {
        normalized[row_offset + feature] = static_cast<scalar_t>(
            (values[index] - mean) * inverse_standard_deviation *
                static_cast<float>(norm_weight[feature]) +
            static_cast<float>(norm_bias[feature]));
      }
    }
  }
}

constexpr int kSingleQueryHiddenWidth = 256;
constexpr int kSingleQueryMaximumTokens = 256;
constexpr int kSingleQueryMaximumHeadWidth = 32;

template <typename mask_t>
__device__ bool single_query_valid(const mask_t* mask, int64_t index) {
  return static_cast<float>(mask[index]) != 0.0f;
}

template <typename memory_t, typename scalar_t, typename mask_t>
__global__ void single_query_seed_input_kernel(const memory_t* memory,
    const memory_t* query_source, const mask_t* valid_mask,
    scalar_t* seed_input, int64_t batch_count, int64_t token_count,
    int64_t query_token_count, int64_t query_index) {
  const int64_t batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }
  __shared__ int32_t valid_count;
  if (threadIdx.x == 0) {
    int32_t count = 0;
    for (int64_t token = 0; token < token_count; ++token) {
      count += single_query_valid(
          valid_mask, batch * token_count + token) ? 1 : 0;
    }
    valid_count = count;
  }
  __syncthreads();
  for (int32_t feature = threadIdx.x; feature < kSingleQueryHiddenWidth;
       feature += blockDim.x) {
    seed_input[batch * (2 * kSingleQueryHiddenWidth) + feature] =
        static_cast<scalar_t>(query_source[
            (batch * query_token_count + query_index) * kSingleQueryHiddenWidth +
            feature]);
    float sum = 0.0f;
    if (valid_count != 0) {
      for (int64_t token = 0; token < token_count; ++token) {
        if (single_query_valid(
                valid_mask, batch * token_count + token)) {
          sum += static_cast<float>(memory[
              (batch * token_count + token) * kSingleQueryHiddenWidth +
              feature]);
        }
      }
      sum /= valid_count;
    }
    seed_input[batch * (2 * kSingleQueryHiddenWidth) +
        kSingleQueryHiddenWidth + feature] = static_cast<scalar_t>(sum);
  }
}

template <typename memory_t, typename scalar_t, typename mask_t>
__global__ void single_query_attention_kernel(
    const scalar_t* seed_products, const scalar_t* seed_bias,
    const memory_t* memory, const mask_t* valid_mask,
    const scalar_t* query_weight, const scalar_t* query_bias,
    const scalar_t* key_value_weight, scalar_t* state, scalar_t* context,
    int64_t batch_count, int64_t readout_count, int64_t token_count,
    int64_t attention_heads, int64_t attention_width) {
  const int64_t batch = blockIdx.x;
  const int64_t readout_head = blockIdx.y;
  const int64_t readout = readout_head / attention_heads;
  const int64_t head = readout_head - readout * attention_heads;
  if (batch >= batch_count || readout >= readout_count) {
    return;
  }
  const int64_t head_width = attention_width / attention_heads;
  __shared__ float query[kSingleQueryMaximumHeadWidth];
  __shared__ float direction[kSingleQueryHiddenWidth];
  __shared__ float probabilities[kSingleQueryMaximumTokens];
  __shared__ float pooled[kSingleQueryHiddenWidth];
  const int64_t seed_offset =
      (batch * readout_count + readout) * kSingleQueryHiddenWidth;
  const int64_t state_offset =
      (readout * batch_count + batch) * kSingleQueryHiddenWidth;
  const int64_t readout_hidden_offset = readout * kSingleQueryHiddenWidth;
  if (head == 0) {
    for (int32_t feature = threadIdx.x; feature < kSingleQueryHiddenWidth;
         feature += blockDim.x) {
      state[state_offset + feature] = seed_products[seed_offset + feature];
    }
  }
  if (threadIdx.x < head_width) {
    const int64_t attention_feature = head * head_width + threadIdx.x;
    float value = static_cast<float>(
        query_bias[readout * attention_width + attention_feature]);
    const int64_t weight_offset =
        (readout * attention_width + attention_feature) *
        kSingleQueryHiddenWidth;
    for (int32_t feature = 0; feature < kSingleQueryHiddenWidth; ++feature) {
      const float seed = static_cast<float>(seed_products[seed_offset + feature]) +
          static_cast<float>(seed_bias[readout_hidden_offset + feature]);
      value += seed * static_cast<float>(query_weight[weight_offset + feature]);
    }
    query[threadIdx.x] = value;
  }
  __syncthreads();

  for (int32_t feature = threadIdx.x; feature < kSingleQueryHiddenWidth;
       feature += blockDim.x) {
    float value = 0.0f;
    for (int64_t head_feature = 0; head_feature < head_width;
         ++head_feature) {
      const int64_t attention_feature = head * head_width + head_feature;
      const int64_t key_weight_offset =
          (readout * (2 * attention_width) + attention_feature) *
          kSingleQueryHiddenWidth + feature;
      value += query[head_feature] *
          static_cast<float>(key_value_weight[key_weight_offset]);
    }
    direction[feature] = value * rsqrtf(static_cast<float>(head_width));
  }
  __syncthreads();

  if (threadIdx.x < token_count) {
    const int64_t token = threadIdx.x;
    const bool valid = single_query_valid(
        valid_mask, batch * token_count + token);
    float score = -INFINITY;
    if (valid) {
      score = 0.0f;
      const int64_t memory_offset =
          (batch * token_count + token) * kSingleQueryHiddenWidth;
      for (int32_t feature = 0; feature < kSingleQueryHiddenWidth; ++feature) {
        score += static_cast<float>(
                     static_cast<scalar_t>(memory[memory_offset + feature])) *
            direction[feature];
      }
    }
    probabilities[token] = score;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int64_t token = 0; token < token_count; ++token) {
      maximum = fmaxf(maximum, probabilities[token]);
    }
    if (isinf(maximum)) {
      for (int64_t token = 0; token < token_count; ++token) {
        probabilities[token] = 0.0f;
      }
    } else {
      float denominator = 0.0f;
      for (int64_t token = 0; token < token_count; ++token) {
        const float probability = expf(probabilities[token] - maximum);
        probabilities[token] = probability;
        denominator += probability;
      }
      for (int64_t token = 0; token < token_count; ++token) {
        probabilities[token] /= denominator;
      }
    }
  }
  __syncthreads();

  for (int32_t feature = threadIdx.x; feature < kSingleQueryHiddenWidth;
       feature += blockDim.x) {
    float value = 0.0f;
    for (int64_t token = 0; token < token_count; ++token) {
      value += probabilities[token] * static_cast<float>(
          static_cast<scalar_t>(memory[
              (batch * token_count + token) * kSingleQueryHiddenWidth +
              feature]));
    }
    pooled[feature] = value;
  }
  __syncthreads();
  if (threadIdx.x < head_width) {
    const int64_t attention_feature = head * head_width + threadIdx.x;
    float value = 0.0f;
    const int64_t value_weight_offset =
        (readout * (2 * attention_width) + attention_width +
            attention_feature) * kSingleQueryHiddenWidth;
    for (int32_t feature = 0; feature < kSingleQueryHiddenWidth; ++feature) {
      value += pooled[feature] *
          static_cast<float>(key_value_weight[value_weight_offset + feature]);
    }
    context[(readout * batch_count + batch) * attention_width +
        attention_feature] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, typename norm_t>
__global__ void single_query_residual_layer_norm_kernel(
    const scalar_t* state, const scalar_t* update, const scalar_t* state_bias,
    const scalar_t* update_bias, const norm_t* norm_weight,
    const norm_t* norm_bias, scalar_t* output, int64_t batch_count,
    int64_t readout_count, int64_t output_batch_stride, float epsilon,
    bool batch_major_output) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int64_t active_rows = batch_count * readout_count;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t readout = row / batch_count;
    const int64_t batch = row - readout * batch_count;
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * kSingleQueryHiddenWidth;
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kSingleQueryHiddenWidth) {
        float value = static_cast<float>(state[row_offset + feature]) +
            static_cast<float>(update[row_offset + feature]) +
            static_cast<float>(update_bias[
                readout * kSingleQueryHiddenWidth + feature]);
        if (state_bias != nullptr) {
          value += static_cast<float>(state_bias[
              readout * kSingleQueryHiddenWidth + feature]);
        }
        values[index] = static_cast<float>(static_cast<scalar_t>(value));
        ++count;
        const float delta = values[index] - mean;
        mean += delta / count;
        moment += delta * (values[index] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(
        __shfl(moment, 0) / kSingleQueryHiddenWidth + epsilon);
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kSingleQueryHiddenWidth) {
        const float normalized =
            (values[index] - mean) * inverse_standard_deviation *
                static_cast<float>(norm_weight[
                    readout * kSingleQueryHiddenWidth + feature]) +
            static_cast<float>(norm_bias[
                readout * kSingleQueryHiddenWidth + feature]);
        const int64_t output_offset = batch_major_output
            ? (readout * output_batch_stride + batch) * kSingleQueryHiddenWidth +
                feature
            : row_offset + feature;
        output[output_offset] = static_cast<scalar_t>(normalized);
      }
    }
  }
}

template <typename scalar_t, typename norm_t>
__global__ void single_query_layer_norm_kernel(const scalar_t* input,
    const norm_t* norm_weight, const norm_t* norm_bias, scalar_t* output,
    int64_t batch_count, int64_t readout_count, float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int64_t active_rows = batch_count * readout_count;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t readout = row / batch_count;
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * kSingleQueryHiddenWidth;
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kSingleQueryHiddenWidth) {
        values[index] = static_cast<float>(input[row_offset + feature]);
        ++count;
        const float delta = values[index] - mean;
        mean += delta / count;
        moment += delta * (values[index] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(
        __shfl(moment, 0) / kSingleQueryHiddenWidth + epsilon);
#pragma unroll
    for (int32_t index = 0; index < kTransformerCachedValuesPerLane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < kSingleQueryHiddenWidth) {
        output[row_offset + feature] = static_cast<scalar_t>(
            (values[index] - mean) * inverse_standard_deviation *
                static_cast<float>(norm_weight[
                    readout * kSingleQueryHiddenWidth + feature]) +
            static_cast<float>(norm_bias[
                readout * kSingleQueryHiddenWidth + feature]));
      }
    }
  }
}

template <typename scalar_t>
__global__ void single_query_bias_silu_kernel(scalar_t* values,
    const scalar_t* bias, int64_t batch_count, int64_t readout_count,
    int64_t width) {
  const int64_t element_count = batch_count * readout_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t readout = index / (batch_count * width);
    const int64_t feature = index % width;
    const float value = static_cast<float>(values[index]) +
        static_cast<float>(bias[readout * width + feature]);
    values[index] = static_cast<scalar_t>(
        value / (1.0f + expf(-value)));
  }
}

constexpr int kIndexedLocalHiddenWidth = 256;
constexpr int kIndexedLocalMaximumTokens = 32;
constexpr int kIndexedLocalAttentionHeads = 4;
constexpr int kIndexedLocalAttentionWidth = 64;
constexpr int kIndexedLocalHeadWidth = 16;

template <typename scalar_t>
struct IndexedLocalDenseInput {
  const scalar_t* data;

  __device__ const scalar_t* row(int64_t dense_row) const {
    return data + dense_row * kIndexedLocalHiddenWidth;
  }
};

template <typename scalar_t>
struct IndexedLocalSegmentedInput {
  const void* data[kMaximumIndexedLocalTransformerSegments];
  int64_t token_offsets[kMaximumIndexedLocalTransformerSegments];
  int64_t token_counts[kMaximumIndexedLocalTransformerSegments];
  int64_t token_count;
  int32_t segment_count;

  __device__ const scalar_t* row(int64_t dense_row) const {
    const int64_t token = dense_row % token_count;
    const int64_t group_row = dense_row / token_count;
    for (int32_t segment = 0; segment < segment_count; ++segment) {
      const int64_t local_token = token - token_offsets[segment];
      if (local_token >= 0 && local_token < token_counts[segment]) {
        const int64_t segment_row = group_row * token_counts[segment] + local_token;
        return static_cast<const scalar_t*>(data[segment]) +
            segment_row * kIndexedLocalHiddenWidth;
      }
    }
    return nullptr;
  }
};

template <typename scalar_t, typename index_t, typename input_t>
__global__ void indexed_local_gather_normalize_kernel(input_t input, const index_t* indices,
    const void* input_norm_weight, const void* input_norm_bias, int32_t input_norm_data_type,
    const void* attention_norm_weight, const void* attention_norm_bias, int32_t attention_norm_data_type,
    scalar_t* output, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const scalar_t* input_row = input.row(dense_row);
    if (input_row == nullptr) {
      continue;
    }
    const int64_t output_offset = dense_row * kIndexedLocalHiddenWidth;
    const int64_t normalized_offset = row * kIndexedLocalHiddenWidth;
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        const float value = static_cast<float>(input_row[feature]);
        values[item] = value;
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / kIndexedLocalHiddenWidth + epsilon);

    float second_mean = 0.0f;
    float second_moment = 0.0f;
    count = 0;
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        const float value = (values[item] - mean) * inverse_standard_deviation *
                                load_fusion_float(input_norm_weight, feature, input_norm_data_type) +
                            load_fusion_float(input_norm_bias, feature, input_norm_data_type);
        const scalar_t rounded = static_cast<scalar_t>(value);
        output[output_offset + feature] = rounded;
        values[item] = static_cast<float>(rounded);
        ++count;
        const float delta = values[item] - second_mean;
        second_mean += delta / count;
        second_moment += delta * (values[item] - second_mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(second_mean, offset);
      const float other_moment = __shfl_down(second_moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - second_mean;
        second_moment += other_moment + delta * delta * count * other_count / combined_count;
        second_mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    second_mean = __shfl(second_mean, 0);
    const float second_inverse_standard_deviation =
        rsqrtf(__shfl(second_moment, 0) / kIndexedLocalHiddenWidth + epsilon);
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        normalized[normalized_offset + feature] =
            static_cast<scalar_t>((values[item] - second_mean) * second_inverse_standard_deviation *
                                      load_fusion_float(attention_norm_weight, feature, attention_norm_data_type) +
                                  load_fusion_float(attention_norm_bias, feature, attention_norm_data_type));
      }
    }
  }
}

template <typename index_t>
__device__ int64_t indexed_local_lower_bound(const index_t* indices, int64_t active_rows, int64_t target) {
  int64_t low = 0;
  int64_t high = active_rows;
  while (low < high) {
    const int64_t middle = low + (high - low) / 2;
    if (static_cast<int64_t>(indices[middle]) < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

template <typename scalar_t, typename index_t>
__global__ void indexed_local_attention_kernel(
    scalar_t* query_key_value, const index_t* indices, int64_t active_rows, int64_t dense_rows, int64_t token_count) {
  const int64_t group = blockIdx.x;
  const int64_t group_count = dense_rows / token_count;
  if (group >= group_count) {
    return;
  }
  __shared__ int64_t group_begin;
  __shared__ int64_t group_end;
  __shared__ float probabilities[kIndexedLocalAttentionHeads * kIndexedLocalMaximumTokens];
  if (threadIdx.x == 0) {
    group_begin = indexed_local_lower_bound(indices, active_rows, group * token_count);
    group_end = indexed_local_lower_bound(indices, active_rows, (group + 1) * token_count);
  }
  __syncthreads();
  const int64_t selected_tokens = group_end - group_begin;
  if (selected_tokens == 0) {
    return;
  }
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const bool active_head = wave < kIndexedLocalAttentionHeads;
  constexpr float scale = 0.25f;
  for (int64_t query = 0; query < selected_tokens; ++query) {
    if (active_head) {
      for (int64_t key = 0; key < selected_tokens; ++key) {
        float product = 0.0f;
        if (lane < kIndexedLocalHeadWidth) {
          const int64_t query_offset =
              (group_begin + query) * (3 * kIndexedLocalAttentionWidth) + wave * kIndexedLocalHeadWidth + lane;
          const int64_t key_offset = (group_begin + key) * (3 * kIndexedLocalAttentionWidth) +
                                     kIndexedLocalAttentionWidth + wave * kIndexedLocalHeadWidth + lane;
          product = static_cast<float>(query_key_value[query_offset]) * static_cast<float>(query_key_value[key_offset]);
        }
        for (int32_t offset = kIndexedLocalHeadWidth / 2; offset > 0; offset /= 2) {
          product += __shfl_down(product, offset, kIndexedLocalHeadWidth);
        }
        if (lane == 0) {
          probabilities[wave * kIndexedLocalMaximumTokens + key] = product * scale;
        }
      }
    }
    __syncthreads();
    if (active_head && lane == 0) {
      float maximum = probabilities[wave * kIndexedLocalMaximumTokens];
      for (int64_t key = 1; key < selected_tokens; ++key) {
        maximum = fmaxf(maximum, probabilities[wave * kIndexedLocalMaximumTokens + key]);
      }
      float denominator = 0.0f;
      for (int64_t key = 0; key < selected_tokens; ++key) {
        const int64_t offset = wave * kIndexedLocalMaximumTokens + key;
        const float value = expf(probabilities[offset] - maximum);
        probabilities[offset] = value;
        denominator += value;
      }
      for (int64_t key = 0; key < selected_tokens; ++key) {
        probabilities[wave * kIndexedLocalMaximumTokens + key] /= denominator;
      }
    }
    __syncthreads();
    if (active_head && lane < kIndexedLocalHeadWidth) {
      float value = 0.0f;
      for (int64_t key = 0; key < selected_tokens; ++key) {
        const int64_t value_offset = (group_begin + key) * (3 * kIndexedLocalAttentionWidth) +
                                     2 * kIndexedLocalAttentionWidth + wave * kIndexedLocalHeadWidth + lane;
        value +=
            probabilities[wave * kIndexedLocalMaximumTokens + key] * static_cast<float>(query_key_value[value_offset]);
      }
      query_key_value[(group_begin + query) * (3 * kIndexedLocalAttentionWidth) + wave * kIndexedLocalHeadWidth +
                      lane] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename index_t>
__global__ void indexed_local_residual_normalize_kernel(scalar_t* output, const scalar_t* update,
    const index_t* indices, const scalar_t* update_bias, const void* norm_weight, const void* norm_bias,
    int32_t norm_data_type, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t compact_offset = row * kIndexedLocalHiddenWidth;
    const int64_t output_offset = dense_row * kIndexedLocalHiddenWidth;
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        const float value = static_cast<float>(output[output_offset + feature]) +
                            static_cast<float>(update[compact_offset + feature]) +
                            static_cast<float>(update_bias[feature]);
        const scalar_t rounded = static_cast<scalar_t>(value);
        output[output_offset + feature] = rounded;
        values[item] = static_cast<float>(rounded);
        ++count;
        const float delta = values[item] - mean;
        mean += delta / count;
        moment += delta * (values[item] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / kIndexedLocalHiddenWidth + epsilon);
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        normalized[compact_offset + feature] =
            static_cast<scalar_t>((values[item] - mean) * inverse_standard_deviation *
                                      load_fusion_float(norm_weight, feature, norm_data_type) +
                                  load_fusion_float(norm_bias, feature, norm_data_type));
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void indexed_local_finalize_kernel(scalar_t* output, const scalar_t* update, const index_t* indices,
    const scalar_t* update_bias, const void* norm_weight, const void* norm_bias, int32_t norm_data_type,
    int64_t active_offset, int64_t active_rows, int64_t dense_rows, float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[kTransformerCachedValuesPerLane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t compact_offset = row * kIndexedLocalHiddenWidth;
    const int64_t output_offset = dense_row * kIndexedLocalHiddenWidth;
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        const float value = static_cast<float>(output[output_offset + feature]) +
                            static_cast<float>(update[compact_offset + feature]) +
                            static_cast<float>(update_bias[feature]);
        values[item] = static_cast<float>(static_cast<scalar_t>(value));
        ++count;
        const float delta = values[item] - mean;
        mean += delta / count;
        moment += delta * (values[item] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / kIndexedLocalHiddenWidth + epsilon);
#pragma unroll
    for (int32_t item = 0; item < kTransformerCachedValuesPerLane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < kIndexedLocalHiddenWidth) {
        output[output_offset + feature] =
            static_cast<scalar_t>((values[item] - mean) * inverse_standard_deviation *
                                      load_fusion_float(norm_weight, feature, norm_data_type) +
                                  load_fusion_float(norm_bias, feature, norm_data_type));
      }
    }
  }
}

void CheckHipLaunch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ", hipGetErrorString(error));
}

}  // namespace

template <typename output_t>
void LaunchOutputPackTyped(const OutputPackArguments& arguments,
    int32_t source_count, uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((output_pack_kernel<output_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream,
      arguments);
}

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  OutputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion output-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      LaunchOutputPackTyped<c10::Half>(
          arguments, source_count, block_count, stream.stream());
      break;
    case torch::kBFloat16:
      LaunchOutputPackTyped<c10::BFloat16>(
          arguments, source_count, block_count, stream.stream());
      break;
    case torch::kFloat32:
      LaunchOutputPackTyped<float>(
          arguments, source_count, block_count, stream.stream());
      break;
    default:
      TORCH_CHECK(false,
          "fusion output pack requires FLOAT16, BFLOAT16, or FLOAT32 output");
  }
  CheckHipLaunch("fusion output pack");
}

template <typename scalar_t>
void LaunchSegmentedOutputPackTyped(
    const SegmentedOutputPackArguments& arguments, int32_t source_count,
    uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((segmented_output_pack_kernel<scalar_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream,
      arguments);
}

void LaunchSegmentedOutputPack(const SegmentedOutputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  SegmentedOutputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    arguments.source_prefix_counts[source_index] = source.source_prefix_count;
    arguments.source_token_counts[source_index] = source.source_token_count;
    arguments.token_offsets[source_index] = source.token_offset;
    arguments.token_counts[source_index] = source.token_count;
    arguments.hidden_widths[source_index] = source.hidden_width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    const int64_t width = CheckedMultiply(
        CheckedMultiply(source.source_prefix_count, source.token_count,
            "fusion segmented output-pack source tokens"),
        source.hidden_width, "fusion segmented output-pack source width");
    maximum_width = std::max(maximum_width, width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion segmented output-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      LaunchSegmentedOutputPackTyped<c10::Half>(
          arguments, source_count, block_count, stream.stream());
      break;
    case torch::kBFloat16:
      LaunchSegmentedOutputPackTyped<c10::BFloat16>(
          arguments, source_count, block_count, stream.stream());
      break;
    case torch::kFloat32:
      LaunchSegmentedOutputPackTyped<float>(
          arguments, source_count, block_count, stream.stream());
      break;
    default:
      TORCH_CHECK(false,
          "segmented output pack supports FLOAT16, BFLOAT16, and FLOAT32");
  }
  CheckHipLaunch("fusion segmented output pack");
}

template <typename baseline_t, typename selected_t>
void LaunchBinaryBranchBlendTyped(const BinaryBranchBlendArguments& arguments,
    uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((binary_branch_blend_kernel<baseline_t, selected_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchBinaryBranchBlend(const torch::Tensor& baseline_context,
    const torch::Tensor& selected_context,
    const torch::Tensor& selected_logit,
    const torch::Tensor& baseline_presence,
    const torch::Tensor& selected_presence, torch::Tensor& output,
    int64_t row_count, int64_t width) {
  if (row_count == 0) {
    return;
  }
  BinaryBranchBlendArguments arguments{};
  arguments.baseline_context = baseline_context.data_ptr();
  arguments.selected_context = selected_context.data_ptr();
  arguments.selected_logit = selected_logit.data_ptr();
  arguments.baseline_presence = baseline_presence.data_ptr();
  arguments.selected_presence = selected_presence.data_ptr();
  arguments.output = output.data_ptr<float>();
  arguments.row_count = row_count;
  arguments.width = width;
  arguments.selected_logit_data_type =
      static_cast<int32_t>(selected_logit.scalar_type());
  arguments.baseline_presence_data_type =
      static_cast<int32_t>(baseline_presence.scalar_type());
  arguments.selected_presence_data_type =
      static_cast<int32_t>(selected_presence.scalar_type());
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(row_count, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
#define LAUNCH_BINARY_BRANCH_BLEND(baseline_type, selected_type) \
  LaunchBinaryBranchBlendTyped<baseline_type, selected_type>( \
      arguments, block_count, stream.stream())
  switch (baseline_context.scalar_type()) {
    case torch::kFloat16:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, float);
      }
      break;
    case torch::kBFloat16:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, float);
      }
      break;
    case torch::kFloat32:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(float, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(float, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(float, float);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported binary branch baseline data type");
  }
#undef LAUNCH_BINARY_BRANCH_BLEND
  CheckHipLaunch("fusion binary branch blend");
}

void LaunchAffineInputPack(const AffineInputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  AffineInputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion affine input-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      hipLaunchKernelGGL(affine_input_pack_kernel<c10::Half>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream.stream(), arguments);
      break;
    case torch::kBFloat16:
      hipLaunchKernelGGL(affine_input_pack_kernel<c10::BFloat16>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream.stream(), arguments);
      break;
    case torch::kFloat32:
      hipLaunchKernelGGL(affine_input_pack_kernel<float>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream.stream(), arguments);
      break;
    default:
      TORCH_CHECK(false, "unsupported affine input-pack data type");
  }
  CheckHipLaunch("fusion affine input pack");
}

void LaunchAffineFinalize(const AffineSumSource* sources, int32_t source_count,
    const void* bias, torch::Tensor& output, int64_t batch_count,
    const int64_t* output_prefix, int32_t output_prefix_rank,
    int64_t output_width, AffineActivation activation) {
  if (batch_count == 0) {
    return;
  }
  AffineFinalizeArguments arguments{};
  arguments.bias = bias;
  arguments.output = output.data_ptr();
  arguments.batch_count = batch_count;
  arguments.output_width = output_width;
  arguments.source_count = source_count;
  arguments.output_prefix_rank = output_prefix_rank;
  arguments.activation = static_cast<int32_t>(activation);
  arguments.output_prefix_rows = 1;
  for (int32_t axis = 0; axis < output_prefix_rank; ++axis) {
    arguments.output_prefix[axis] = output_prefix[axis];
    arguments.output_prefix_rows = CheckedMultiply(arguments.output_prefix_rows,
        output_prefix[axis], "fusion affine output prefix size");
  }
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    arguments.sources[source_index] = sources[source_index].data;
    arguments.source_rows_per_batch[source_index] =
        sources[source_index].rows_per_batch;
    for (int32_t axis = 0; axis < output_prefix_rank; ++axis) {
      arguments.source_strides[source_index][axis] =
          sources[source_index].output_strides[axis];
    }
  }

  const int64_t output_rows = CheckedMultiply(batch_count,
      arguments.output_prefix_rows, "fusion affine output row count");
  const int64_t element_count = CheckedMultiply(
      output_rows, output_width, "fusion affine launch element count");
  const int64_t required_blocks = RequiredBlockCount(element_count);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      hipLaunchKernelGGL(affine_finalize_kernel<c10::Half>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream.stream(), arguments);
      break;
    case torch::kBFloat16:
      hipLaunchKernelGGL(affine_finalize_kernel<c10::BFloat16>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream.stream(), arguments);
      break;
    case torch::kFloat32:
      hipLaunchKernelGGL(affine_finalize_kernel<float>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream.stream(), arguments);
      break;
    default:
      TORCH_CHECK(false, "unsupported affine finalize data type");
  }
  CheckHipLaunch("fusion affine finalize");
}

template <typename output_t, typename index_t>
void LaunchIndexedAffineInputPackTyped(const IndexedAffineInputPackArguments& arguments,
    uint32_t block_count, int32_t source_count, hipStream_t stream) {
  hipLaunchKernelGGL((indexed_affine_input_pack_kernel<output_t, index_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchIndexedAffineInputPack(const IndexedAffineSource* sources,
    int32_t source_count, const void* indices, torch::ScalarType index_data_type,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t output_width) {
  if (active_rows == 0) {
    return;
  }
  IndexedAffineInputPackArguments arguments{};
  arguments.indices = indices;
  arguments.output = output.data_ptr();
  arguments.active_rows = active_rows;
  arguments.destination_rows = destination_rows;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.index_divisors[source_index] = source.index_divisor;
    arguments.source_rows[source_index] = source.row_count;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      active_rows, maximum_width, "fusion indexed-affine input-pack element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
#define LAUNCH_INDEXED_PACK(output_type, index_type) \
  LaunchIndexedAffineInputPackTyped<output_type, index_type>( \
      arguments, block_count, source_count, stream.stream())
  switch (output.scalar_type()) {
    case torch::kFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(c10::Half, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(c10::Half, int64_t);
      }
      break;
    case torch::kBFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(c10::BFloat16, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(c10::BFloat16, int64_t);
      }
      break;
    case torch::kFloat32:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(float, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(float, int64_t);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported indexed-affine input-pack data type");
  }
#undef LAUNCH_INDEXED_PACK
  CheckHipLaunch("fusion indexed-affine input pack");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedAffineFinalizeTyped(const IndexedAffineFinalizeArguments& arguments,
    uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((indexed_affine_finalize_kernel<scalar_t, index_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchIndexedAffineFinalize(const void* indices,
    torch::ScalarType index_data_type, const torch::Tensor& hidden,
    const void* hidden_bias, const void* output_weight, const void* output_bias,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t hidden_width, int64_t output_width, AffineActivation activation) {
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  const int64_t output_elements = CheckedMultiply(destination_rows, output_width,
      "fusion indexed-affine output element count");
  const int64_t output_bytes = CheckedMultiply(output_elements,
      static_cast<int64_t>(output.element_size()),
      "fusion indexed-affine output byte count");
  if (output_bytes != 0) {
    const hipError_t memset_error = hipMemsetAsync(
        output.data_ptr(), 0, static_cast<std::size_t>(output_bytes), stream.stream());
    TORCH_CHECK(memset_error == hipSuccess, "fusion indexed-affine output clear failed: ",
        hipGetErrorString(memset_error));
  }
  if (active_rows == 0) {
    return;
  }

  IndexedAffineFinalizeArguments arguments{};
  arguments.indices = indices;
  arguments.hidden = hidden.data_ptr();
  arguments.hidden_bias = hidden_bias;
  arguments.output_weight = output_weight;
  arguments.output_bias = output_bias;
  arguments.output = output.data_ptr();
  arguments.active_rows = active_rows;
  arguments.destination_rows = destination_rows;
  arguments.hidden_width = hidden_width;
  arguments.output_width = output_width;
  arguments.activation = static_cast<int32_t>(activation);
  const int64_t active_elements = CheckedMultiply(active_rows, output_width,
      "fusion indexed-affine active output element count");
  constexpr int64_t maximum_wavefront_size = 64;
  const int64_t minimum_waves_per_block = kThreadsPerBlock / maximum_wavefront_size;
  const int64_t required_blocks = active_elements / minimum_waves_per_block +
      (active_elements % minimum_waves_per_block == 0 ? 0 : 1);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
#define LAUNCH_INDEXED_FINALIZE(scalar_type, index_type) \
  LaunchIndexedAffineFinalizeTyped<scalar_type, index_type>( \
      arguments, block_count, stream.stream())
  switch (output.scalar_type()) {
    case torch::kFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(c10::Half, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(c10::Half, int64_t);
      }
      break;
    case torch::kBFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(c10::BFloat16, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(c10::BFloat16, int64_t);
      }
      break;
    case torch::kFloat32:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(float, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(float, int64_t);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported indexed-affine finalize data type");
  }
#undef LAUNCH_INDEXED_FINALIZE
  CheckHipLaunch("fusion indexed-affine finalize");
}

template <typename scalar_t, typename norm_t>
void LaunchTransformerCopyAndLayerNormTyped(const torch::Tensor& input,
    torch::Tensor& state, torch::Tensor& normalized,
    const torch::Tensor& weight, const torch::Tensor& bias,
    int64_t active_rows, float epsilon, bool copy_input,
    hipStream_t stream) {
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
  hipLaunchKernelGGL(
      (transformer_copy_layer_norm_kernel<scalar_t, norm_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      input.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
      normalized.data_ptr<scalar_t>(), weight.data_ptr<norm_t>(),
      bias.data_ptr<norm_t>(), active_rows, epsilon, copy_input);
}

void LaunchTransformerCopyAndLayerNorm(const torch::Tensor& input,
    torch::Tensor& state, torch::Tensor& normalized,
    const torch::Tensor& weight, const torch::Tensor& bias,
    int64_t batch_count, int64_t token_count, int64_t hidden_width,
    float epsilon, bool copy_input) {
  TORCH_CHECK(hidden_width == kTransformerHiddenWidth,
      "transformer encoder lowering requires hidden width 256");
  const int64_t active_rows = CheckedMultiply(
      batch_count, token_count, "transformer LayerNorm row count");
  if (active_rows == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(state.get_device());
#define LAUNCH_TRANSFORMER_COPY_LAYER_NORM(scalar_type, norm_type) \
  LaunchTransformerCopyAndLayerNormTyped<scalar_type, norm_type>( \
      input, state, normalized, weight, bias, active_rows, epsilon, \
      copy_input, stream.stream())
  if (input.scalar_type() == torch::kFloat16) {
    if (weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::Half, float);
    } else {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::Half, c10::Half);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_COPY_LAYER_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_COPY_LAYER_NORM
  CheckHipLaunch("fusion transformer copy LayerNorm");
}

template <typename scalar_t>
void LaunchTransformerAttentionTyped(const torch::Tensor& query_key_value,
    torch::Tensor& context, int64_t batch_count, int64_t token_count,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      (transformer_attention_kernel<scalar_t>),
      dim3(batch_count), dim3(kThreadsPerBlock), 0, stream,
      query_key_value.data_ptr<scalar_t>(), context.data_ptr<scalar_t>(),
      batch_count, token_count);
}

void LaunchTransformerAttention(const torch::Tensor& query_key_value,
    torch::Tensor& context, int64_t batch_count, int64_t token_count,
    int64_t attention_heads, int64_t attention_width, int64_t hidden_width) {
  TORCH_CHECK(token_count > 0 && token_count <= kTransformerMaximumTokens,
      "transformer encoder lowering supports at most eight tokens");
  TORCH_CHECK(attention_heads == kTransformerAttentionHeads &&
          attention_width == kTransformerAttentionWidth &&
          hidden_width == kTransformerHiddenWidth,
      "transformer encoder lowering requires heads=4, attention=128, hidden=256");
  if (batch_count == 0) {
    return;
  }
  const auto stream =
      at::hip::getCurrentHIPStream(query_key_value.get_device());
#define LAUNCH_TRANSFORMER_ATTENTION(scalar_type) \
  LaunchTransformerAttentionTyped<scalar_type>(query_key_value, context, \
      batch_count, token_count, stream.stream())
  if (query_key_value.scalar_type() == torch::kFloat16) {
    LAUNCH_TRANSFORMER_ATTENTION(c10::Half);
  } else if (query_key_value.scalar_type() == torch::kBFloat16) {
    LAUNCH_TRANSFORMER_ATTENTION(c10::BFloat16);
  } else if (query_key_value.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_ATTENTION(float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_ATTENTION
  CheckHipLaunch("fusion transformer attention");
}

template <typename scalar_t>
void LaunchIndexedRelationAttentionBiasTyped(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_ids,
    const torch::Tensor& pair_bias, torch::Tensor& attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale,
    uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL(
      (indexed_relation_attention_bias_kernel<scalar_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      relation_logits.data_ptr<scalar_t>(), relation_ids.data_ptr<int16_t>(),
      pair_bias.data_ptr<scalar_t>(), attention_bias.data_ptr<scalar_t>(),
      batch_count, attention_heads, token_count, relation_count,
      padded_token_count, scale);
}

void LaunchIndexedRelationAttentionBias(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_ids,
    const torch::Tensor& pair_bias, torch::Tensor& attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale) {
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt16,
      "indexed-relation IDs must use INT16");
  TORCH_CHECK(relation_logits.scalar_type() == pair_bias.scalar_type() &&
          relation_logits.scalar_type() == attention_bias.scalar_type(),
      "indexed-relation attention tensors must use one data type");
  const int64_t element_count = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(batch_count, attention_heads,
              "indexed-relation attention bias elements"),
          token_count, "indexed-relation attention bias elements"),
      padded_token_count, "indexed-relation attention bias elements");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = at::hip::getCurrentHIPStream(attention_bias.get_device());
#define LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(scalar_type) \
  LaunchIndexedRelationAttentionBiasTyped<scalar_type>(relation_logits, \
      relation_ids, pair_bias, attention_bias, batch_count, attention_heads, \
      token_count, relation_count, padded_token_count, scale, block_count, \
      stream.stream())
  if (attention_bias.scalar_type() == torch::kFloat16) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(c10::Half);
  } else if (attention_bias.scalar_type() == torch::kBFloat16) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(c10::BFloat16);
  } else if (attention_bias.scalar_type() == torch::kFloat32) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(float);
  } else {
    TORCH_CHECK(false, "unsupported indexed-relation attention data type");
  }
#undef LAUNCH_INDEXED_RELATION_ATTENTION_BIAS
  CheckHipLaunch("fusion indexed-relation attention bias");
}

void LaunchTransformerBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t active_rows, int64_t width,
    bool round_bias_before_activation) {
  const int64_t element_count = CheckedMultiply(
      active_rows, width, "transformer SiLU element count");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = at::hip::getCurrentHIPStream(values.get_device());
#define LAUNCH_TRANSFORMER_SILU(scalar_type) \
  hipLaunchKernelGGL((transformer_bias_silu_kernel<scalar_type>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream.stream(), \
      values.data_ptr<scalar_type>(), bias.data_ptr<scalar_type>(), \
      element_count, width, round_bias_before_activation)
  if (values.scalar_type() == torch::kFloat16) {
    LAUNCH_TRANSFORMER_SILU(c10::Half);
  } else if (values.scalar_type() == torch::kBFloat16) {
    LAUNCH_TRANSFORMER_SILU(c10::BFloat16);
  } else if (values.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_SILU(float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_SILU
  CheckHipLaunch("fusion transformer bias SiLU");
}

template <typename scalar_t, typename norm_t>
void LaunchTransformerResidualLayerNormTyped(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, float epsilon,
    bool round_update_before_residual, hipStream_t stream) {
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
  hipLaunchKernelGGL(
      (transformer_residual_layer_norm_kernel<scalar_t, norm_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      state.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
      update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(),
      norm_bias.data_ptr<norm_t>(), normalized.data_ptr<scalar_t>(),
      active_rows, epsilon, round_update_before_residual);
}

void LaunchTransformerResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, int64_t hidden_width,
    float epsilon, bool round_update_before_residual) {
  TORCH_CHECK(hidden_width == kTransformerHiddenWidth,
      "transformer encoder lowering requires hidden width 256");
  if (active_rows == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(state.get_device());
#define LAUNCH_TRANSFORMER_RESIDUAL_NORM(scalar_type, norm_type) \
  LaunchTransformerResidualLayerNormTyped<scalar_type, norm_type>( \
      state, update, update_bias, norm_weight, norm_bias, normalized, \
      active_rows, epsilon, round_update_before_residual, stream.stream())
  if (state.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::Half, float);
    } else {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::Half, c10::Half);
    }
  } else if (state.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (state.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_RESIDUAL_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_RESIDUAL_NORM
  CheckHipLaunch("fusion transformer residual LayerNorm");
}

template <typename memory_t, typename scalar_t, typename mask_t>
void LaunchSingleQueryReadoutSeedInputTyped(const torch::Tensor& memory,
    const torch::Tensor& query_source, const torch::Tensor& valid_mask,
    torch::Tensor& seed_input, int64_t batch_count, int64_t token_count,
    int64_t query_token_count, int64_t query_index, hipStream_t stream) {
  hipLaunchKernelGGL(
      (single_query_seed_input_kernel<memory_t, scalar_t, mask_t>),
      dim3(batch_count), dim3(kThreadsPerBlock), 0, stream,
      memory.data_ptr<memory_t>(), query_source.data_ptr<memory_t>(),
      valid_mask.data_ptr<mask_t>(), seed_input.data_ptr<scalar_t>(),
      batch_count, token_count, query_token_count, query_index);
}

void LaunchSingleQueryReadoutSeedInput(const torch::Tensor& memory,
    const torch::Tensor& query_source, const torch::Tensor& valid_mask,
    torch::Tensor& seed_input, int64_t batch_count, int64_t token_count,
    int64_t hidden_width, int64_t query_token_count, int64_t query_index) {
  TORCH_CHECK(hidden_width == kSingleQueryHiddenWidth && token_count > 0 &&
          token_count <= kSingleQueryMaximumTokens && query_token_count > 0 &&
          query_index >= 0 && query_index < query_token_count,
      "single-query readout seed lowering requires hidden=256 and tokens<=256");
  if (batch_count == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(memory.get_device());
#define LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, mask_type) \
  LaunchSingleQueryReadoutSeedInputTyped<memory_type, compute_t, mask_type>( \
      memory, query_source, valid_mask, seed_input, batch_count, token_count, \
      query_token_count, query_index, stream.stream())
#define DISPATCH_SINGLE_QUERY_SEED_MASK(memory_type, compute_t) \
  switch (valid_mask.scalar_type()) { \
    case torch::kBool: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, bool); \
      break; \
    case torch::kUInt8: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, uint8_t); \
      break; \
    case torch::kFloat16: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout mask data type"); \
  }
#define DISPATCH_SINGLE_QUERY_SEED_MEMORY(compute_type) \
  switch (memory.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(c10::Half, compute_type); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(c10::BFloat16, compute_type); \
      break; \
    case torch::kFloat32: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(float, compute_type); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout memory data type"); \
  }
  if (seed_input.scalar_type() == torch::kFloat16) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(c10::Half);
  } else if (seed_input.scalar_type() == torch::kBFloat16) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(c10::BFloat16);
  } else if (seed_input.scalar_type() == torch::kFloat32) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout compute data type");
  }
#undef DISPATCH_SINGLE_QUERY_SEED_MEMORY
#undef DISPATCH_SINGLE_QUERY_SEED_MASK
#undef LAUNCH_SINGLE_QUERY_SEED
  CheckHipLaunch("fusion single-query readout seed input");
}

template <typename memory_t, typename scalar_t, typename mask_t>
void LaunchSingleQueryReadoutAttentionTyped(
    const torch::Tensor& seed_products, const torch::Tensor& seed_bias,
    const torch::Tensor& memory, const torch::Tensor& valid_mask,
    const torch::Tensor& query_weight, const torch::Tensor& query_bias,
    const torch::Tensor& key_value_weight, torch::Tensor& state,
    torch::Tensor& context, int64_t batch_count, int64_t readout_count,
    int64_t token_count, int64_t attention_heads, int64_t attention_width,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      (single_query_attention_kernel<memory_t, scalar_t, mask_t>),
      dim3(batch_count, readout_count * attention_heads),
      dim3(kThreadsPerBlock), 0, stream,
      seed_products.data_ptr<scalar_t>(), seed_bias.data_ptr<scalar_t>(),
      memory.data_ptr<memory_t>(), valid_mask.data_ptr<mask_t>(),
      query_weight.data_ptr<scalar_t>(), query_bias.data_ptr<scalar_t>(),
      key_value_weight.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
      context.data_ptr<scalar_t>(), batch_count, readout_count, token_count,
      attention_heads, attention_width);
}

void LaunchSingleQueryReadoutAttention(const torch::Tensor& seed_products,
    const torch::Tensor& seed_bias, const torch::Tensor& memory,
    const torch::Tensor& valid_mask, const torch::Tensor& query_weight,
    const torch::Tensor& query_bias, const torch::Tensor& key_value_weight,
    torch::Tensor& state, torch::Tensor& context, int64_t batch_count,
    int64_t readout_count, int64_t token_count, int64_t hidden_width,
    int64_t attention_heads, int64_t attention_width) {
  TORCH_CHECK(hidden_width == kSingleQueryHiddenWidth && token_count > 0 &&
          token_count <= kSingleQueryMaximumTokens && attention_heads > 0 &&
          attention_width > 0 && attention_width % attention_heads == 0 &&
          attention_width / attention_heads <= kSingleQueryMaximumHeadWidth,
      "single-query readout attention metadata is unsupported");
  if (batch_count == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(memory.get_device());
#define LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, mask_type) \
  LaunchSingleQueryReadoutAttentionTyped<memory_type, compute_t, mask_type>( \
      seed_products, seed_bias, memory, valid_mask, query_weight, query_bias, \
      key_value_weight, state, context, batch_count, readout_count, token_count, \
      attention_heads, attention_width, stream.stream())
#define DISPATCH_SINGLE_QUERY_ATTENTION_MASK(memory_type, compute_t) \
  switch (valid_mask.scalar_type()) { \
    case torch::kBool: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, bool); \
      break; \
    case torch::kUInt8: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, uint8_t); \
      break; \
    case torch::kFloat16: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout mask data type"); \
  }
#define DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(compute_type) \
  switch (memory.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(c10::Half, compute_type); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(c10::BFloat16, compute_type); \
      break; \
    case torch::kFloat32: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(float, compute_type); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout memory data type"); \
  }
  if (state.scalar_type() == torch::kFloat16) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(c10::Half);
  } else if (state.scalar_type() == torch::kBFloat16) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(c10::BFloat16);
  } else if (state.scalar_type() == torch::kFloat32) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout compute data type");
  }
#undef DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY
#undef DISPATCH_SINGLE_QUERY_ATTENTION_MASK
#undef LAUNCH_SINGLE_QUERY_ATTENTION
  CheckHipLaunch("fusion single-query readout attention");
}

template <typename scalar_t, typename norm_t>
void LaunchSingleQueryReadoutResidualLayerNormTyped(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor* state_bias,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output,
    int64_t batch_count, int64_t readout_count, int64_t output_batch_stride,
    float epsilon, bool batch_major_output, hipStream_t stream) {
  const int64_t active_rows = batch_count * readout_count;
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
  hipLaunchKernelGGL(
      (single_query_residual_layer_norm_kernel<scalar_t, norm_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      state.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
      state_bias == nullptr ? nullptr : state_bias->data_ptr<scalar_t>(),
      update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(),
      norm_bias.data_ptr<norm_t>(), output.data_ptr<scalar_t>(), batch_count,
      readout_count, output_batch_stride, epsilon, batch_major_output);
}

void LaunchSingleQueryReadoutResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor* state_bias,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output,
    int64_t batch_count, int64_t readout_count, int64_t hidden_width,
    int64_t output_batch_stride, float epsilon, bool batch_major_output) {
  TORCH_CHECK(hidden_width == kSingleQueryHiddenWidth,
      "single-query readout residual LayerNorm requires hidden width 256");
  if (batch_count == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(state.get_device());
#define LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(scalar_type, norm_type) \
  LaunchSingleQueryReadoutResidualLayerNormTyped<scalar_type, norm_type>( \
      state, update, state_bias, update_bias, norm_weight, norm_bias, output, \
      batch_count, readout_count, output_batch_stride, epsilon, \
      batch_major_output, stream.stream())
  if (state.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::Half, float);
    } else {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::Half, c10::Half);
    }
  } else if (state.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (state.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_RESIDUAL_NORM
  CheckHipLaunch("fusion single-query readout residual LayerNorm");
}

template <typename scalar_t, typename norm_t>
void LaunchSingleQueryReadoutLayerNormTyped(const torch::Tensor& input,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    float epsilon, hipStream_t stream) {
  const int64_t active_rows = batch_count * readout_count;
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
  hipLaunchKernelGGL(
      (single_query_layer_norm_kernel<scalar_t, norm_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      input.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(),
      norm_bias.data_ptr<norm_t>(), output.data_ptr<scalar_t>(), batch_count,
      readout_count, epsilon);
}

void LaunchSingleQueryReadoutLayerNorm(const torch::Tensor& input,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width == kSingleQueryHiddenWidth,
      "single-query readout LayerNorm requires hidden width 256");
  if (batch_count == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(input.get_device());
#define LAUNCH_SINGLE_QUERY_NORM(scalar_type, norm_type) \
  LaunchSingleQueryReadoutLayerNormTyped<scalar_type, norm_type>( \
      input, norm_weight, norm_bias, output, batch_count, readout_count, \
      epsilon, stream.stream())
  if (input.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_NORM(c10::Half, float);
    } else {
      LAUNCH_SINGLE_QUERY_NORM(c10::Half, c10::Half);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_SINGLE_QUERY_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_NORM
  CheckHipLaunch("fusion single-query readout LayerNorm");
}

void LaunchSingleQueryReadoutBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t batch_count, int64_t readout_count,
    int64_t width) {
  const int64_t element_count = CheckedMultiply(
      CheckedMultiply(batch_count, readout_count,
          "single-query readout SiLU row count"),
      width, "single-query readout SiLU element count");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = at::hip::getCurrentHIPStream(values.get_device());
#define LAUNCH_SINGLE_QUERY_SILU(scalar_type) \
  hipLaunchKernelGGL((single_query_bias_silu_kernel<scalar_type>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream.stream(), \
      values.data_ptr<scalar_type>(), bias.data_ptr<scalar_type>(), \
      batch_count, readout_count, width)
  if (values.scalar_type() == torch::kFloat16) {
    LAUNCH_SINGLE_QUERY_SILU(c10::Half);
  } else if (values.scalar_type() == torch::kBFloat16) {
    LAUNCH_SINGLE_QUERY_SILU(c10::BFloat16);
  } else if (values.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_SILU(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_SILU
  CheckHipLaunch("fusion single-query readout bias SiLU");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalGatherNormalizeTyped(const torch::Tensor& input, const torch::Tensor& indices,
    const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, float epsilon,
    uint32_t block_count, hipStream_t stream) {
  const IndexedLocalDenseInput<scalar_t> dense_input{input.data_ptr<scalar_t>()};
  hipLaunchKernelGGL((indexed_local_gather_normalize_kernel<scalar_t, index_t,
                         IndexedLocalDenseInput<scalar_t>>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, dense_input, indices.data_ptr<index_t>(),
      input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()),
      attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),
      static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),
      normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, epsilon);
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalGatherNormalizeSegmentsTyped(
    const IndexedLocalTransformerInputSegment* input_segments, int32_t input_segment_count,
    const torch::Tensor& indices, const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t token_count,
    float epsilon, uint32_t block_count, hipStream_t stream) {
  IndexedLocalSegmentedInput<scalar_t> segmented_input{};
  segmented_input.segment_count = input_segment_count;
  segmented_input.token_count = token_count;
  for (int32_t segment = 0; segment < input_segment_count; ++segment) {
    segmented_input.data[segment] = input_segments[segment].data;
    segmented_input.token_offsets[segment] = input_segments[segment].token_offset;
    segmented_input.token_counts[segment] = input_segments[segment].token_count;
  }
  hipLaunchKernelGGL((indexed_local_gather_normalize_kernel<scalar_t, index_t,
                         IndexedLocalSegmentedInput<scalar_t>>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, segmented_input, indices.data_ptr<index_t>(),
      input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()),
      attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),
      static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),
      normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, epsilon);
}

void LaunchIndexedLocalTransformerClear(torch::Tensor& output, int64_t dense_rows, int64_t hidden_width) {
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  const int64_t output_bytes =
      CheckedMultiply(CheckedMultiply(dense_rows, hidden_width, "indexed local transformer output elements"),
          static_cast<int64_t>(output.element_size()), "indexed local transformer output bytes");
  const hipError_t error =
      hipMemsetAsync(output.data_ptr(), 0, static_cast<std::size_t>(output_bytes), stream.stream());
  TORCH_CHECK(error == hipSuccess, "indexed local transformer output clear failed: ", hipGetErrorString(error));
}

void LaunchIndexedLocalTransformerGatherNormalize(const torch::Tensor& input, const torch::Tensor& indices,
    torch::ScalarType index_data_type, const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t hidden_width,
    float epsilon) {
  TORCH_CHECK(hidden_width == kIndexedLocalHiddenWidth, "indexed local transformer requires hidden width 256");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = at::hip::getCurrentHIPStream(input.get_device());
#define LAUNCH_INDEXED_LOCAL_GATHER(scalar_type, index_type)                                                           \
  LaunchIndexedLocalGatherNormalizeTyped<scalar_type, index_type>(input, indices, input_norm_weight, input_norm_bias,  \
      attention_norm_weight, attention_norm_bias, output, normalized, active_offset, active_rows, dense_rows, epsilon, \
      block_count, stream.stream())
  if (input.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::Half, int64_t);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::BFloat16, int64_t);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_GATHER
  CheckHipLaunch("fusion indexed local transformer gather LayerNorm");
}

void LaunchIndexedLocalTransformerGatherNormalizeSegments(
    const IndexedLocalTransformerInputSegment* input_segments, int32_t input_segment_count,
    torch::ScalarType input_data_type, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t group_count,
    int64_t token_count, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(input_segment_count > 1 && input_segment_count <= kMaximumIndexedLocalTransformerSegments,
      "segmented indexed local transformer requires two to eight input segments");
  TORCH_CHECK(group_count > 0 && token_count > 0 && hidden_width == kIndexedLocalHiddenWidth,
      "segmented indexed local transformer metadata is invalid");
  int64_t next_token_offset = 0;
  for (int32_t segment = 0; segment < input_segment_count; ++segment) {
    TORCH_CHECK(input_segments[segment].data != nullptr &&
            input_segments[segment].token_offset == next_token_offset &&
            input_segments[segment].token_count > 0,
        "segmented indexed local transformer input metadata is invalid");
    next_token_offset += input_segments[segment].token_count;
  }
  TORCH_CHECK(next_token_offset == token_count,
      "segmented indexed local transformer token count does not match its inputs");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
#define LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(scalar_type, index_type)                               \
  LaunchIndexedLocalGatherNormalizeSegmentsTyped<scalar_type, index_type>(                           \
      input_segments, input_segment_count, indices, input_norm_weight, input_norm_bias,              \
      attention_norm_weight, attention_norm_bias, output, normalized, active_offset, active_rows,   \
      dense_rows, token_count, epsilon, block_count, stream.stream())
  if (input_data_type == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::Half, int64_t);
    }
  } else if (input_data_type == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::BFloat16, int64_t);
    }
  } else if (input_data_type == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported segmented indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER
  CheckHipLaunch("fusion segmented indexed local transformer gather LayerNorm");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalAttentionTyped(torch::Tensor& query_key_value, const torch::Tensor& indices, int64_t active_rows,
    int64_t dense_rows, int64_t token_count, uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((indexed_local_attention_kernel<scalar_t, index_t>), dim3(block_count), dim3(kThreadsPerBlock), 0,
      stream, query_key_value.data_ptr<scalar_t>(), indices.data_ptr<index_t>(), active_rows, dense_rows, token_count);
}

void LaunchIndexedLocalTransformerAttention(torch::Tensor& query_key_value, const torch::Tensor& indices,
    torch::ScalarType index_data_type, int64_t active_rows, int64_t dense_rows, int64_t token_count,
    int64_t attention_heads, int64_t attention_width) {
  TORCH_CHECK(token_count > 0 && token_count <= kIndexedLocalMaximumTokens &&
                  attention_heads == kIndexedLocalAttentionHeads && attention_width == kIndexedLocalAttentionWidth,
      "indexed local transformer attention metadata is unsupported");
  if (active_rows == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(dense_rows / token_count);
  const auto stream = at::hip::getCurrentHIPStream(query_key_value.get_device());
#define LAUNCH_INDEXED_LOCAL_ATTENTION(scalar_type, index_type) \
  LaunchIndexedLocalAttentionTyped<scalar_type, index_type>(    \
      query_key_value, indices, active_rows, dense_rows, token_count, block_count, stream.stream())
  if (query_key_value.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::Half, int64_t);
    }
  } else if (query_key_value.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::BFloat16, int64_t);
    }
  } else if (query_key_value.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_ATTENTION
  CheckHipLaunch("fusion indexed local transformer attention");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalResidualNormalizeTyped(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& indices, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& normalized, int64_t active_offset, int64_t active_rows,
    int64_t dense_rows, float epsilon, uint32_t block_count, hipStream_t stream) {
  hipLaunchKernelGGL((indexed_local_residual_normalize_kernel<scalar_t, index_t>), dim3(block_count),
      dim3(kThreadsPerBlock), 0, stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
      indices.data_ptr<index_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),
      static_cast<int32_t>(norm_weight.scalar_type()), normalized.data_ptr<scalar_t>(), active_offset, active_rows,
      dense_rows, epsilon);
}

void LaunchIndexedLocalTransformerResidualLayerNorm(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, torch::Tensor& normalized, int64_t active_offset,
    int64_t active_rows, int64_t dense_rows, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width == kIndexedLocalHiddenWidth, "indexed local transformer requires hidden width 256");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
#define LAUNCH_INDEXED_LOCAL_RESIDUAL(scalar_type, index_type)                                                         \
  LaunchIndexedLocalResidualNormalizeTyped<scalar_type, index_type>(output, update, update_bias, indices, norm_weight, \
      norm_bias, normalized, active_offset, active_rows, dense_rows, epsilon, block_count, stream.stream())
  if (output.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::Half, int64_t);
    }
  } else if (output.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::BFloat16, int64_t);
    }
  } else if (output.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_RESIDUAL
  CheckHipLaunch("fusion indexed local transformer residual LayerNorm");
}

void LaunchIndexedLocalTransformerBiasSilu(
    torch::Tensor& values, const torch::Tensor& bias, int64_t active_rows, int64_t width) {
  LaunchTransformerBiasSilu(values, bias, active_rows, width);
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalFinalizeTyped(torch::Tensor& output, const torch::Tensor& update, const torch::Tensor& indices,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    int64_t active_offset, int64_t active_rows, int64_t dense_rows, float epsilon, uint32_t block_count,
    hipStream_t stream) {
  hipLaunchKernelGGL((indexed_local_finalize_kernel<scalar_t, index_t>), dim3(block_count), dim3(kThreadsPerBlock), 0,
      stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(), indices.data_ptr<index_t>(),
      update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),
      static_cast<int32_t>(norm_weight.scalar_type()), active_offset, active_rows, dense_rows, epsilon);
}

void LaunchIndexedLocalTransformerFinalize(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& indices, torch::ScalarType index_data_type, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, int64_t active_offset, int64_t active_rows,
    int64_t dense_rows, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width == kIndexedLocalHiddenWidth, "indexed local transformer requires hidden width 256");
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
#define LAUNCH_INDEXED_LOCAL_FINALIZE(scalar_type, index_type)                                                \
  LaunchIndexedLocalFinalizeTyped<scalar_type, index_type>(output, update, indices, update_bias, norm_weight, \
      norm_bias, active_offset, active_rows, dense_rows, epsilon, block_count, stream.stream())
  if (output.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::Half, int64_t);
    }
  } else if (output.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::BFloat16, int64_t);
    }
  } else if (output.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_FINALIZE
  CheckHipLaunch("fusion indexed local transformer finalize");
}

template <typename score_t, typename mask_t, typename value_t>
void LaunchMappedGroupedMaskedSoftmaxPoolTyped(
    const torch::Tensor& scores, const torch::Tensor& masks,
    const torch::Tensor& values, const torch::Tensor& destination_metadata,
    torch::Tensor& contexts, torch::Tensor& presence, int64_t batch_count,
    int64_t candidate_count, int64_t group_count, int64_t width,
    int64_t destination_count, hipStream_t stream) {
  const int64_t row_count = CheckedMultiply(
      batch_count, destination_count, "mapped grouped pool row count");
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(row_count, 65535));
  hipLaunchKernelGGL(
      (mapped_grouped_masked_softmax_pool_kernel<
          score_t, mask_t, value_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      scores.data_ptr<score_t>(), masks.data_ptr<mask_t>(),
      values.data_ptr<value_t>(), destination_metadata.data_ptr<int32_t>(),
      contexts.data_ptr<float>(), presence.data_ptr<mask_t>(), batch_count,
      candidate_count, group_count, width, destination_count);
}

void LaunchMappedGroupedMaskedSoftmaxPool(const torch::Tensor& scores,
    const torch::Tensor& masks, const torch::Tensor& values,
    const torch::Tensor& destination_metadata, torch::Tensor& contexts,
    torch::Tensor& presence, int64_t batch_count, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count) {
  TORCH_CHECK(candidate_count > 0 && candidate_count <= 256 &&
          group_count > 0 && width > 0 && destination_count > 0,
      "mapped grouped pool metadata is unsupported");
  TORCH_CHECK(destination_metadata.scalar_type() == torch::kInt32 &&
          destination_metadata.is_contiguous() &&
          destination_metadata.dim() == 2 &&
          destination_metadata.size(0) == destination_count &&
          destination_metadata.size(1) == 4,
      "mapped grouped pool destination metadata is invalid");
  TORCH_CHECK(contexts.scalar_type() == torch::kFloat32 &&
          presence.scalar_type() == masks.scalar_type(),
      "mapped grouped pool result data types are invalid");
  if (batch_count == 0) {
    return;
  }
  const auto stream = at::hip::getCurrentHIPStream(scores.get_device());
#define LAUNCH_MAPPED_POOL(score_type, mask_type, value_type) \
  LaunchMappedGroupedMaskedSoftmaxPoolTyped<score_type, mask_type, value_type>( \
      scores, masks, values, destination_metadata, contexts, presence, \
      batch_count, candidate_count, group_count, width, destination_count, \
      stream.stream())
#define DISPATCH_MAPPED_POOL_VALUE(score_type, mask_type) \
  switch (values.scalar_type()) { \
    case torch::kFloat16: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported mapped grouped pool value data type"); \
  }
#define DISPATCH_MAPPED_POOL_MASK(score_type) \
  switch (masks.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, c10::Half); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported mapped grouped pool mask data type"); \
  }
  switch (scores.scalar_type()) {
    case torch::kFloat16:
      DISPATCH_MAPPED_POOL_MASK(c10::Half);
      break;
    case torch::kBFloat16:
      DISPATCH_MAPPED_POOL_MASK(c10::BFloat16);
      break;
    case torch::kFloat32:
      DISPATCH_MAPPED_POOL_MASK(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported mapped grouped pool score data type");
  }
#undef DISPATCH_MAPPED_POOL_MASK
#undef DISPATCH_MAPPED_POOL_VALUE
#undef LAUNCH_MAPPED_POOL
  CheckHipLaunch("fusion mapped grouped masked-softmax pool");
}

}  // namespace djl::pytorch::fusion
