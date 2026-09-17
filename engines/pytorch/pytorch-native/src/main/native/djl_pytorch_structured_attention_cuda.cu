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

#include "djl_pytorch_structured_attention_cuda.h"

#include "djl_pytorch_structured_attention.h"

#include <ATen/Dispatch.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace djl::pytorch::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr size_t kSharedMemoryLimit = 48 * 1024;

struct MappedGroupedIndexedAttentionLayout {
  int64_t heads;
  int64_t key_features;
  int64_t value_features;
  int64_t shared_tokens;
  int64_t indexed_tokens;
  int64_t packed_width;
  int64_t groups;
  int64_t delta_rows;
};

template <typename scalar_t>
__device__ __forceinline__ float round_mapped_attention_value(float value) {
  // Preserve eager materialization boundaries for both low-precision input types.
  if constexpr (!std::is_same_v<scalar_t, float>) {
    return static_cast<float>(static_cast<scalar_t>(value));
  }
  return value;
}

template <typename scalar_t, typename output_t, bool fp32_reductions>
__global__ void mapped_grouped_indexed_attention_kernel(const scalar_t* query,
    const scalar_t* shared_key_values, const int32_t* shared_group_indices,
    const scalar_t* shared_delta_table, const int32_t* shared_delta_indices,
    const scalar_t* indexed_deltas, const int32_t* indexed_shared_ids, output_t* output,
    float* probabilities,
    int64_t query_head_count, MappedGroupedIndexedAttentionLayout layout, float scale) {
  extern __shared__ unsigned char shared_storage[];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t total_tokens = layout.shared_tokens + layout.indexed_tokens;
  float* all_token_weights = reinterpret_cast<float*>(shared_storage);
  int32_t* all_shared_indices =
      reinterpret_cast<int32_t*>(all_token_weights + waves_per_block * total_tokens);
  float* all_query_values =
      reinterpret_cast<float*>(all_shared_indices + waves_per_block * total_tokens);
  float* token_weights = all_token_weights + wave * total_tokens;
  int32_t* shared_indices = all_shared_indices + wave * total_tokens;
  float* query_values = all_query_values + wave * layout.key_features;

  const int64_t query_head = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  const bool active = query_head < query_head_count;
  const int64_t query_index = query_head / layout.heads;
  const int64_t head = query_head % layout.heads;
  const int64_t group = active ? static_cast<int64_t>(shared_group_indices[query_index]) : 0;
  CUDA_KERNEL_ASSERT(!active || (group >= 0 && group < layout.groups));
  const int64_t key_offset = head * layout.key_features;
  const int64_t value_offset = layout.heads * layout.key_features + head * layout.value_features;

  if (active) {
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      query_values[feature] = static_cast<float>(
          query[(query_index * layout.heads + head) * layout.key_features + feature]);
    }
  }
  __syncthreads();

  float local_maximum = -std::numeric_limits<float>::infinity();
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const bool shared_token = attended_token < layout.shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
      const int64_t stored_id =
          shared_token ? attended_token + 1
                       : static_cast<int64_t>(
                             indexed_shared_ids[query_index * layout.indexed_tokens + token]);
      CUDA_KERNEL_ASSERT(stored_id <= layout.shared_tokens);
      const bool present = stored_id != 0;
      const int64_t shared_index = stored_id > 0 ? stored_id - 1 : 0;
      float score = -std::numeric_limits<float>::infinity();
      if (present) {
        const int64_t base_offset =
            (group * layout.shared_tokens + shared_index) * layout.packed_width;
        if (shared_token) {
          const int32_t delta_index = shared_delta_indices[query_index * layout.shared_tokens + token];
          CUDA_KERNEL_ASSERT(delta_index >= 0 && delta_index < layout.delta_rows);
        }
        const int64_t delta_offset =
            shared_token
                ? static_cast<int64_t>(
                      shared_delta_indices[query_index * layout.shared_tokens + token]) *
                      layout.packed_width
                : (query_index * layout.indexed_tokens + token) * layout.packed_width;
        score = 0.0f;
        for (int64_t feature = 0; feature < layout.key_features; feature++) {
          const int64_t packed_feature = key_offset + feature;
          const float base =
              static_cast<float>(shared_key_values[base_offset + packed_feature]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_delta_table[delta_offset + packed_feature])
                  : static_cast<float>(indexed_deltas[delta_offset + packed_feature]);
          const float key = round_mapped_attention_value<scalar_t>(base + delta);
          score += round_mapped_attention_value<scalar_t>(query_values[feature] * key);
        }
        if constexpr (fp32_reductions) {
          score *= scale;
        } else {
          score = round_mapped_attention_value<scalar_t>(score);
          score = round_mapped_attention_value<scalar_t>(score * scale);
        }
      }
      shared_indices[attended_token] = present ? static_cast<int32_t>(shared_index) : -1;
      token_weights[attended_token] = score;
      local_maximum = fmaxf(local_maximum, score);
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_maximum = fmaxf(local_maximum, __shfl_down_sync(0xffffffff, local_maximum, offset));
  }
  const float maximum = __shfl_sync(0xffffffff, local_maximum, 0);

  float local_sum = 0.0f;
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const float score = token_weights[attended_token];
      const float weight = score == -std::numeric_limits<float>::infinity()
                               ? 0.0f
                               : expf(score - maximum);
      token_weights[attended_token] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  const float inverse_sum = active ? 1.0f / __shfl_sync(0xffffffff, local_sum, 0) : 0.0f;
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      float probability = token_weights[attended_token] * inverse_sum;
      if constexpr (!fp32_reductions) {
        probability = round_mapped_attention_value<scalar_t>(probability);
      }
      token_weights[attended_token] = probability;
      if (probabilities != nullptr) {
        probabilities[query_head * total_tokens + attended_token] = probability;
      }
    }
  }
  __syncthreads();

  if (!active) {
    return;
  }
  for (int64_t feature = lane; feature < layout.value_features; feature += warpSize) {
    const int64_t packed_feature = value_offset + feature;
    float context = 0.0f;
    for (int64_t attended_token = 0; attended_token < total_tokens; attended_token++) {
      const int32_t shared_index = shared_indices[attended_token];
      if (shared_index >= 0) {
        const bool shared_token = attended_token < layout.shared_tokens;
        const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
        const int64_t base_offset =
            (group * layout.shared_tokens + shared_index) * layout.packed_width;
        if (shared_token) {
          const int32_t delta_index = shared_delta_indices[query_index * layout.shared_tokens + token];
          CUDA_KERNEL_ASSERT(delta_index >= 0 && delta_index < layout.delta_rows);
        }
        const int64_t delta_offset =
            shared_token
                ? static_cast<int64_t>(
                      shared_delta_indices[query_index * layout.shared_tokens + token]) *
                      layout.packed_width
                : (query_index * layout.indexed_tokens + token) * layout.packed_width;
        const float base =
            static_cast<float>(shared_key_values[base_offset + packed_feature]);
        const float delta =
            shared_token
                ? static_cast<float>(shared_delta_table[delta_offset + packed_feature])
                : static_cast<float>(indexed_deltas[delta_offset + packed_feature]);
        const float value = round_mapped_attention_value<scalar_t>(base + delta);
        const float contribution = token_weights[attended_token] * value;
        context += fp32_reductions ? contribution : round_mapped_attention_value<scalar_t>(contribution);
      }
    }
    output[query_head * layout.value_features + feature] = static_cast<output_t>(context);
  }
}

template <typename scalar_t>
__device__ __forceinline__ void store_mapped_grouped_indexed_attention_gradient(
    float gradient, int64_t query_index, int64_t attended_token,
    int64_t packed_feature, bool shared_token, int64_t token,
    int64_t group, int32_t token_mapping, float* shared_key_value_gradient,
    float* shared_delta_table_gradient, scalar_t* indexed_delta_gradient,
    const MappedGroupedIndexedAttentionLayout& layout) {
  const bool present = shared_token || token_mapping >= 0;
  if (present) {
    const int64_t shared_index = shared_token ? attended_token : token_mapping;
    if (shared_key_value_gradient != nullptr) {
      atomicAdd(shared_key_value_gradient +
              (group * layout.shared_tokens + shared_index) * layout.packed_width +
              packed_feature,
          gradient);
    }
    if (shared_token && shared_delta_table_gradient != nullptr) {
      atomicAdd(shared_delta_table_gradient +
              static_cast<int64_t>(token_mapping) * layout.packed_width + packed_feature,
          gradient);
    }
  }
  if (!shared_token && indexed_delta_gradient != nullptr) {
    indexed_delta_gradient[
        (query_index * layout.indexed_tokens + token) * layout.packed_width +
        packed_feature] = static_cast<scalar_t>(present ? gradient : 0.0f);
  }
}

template <typename scalar_t, typename gradient_t, bool fp32_reductions>
__global__ void mapped_grouped_indexed_attention_backward_kernel(
    const scalar_t* query, const scalar_t* shared_key_values,
    const int32_t* shared_group_indices, const scalar_t* shared_delta_table,
    const int32_t* shared_delta_indices, const scalar_t* indexed_deltas,
    const int32_t* indexed_shared_ids, const float* probabilities,
    const gradient_t* gradient_output, scalar_t* query_gradient,
    float* shared_key_value_gradient, float* shared_delta_table_gradient,
    scalar_t* indexed_delta_gradient, int64_t query_head_count,
    int64_t delta_gradient_shards, int64_t delta_gradient_elements,
    MappedGroupedIndexedAttentionLayout layout, float scale) {
  extern __shared__ unsigned char shared_storage[];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  // Queries often share the same relation code. Spread their atomic writes over
  // independent tables, then reduce those FP32 partials once after this kernel.
  if (shared_delta_table_gradient != nullptr) {
    shared_delta_table_gradient +=
        (blockIdx.x % delta_gradient_shards) * delta_gradient_elements;
  }
  const int64_t total_tokens = layout.shared_tokens + layout.indexed_tokens;
  float* all_probabilities = reinterpret_cast<float*>(shared_storage);
  float* all_score_gradients = all_probabilities + waves_per_block * total_tokens;
  int32_t* all_token_mappings =
      reinterpret_cast<int32_t*>(all_score_gradients + waves_per_block * total_tokens);
  float* all_query_values =
      reinterpret_cast<float*>(all_token_mappings + waves_per_block * total_tokens);
  float* all_output_gradients = all_query_values + waves_per_block * layout.key_features;
  float* token_probabilities = all_probabilities + wave * total_tokens;
  float* token_score_gradients = all_score_gradients + wave * total_tokens;
  int32_t* token_mappings = all_token_mappings + wave * total_tokens;
  float* query_values = all_query_values + wave * layout.key_features;
  float* output_gradients = all_output_gradients + wave * layout.value_features;

  const int64_t query_head = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  const bool active = query_head < query_head_count;
  const int64_t query_index = query_head / layout.heads;
  const int64_t head = query_head % layout.heads;
  const int64_t group = active ? static_cast<int64_t>(shared_group_indices[query_index]) : 0;
  CUDA_KERNEL_ASSERT(!active || (group >= 0 && group < layout.groups));
  const int64_t key_offset = head * layout.key_features;
  const int64_t value_offset = layout.heads * layout.key_features + head * layout.value_features;

  if (active) {
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      query_values[feature] = static_cast<float>(
          query[(query_index * layout.heads + head) * layout.key_features + feature]);
    }
    for (int64_t feature = lane; feature < layout.value_features; feature += warpSize) {
      output_gradients[feature] = static_cast<float>(
          gradient_output[query_head * layout.value_features + feature]);
    }
  }
  __syncthreads();

  float local_weighted_gradient = 0.0f;
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const bool shared_token = attended_token < layout.shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
      const int64_t stored_id = shared_token
          ? attended_token + 1
          : static_cast<int64_t>(
                indexed_shared_ids[query_index * layout.indexed_tokens + token]);
      CUDA_KERNEL_ASSERT(stored_id <= layout.shared_tokens);
      const bool present = shared_token || stored_id != 0;
      const int64_t shared_index = shared_token ? attended_token : (stored_id > 0 ? stored_id - 1 : 0);
      const int32_t token_mapping = shared_token
          ? shared_delta_indices[query_index * layout.shared_tokens + token]
          : (present ? static_cast<int32_t>(shared_index) : -1);
      CUDA_KERNEL_ASSERT(!shared_token || (token_mapping >= 0 && token_mapping < layout.delta_rows));
      const float probability = probabilities[query_head * total_tokens + attended_token];
      float probability_gradient = 0.0f;
      if (present) {
        const int64_t base_offset =
            (group * layout.shared_tokens + shared_index) * layout.packed_width;
        if (shared_token) {
          const int32_t delta_index = shared_delta_indices[query_index * layout.shared_tokens + token];
          CUDA_KERNEL_ASSERT(delta_index >= 0 && delta_index < layout.delta_rows);
        }
        const int64_t delta_offset =
            shared_token
                ? static_cast<int64_t>(token_mapping) * layout.packed_width
                : (query_index * layout.indexed_tokens + token) * layout.packed_width;
        for (int64_t feature = 0; feature < layout.value_features; feature++) {
          const int64_t packed_feature = value_offset + feature;
          const float base =
              static_cast<float>(shared_key_values[base_offset + packed_feature]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_delta_table[delta_offset + packed_feature])
                  : static_cast<float>(indexed_deltas[delta_offset + packed_feature]);
          const float value = round_mapped_attention_value<scalar_t>(base + delta);
          const float contribution = output_gradients[feature] * value;
          probability_gradient += fp32_reductions
              ? contribution : round_mapped_attention_value<scalar_t>(contribution);
        }
      }
      if constexpr (!fp32_reductions) {
        probability_gradient = round_mapped_attention_value<scalar_t>(probability_gradient);
      }
      token_probabilities[attended_token] = probability;
      token_score_gradients[attended_token] = probability_gradient;
      token_mappings[attended_token] = token_mapping;
      local_weighted_gradient += probability * probability_gradient;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_weighted_gradient += __shfl_down_sync(0xffffffff, local_weighted_gradient, offset);
  }
  const float weighted_gradient = __shfl_sync(0xffffffff, local_weighted_gradient, 0);
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      float score_gradient = token_probabilities[attended_token] *
          (token_score_gradients[attended_token] - weighted_gradient);
      if constexpr (!fp32_reductions) {
        score_gradient = round_mapped_attention_value<scalar_t>(score_gradient);
      }
      token_score_gradients[attended_token] = round_mapped_attention_value<scalar_t>(score_gradient * scale);
    }
  }
  __syncthreads();

  if (!active) {
    return;
  }
  if (query_gradient != nullptr) {
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      float gradient = 0.0f;
      for (int64_t attended_token = 0; attended_token < total_tokens; attended_token++) {
        const bool shared_token = attended_token < layout.shared_tokens;
        const int64_t token =
            shared_token ? attended_token : attended_token - layout.shared_tokens;
        const int32_t token_mapping = token_mappings[attended_token];
        if (!shared_token && token_mapping < 0) {
          continue;
        }
        const int64_t shared_index = shared_token ? attended_token : token_mapping;
        const int64_t base_offset =
            (group * layout.shared_tokens + shared_index) * layout.packed_width;
        if (shared_token) {
          const int32_t delta_index = shared_delta_indices[query_index * layout.shared_tokens + token];
          CUDA_KERNEL_ASSERT(delta_index >= 0 && delta_index < layout.delta_rows);
        }
        const int64_t delta_offset =
            shared_token
                ? static_cast<int64_t>(token_mapping) * layout.packed_width
                : (query_index * layout.indexed_tokens + token) * layout.packed_width;
        const int64_t packed_feature = key_offset + feature;
        const float base =
            static_cast<float>(shared_key_values[base_offset + packed_feature]);
        const float delta = shared_token
            ? static_cast<float>(shared_delta_table[delta_offset + packed_feature])
            : static_cast<float>(indexed_deltas[delta_offset + packed_feature]);
        const float key = round_mapped_attention_value<scalar_t>(base + delta);
        const float contribution = token_score_gradients[attended_token] * key;
        gradient += round_mapped_attention_value<scalar_t>(contribution);
      }
      query_gradient[query_head * layout.key_features + feature] =
          static_cast<scalar_t>(gradient);
    }
  }

  if (shared_key_value_gradient == nullptr && shared_delta_table_gradient == nullptr &&
      indexed_delta_gradient == nullptr) {
    return;
  }
  for (int64_t attended_token = 0; attended_token < total_tokens; attended_token++) {
    const bool shared_token = attended_token < layout.shared_tokens;
    const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
    const int32_t token_mapping = token_mappings[attended_token];
    const bool present = shared_token || token_mapping >= 0;
    const float score_gradient = token_score_gradients[attended_token];
    const float probability = token_probabilities[attended_token];
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      const int64_t packed_feature = key_offset + feature;
      const float gradient = present
          ? round_mapped_attention_value<scalar_t>(score_gradient * query_values[feature]) : 0.0f;
      store_mapped_grouped_indexed_attention_gradient(gradient, query_index,
          attended_token, packed_feature, shared_token, token, group, token_mapping,
          shared_key_value_gradient, shared_delta_table_gradient,
          indexed_delta_gradient, layout);
    }
    for (int64_t feature = lane; feature < layout.value_features; feature += warpSize) {
      const int64_t packed_feature = value_offset + feature;
      const float contribution = probability * output_gradients[feature];
      const float gradient = present
          ? round_mapped_attention_value<scalar_t>(contribution) : 0.0f;
      store_mapped_grouped_indexed_attention_gradient(gradient, query_index,
          attended_token, packed_feature, shared_token, token, group, token_mapping,
          shared_key_value_gradient, shared_delta_table_gradient,
          indexed_delta_gradient, layout);
    }
  }
}


size_t shared_memory_bytes_per_warp(const MappedGroupedIndexedAttentionLayout& layout, bool backward) {
  const int64_t tokens = layout.shared_tokens + layout.indexed_tokens;
  const int64_t per_warp = backward
      ? tokens * (2 * sizeof(float) + sizeof(int32_t)) +
            (layout.key_features + layout.value_features) * sizeof(float)
      : tokens * (sizeof(float) + sizeof(int32_t)) + layout.key_features * sizeof(float);
  return static_cast<size_t>(per_warp);
}

int attention_warps_per_block(const MappedGroupedIndexedAttentionLayout& layout, bool backward) {
  const size_t per_warp = shared_memory_bytes_per_warp(layout, backward);
  // Each warp owns its scratch. Reduce independent queries per block before
  // falling back, so larger token counts and feature widths remain fused.
  for (int warps = kWarpsPerBlock; warps > 0; warps /= 2) {
    if (per_warp <= kSharedMemoryLimit / warps) {
      return warps;
    }
  }
  return 0;
}

MappedGroupedIndexedAttentionLayout attention_layout(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_delta_table,
    const torch::Tensor& indexed_deltas) {
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t packed_width = shared_key_values.size(2);
  return {heads, key_features, (packed_width - heads * key_features) / heads,
      shared_key_values.size(1), indexed_deltas.size(1), packed_width,
      shared_key_values.size(0), shared_delta_table.size(0)};
}

struct GroupedPackedAttentionLaunchPlan {
  int block_threads;
  size_t shared_memory_bytes;
};

std::optional<GroupedPackedAttentionLaunchPlan> grouped_packed_attention_launch_plan(
    const torch::Tensor& query, const torch::Tensor& packed_key_value, int64_t heads) {
  constexpr int kRequestedThreadsPerBlock = 256;
  // A logical group owns one query and keeps every short-row probability in registers.
  constexpr int kQueryGroupWidth = 32;
  const auto& properties = *at::cuda::getDeviceProperties(query.get_device());
  const int64_t key_features = query.size(2) / heads;
  const int64_t value_features = (packed_key_value.size(3) - query.size(2)) / heads;
  const int64_t key_tokens = packed_key_value.size(2);
  const int64_t maximum_int = std::numeric_limits<int>::max();
  const int maximum_threads =
      std::min(kRequestedThreadsPerBlock, properties.maxThreadsPerBlock);
  const int block_threads =
      maximum_threads - maximum_threads % kQueryGroupWidth;
  if (properties.warpSize % kQueryGroupWidth != 0 || block_threads <= 0 ||
      key_tokens > kQueryGroupWidth || key_features > kQueryGroupWidth ||
      value_features <= 0 ||
      query.size(0) > maximum_int || query.size(1) > maximum_int ||
      packed_key_value.size(1) > maximum_int ||
      key_tokens > maximum_int || heads > maximum_int || value_features > maximum_int ||
      query.size(2) > maximum_int || packed_key_value.size(3) > maximum_int ||
      key_tokens > maximum_int / key_features ||
      key_tokens > maximum_int / value_features ||
      heads > properties.maxGridSize[0] ||
      packed_key_value.size(1) > properties.maxGridSize[1] ||
      query.size(0) > properties.maxGridSize[2]) {
    return std::nullopt;
  }
  const size_t key_value_bytes = static_cast<size_t>(key_tokens) *
      static_cast<size_t>(key_features + value_features) * sizeof(float);
  const size_t mask_bytes = static_cast<size_t>(key_tokens) * sizeof(unsigned char);
  const size_t shared_memory_bytes = key_value_bytes + mask_bytes;
  if (shared_memory_bytes + sizeof(uint32_t) >
      static_cast<size_t>(properties.sharedMemPerBlock)) {
    return std::nullopt;
  }
  return GroupedPackedAttentionLaunchPlan{block_threads, shared_memory_bytes};
}

template <typename scalar_t>
__device__ __forceinline__ float round_grouped_attention_boundary(float value) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return value;
  }
  return static_cast<float>(static_cast<scalar_t>(value));
}

template <typename mask_t>
__device__ __forceinline__ bool grouped_attention_mask_present(mask_t value) {
  return static_cast<float>(value) != 0.0f;
}

__device__ float grouped_attention_maximum(float left, float right) {
  return isnan(left) || isnan(right) ? NAN : fmaxf(left, right);
}

template <int query_group_width, typename scalar_t, typename mask_t>
__global__ void grouped_packed_attention_kernel(const scalar_t* query,
    const scalar_t* packed_key_value, const mask_t* mask, scalar_t* output,
    int query_tokens, int groups, int key_tokens, int heads,
    int key_features, int value_features, float scale) {
  extern __shared__ unsigned char shared_storage[];
  __shared__ uint32_t present_key_bits;
  const int lane = threadIdx.x % query_group_width;
  const int query_group = threadIdx.x / query_group_width;
  const int query_groups_per_block = blockDim.x / query_group_width;
  const int head = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int batch = static_cast<int>(blockIdx.z);
  const int query_width = heads * key_features;
  const int packed_width = query_width + heads * value_features;
  const int64_t matrix = static_cast<int64_t>(batch) * groups + group;
  const int key_elements = key_tokens * key_features;
  const int value_elements = key_tokens * value_features;
  auto* shared_keys = reinterpret_cast<float*>(shared_storage);
  auto* shared_values = shared_keys + key_elements;
  const size_t key_value_bytes =
      static_cast<size_t>(key_elements + value_elements) * sizeof(float);
  auto* shared_mask = shared_storage + key_value_bytes;

  // Feature-major keys let adjacent token lanes read adjacent shared-memory banks.
  for (int index = threadIdx.x; index < key_elements; index += blockDim.x) {
    const int key_token = index / key_features;
    const int feature = index - key_token * key_features;
    const int64_t source_offset =
        (matrix * key_tokens + key_token) * packed_width + head * key_features + feature;
    shared_keys[feature * key_tokens + key_token] = grouped_attention_mask_present(
            mask[matrix * key_tokens + key_token])
        ? static_cast<float>(packed_key_value[source_offset])
        : 0.0f;
  }
  for (int index = threadIdx.x; index < value_elements; index += blockDim.x) {
    const int key_token = index / value_features;
    const int feature = index - key_token * value_features;
    const int64_t source_offset =
        (matrix * key_tokens + key_token) * packed_width + query_width +
        head * value_features + feature;
    shared_values[key_token * value_features + feature] = grouped_attention_mask_present(
            mask[matrix * key_tokens + key_token])
        ? static_cast<float>(packed_key_value[source_offset])
        : 0.0f;
  }
  for (int key_token = threadIdx.x; key_token < key_tokens;
       key_token += blockDim.x) {
    shared_mask[key_token] = static_cast<unsigned char>(grouped_attention_mask_present(
        mask[matrix * key_tokens + key_token]));
  }
  // The first logical group reads its own mask stores, then publishes the ballot.
  if (threadIdx.x < query_group_width) {
    const auto present = __ballot_sync(0xffffffff,
        threadIdx.x < key_tokens && shared_mask[threadIdx.x] != 0);
    if (threadIdx.x == 0) {
      present_key_bits = static_cast<uint32_t>(present);
    }
  }
  __syncthreads();

  for (int64_t query_token = query_group; query_token < query_tokens;
       query_token += query_groups_per_block) {
    const int64_t query_offset =
        (static_cast<int64_t>(batch) * query_tokens + query_token) * query_width +
        head * key_features;
    const float query_value = lane < key_features
        ? static_cast<float>(query[query_offset + lane])
        : 0.0f;
    const bool present = lane < key_tokens && shared_mask[lane] != 0;
    float score = 0.0f;
    for (int feature = 0; feature < key_features; ++feature) {
      // A masked key lane can still supply a query feature to the other lanes.
      const float query_feature = __shfl_sync(0xffffffff, query_value, feature, query_group_width);
      if (present) {
        score += query_feature * shared_keys[feature * key_tokens + lane];
      }
    }
    if (present) {
      score = round_grouped_attention_boundary<scalar_t>(score);
      score = round_grouped_attention_boundary<scalar_t>(score * scale);
    } else {
      score = -std::numeric_limits<float>::infinity();
    }
    float local_maximum = score;
    for (int offset = query_group_width / 2; offset > 0; offset /= 2) {
      local_maximum = grouped_attention_maximum(
          local_maximum, __shfl_down_sync(0xffffffff, local_maximum, offset, query_group_width));
    }
    const float maximum = __shfl_sync(0xffffffff, local_maximum, 0, query_group_width);

    const float weight = present ? expf(score - maximum) : 0.0f;
    float local_sum = weight;
    for (int offset = query_group_width / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset, query_group_width);
    }
    const float sum = __shfl_sync(0xffffffff, local_sum, 0, query_group_width);
    const float inverse_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
    const float probability = present
        ? round_grouped_attention_boundary<scalar_t>(weight * inverse_sum)
        : 0.0f;

    const int64_t output_offset =
        ((matrix * query_tokens + query_token) * heads + head) * value_features;
    for (int feature_base = 0; feature_base < value_features;
         feature_base += query_group_width) {
      const int feature = feature_base + lane;
      float context = 0.0f;
      uint32_t remaining = present_key_bits;
      while (remaining != 0) {
        const int key_token = __ffs(remaining) - 1;
        remaining &= remaining - 1;
        const float token_probability =
            __shfl_sync(0xffffffff, probability, key_token, query_group_width);
        if (feature < value_features) {
          context += token_probability *
              shared_values[key_token * value_features + feature];
        }
      }
      if (feature < value_features) {
        output[output_offset + feature] = static_cast<scalar_t>(context);
      }
    }
  }
}

template <typename scalar_t, typename mask_t>
void launch_grouped_packed_attention(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    torch::Tensor& output, const GroupedPackedAttentionLaunchPlan& launch_plan,
    int64_t heads, float scale, cudaStream_t stream) {
  const int64_t key_features = query.size(2) / heads;
  const int64_t value_features = (packed_key_value.size(3) - query.size(2)) / heads;
  const dim3 grid(static_cast<unsigned int>(heads),
      static_cast<unsigned int>(packed_key_value.size(1)),
      static_cast<unsigned int>(query.size(0)));
  grouped_packed_attention_kernel<32, scalar_t, mask_t>
      <<<grid, launch_plan.block_threads, launch_plan.shared_memory_bytes, stream>>>(
          query.data_ptr<scalar_t>(), packed_key_value.data_ptr<scalar_t>(),
          mask.data_ptr<mask_t>(), output.data_ptr<scalar_t>(),
          static_cast<int>(query.size(1)), static_cast<int>(packed_key_value.size(1)),
          static_cast<int>(packed_key_value.size(2)), static_cast<int>(heads),
          static_cast<int>(key_features), static_cast<int>(value_features), scale);
}

template <typename scalar_t>
void dispatch_grouped_packed_attention(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    torch::Tensor& output,
    const GroupedPackedAttentionLaunchPlan& launch_plan,
    int64_t heads, float scale, cudaStream_t stream) {
  switch (mask.scalar_type()) {
    case torch::kBool:
      launch_grouped_packed_attention<scalar_t, bool>(query, packed_key_value, mask,
          output, launch_plan, heads, scale, stream);
      return;
    case torch::kInt32:
      launch_grouped_packed_attention<scalar_t, int32_t>(query, packed_key_value, mask,
          output, launch_plan, heads, scale, stream);
      return;
    case torch::kFloat32:
      launch_grouped_packed_attention<scalar_t, float>(query, packed_key_value, mask,
          output, launch_plan, heads, scale, stream);
      return;
    case torch::kFloat16:
      launch_grouped_packed_attention<scalar_t, c10::Half>(query, packed_key_value, mask,
          output, launch_plan, heads, scale, stream);
      return;
    case torch::kBFloat16:
      launch_grouped_packed_attention<scalar_t, c10::BFloat16>(query, packed_key_value, mask,
          output, launch_plan, heads, scale, stream);
      return;
    default:
      TORCH_CHECK(false, "unsupported grouped packed attention mask dtype");
  }
}

}  // namespace

bool supports_grouped_packed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask, int64_t heads) {
  if (query.is_neg() || query.is_conj() || packed_key_value.is_neg() ||
      packed_key_value.is_conj() || mask.is_neg() || mask.is_conj() || !query.is_cuda() ||
      !query.is_contiguous() || !packed_key_value.is_contiguous() ||
      !mask.is_contiguous() ||
      (query.scalar_type() != torch::kFloat32 && query.scalar_type() != torch::kFloat16 &&
          query.scalar_type() != torch::kBFloat16) ||
      query.scalar_type() != packed_key_value.scalar_type() ||
      query.device() != packed_key_value.device() || query.device() != mask.device() ||
      query.dim() != 3 || packed_key_value.dim() != 4 || mask.dim() != 3 ||
      query.size(0) != packed_key_value.size(0) ||
      mask.size(0) != packed_key_value.size(0) ||
      mask.size(1) != packed_key_value.size(1) ||
      mask.size(2) != packed_key_value.size(2) || heads <= 0 || query.size(0) <= 0 ||
      query.size(1) <= 0 || query.size(2) <= 0 || query.size(2) % heads != 0 ||
      packed_key_value.size(1) <= 0 || packed_key_value.size(2) <= 0 ||
      packed_key_value.size(3) <= query.size(2) ||
      (packed_key_value.size(3) - query.size(2)) % heads != 0) {
    return false;
  }
  // CUDA autocast may change matmul's input and output type independently of these tensors.
  if (at::autocast::is_autocast_enabled(at::DeviceType::CUDA) &&
      query.scalar_type() != at::autocast::get_autocast_dtype(at::DeviceType::CUDA)) {
    return false;
  }
  const bool supported_mask = mask.scalar_type() == torch::kBool ||
      mask.scalar_type() == torch::kInt32 || mask.scalar_type() == torch::kFloat32 ||
      mask.scalar_type() == torch::kFloat16 || mask.scalar_type() == torch::kBFloat16;
  return supported_mask &&
      grouped_packed_attention_launch_plan(query, packed_key_value, heads).has_value();
}

torch::Tensor grouped_packed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    int64_t heads, float scale) {
  TORCH_CHECK(supports_grouped_packed_attention_forward(query, packed_key_value, mask, heads),
      "grouped packed attention received an unsupported CUDA tensor layout");
  c10::cuda::CUDAGuard device_guard(query.device());
  const auto launch_plan = grouped_packed_attention_launch_plan(query, packed_key_value, heads);
  TORCH_CHECK(launch_plan.has_value(), "grouped packed attention has no valid CUDA launch plan");
  auto output = torch::empty(
      {query.size(0), packed_key_value.size(1), query.size(1),
          packed_key_value.size(3) - query.size(2)},
      query.options().memory_format(torch::MemoryFormat::Contiguous));
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, query.scalar_type(),
      "grouped_packed_attention_cuda", [&] {
        dispatch_grouped_packed_attention<scalar_t>(
            query, packed_key_value, mask, output, *launch_plan, heads, scale, stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

bool supports_mapped_grouped_indexed_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, bool backward) {
  if (!detail::is_mapped_grouped_indexed_attention_layout_supported(query, shared_key_values,
          shared_group_indices, shared_delta_table, shared_delta_indices, indexed_deltas,
          indexed_shared_ids)) {
    return false;
  }
  const int64_t queries = query.size(0);
  const int64_t heads = query.size(1);
  const auto layout = attention_layout(query, shared_key_values, shared_delta_table, indexed_deltas);
  const int warps = attention_warps_per_block(layout, backward);
  return warps > 0 &&
      (queries * heads + warps - 1) / warps <= std::numeric_limits<int32_t>::max();
}

MappedGroupedIndexedAttentionForwardResult mapped_grouped_indexed_attention_forward(
    const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_group_indices, const torch::Tensor& shared_delta_table,
    const torch::Tensor& shared_delta_indices, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, float scale, bool capture_probabilities) {
  TORCH_CHECK(supports_mapped_grouped_indexed_attention(query, shared_key_values, shared_group_indices,
          shared_delta_table, shared_delta_indices, indexed_deltas, indexed_shared_ids, capture_probabilities),
      "mapped grouped indexed attention requires supported contiguous CUDA tensors with int32 mappings");
  c10::cuda::CUDAGuard device_guard(query.device());
  const auto layout = attention_layout(query, shared_key_values, shared_delta_table, indexed_deltas);
  const int64_t queries = query.size(0);
  const int64_t query_heads = queries * layout.heads;
  // The reference's sum and softmax follow CUDA autocast's FP32 reduction policy.
  const bool fp32_reductions = at::autocast::is_autocast_enabled(at::DeviceType::CUDA);
  const auto output_type = fp32_reductions ? torch::kFloat32 : query.scalar_type();
  auto output = torch::empty({queries, layout.heads, layout.value_features}, query.options().dtype(output_type));
  auto probabilities = capture_probabilities
      ? torch::empty({queries, layout.heads, layout.shared_tokens + layout.indexed_tokens},
            query.options().dtype(torch::kFloat32))
      : torch::Tensor();
  float* probability_data = capture_probabilities ? probabilities.data_ptr<float>() : nullptr;
  const int warps = attention_warps_per_block(layout, false);
  const int blocks = static_cast<int>((query_heads + warps - 1) / warps);
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  const size_t shared_bytes = shared_memory_bytes_per_warp(layout, false) * warps;
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, query.scalar_type(),
      "mapped_grouped_indexed_attention_cuda", [&] {
        const auto launch = [&](auto fp32) {
          constexpr bool use_fp32 = decltype(fp32)::value;
          using output_t = std::conditional_t<use_fp32, float, scalar_t>;
          mapped_grouped_indexed_attention_kernel<scalar_t, output_t, use_fp32>
              <<<blocks, warps * 32, shared_bytes, stream>>>(query.data_ptr<scalar_t>(),
                  shared_key_values.data_ptr<scalar_t>(), shared_group_indices.data_ptr<int32_t>(),
                  shared_delta_table.data_ptr<scalar_t>(), shared_delta_indices.data_ptr<int32_t>(),
                  indexed_deltas.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<int32_t>(),
                  output.data_ptr<output_t>(), probability_data, query_heads, layout, scale);
        };
        if (fp32_reductions) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, probabilities};
}

MappedGroupedIndexedAttentionGradients mapped_grouped_indexed_attention_backward(
    const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_group_indices, const torch::Tensor& shared_delta_table,
    const torch::Tensor& shared_delta_indices, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, const torch::Tensor& probabilities,
    const torch::Tensor& gradient_output, float scale, bool needs_query_gradient,
    bool needs_shared_key_value_gradient, bool needs_shared_delta_table_gradient,
    bool needs_indexed_delta_gradient) {
  c10::cuda::CUDAGuard device_guard(query.device());
  const auto layout = attention_layout(query, shared_key_values, shared_delta_table, indexed_deltas);
  const int64_t queries = query.size(0);
  const int64_t query_heads = queries * layout.heads;
  const bool fp32_reductions = gradient_output.scalar_type() == torch::kFloat32;
  TORCH_CHECK(fp32_reductions || gradient_output.scalar_type() == query.scalar_type(),
      "mapped grouped indexed attention output gradient dtype mismatch");
  auto query_gradient = needs_query_gradient ? torch::empty_like(query) : torch::Tensor();
  auto shared_accumulator = needs_shared_key_value_gradient
      ? torch::zeros(shared_key_values.sizes(), query.options().dtype(torch::kFloat32)) : torch::Tensor();
  const int64_t delta_elements = shared_delta_table.numel();
  constexpr int64_t maximum_delta_gradient_bytes = 32 * 1024 * 1024;
  const int64_t delta_shards = needs_shared_delta_table_gradient
      ? std::max<int64_t>(1, std::min<int64_t>({32, (queries + 255) / 256,
            maximum_delta_gradient_bytes / (delta_elements * static_cast<int64_t>(sizeof(float)))})) : 1;
  auto delta_accumulator = needs_shared_delta_table_gradient
      ? torch::zeros({delta_shards, shared_delta_table.size(0), layout.packed_width},
            query.options().dtype(torch::kFloat32)) : torch::Tensor();
  auto indexed_gradient = needs_indexed_delta_gradient ? torch::empty_like(indexed_deltas) : torch::Tensor();
  const auto gradient = gradient_output.contiguous();
  const int warps = attention_warps_per_block(layout, true);
  TORCH_CHECK(warps > 0, "mapped grouped indexed attention backward exceeds CUDA shared memory");
  const int blocks = static_cast<int>((query_heads + warps - 1) / warps);
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  const size_t shared_bytes = shared_memory_bytes_per_warp(layout, true) * warps;
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, query.scalar_type(),
      "mapped_grouped_indexed_attention_backward_cuda", [&] {
        const auto launch = [&](auto fp32) {
          constexpr bool use_fp32 = decltype(fp32)::value;
          using gradient_t = std::conditional_t<use_fp32, float, scalar_t>;
          mapped_grouped_indexed_attention_backward_kernel<scalar_t, gradient_t, use_fp32>
              <<<blocks, warps * 32, shared_bytes, stream>>>(query.data_ptr<scalar_t>(),
                  shared_key_values.data_ptr<scalar_t>(), shared_group_indices.data_ptr<int32_t>(),
                  shared_delta_table.data_ptr<scalar_t>(), shared_delta_indices.data_ptr<int32_t>(),
                  indexed_deltas.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<int32_t>(),
                  probabilities.data_ptr<float>(), gradient.data_ptr<gradient_t>(),
                  needs_query_gradient ? query_gradient.data_ptr<scalar_t>() : nullptr,
                  needs_shared_key_value_gradient ? shared_accumulator.data_ptr<float>() : nullptr,
                  needs_shared_delta_table_gradient ? delta_accumulator.data_ptr<float>() : nullptr,
                  needs_indexed_delta_gradient ? indexed_gradient.data_ptr<scalar_t>() : nullptr,
                  query_heads, delta_shards, delta_elements, layout, scale);
        };
        if (fp32_reductions) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto shared_gradient = needs_shared_key_value_gradient
      ? shared_accumulator.to(query.scalar_type()) : torch::Tensor();
  auto delta_gradient = needs_shared_delta_table_gradient
      ? (delta_shards == 1 ? delta_accumulator.select(0, 0) : delta_accumulator.sum(0))
            .to(query.scalar_type()) : torch::Tensor();
  return {query_gradient, shared_gradient, delta_gradient, indexed_gradient};
}

}  // namespace djl::pytorch::cuda
