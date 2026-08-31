#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "djl_pytorch_rocm_kernels.h"
#include "djl_pytorch_rocm_launch_config.h"

namespace djl::pytorch::rocm {
namespace {

// Bound register-backed specializations; wider rows retain the original two-pass kernel.
constexpr int kResidualCachedValuesPerLaneLimit = 16;
constexpr int kAutocastLayerNormCachedValuesPerLaneLimit = 16;

void check_hip_kernel_launch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ", hipGetErrorString(error));
}

bool is_fused_floating(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16 ||
         tensor.scalar_type() == torch::kBFloat16;
}

bool supports_owned_residual_layer_norm_types(
    torch::ScalarType residual_type, torch::ScalarType update_type, torch::ScalarType parameter_type) {
  const bool supports_existing_parameter_type =
      parameter_type == torch::kFloat32 || parameter_type == residual_type;
  const bool supports_inference_parameter_type = residual_type == torch::kFloat32 &&
      update_type == parameter_type &&
      (parameter_type == torch::kFloat16 || parameter_type == torch::kBFloat16);
  return supports_existing_parameter_type || supports_inference_parameter_type;
}

__device__ float maximum_propagating_nan(float left, float right) {
  return isnan(left) || isnan(right) ? NAN : fmaxf(left, right);
}

template <typename scalar_t>
__global__ void fused_adam_update_kernel(scalar_t* weight, const scalar_t* gradient, scalar_t* mean,
    scalar_t* variance, int64_t element_count, float learning_rate, float learning_rate_bias_correction,
    float weight_decay, float rescale_gradient, float clip_gradient, float beta1, float beta2, float epsilon,
    bool adamw) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float weight_value = static_cast<float>(weight[index]);
    float gradient_value = static_cast<float>(gradient[index]) * rescale_gradient;
    if (clip_gradient >= 0.0f && gradient_value > clip_gradient) {
      gradient_value = clip_gradient;
    }
    if (adamw) {
      weight_value -= weight_value * learning_rate * weight_decay;
    } else {
      gradient_value += weight_value * weight_decay;
    }
    const float mean_value = beta1 * static_cast<float>(mean[index]) + (1.0f - beta1) * gradient_value;
    const float variance_value =
        beta2 * static_cast<float>(variance[index]) + (1.0f - beta2) * gradient_value * gradient_value;
    weight_value -= learning_rate_bias_correction * mean_value / (sqrtf(variance_value) + epsilon);
    weight[index] = static_cast<scalar_t>(weight_value);
    mean[index] = static_cast<scalar_t>(mean_value);
    variance[index] = static_cast<scalar_t>(variance_value);
  }
}

template <typename scalar_t, bool write_probabilities>
__global__ void masked_categorical_forward_kernel(const scalar_t* logits, const bool* mask,
    float* output, int64_t row_count, int64_t width) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t row_offset = row * width;
    float local_maximum = -INFINITY;
    int local_choice_count = 0;
    for (int64_t column = lane; column < width; column += warpSize) {
      if (mask[row_offset + column]) {
        local_maximum = maximum_propagating_nan(
            local_maximum, static_cast<float>(logits[row_offset + column]));
        ++local_choice_count;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_maximum = maximum_propagating_nan(
          local_maximum, __shfl_down(local_maximum, offset));
      local_choice_count += __shfl_down(local_choice_count, offset);
    }
    const float maximum = __shfl(local_maximum, 0);
    const int choice_count = __shfl(local_choice_count, 0);
    float local_sum = 0.0f;
    if (choice_count > 0) {
      for (int64_t column = lane; column < width; column += warpSize) {
        if (mask[row_offset + column]) {
          local_sum += expf(static_cast<float>(logits[row_offset + column]) - maximum);
        }
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down(local_sum, offset);
    }
    const float denominator = __shfl(local_sum, 0);
    if constexpr (write_probabilities) {
      for (int64_t column = lane; column < width; column += warpSize) {
        output[row_offset + column] = mask[row_offset + column] && choice_count > 0
            ? expf(static_cast<float>(logits[row_offset + column]) - maximum) / denominator
            : 0.0f;
      }
    } else if (lane == 0) {
      output[row] = choice_count > 0 ? logf(denominator) + maximum : 0.0f;
    }
  }
}

template <int subwave_width, typename scalar_t, bool write_probabilities>
__global__ void masked_categorical_forward_subwave_kernel(const scalar_t* logits, const bool* mask,
    float* output, int64_t row_count, int64_t width) {
  const int lane = threadIdx.x % subwave_width;
  const int group = threadIdx.x / subwave_width;
  const int groups_per_block = blockDim.x / subwave_width;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * groups_per_block + group;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t row_offset = row * width;
    float local_maximum = -INFINITY;
    int local_choice_count = 0;
    for (int64_t column = lane; column < width; column += subwave_width) {
      if (mask[row_offset + column]) {
        local_maximum = maximum_propagating_nan(
            local_maximum, static_cast<float>(logits[row_offset + column]));
        ++local_choice_count;
      }
    }
    for (int offset = subwave_width / 2; offset > 0; offset /= 2) {
      local_maximum = maximum_propagating_nan(
          local_maximum, __shfl_down(local_maximum, offset, subwave_width));
      local_choice_count += __shfl_down(local_choice_count, offset, subwave_width);
    }
    const float maximum = __shfl(local_maximum, 0, subwave_width);
    const int choice_count = __shfl(local_choice_count, 0, subwave_width);
    float local_sum = 0.0f;
    if (choice_count > 0) {
      for (int64_t column = lane; column < width; column += subwave_width) {
        if (mask[row_offset + column]) {
          local_sum += expf(static_cast<float>(logits[row_offset + column]) - maximum);
        }
      }
    }
    for (int offset = subwave_width / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down(local_sum, offset, subwave_width);
    }
    const float denominator = __shfl(local_sum, 0, subwave_width);
    if constexpr (write_probabilities) {
      for (int64_t column = lane; column < width; column += subwave_width) {
        output[row_offset + column] = mask[row_offset + column] && choice_count > 0
            ? expf(static_cast<float>(logits[row_offset + column]) - maximum) / denominator
            : 0.0f;
      }
    } else if (lane == 0) {
      output[row] = choice_count > 0 ? logf(denominator) + maximum : 0.0f;
    }
  }
}

template <typename logit_t, typename value_t>
__global__ void grouped_masked_softmax_pool_kernel(const logit_t* logits, const bool* mask,
    const value_t* values, float* output, int64_t row_count, int64_t choice_count,
    int64_t group_count, int64_t feature_count) {
  extern __shared__ float probabilities[];
  __shared__ int has_choice;
  const int lane = threadIdx.x % warpSize;
  const int64_t pool_count = row_count * group_count;
  for (int64_t pool = blockIdx.x; pool < pool_count; pool += gridDim.x) {
    const int64_t row = pool / group_count;
    const int64_t group = pool % group_count;
    if (threadIdx.x < warpSize) {
      const bool selected = lane < choice_count &&
          mask[(row * choice_count + lane) * group_count + group];
      const float logit = selected
          ? static_cast<float>(logits[row * choice_count + lane])
          : -INFINITY;
      float maximum = logit;
      int selected_count = selected ? 1 : 0;
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        maximum = maximum_propagating_nan(maximum, __shfl_down(maximum, offset));
        selected_count += __shfl_down(selected_count, offset);
      }
      maximum = __shfl(maximum, 0);
      selected_count = __shfl(selected_count, 0);
      float exponential = selected && selected_count > 0 ? expf(logit - maximum) : 0.0f;
      float denominator = exponential;
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        denominator += __shfl_down(denominator, offset);
      }
      denominator = __shfl(denominator, 0);
      if (lane == 0) {
        has_choice = selected_count > 0;
      }
      if (lane < choice_count) {
        probabilities[lane] = selected_count > 0 ? exponential / denominator : 0.0f;
      }
    }
    __syncthreads();
    // Keep each group in one contiguous slab. Consumers commonly select one group before
    // submitting it to another fused operation, so a group-major backing layout avoids a copy.
    const int64_t output_offset = (group * row_count + row) * feature_count;
    for (int64_t feature = threadIdx.x; feature < feature_count; feature += blockDim.x) {
      float pooled = 0.0f;
      if (has_choice) {
        for (int64_t choice = 0; choice < choice_count; ++choice) {
          pooled += probabilities[choice] * static_cast<float>(
              values[(row * choice_count + choice) * feature_count + feature]);
        }
      }
      output[output_offset + feature] = pooled;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void masked_softmax_backward_kernel(const float* gradient_output,
    const float* probabilities, const bool* mask, scalar_t* gradient_input,
    int64_t row_count, int64_t width) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t row_offset = row * width;
    float local_dot = 0.0f;
    for (int64_t column = lane; column < width; column += warpSize) {
      local_dot += gradient_output[row_offset + column] * probabilities[row_offset + column];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_dot += __shfl_down(local_dot, offset);
    }
    const float dot = __shfl(local_dot, 0);
    for (int64_t column = lane; column < width; column += warpSize) {
      const int64_t index = row_offset + column;
      gradient_input[index] = mask[index]
          ? static_cast<scalar_t>(probabilities[index] * (gradient_output[index] - dot))
          : static_cast<scalar_t>(0.0f);
    }
  }
}

template <typename scalar_t>
__global__ void masked_log_sum_exp_backward_kernel(const float* gradient_output,
    const scalar_t* logits, const bool* mask, const float* normalizers, scalar_t* gradient_input,
    int64_t row_count, int64_t width) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < row_count;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t row_offset = row * width;
    const float row_gradient = gradient_output[row];
    const float normalizer = normalizers[row];
    for (int64_t column = lane; column < width; column += warpSize) {
      const int64_t index = row_offset + column;
      gradient_input[index] = mask[index]
          ? static_cast<scalar_t>(row_gradient * expf(static_cast<float>(logits[index]) - normalizer))
          : static_cast<scalar_t>(0.0f);
    }
  }
}

template <typename scalar_t>
__global__ void scatter_rows_kernel(const scalar_t* rows, const int64_t* row_indices,
    scalar_t* output, int64_t compact_element_count, int64_t row_width, int64_t row_count) {
  for (int64_t source_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       source_index < compact_element_count;
       source_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t source_row = source_index / row_width;
    const int64_t destination_row = row_indices[source_row];
    if (destination_row >= 0 && destination_row < row_count) {
      output[destination_row * row_width + source_index % row_width] = rows[source_index];
    }
  }
}

template <typename scalar_t>
__global__ void gather_scattered_row_gradients_kernel(const scalar_t* gradient_output,
    const int64_t* row_indices, scalar_t* gradient_rows, int64_t compact_element_count,
    int64_t row_width) {
  for (int64_t destination_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination_index < compact_element_count;
       destination_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t compact_row = destination_index / row_width;
    gradient_rows[destination_index] =
        gradient_output[row_indices[compact_row] * row_width + destination_index % row_width];
  }
}

template <int group_width, typename scalar_t>
__global__ void scatter_rows_grouped_kernel(const scalar_t* rows, const int64_t* row_indices,
    scalar_t* output, int64_t compact_row_count, int64_t row_width, int64_t row_count) {
  const int lane = threadIdx.x % group_width;
  const int group = threadIdx.x / group_width;
  const int groups_per_block = blockDim.x / group_width;
  for (int64_t source_row = static_cast<int64_t>(blockIdx.x) * groups_per_block + group;
       source_row < compact_row_count;
       source_row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t destination_row = row_indices[source_row];
    if (destination_row >= 0 && destination_row < row_count) {
      const int64_t source_offset = source_row * row_width;
      const int64_t destination_offset = destination_row * row_width;
      for (int64_t column = lane; column < row_width; column += group_width) {
        output[destination_offset + column] = rows[source_offset + column];
      }
    }
  }
}

template <int group_width, typename scalar_t>
__global__ void gather_scattered_row_gradients_grouped_kernel(const scalar_t* gradient_output,
    const int64_t* row_indices, scalar_t* gradient_rows, int64_t compact_row_count,
    int64_t row_width) {
  const int lane = threadIdx.x % group_width;
  const int group = threadIdx.x / group_width;
  const int groups_per_block = blockDim.x / group_width;
  for (int64_t compact_row = static_cast<int64_t>(blockIdx.x) * groups_per_block + group;
       compact_row < compact_row_count;
       compact_row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t source_offset = row_indices[compact_row] * row_width;
    const int64_t destination_offset = compact_row * row_width;
    for (int64_t column = lane; column < row_width; column += group_width) {
      gradient_rows[destination_offset + column] = gradient_output[source_offset + column];
    }
  }
}

constexpr int kPaddedBatchGatherMaximumIndexRank = 8;

struct PaddedBatchGatherIndexLayout {
  int64_t rank;
  int64_t sizes[kPaddedBatchGatherMaximumIndexRank];
  int64_t strides[kPaddedBatchGatherMaximumIndexRank];
};

enum class PaddedBatchGatherIndexType : int32_t {
  kInt16,
  kInt32,
  kInt64,
};

__device__ int64_t padded_batch_gather_index_offset(
    int64_t logical_index, const PaddedBatchGatherIndexLayout& layout) {
  int64_t offset = 0;
  for (int64_t axis = layout.rank - 1; axis >= 0; --axis) {
    const int64_t coordinate = logical_index % layout.sizes[axis];
    logical_index /= layout.sizes[axis];
    offset += coordinate * layout.strides[axis];
  }
  return offset;
}

__device__ int64_t padded_batch_gather_index_value(
    const void* data, int64_t offset, PaddedBatchGatherIndexType type) {
  switch (type) {
    case PaddedBatchGatherIndexType::kInt16:
      return static_cast<const int16_t*>(data)[offset];
    case PaddedBatchGatherIndexType::kInt32:
      return static_cast<const int32_t*>(data)[offset];
    case PaddedBatchGatherIndexType::kInt64:
      return static_cast<const int64_t*>(data)[offset];
  }
  return 0;
}

template <int group_width, typename scalar_t, typename stored_index_t,
    bool has_inner_indices, bool has_explicit_batch_indices>
__global__ void padded_batch_gather_kernel(const scalar_t* source,
    const void* batch_indices, const stored_index_t* outer_stored_indices,
    const stored_index_t* inner_stored_indices, scalar_t* output,
    PaddedBatchGatherIndexType batch_index_type,
    PaddedBatchGatherIndexLayout batch_layout, PaddedBatchGatherIndexLayout outer_layout,
    PaddedBatchGatherIndexLayout inner_layout, int64_t output_rows, int64_t indices_per_batch,
    int64_t batch_count, int64_t outer_entries, int64_t inner_entries, int64_t row_width,
    int64_t source_stride_batch, int64_t source_stride_outer, int64_t source_stride_inner) {
  const int lane = threadIdx.x % group_width;
  const int group = threadIdx.x / group_width;
  const int groups_per_block = blockDim.x / group_width;
  extern __shared__ int64_t source_offsets[];
  for (int64_t first_row = static_cast<int64_t>(blockIdx.x) * groups_per_block;
       first_row < output_rows;
       first_row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t output_row = first_row + group;
    if (lane == 0) {
      int64_t source_offset = -1;
      if (output_row < output_rows) {
        const int64_t outer_offset =
            padded_batch_gather_index_offset(output_row, outer_layout);
        const int64_t outer_index =
            static_cast<int64_t>(outer_stored_indices[outer_offset]);
        int64_t inner_index = 1;
        if constexpr (has_inner_indices) {
          const int64_t inner_offset =
              padded_batch_gather_index_offset(output_row, inner_layout);
          inner_index = static_cast<int64_t>(inner_stored_indices[inner_offset]);
        }
        int64_t batch_index = output_row / indices_per_batch;
        if constexpr (has_explicit_batch_indices) {
          const int64_t batch_offset =
              padded_batch_gather_index_offset(output_row, batch_layout);
          batch_index = padded_batch_gather_index_value(
              batch_indices, batch_offset, batch_index_type);
        }
        if (batch_index >= 0 && batch_index < batch_count &&
            outer_index > 0 && outer_index <= outer_entries &&
            inner_index > 0 && inner_index <= inner_entries) {
          source_offset = batch_index * source_stride_batch +
              (outer_index - 1) * source_stride_outer;
          if constexpr (has_inner_indices) {
            source_offset += (inner_index - 1) * source_stride_inner;
          }
        }
      }
      source_offsets[group] = source_offset;
    }
    __syncthreads();
    if (output_row < output_rows) {
      const int64_t source_offset = source_offsets[group];
      const int64_t output_offset = output_row * row_width;
      for (int64_t column = lane; column < row_width; column += group_width) {
        output[output_offset + column] = source_offset >= 0
            ? source[source_offset + column]
            : static_cast<scalar_t>(0);
      }
    }
    __syncthreads();
  }
}

template <int queries_per_wave, typename scalar_t, typename index_t>
__global__ void indexed_relation_bias_kernel(const scalar_t* relation_logits, const scalar_t* relation_bias,
    const index_t* relation_ids, scalar_t* output, int64_t query_chunk_count, int64_t work_item_count,
    int64_t heads, int64_t query_tokens, int64_t key_tokens, int64_t relation_count, int64_t logits_stride_batch,
    int64_t logits_stride_head,
    int64_t logits_stride_query, int64_t logits_stride_relation, int64_t bias_stride_batch, int64_t bias_stride_head,
    int64_t bias_stride_query, int64_t bias_stride_key, int64_t relation_id_stride_batch,
    int64_t relation_id_stride_query, int64_t relation_id_stride_key, float scale) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t work_item = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (work_item >= work_item_count) {
    return;
  }
  const int64_t query_chunk = work_item % query_chunk_count;
  const int64_t batch_head = work_item / query_chunk_count;
  const int64_t head = batch_head % heads;
  const int64_t batch = batch_head / heads;
  const int64_t first_query = query_chunk * queries_per_wave;
  const int64_t query_limit = min(first_query + queries_per_wave, query_tokens);
  for (int64_t query_token = first_query; query_token < query_limit; ++query_token) {
    for (int64_t key_token = lane; key_token < key_tokens; key_token += warpSize) {
      const int64_t relation = static_cast<int64_t>(
          relation_ids[batch * relation_id_stride_batch + query_token * relation_id_stride_query +
                       key_token * relation_id_stride_key]);
      const int64_t output_index =
          ((batch_head * query_tokens + query_token) * key_tokens) + key_token;
      if (relation < 0 || relation >= relation_count) {
        output[output_index] = static_cast<scalar_t>(NAN);
        continue;
      }
      const float logit =
          static_cast<float>(relation_logits[batch * logits_stride_batch + head * logits_stride_head +
                                             query_token * logits_stride_query + relation * logits_stride_relation]);
      const float bias = static_cast<float>(relation_bias[batch * bias_stride_batch + head * bias_stride_head +
                                                          query_token * bias_stride_query +
                                                          key_token * bias_stride_key]);
      output[output_index] = static_cast<scalar_t>(logit * scale + bias);
    }
  }
}

template <int queries_per_wave, typename scalar_t, typename index_t>
__global__ void indexed_relation_bias_logit_gradient_kernel(const scalar_t* gradient_output,
    const index_t* relation_ids, scalar_t* relation_logits_gradient, int64_t query_chunk_count,
    int64_t work_item_count, int64_t heads, int64_t query_tokens, int64_t key_tokens, int64_t relation_count,
    int64_t relation_id_stride_batch, int64_t relation_id_stride_query, int64_t relation_id_stride_key,
    float scale) {
  extern __shared__ float shared_gradients[];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t work_item = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  const bool active = work_item < work_item_count;
  const int64_t query_chunk = active ? work_item % query_chunk_count : 0;
  const int64_t batch_head = active ? work_item / query_chunk_count : 0;
  const int64_t first_query = query_chunk * queries_per_wave;
  const int64_t batch = batch_head / heads;
  float* wave_gradients = shared_gradients + wave * relation_count;

  for (int query_offset = 0; query_offset < queries_per_wave; ++query_offset) {
    const int64_t query_token = first_query + query_offset;
    const bool active_query = active && query_token < query_tokens;
    for (int64_t relation = lane; relation < relation_count; relation += warpSize) {
      wave_gradients[relation] = 0.0f;
    }
    __syncthreads();
    if (active_query) {
      for (int64_t key_token = lane; key_token < key_tokens; key_token += warpSize) {
        const int64_t relation = static_cast<int64_t>(
            relation_ids[batch * relation_id_stride_batch + query_token * relation_id_stride_query +
                         key_token * relation_id_stride_key]);
        if (relation >= 0 && relation < relation_count) {
          const int64_t gradient_index =
              ((batch_head * query_tokens + query_token) * key_tokens) + key_token;
          atomicAdd(wave_gradients + relation, static_cast<float>(gradient_output[gradient_index]) * scale);
        }
      }
    }
    __syncthreads();
    if (active_query) {
      for (int64_t relation = lane; relation < relation_count; relation += warpSize) {
        const int64_t gradient_index =
            ((batch_head * query_tokens + query_token) * relation_count) + relation;
        relation_logits_gradient[gradient_index] = static_cast<scalar_t>(wave_gradients[relation]);
      }
    }
    __syncthreads();
  }
}

struct GroupedIndexedAttentionLayout {
  int64_t queries_per_group;
  int64_t heads;
  int64_t key_features;
  int64_t value_features;
  int64_t shared_tokens;
  int64_t indexed_tokens;
  int64_t packed_width;
  int64_t query_stride_query;
  int64_t query_stride_head;
  int64_t query_stride_feature;
  int64_t shared_stride_group;
  int64_t shared_stride_token;
  int64_t shared_stride_feature;
  int64_t shared_delta_stride_query;
  int64_t shared_delta_stride_token;
  int64_t shared_delta_stride_feature;
  int64_t indexed_delta_stride_query;
  int64_t indexed_delta_stride_token;
  int64_t indexed_delta_stride_feature;
  int64_t indexed_id_stride_query;
  int64_t indexed_id_stride_token;
};

template <bool contiguous>
__device__ __forceinline__ int64_t query_offset(
    const GroupedIndexedAttentionLayout& layout, int64_t query, int64_t head, int64_t feature) {
  if constexpr (contiguous) {
    return (query * layout.heads + head) * layout.key_features + feature;
  }
  return query * layout.query_stride_query + head * layout.query_stride_head +
         feature * layout.query_stride_feature;
}

template <bool contiguous>
__device__ __forceinline__ int64_t shared_offset(const GroupedIndexedAttentionLayout& layout, int64_t group,
    int64_t token, int64_t feature) {
  if constexpr (contiguous) {
    return (group * layout.shared_tokens + token) * layout.packed_width + feature;
  }
  return group * layout.shared_stride_group + token * layout.shared_stride_token +
         feature * layout.shared_stride_feature;
}

template <bool contiguous>
__device__ __forceinline__ int64_t shared_delta_offset(const GroupedIndexedAttentionLayout& layout, int64_t query,
    int64_t token, int64_t feature) {
  if constexpr (contiguous) {
    return (query * layout.shared_tokens + token) * layout.packed_width + feature;
  }
  return query * layout.shared_delta_stride_query + token * layout.shared_delta_stride_token +
         feature * layout.shared_delta_stride_feature;
}

template <bool contiguous>
__device__ __forceinline__ int64_t indexed_delta_offset(const GroupedIndexedAttentionLayout& layout, int64_t query,
    int64_t token, int64_t feature) {
  if constexpr (contiguous) {
    return (query * layout.indexed_tokens + token) * layout.packed_width + feature;
  }
  return query * layout.indexed_delta_stride_query + token * layout.indexed_delta_stride_token +
         feature * layout.indexed_delta_stride_feature;
}

template <bool contiguous>
__device__ __forceinline__ int64_t indexed_id_offset(
    const GroupedIndexedAttentionLayout& layout, int64_t query, int64_t token) {
  if constexpr (contiguous) {
    return query * layout.indexed_tokens + token;
  }
  return query * layout.indexed_id_stride_query + token * layout.indexed_id_stride_token;
}

template <typename scalar_t, typename index_t, bool contiguous>
__global__ void grouped_indexed_attention_kernel(const scalar_t* query, const scalar_t* shared_key_values,
    const scalar_t* shared_deltas, const scalar_t* indexed_deltas, const index_t* indexed_shared_ids, scalar_t* output,
    float* log_sum_exp, int64_t query_head_count, GroupedIndexedAttentionLayout layout, float scale) {
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
  const int64_t group = query_index / layout.queries_per_group;
  const int64_t key_offset = head * layout.key_features;
  const int64_t value_offset = layout.heads * layout.key_features + head * layout.value_features;

  if (active) {
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      query_values[feature] = static_cast<float>(
          query[query_offset<contiguous>(layout, query_index, head, feature)]);
    }
  }
  __syncthreads();

  float local_maximum = -std::numeric_limits<float>::infinity();
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const bool shared_token = attended_token < layout.shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
      const int64_t stored_id =
          shared_token
              ? attended_token + 1
              : static_cast<int64_t>(
                    indexed_shared_ids[indexed_id_offset<contiguous>(layout, query_index, token)]);
      const bool present = stored_id > 0 && stored_id <= layout.shared_tokens;
      const int64_t shared_index = stored_id - 1;
      float score = -std::numeric_limits<float>::infinity();
      if (present) {
        score = 0.0f;
        for (int64_t feature = 0; feature < layout.key_features; feature++) {
          const int64_t packed_feature = key_offset + feature;
          const float base = static_cast<float>(shared_key_values[shared_offset<contiguous>(
              layout, group, shared_index, packed_feature)]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_deltas[shared_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)])
                  : static_cast<float>(indexed_deltas[indexed_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)]);
          score += query_values[feature] * (base + delta);
        }
        score *= scale;
      }
      shared_indices[attended_token] = present ? static_cast<int32_t>(shared_index) : -1;
      token_weights[attended_token] = score;
      local_maximum = fmaxf(local_maximum, score);
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, offset));
  }
  const float maximum = __shfl(local_maximum, 0);

  float local_sum = 0.0f;
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const float score = token_weights[attended_token];
      const float weight = score == -std::numeric_limits<float>::infinity() ? 0.0f : expf(score - maximum);
      token_weights[attended_token] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down(local_sum, offset);
  }
  const float sum = __shfl(local_sum, 0);
  const float inverse_sum = active ? 1.0f / sum : 0.0f;
  if (active && lane == 0 && log_sum_exp != nullptr) {
    log_sum_exp[query_head] = maximum + logf(sum);
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
        const float base = static_cast<float>(shared_key_values[shared_offset<contiguous>(
            layout, group, shared_index, packed_feature)]);
        const float delta =
            shared_token
                ? static_cast<float>(shared_deltas[shared_delta_offset<contiguous>(
                      layout, query_index, token, packed_feature)])
                : static_cast<float>(indexed_deltas[indexed_delta_offset<contiguous>(
                      layout, query_index, token, packed_feature)]);
        context += token_weights[attended_token] * (base + delta);
      }
    }
    output[query_head * layout.value_features + feature] = static_cast<scalar_t>(context * inverse_sum);
  }
}

template <typename scalar_t, typename index_t, bool contiguous>
void launch_grouped_indexed_attention(const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, torch::Tensor& output, int64_t grid_size, int block_threads,
    size_t shared_memory_bytes, hipStream_t stream, int64_t query_head_count,
    const GroupedIndexedAttentionLayout& layout, float scale, float* log_sum_exp) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_indexed_attention_kernel<scalar_t, index_t, contiguous>),
      dim3(grid_size), dim3(block_threads), shared_memory_bytes, stream, query.data_ptr<scalar_t>(),
      shared_key_values.data_ptr<scalar_t>(), shared_deltas.data_ptr<scalar_t>(),
      indexed_deltas.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<index_t>(), output.data_ptr<scalar_t>(),
      log_sum_exp, query_head_count, layout, scale);
}

template <typename scalar_t, typename index_t, bool contiguous>
__global__ void grouped_indexed_attention_backward_query_kernel(const scalar_t* query,
    const scalar_t* shared_key_values, const scalar_t* shared_deltas, const scalar_t* indexed_deltas,
    const index_t* indexed_shared_ids, const float* log_sum_exp, const scalar_t* gradient_output,
    scalar_t* query_gradient, scalar_t* shared_delta_gradient, scalar_t* indexed_delta_gradient,
    int64_t query_head_count, GroupedIndexedAttentionLayout layout, float scale) {
  extern __shared__ unsigned char shared_storage[];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t total_tokens = layout.shared_tokens + layout.indexed_tokens;
  float* all_token_weights = reinterpret_cast<float*>(shared_storage);
  float* all_score_gradients = all_token_weights + waves_per_block * total_tokens;
  int32_t* all_shared_indices =
      reinterpret_cast<int32_t*>(all_score_gradients + waves_per_block * total_tokens);
  float* all_query_values =
      reinterpret_cast<float*>(all_shared_indices + waves_per_block * total_tokens);
  float* all_output_gradients = all_query_values + waves_per_block * layout.key_features;
  float* token_weights = all_token_weights + wave * total_tokens;
  float* score_gradients = all_score_gradients + wave * total_tokens;
  int32_t* shared_indices = all_shared_indices + wave * total_tokens;
  float* query_values = all_query_values + wave * layout.key_features;
  float* output_gradients = all_output_gradients + wave * layout.value_features;

  const int64_t query_head = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  const bool active = query_head < query_head_count;
  const int64_t query_index = query_head / layout.heads;
  const int64_t head = query_head % layout.heads;
  const int64_t group = query_index / layout.queries_per_group;
  const int64_t key_offset = head * layout.key_features;
  const int64_t value_offset = layout.heads * layout.key_features + head * layout.value_features;

  if (active) {
    for (int64_t feature = lane; feature < layout.key_features; feature += warpSize) {
      query_values[feature] =
          static_cast<float>(query[query_offset<contiguous>(layout, query_index, head, feature)]);
    }
    for (int64_t feature = lane; feature < layout.value_features; feature += warpSize) {
      output_gradients[feature] = static_cast<float>(
          gradient_output[(query_head * layout.value_features) + feature]);
    }
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const bool shared_token = attended_token < layout.shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
      const int64_t stored_id =
          shared_token
              ? attended_token + 1
              : static_cast<int64_t>(
                    indexed_shared_ids[indexed_id_offset<contiguous>(layout, query_index, token)]);
      const bool present = stored_id > 0 && stored_id <= layout.shared_tokens;
      const int64_t shared_index = stored_id - 1;
      float score = -std::numeric_limits<float>::infinity();
      if (present) {
        score = 0.0f;
        for (int64_t feature = 0; feature < layout.key_features; feature++) {
          const int64_t packed_feature = key_offset + feature;
          const float base = static_cast<float>(shared_key_values[shared_offset<contiguous>(
              layout, group, shared_index, packed_feature)]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_deltas[shared_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)])
                  : static_cast<float>(indexed_deltas[indexed_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)]);
          score += query_values[feature] * (base + delta);
        }
        score *= scale;
      }
      shared_indices[attended_token] = present ? static_cast<int32_t>(shared_index) : -1;
      token_weights[attended_token] = present ? expf(score - log_sum_exp[query_head]) : 0.0f;
    }
  }
  __syncthreads();

  float local_weighted_gradient = 0.0f;
  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const int32_t shared_index = shared_indices[attended_token];
      float weight_gradient = 0.0f;
      if (shared_index >= 0) {
        const bool shared_token = attended_token < layout.shared_tokens;
        const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
        for (int64_t feature = 0; feature < layout.value_features; feature++) {
          const int64_t packed_feature = value_offset + feature;
          const float base = static_cast<float>(shared_key_values[shared_offset<contiguous>(
              layout, group, shared_index, packed_feature)]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_deltas[shared_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)])
                  : static_cast<float>(indexed_deltas[indexed_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)]);
          weight_gradient += output_gradients[feature] * (base + delta);
        }
      }
      score_gradients[attended_token] = weight_gradient;
      local_weighted_gradient += token_weights[attended_token] * weight_gradient;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_weighted_gradient += __shfl_down(local_weighted_gradient, offset);
  }
  const float weighted_gradient = __shfl(local_weighted_gradient, 0);

  if (active) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      score_gradients[attended_token] =
          token_weights[attended_token] * (score_gradients[attended_token] - weighted_gradient);
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
        const int32_t shared_index = shared_indices[attended_token];
        if (shared_index >= 0) {
          const bool shared_token = attended_token < layout.shared_tokens;
          const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
          const int64_t packed_feature = key_offset + feature;
          const float base = static_cast<float>(shared_key_values[shared_offset<contiguous>(
              layout, group, shared_index, packed_feature)]);
          const float delta =
              shared_token
                  ? static_cast<float>(shared_deltas[shared_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)])
                  : static_cast<float>(indexed_deltas[indexed_delta_offset<contiguous>(
                        layout, query_index, token, packed_feature)]);
          gradient += score_gradients[attended_token] * (base + delta);
        }
      }
      query_gradient[(query_head * layout.key_features) + feature] = static_cast<scalar_t>(gradient * scale);
    }
  }

  if (shared_delta_gradient != nullptr || indexed_delta_gradient != nullptr) {
    for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
      const bool shared_token = attended_token < layout.shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
      const bool present = shared_indices[attended_token] >= 0;
      const float score_gradient = score_gradients[attended_token] * scale;
      const float weight = token_weights[attended_token];
      for (int64_t feature = 0; feature < layout.key_features; feature++) {
        const int64_t packed_feature = key_offset + feature;
        const scalar_t gradient = static_cast<scalar_t>(present ? score_gradient * query_values[feature] : 0.0f);
        if (shared_token && shared_delta_gradient != nullptr) {
          shared_delta_gradient[(query_index * layout.shared_tokens + token) * layout.packed_width + packed_feature] =
              gradient;
        } else if (!shared_token && indexed_delta_gradient != nullptr) {
          indexed_delta_gradient[(query_index * layout.indexed_tokens + token) * layout.packed_width +
                                 packed_feature] = gradient;
        }
      }
      for (int64_t feature = 0; feature < layout.value_features; feature++) {
        const int64_t packed_feature = value_offset + feature;
        const scalar_t gradient = static_cast<scalar_t>(present ? weight * output_gradients[feature] : 0.0f);
        if (shared_token && shared_delta_gradient != nullptr) {
          shared_delta_gradient[(query_index * layout.shared_tokens + token) * layout.packed_width + packed_feature] =
              gradient;
        } else if (!shared_token && indexed_delta_gradient != nullptr) {
          indexed_delta_gradient[(query_index * layout.indexed_tokens + token) * layout.packed_width +
                                 packed_feature] = gradient;
        }
      }
    }
  }
}

template <int features_per_block, typename scalar_t, typename index_t, bool contiguous_ids>
__global__ void grouped_indexed_attention_shared_key_value_gradient_kernel(
    const scalar_t* shared_delta_gradient,
    const scalar_t* indexed_delta_gradient, const index_t* indexed_shared_ids, scalar_t* shared_gradient,
    int64_t feature_block_count_per_group, GroupedIndexedAttentionLayout layout) {
  extern __shared__ float partial_sums[];
  const int64_t group = static_cast<int64_t>(blockIdx.x) / feature_block_count_per_group;
  const int64_t feature_block = static_cast<int64_t>(blockIdx.x) % feature_block_count_per_group;
  const int64_t first_feature = feature_block * features_per_block;
  const int64_t contribution_count =
      layout.queries_per_group * (layout.shared_tokens + layout.indexed_tokens);
  const int64_t partial_sum_count = layout.shared_tokens * features_per_block;

  for (int64_t index = threadIdx.x; index < partial_sum_count; index += blockDim.x) {
    partial_sums[index] = 0.0f;
  }
  __syncthreads();

  for (int64_t contribution = threadIdx.x; contribution < contribution_count; contribution += blockDim.x) {
    const int64_t local_query = contribution / (layout.shared_tokens + layout.indexed_tokens);
    const int64_t attended_token = contribution % (layout.shared_tokens + layout.indexed_tokens);
    const int64_t query_index = group * layout.queries_per_group + local_query;
    const bool shared_token = attended_token < layout.shared_tokens;
    const int64_t token = shared_token ? attended_token : attended_token - layout.shared_tokens;
    const int64_t stored_id =
        shared_token
            ? attended_token + 1
            : static_cast<int64_t>(
                  indexed_shared_ids[indexed_id_offset<contiguous_ids>(layout, query_index, token)]);
    if (stored_id <= 0 || stored_id > layout.shared_tokens) {
      continue;
    }
    const int64_t shared_index = stored_id - 1;
    const int64_t source_offset =
        shared_token ? (query_index * layout.shared_tokens + token) * layout.packed_width
                     : (query_index * layout.indexed_tokens + token) * layout.packed_width;
    const scalar_t* source = shared_token ? shared_delta_gradient : indexed_delta_gradient;
    for (int64_t local_feature = 0; local_feature < features_per_block; local_feature++) {
      const int64_t feature = first_feature + local_feature;
      if (feature < layout.packed_width) {
        atomicAdd(partial_sums + shared_index * features_per_block + local_feature,
            static_cast<float>(source[source_offset + feature]));
      }
    }
  }
  __syncthreads();

  for (int64_t index = threadIdx.x; index < partial_sum_count; index += blockDim.x) {
    const int64_t shared_token = index / features_per_block;
    const int64_t local_feature = index % features_per_block;
    const int64_t feature = first_feature + local_feature;
    if (feature < layout.packed_width) {
      shared_gradient[(group * layout.shared_tokens + shared_token) * layout.packed_width + feature] =
          static_cast<scalar_t>(partial_sums[index]);
    }
  }
}

template <typename residual_t, typename update_t, typename parameter_t>
__global__ void residual_add_layer_norm_in_place_kernel(residual_t* residual, const update_t* update,
    const parameter_t* weight, const parameter_t* bias, residual_t* output, int64_t row_count, int64_t width,
    float epsilon) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (row >= row_count) {
    return;
  }

  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * width;
  for (int64_t feature = lane; feature < width; feature += warpSize) {
    const int64_t index = row_offset + feature;
    const float value = static_cast<float>(residual[index]) + static_cast<float>(update[index]);
    residual[index] = static_cast<residual_t>(value);
    count++;
    const float delta = value - mean;
    mean += delta / count;
    moment += delta * (value - mean);
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int other_count = __shfl_down(count, offset);
    if (lane + offset < warpSize && other_count > 0) {
      const int combined_count = count + other_count;
      const float delta = other_mean - mean;
      moment += other_moment + delta * delta * static_cast<float>(count) * static_cast<float>(other_count) /
                                   static_cast<float>(combined_count);
      mean += delta * static_cast<float>(other_count) / static_cast<float>(combined_count);
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / static_cast<float>(width) + epsilon);

  for (int64_t feature = lane; feature < width; feature += warpSize) {
    const int64_t index = row_offset + feature;
    const float normalized = (static_cast<float>(residual[index]) - mean) * inverse_standard_deviation;
    output[index] =
        static_cast<residual_t>(normalized * static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]));
  }
}

template <int cached_values_per_lane, typename residual_t, typename update_t, typename parameter_t>
__global__ void residual_add_layer_norm_cached_in_place_kernel(residual_t* residual,
    const update_t* update, const parameter_t* weight, const parameter_t* bias, residual_t* output,
    int64_t row_count, int64_t width, float epsilon) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (row >= row_count) {
    return;
  }

  float cached_values[cached_values_per_lane];
  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * width;
#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; cached_index++) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * warpSize;
    if (feature < width) {
      const int64_t index = row_offset + feature;
      const float value = static_cast<float>(residual[index]) + static_cast<float>(update[index]);
      const residual_t rounded_value = static_cast<residual_t>(value);
      residual[index] = rounded_value;
      cached_values[cached_index] = static_cast<float>(rounded_value);
      count++;
      const float delta = value - mean;
      mean += delta / count;
      moment += delta * (value - mean);
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int other_count = __shfl_down(count, offset);
    if (lane + offset < warpSize && other_count > 0) {
      const int combined_count = count + other_count;
      const float delta = other_mean - mean;
      moment += other_moment + delta * delta * static_cast<float>(count) * static_cast<float>(other_count) /
                                   static_cast<float>(combined_count);
      mean += delta * static_cast<float>(other_count) / static_cast<float>(combined_count);
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  const float inverse_standard_deviation =
      rsqrtf(__shfl(moment, 0) / static_cast<float>(width) + epsilon);

#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; cached_index++) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * warpSize;
    if (feature < width) {
      const float normalized = (cached_values[cached_index] - mean) * inverse_standard_deviation;
      output[row_offset + feature] = static_cast<residual_t>(
          normalized * static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]));
    }
  }
}

template <int cached_values_per_lane, typename input_t, typename parameter_t>
__global__ void autocast_layer_norm_kernel(const input_t* input, const parameter_t* weight,
    const parameter_t* bias, float* output, int64_t row_count, int64_t width, float epsilon) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (row >= row_count) {
    return;
  }

  float cached_values[cached_values_per_lane];
  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * width;
#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; cached_index++) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * warpSize;
    if (feature < width) {
      const float value = static_cast<float>(input[row_offset + feature]);
      cached_values[cached_index] = value;
      count++;
      const float delta = value - mean;
      mean += delta / count;
      moment += delta * (value - mean);
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int other_count = __shfl_down(count, offset);
    if (lane + offset < warpSize && other_count > 0) {
      const int combined_count = count + other_count;
      const float delta = other_mean - mean;
      moment += other_moment + delta * delta * static_cast<float>(count) *
                                   static_cast<float>(other_count) /
                                   static_cast<float>(combined_count);
      mean += delta * static_cast<float>(other_count) /
              static_cast<float>(combined_count);
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  const float inverse_standard_deviation =
      rsqrtf(__shfl(moment, 0) / static_cast<float>(width) + epsilon);

#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; cached_index++) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * warpSize;
    if (feature < width) {
      const float normalized =
          (cached_values[cached_index] - mean) * inverse_standard_deviation;
      output[row_offset + feature] =
          normalized * static_cast<float>(weight[feature]) +
          static_cast<float>(bias[feature]);
    }
  }
}

const hipDeviceProp_t& device_properties(int device) {
  static const std::vector<hipDeviceProp_t> all_properties = [] {
    int device_count;
    C10_HIP_CHECK(hipGetDeviceCount(&device_count));
    std::vector<hipDeviceProp_t> properties(device_count);
    for (int index = 0; index < device_count; ++index) {
      C10_HIP_CHECK(hipGetDeviceProperties(&properties[index], index));
    }
    return properties;
  }();
  TORCH_CHECK(device >= 0 && device < static_cast<int>(all_properties.size()), "invalid ROCm device index");
  const auto& properties = all_properties[device];
  TORCH_CHECK(
      properties.warpSize == 32 || properties.warpSize == 64, "ROCm kernels require a 32- or 64-lane wavefront");
  return properties;
}

int configured_threads_per_block(
    const hipDeviceProp_t& properties, int requested_threads, const char* setting, bool require_complete_waves) {
  TORCH_CHECK(requested_threads <= properties.maxThreadsPerBlock,
      setting, " exceeds this device's maximum threads per block: ", requested_threads, " > ",
      properties.maxThreadsPerBlock);
  TORCH_CHECK(!require_complete_waves || requested_threads % properties.warpSize == 0,
      setting, " must be divisible by the device wavefront size ", properties.warpSize,
      ", but was ", requested_threads);
  return requested_threads;
}

int configured_waves_per_block(
    const hipDeviceProp_t& properties, int requested_waves, const char* setting) {
  TORCH_CHECK(requested_waves <= properties.maxThreadsPerBlock / properties.warpSize,
      setting, " exceeds this device's maximum waves per block: ", requested_waves, " > ",
      properties.maxThreadsPerBlock / properties.warpSize);
  return requested_waves;
}

template <typename Callback>
decltype(auto) dispatch_compiled_work_size(int value, const char* setting, Callback&& callback) {
  switch (value) {
    case 1:
      return callback(std::integral_constant<int, 1>{});
    case 2:
      return callback(std::integral_constant<int, 2>{});
    case 4:
      return callback(std::integral_constant<int, 4>{});
    case 8:
      return callback(std::integral_constant<int, 8>{});
    case 16:
      return callback(std::integral_constant<int, 16>{});
    case 32:
      return callback(std::integral_constant<int, 32>{});
    default:
      TORCH_CHECK(false, setting, " has no compiled kernel specialization for ", value);
  }
}

int masked_categorical_group_width(int64_t width, int hardware_wave_width) {
  if (width <= 4) {
    return 4;
  }
  if (width <= 8) {
    return 8;
  }
  if (width <= 16) {
    return 16;
  }
  return hardware_wave_width;
}

template <typename Callback>
void dispatch_masked_categorical_subwave_width(int64_t width, Callback&& callback) {
  if (width <= 4) {
    callback(std::integral_constant<int, 4>{});
  } else if (width <= 8) {
    callback(std::integral_constant<int, 8>{});
  } else if (width <= 16) {
    callback(std::integral_constant<int, 16>{});
  } else {
    TORCH_CHECK(false, "masked categorical subwave width is unavailable for width ", width);
  }
}

template <typename Callback>
void dispatch_scatter_rows_group_width(int group_width, Callback&& callback) {
  switch (group_width) {
    case 32:
      callback(std::integral_constant<int, 32>{});
      break;
    case 64:
      callback(std::integral_constant<int, 64>{});
      break;
    default:
      TORCH_CHECK(false, "unsupported scatter rows group width: ", group_width);
  }
}

int padded_batch_gather_group_width(int64_t row_width, int hardware_wave_width) {
  int group_width = 1;
  while (group_width < row_width && group_width < hardware_wave_width) {
    group_width *= 2;
  }
  return group_width;
}

template <typename Callback>
void dispatch_padded_batch_gather_group_width(int group_width, Callback&& callback) {
  switch (group_width) {
    case 1:
      callback(std::integral_constant<int, 1>{});
      break;
    case 2:
      callback(std::integral_constant<int, 2>{});
      break;
    case 4:
      callback(std::integral_constant<int, 4>{});
      break;
    case 8:
      callback(std::integral_constant<int, 8>{});
      break;
    case 16:
      callback(std::integral_constant<int, 16>{});
      break;
    case 32:
      callback(std::integral_constant<int, 32>{});
      break;
    case 64:
      callback(std::integral_constant<int, 64>{});
      break;
    default:
      TORCH_CHECK(false, "unsupported padded batch gather group width: ", group_width);
  }
}

template <int compiled_values_per_lane = 1, typename Callback>
void dispatch_residual_cached_values_per_lane(int values_per_lane, Callback&& callback) {
  if (values_per_lane == compiled_values_per_lane) {
    callback(std::integral_constant<int, compiled_values_per_lane>{});
  } else if constexpr (compiled_values_per_lane < kResidualCachedValuesPerLaneLimit) {
    dispatch_residual_cached_values_per_lane<compiled_values_per_lane + 1>(
        values_per_lane, callback);
  } else {
    TORCH_CHECK(false, "unsupported residual cache values per lane: ", values_per_lane);
  }
}

template <int compiled_values_per_lane = 1, typename Callback>
void dispatch_autocast_layer_norm_cached_values_per_lane(
    int values_per_lane, Callback&& callback) {
  if (values_per_lane == compiled_values_per_lane) {
    callback(std::integral_constant<int, compiled_values_per_lane>{});
  } else if constexpr (
      compiled_values_per_lane < kAutocastLayerNormCachedValuesPerLaneLimit) {
    dispatch_autocast_layer_norm_cached_values_per_lane<compiled_values_per_lane + 1>(
        values_per_lane, callback);
  } else {
    TORCH_CHECK(false, "unsupported autocast LayerNorm cache values per lane: ", values_per_lane);
  }
}

struct IndexedRelationBiasLaunchPlan {
  int block_threads;
  int64_t query_chunk_count;
  int64_t work_item_count;
  int64_t grid_size;
  size_t shared_memory_bytes;
};

std::optional<IndexedRelationBiasLaunchPlan> indexed_relation_bias_launch_plan(
    int device, int64_t batch, int64_t heads,
    int64_t query_tokens, int64_t relation_count, int requested_waves,
    int queries_per_wave, bool backward) {
  const auto& properties = device_properties(device);
  const int waves_per_block = configured_waves_per_block(properties, requested_waves,
      backward ? launch_environment::kIndexedRelationBiasBackwardWavesPerBlock
               : launch_environment::kIndexedRelationBiasForwardWavesPerBlock);
  const int64_t query_chunk_count =
      (query_tokens + queries_per_wave - 1) / queries_per_wave;
  const int64_t work_item_count = batch * heads * query_chunk_count;
  const int64_t grid_size = (work_item_count + waves_per_block - 1) / waves_per_block;
  const size_t shared_memory_bytes =
      backward ? static_cast<size_t>(waves_per_block) * relation_count * sizeof(float) : 0;
  if (grid_size > static_cast<int64_t>(properties.maxGridSize[0]) ||
      shared_memory_bytes > static_cast<size_t>(properties.sharedMemPerBlock)) {
    return std::nullopt;
  }
  return IndexedRelationBiasLaunchPlan{waves_per_block * properties.warpSize,
      query_chunk_count, work_item_count, grid_size, shared_memory_bytes};
}

struct GroupedIndexedAttentionLaunchPlan {
  int block_threads;
  int64_t grid_size;
  size_t shared_memory_bytes;
};

std::optional<GroupedIndexedAttentionLaunchPlan> grouped_indexed_attention_launch_plan(
    const torch::Tensor& query,
    int64_t shared_tokens, int64_t indexed_tokens, int64_t value_features, bool backward) {
  const auto& properties = device_properties(query.get_device());
  const auto& config = rocm_kernel_launch_config();
  const int requested_maximum_threads =
      backward ? config.grouped_indexed_attention_backward_max_threads_per_block
               : config.grouped_indexed_attention_forward_max_threads_per_block;
  const int maximum_threads_per_block = configured_threads_per_block(properties, requested_maximum_threads,
      backward ? launch_environment::kGroupedIndexedAttentionBackwardMaxThreadsPerBlock
               : launch_environment::kGroupedIndexedAttentionForwardMaxThreadsPerBlock,
      true);
  const int64_t total_tokens = shared_tokens + indexed_tokens;
  const size_t shared_memory_per_wave =
      backward
          ? static_cast<size_t>(total_tokens) * (2 * sizeof(float) + sizeof(int32_t)) +
                static_cast<size_t>(query.size(2) + value_features) * sizeof(float)
          : static_cast<size_t>(total_tokens) * (sizeof(float) + sizeof(int32_t)) +
                static_cast<size_t>(query.size(2)) * sizeof(float);
  const int maximum_waves_by_threads = maximum_threads_per_block / properties.warpSize;
  const int maximum_waves_by_memory =
      static_cast<int>(static_cast<size_t>(properties.sharedMemPerBlock) / shared_memory_per_wave);
  const int64_t query_head_count = query.size(0) * query.size(1);
  const int waves_per_block = static_cast<int>(std::min<int64_t>(
      std::min(maximum_waves_by_threads, maximum_waves_by_memory), query_head_count));
  if (waves_per_block <= 0) {
    return std::nullopt;
  }
  const int64_t grid_size = (query_head_count + waves_per_block - 1) / waves_per_block;
  if (grid_size > static_cast<int64_t>(properties.maxGridSize[0])) {
    return std::nullopt;
  }
  return GroupedIndexedAttentionLaunchPlan{waves_per_block * properties.warpSize,
      grid_size, shared_memory_per_wave * waves_per_block};
}

struct GroupedIndexedAttentionSharedKeyValueGradientLaunchPlan {
  int threads_per_block;
  int64_t feature_block_count_per_group;
  int64_t grid_size;
  size_t shared_memory_bytes;
};

std::optional<GroupedIndexedAttentionSharedKeyValueGradientLaunchPlan>
grouped_indexed_attention_shared_key_value_gradient_launch_plan(
    const torch::Tensor& query, int64_t group_count, int64_t shared_tokens, int64_t packed_width) {
  const auto& properties = device_properties(query.get_device());
  const auto& config = rocm_kernel_launch_config();
  const int threads_per_block = configured_threads_per_block(properties,
      config.grouped_indexed_attention_shared_key_value_gradient_threads_per_block,
      launch_environment::kGroupedIndexedAttentionSharedKeyValueGradientThreadsPerBlock, false);
  const int features_per_block =
      config.grouped_indexed_attention_shared_key_value_gradient_features_per_block;
  const int64_t feature_block_count_per_group =
      (packed_width + features_per_block - 1) / features_per_block;
  const int64_t grid_size = group_count * feature_block_count_per_group;
  const size_t shared_memory_bytes =
      static_cast<size_t>(shared_tokens * features_per_block) * sizeof(float);
  if (grid_size > static_cast<int64_t>(properties.maxGridSize[0]) ||
      shared_memory_bytes > static_cast<size_t>(properties.sharedMemPerBlock)) {
    return std::nullopt;
  }
  return GroupedIndexedAttentionSharedKeyValueGradientLaunchPlan{
      threads_per_block, feature_block_count_per_group, grid_size, shared_memory_bytes};
}

bool is_indexed_relation_bias_layout_supported(const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias, const torch::Tensor& relation_ids) {
  return relation_logits.is_cuda() && relation_logits.dim() == 4 && is_fused_floating(relation_logits) &&
         relation_logits.scalar_type() == relation_bias.scalar_type() &&
         (relation_ids.scalar_type() == torch::kInt32 || relation_ids.scalar_type() == torch::kInt64) &&
         (relation_ids.dim() == 2 || relation_ids.dim() == 3) && relation_logits.size(2) > 0 &&
         relation_logits.size(3) > 0 && relation_ids.size(-1) > 0 &&
         relation_ids.size(-2) == relation_logits.size(2) &&
         (relation_ids.dim() != 3 || relation_ids.size(0) == relation_logits.size(0)) &&
         relation_bias.device() == relation_logits.device() && relation_ids.device() == relation_logits.device();
}

bool is_grouped_indexed_attention_layout_supported(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids,
    int64_t queries_per_group) {
  if (!query.is_cuda() || query.dim() != 3 || shared_key_values.dim() != 3 || shared_deltas.dim() != 3 ||
      indexed_deltas.dim() != 3 || indexed_shared_ids.dim() != 2 || queries_per_group <= 0 ||
      !is_fused_floating(query) || query.scalar_type() != shared_key_values.scalar_type() ||
      query.scalar_type() != shared_deltas.scalar_type() || query.scalar_type() != indexed_deltas.scalar_type() ||
      (indexed_shared_ids.scalar_type() != torch::kInt32 && indexed_shared_ids.scalar_type() != torch::kInt64)) {
    return false;
  }
  const int64_t query_count = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t group_count = shared_key_values.size(0);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t key_width = heads * key_features;
  return query_count > 0 && heads > 0 && key_features > 0 && shared_tokens > 0 &&
         query_count == group_count * queries_per_group && packed_width > key_width &&
         (packed_width - key_width) % heads == 0 && shared_deltas.size(0) == query_count &&
         shared_deltas.size(1) == shared_tokens && shared_deltas.size(2) == packed_width &&
         indexed_deltas.size(0) == query_count && indexed_deltas.size(2) == packed_width &&
         indexed_shared_ids.size(0) == query_count && indexed_shared_ids.size(1) == indexed_tokens &&
         shared_key_values.device() == query.device() && shared_deltas.device() == query.device() &&
         indexed_deltas.device() == query.device() && indexed_shared_ids.device() == query.device();
}

bool has_supported_padded_gather_index_layout(
    const torch::Tensor& indices, const torch::Device& device) {
  if (!indices.is_cuda() || indices.device() != device || indices.dim() < 1 ||
      indices.dim() > kPaddedBatchGatherMaximumIndexRank || indices.numel() == 0 ||
      (indices.scalar_type() != torch::kInt16 && indices.scalar_type() != torch::kInt32 &&
          indices.scalar_type() != torch::kInt64)) {
    return false;
  }
  for (int64_t axis = 0; axis < indices.dim(); ++axis) {
    if (indices.size(axis) <= 0 || indices.stride(axis) < 0) {
      return false;
    }
  }
  return true;
}

bool has_contiguous_padded_gather_rows(
    const torch::Tensor& source, int64_t indexed_dimensions) {
  if (!source.is_cuda() || !is_fused_floating(source) ||
      source.dim() < indexed_dimensions || source.numel() == 0) {
    return false;
  }
  int64_t expected_stride = 1;
  for (int64_t axis = source.dim() - 1; axis >= indexed_dimensions; --axis) {
    if (source.size(axis) <= 0 || source.stride(axis) != expected_stride) {
      return false;
    }
    expected_stride *= source.size(axis);
  }
  for (int64_t axis = 0; axis < indexed_dimensions; ++axis) {
    if (source.size(axis) <= 0 || source.stride(axis) < 0) {
      return false;
    }
  }
  return true;
}

PaddedBatchGatherIndexLayout padded_batch_gather_index_layout(
    const torch::Tensor& indices) {
  PaddedBatchGatherIndexLayout layout{};
  layout.rank = indices.dim();
  for (int64_t axis = 0; axis < indices.dim(); ++axis) {
    layout.sizes[axis] = indices.size(axis);
    layout.strides[axis] = indices.stride(axis);
  }
  return layout;
}

PaddedBatchGatherIndexType padded_batch_gather_index_type(
    const torch::Tensor& indices) {
  switch (indices.scalar_type()) {
    case torch::kInt16:
      return PaddedBatchGatherIndexType::kInt16;
    case torch::kInt32:
      return PaddedBatchGatherIndexType::kInt32;
    case torch::kInt64:
      return PaddedBatchGatherIndexType::kInt64;
    default:
      TORCH_CHECK(false, "unsupported padded batch gather index data type");
  }
}

template <bool has_inner_indices, bool has_explicit_batch_indices, typename stored_index_t>
torch::Tensor launch_padded_batch_gather(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  constexpr int64_t indexed_dimensions = has_inner_indices ? 3 : 2;
  auto output_shape = outer_stored_indices.sizes().vec();
  output_shape.insert(
      output_shape.end(), source.sizes().begin() + indexed_dimensions, source.sizes().end());
  auto output = torch::empty(output_shape, source.options());
  if (output.numel() == 0) {
    return output;
  }

  const auto& properties = device_properties(source.get_device());
  const int group_width = padded_batch_gather_group_width(
      output.numel() / outer_stored_indices.numel(), properties.warpSize);
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().padded_batch_gather_threads_per_block,
      launch_environment::kPaddedBatchGatherThreadsPerBlock, false);
  TORCH_CHECK(threads_per_block % group_width == 0,
      launch_environment::kPaddedBatchGatherThreadsPerBlock,
      " must be divisible by the selected gather group width ", group_width);
  const int groups_per_block = threads_per_block / group_width;
  const int64_t required_blocks =
      (outer_stored_indices.numel() + groups_per_block - 1) / groups_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const size_t shared_memory_bytes =
      static_cast<size_t>(groups_per_block) * sizeof(int64_t);
  c10::DeviceGuard device_guard(source.device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(source.get_device()).stream();

  PaddedBatchGatherIndexLayout batch_layout{};
  PaddedBatchGatherIndexLayout inner_layout{};
  if constexpr (has_explicit_batch_indices) {
    batch_layout = padded_batch_gather_index_layout(batch_indices);
  }
  if constexpr (has_inner_indices) {
    inner_layout = padded_batch_gather_index_layout(inner_stored_indices);
  }
  const auto outer_layout = padded_batch_gather_index_layout(outer_stored_indices);
  const void* batch_data = nullptr;
  PaddedBatchGatherIndexType batch_index_type = PaddedBatchGatherIndexType::kInt16;
  const stored_index_t* inner_data = nullptr;
  if constexpr (has_explicit_batch_indices) {
    batch_data = batch_indices.data_ptr();
    batch_index_type = padded_batch_gather_index_type(batch_indices);
  }
  if constexpr (has_inner_indices) {
    inner_data = inner_stored_indices.data_ptr<stored_index_t>();
  }
  const int64_t row_width = output.numel() / outer_stored_indices.numel();
  const int64_t indices_per_batch = has_explicit_batch_indices
      ? 1
      : outer_stored_indices.numel() / source.size(0);
  const int64_t inner_entries = has_inner_indices ? source.size(2) : 1;
  const int64_t source_stride_inner = has_inner_indices ? source.stride(2) : 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, source.scalar_type(), "padded_batch_gather", [&] {
        dispatch_padded_batch_gather_group_width(group_width, [&](auto width) {
          constexpr int compiled_group_width = decltype(width)::value;
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(padded_batch_gather_kernel<compiled_group_width, scalar_t,
                  stored_index_t, has_inner_indices, has_explicit_batch_indices>),
              dim3(blocks), dim3(threads_per_block), shared_memory_bytes, stream,
              source.data_ptr<scalar_t>(), batch_data,
              outer_stored_indices.data_ptr<stored_index_t>(), inner_data,
              output.data_ptr<scalar_t>(), batch_index_type, batch_layout, outer_layout, inner_layout,
              outer_stored_indices.numel(), indices_per_batch, source.size(0), source.size(1),
              inner_entries, row_width, source.stride(0), source.stride(1),
              source_stride_inner);
        });
      });
  check_hip_kernel_launch("padded batch gather");
  return output;
}

template <bool has_inner_indices, bool has_explicit_batch_indices>
torch::Tensor dispatch_padded_batch_gather(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  switch (outer_stored_indices.scalar_type()) {
    case torch::kInt16:
      return launch_padded_batch_gather<
          has_inner_indices, has_explicit_batch_indices, int16_t>(
          source, batch_indices, outer_stored_indices, inner_stored_indices);
    case torch::kInt32:
      return launch_padded_batch_gather<
          has_inner_indices, has_explicit_batch_indices, int32_t>(
          source, batch_indices, outer_stored_indices, inner_stored_indices);
    case torch::kInt64:
      return launch_padded_batch_gather<
          has_inner_indices, has_explicit_batch_indices, int64_t>(
          source, batch_indices, outer_stored_indices, inner_stored_indices);
    default:
      TORCH_CHECK(false, "unsupported padded batch gather stored-index data type");
  }
}

}  // namespace

bool supports_indexed_relation_bias_forward(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids) {
  if (!is_indexed_relation_bias_layout_supported(relation_logits, relation_bias, relation_ids)) {
    return false;
  }
  const auto& config = rocm_kernel_launch_config();
  return indexed_relation_bias_launch_plan(relation_logits.get_device(), relation_logits.size(0),
      relation_logits.size(1), relation_logits.size(2), relation_logits.size(3),
      config.indexed_relation_bias_forward_waves_per_block,
      config.indexed_relation_bias_forward_queries_per_wave, false).has_value();
}

bool supports_indexed_relation_bias_logit_gradient(const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias, const torch::Tensor& relation_ids) {
  if (!is_indexed_relation_bias_layout_supported(relation_logits, relation_bias, relation_ids)) {
    return false;
  }
  const auto& config = rocm_kernel_launch_config();
  return indexed_relation_bias_launch_plan(relation_logits.get_device(), relation_logits.size(0),
      relation_logits.size(1), relation_logits.size(2), relation_logits.size(3),
      config.indexed_relation_bias_backward_waves_per_block,
      config.indexed_relation_bias_backward_queries_per_wave, true).has_value();
}

bool supports_grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group) {
  if (!is_grouped_indexed_attention_layout_supported(query, shared_key_values, shared_deltas,
          indexed_deltas, indexed_shared_ids, queries_per_group)) {
    return false;
  }
  const int64_t heads = query.size(1);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t value_features = (packed_width - heads * query.size(2)) / heads;
  return grouped_indexed_attention_launch_plan(
      query, shared_tokens, indexed_tokens, value_features, false).has_value();
}

bool supports_grouped_indexed_attention_backward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids,
    int64_t queries_per_group, bool needs_shared_key_value_gradient) {
  if (!is_grouped_indexed_attention_layout_supported(query, shared_key_values, shared_deltas,
          indexed_deltas, indexed_shared_ids, queries_per_group)) {
    return false;
  }
  const int64_t heads = query.size(1);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t value_features = (packed_width - heads * query.size(2)) / heads;
  if (!grouped_indexed_attention_launch_plan(
          query, shared_tokens, indexed_tokens, value_features, true).has_value()) {
    return false;
  }
  return !needs_shared_key_value_gradient ||
      grouped_indexed_attention_shared_key_value_gradient_launch_plan(
          query, shared_key_values.size(0), shared_tokens, packed_width).has_value();
}

bool supports_owned_residual_layer_norm(const torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias) {
  return residual.is_cuda() && residual.is_contiguous() && update.is_contiguous() && weight.is_contiguous() &&
         bias.is_contiguous() && residual.dim() >= 1 && residual.size(-1) > 0 &&
         residual.sizes() == update.sizes() && weight.dim() == 1 && weight.size(0) == residual.size(-1) &&
         bias.sizes() == weight.sizes() && is_fused_floating(residual) && is_fused_floating(update) &&
         is_fused_floating(weight) && weight.scalar_type() == bias.scalar_type() &&
         supports_owned_residual_layer_norm_types(
             residual.scalar_type(), update.scalar_type(), weight.scalar_type()) &&
         update.device() == residual.device() && weight.device() == residual.device() &&
         bias.device() == residual.device();
}

bool supports_autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, at::IntArrayRef normalized_shape) {
  if (!input.is_cuda() || !input.is_contiguous() || input.dim() < 1 || input.size(-1) <= 0 ||
      normalized_shape.size() != 1 || normalized_shape[0] != input.size(-1) ||
      !weight.defined() || !bias.defined() || !weight.is_contiguous() || !bias.is_contiguous() ||
      weight.dim() != 1 || weight.size(0) != input.size(-1) || bias.sizes() != weight.sizes() ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kBFloat16) ||
      (weight.scalar_type() != torch::kFloat16 && weight.scalar_type() != torch::kBFloat16) ||
      bias.scalar_type() != weight.scalar_type() || weight.device() != input.device() ||
      bias.device() != input.device()) {
    return false;
  }
  const auto& properties = device_properties(input.get_device());
  const int64_t width = input.size(-1);
  const int64_t cached_values_per_lane =
      (width + properties.warpSize - 1) / properties.warpSize;
  if (cached_values_per_lane > kAutocastLayerNormCachedValuesPerLaneLimit) {
    return false;
  }
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().residual_add_layer_norm_threads_per_block,
      launch_environment::kResidualAddLayerNormThreadsPerBlock, true);
  const int64_t waves_per_block = threads_per_block / properties.warpSize;
  const int64_t row_count = input.numel() / width;
  const int64_t grid_size = (row_count + waves_per_block - 1) / waves_per_block;
  return grid_size <= static_cast<int64_t>(properties.maxGridSize[0]);
}

bool supports_fused_adam_update(const torch::Tensor& weight, const torch::Tensor& gradient,
    const torch::Tensor& mean, const torch::Tensor& variance) {
  return weight.is_cuda() && weight.is_contiguous() && gradient.is_contiguous() && mean.is_contiguous() &&
         variance.is_contiguous() && is_fused_floating(weight) && gradient.scalar_type() == weight.scalar_type() &&
         mean.scalar_type() == weight.scalar_type() && variance.scalar_type() == weight.scalar_type() &&
         gradient.sizes() == weight.sizes() && mean.sizes() == weight.sizes() &&
         variance.sizes() == weight.sizes() && gradient.device() == weight.device() &&
         mean.device() == weight.device() && variance.device() == weight.device();
}

bool supports_masked_categorical(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  return logits.is_cuda() && logits.is_contiguous() && mask.is_contiguous() &&
         is_fused_floating(logits) && mask.scalar_type() == torch::kBool &&
         mask.sizes() == logits.sizes() && mask.device() == logits.device() && logits.dim() > 0 &&
         logits.size(-1) > 0 && axis == logits.dim() - 1;
}

bool supports_grouped_masked_softmax_pool(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values) {
  if (!logits.is_cuda() || !logits.is_contiguous() || !mask.is_contiguous() ||
      !values.is_contiguous() || !is_fused_floating(logits) ||
      !is_fused_floating(values) || mask.scalar_type() != torch::kBool ||
      logits.device() != mask.device() || logits.device() != values.device() ||
      logits.dim() == 0 || mask.dim() != logits.dim() + 1 ||
      values.dim() != logits.dim() + 1 || logits.size(-1) <= 0 ||
      mask.size(-2) != logits.size(-1) || values.size(-2) != logits.size(-1) ||
      mask.size(-1) <= 0 || values.size(-1) <= 0) {
    return false;
  }
  for (int64_t dimension = 0; dimension < logits.dim() - 1; ++dimension) {
    if (mask.size(dimension) != logits.size(dimension) ||
        values.size(dimension) != logits.size(dimension)) {
      return false;
    }
  }
  return logits.size(-1) <= device_properties(logits.get_device()).warpSize;
}

bool supports_scatter_rows(const torch::Tensor& rows, const torch::Tensor& row_indices) {
  return rows.is_cuda() && rows.is_contiguous() && is_fused_floating(rows) &&
         row_indices.is_cuda() && row_indices.is_contiguous() &&
         row_indices.scalar_type() == torch::kInt64 && row_indices.device() == rows.device();
}

bool supports_padded_batch_gather(
    const torch::Tensor& source, const torch::Tensor& stored_indices) {
  return has_contiguous_padded_gather_rows(source, 2) &&
      has_supported_padded_gather_index_layout(stored_indices, source.device());
}

bool supports_padded_batch_gather_2d(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices) {
  return has_contiguous_padded_gather_rows(source, 3) &&
      outer_stored_indices.scalar_type() == inner_stored_indices.scalar_type() &&
      has_supported_padded_gather_index_layout(outer_stored_indices, source.device()) &&
      has_supported_padded_gather_index_layout(inner_stored_indices, source.device());
}

bool supports_padded_batch_gather_by_batch_indices(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices) {
  return has_contiguous_padded_gather_rows(source, 2) &&
      has_supported_padded_gather_index_layout(batch_indices, source.device()) &&
      has_supported_padded_gather_index_layout(stored_indices, source.device());
}

void fused_adam_update(torch::Tensor& weight, const torch::Tensor& gradient, torch::Tensor& mean,
    torch::Tensor& variance, float learning_rate, float learning_rate_bias_correction, float weight_decay,
    float rescale_gradient, float clip_gradient, float beta1, float beta2, float epsilon, bool adamw) {
  TORCH_CHECK(supports_fused_adam_update(weight, gradient, mean, variance),
      "fused Adam update received an unsupported ROCm tensor layout");
  if (weight.numel() == 0) {
    return;
  }
  c10::DeviceGuard device_guard(weight.device());
  const auto& properties = device_properties(weight.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().fused_adam_update_threads_per_block,
      launch_environment::kFusedAdamUpdateThreadsPerBlock, false);
  const int64_t required_blocks = (weight.numel() + threads_per_block - 1) / threads_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(weight.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, weight.scalar_type(), "fused_adam_update", [&] {
        hipLaunchKernelGGL(fused_adam_update_kernel<scalar_t>, dim3(blocks), dim3(threads_per_block), 0,
            stream, weight.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), mean.data_ptr<scalar_t>(),
            variance.data_ptr<scalar_t>(), weight.numel(), learning_rate, learning_rate_bias_correction,
            weight_decay, rescale_gradient, clip_gradient, beta1, beta2, epsilon, adamw);
      });
  check_hip_kernel_launch("fused_adam_update");
}

torch::Tensor masked_softmax_forward(const torch::Tensor& logits, const torch::Tensor& mask) {
  TORCH_CHECK(supports_masked_categorical(logits, mask, logits.dim() - 1),
      "masked softmax received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(logits.device());
  auto output = torch::empty(logits.sizes(), logits.options().dtype(torch::kFloat32));
  const int64_t width = logits.size(-1);
  const int64_t row_count = logits.numel() / width;
  if (row_count == 0) {
    return output;
  }
  const auto& properties = device_properties(logits.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().masked_categorical_threads_per_block,
      launch_environment::kMaskedCategoricalThreadsPerBlock, true);
  const int groups_per_block =
      threads_per_block / masked_categorical_group_width(width, properties.warpSize);
  const int64_t required_blocks =
      (row_count + groups_per_block - 1) / groups_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(), "masked_softmax", [&] {
        if (width <= 16) {
          dispatch_masked_categorical_subwave_width(width, [&](auto subwave_width) {
            constexpr int compiled_subwave_width = decltype(subwave_width)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(masked_categorical_forward_subwave_kernel<
                                   compiled_subwave_width, scalar_t, true>),
                dim3(blocks), dim3(threads_per_block), 0, stream, logits.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(), output.data_ptr<float>(), row_count, width);
          });
        } else {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(masked_categorical_forward_kernel<scalar_t, true>),
              dim3(blocks), dim3(threads_per_block), 0, stream, logits.data_ptr<scalar_t>(),
              mask.data_ptr<bool>(), output.data_ptr<float>(), row_count, width);
        }
      });
  check_hip_kernel_launch("masked_softmax");
  return output;
}

torch::Tensor grouped_masked_softmax_pool_forward(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values) {
  TORCH_CHECK(supports_grouped_masked_softmax_pool(logits, mask, values),
      "grouped masked softmax pool received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(logits.device());
  const int64_t choice_count = logits.size(-1);
  const int64_t row_count = logits.numel() / choice_count;
  const int64_t group_count = mask.size(-1);
  const int64_t feature_count = values.size(-1);
  auto output_shape = logits.sizes().vec();
  output_shape.pop_back();
  output_shape.insert(output_shape.begin(), group_count);
  output_shape.push_back(feature_count);
  auto output = torch::empty(output_shape, logits.options().dtype(torch::kFloat32));
  if (row_count == 0) {
    return output;
  }
  const auto& properties = device_properties(logits.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().masked_categorical_threads_per_block,
      launch_environment::kMaskedCategoricalThreadsPerBlock, true);
  const int64_t pool_count = row_count * group_count;
  const int blocks = static_cast<int>(
      std::min<int64_t>(pool_count, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(), "grouped_masked_softmax_pool_logits", [&] {
        using logit_t = scalar_t;
        switch (values.scalar_type()) {
          case torch::kFloat32:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_masked_softmax_pool_kernel<logit_t, float>),
                dim3(blocks), dim3(threads_per_block), choice_count * sizeof(float), stream,
                logits.data_ptr<logit_t>(), mask.data_ptr<bool>(), values.data_ptr<float>(),
                output.data_ptr<float>(), row_count, choice_count, group_count, feature_count);
            break;
          case torch::kFloat16:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_masked_softmax_pool_kernel<logit_t, c10::Half>),
                dim3(blocks), dim3(threads_per_block), choice_count * sizeof(float), stream,
                logits.data_ptr<logit_t>(), mask.data_ptr<bool>(), values.data_ptr<c10::Half>(),
                output.data_ptr<float>(), row_count, choice_count, group_count, feature_count);
            break;
          case torch::kBFloat16:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_masked_softmax_pool_kernel<logit_t, c10::BFloat16>),
                dim3(blocks), dim3(threads_per_block), choice_count * sizeof(float), stream,
                logits.data_ptr<logit_t>(), mask.data_ptr<bool>(), values.data_ptr<c10::BFloat16>(),
                output.data_ptr<float>(), row_count, choice_count, group_count, feature_count);
            break;
          default:
            TORCH_CHECK(false, "unsupported grouped masked softmax pool value data type");
        }
      });
  check_hip_kernel_launch("grouped_masked_softmax_pool");
  return output;
}

torch::Tensor masked_softmax_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& probabilities, const torch::Tensor& mask,
    torch::ScalarType input_type) {
  c10::DeviceGuard device_guard(probabilities.device());
  auto contiguous_gradient = gradient_output.to(torch::kFloat32).contiguous();
  auto gradient_input = torch::empty(probabilities.sizes(), probabilities.options().dtype(input_type));
  const int64_t width = probabilities.size(-1);
  const int64_t row_count = probabilities.numel() / width;
  if (row_count == 0) {
    return gradient_input;
  }
  const auto& properties = device_properties(probabilities.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().masked_categorical_threads_per_block,
      launch_environment::kMaskedCategoricalThreadsPerBlock, true);
  const int waves_per_block = threads_per_block / properties.warpSize;
  const int64_t required_blocks =
      (row_count + waves_per_block - 1) / waves_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(probabilities.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, input_type, "masked_softmax_backward", [&] {
        hipLaunchKernelGGL(masked_softmax_backward_kernel<scalar_t>, dim3(blocks),
            dim3(threads_per_block), 0, stream, contiguous_gradient.data_ptr<float>(),
            probabilities.data_ptr<float>(), mask.data_ptr<bool>(),
            gradient_input.data_ptr<scalar_t>(), row_count, width);
      });
  check_hip_kernel_launch("masked_softmax_backward");
  return gradient_input;
}

torch::Tensor masked_log_sum_exp_forward(const torch::Tensor& logits, const torch::Tensor& mask) {
  TORCH_CHECK(supports_masked_categorical(logits, mask, logits.dim() - 1),
      "masked log-sum-exp received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(logits.device());
  auto output_shape = logits.sizes().vec();
  output_shape.back() = 1;
  auto output = torch::empty(output_shape, logits.options().dtype(torch::kFloat32));
  const int64_t width = logits.size(-1);
  const int64_t row_count = logits.numel() / width;
  if (row_count == 0) {
    return output;
  }
  const auto& properties = device_properties(logits.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().masked_categorical_threads_per_block,
      launch_environment::kMaskedCategoricalThreadsPerBlock, true);
  const int waves_per_block = threads_per_block / properties.warpSize;
  const int64_t required_blocks =
      (row_count + waves_per_block - 1) / waves_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(), "masked_log_sum_exp", [&] {
        if (width <= 16) {
          dispatch_masked_categorical_subwave_width(width, [&](auto subwave_width) {
            constexpr int compiled_subwave_width = decltype(subwave_width)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(masked_categorical_forward_subwave_kernel<
                                   compiled_subwave_width, scalar_t, false>),
                dim3(blocks), dim3(threads_per_block), 0, stream, logits.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(), output.data_ptr<float>(), row_count, width);
          });
        } else {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(masked_categorical_forward_kernel<scalar_t, false>),
              dim3(blocks), dim3(threads_per_block), 0, stream, logits.data_ptr<scalar_t>(),
              mask.data_ptr<bool>(), output.data_ptr<float>(), row_count, width);
        }
      });
  check_hip_kernel_launch("masked_log_sum_exp");
  return output;
}

torch::Tensor masked_log_sum_exp_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& normalizers) {
  c10::DeviceGuard device_guard(logits.device());
  auto contiguous_gradient = gradient_output.to(torch::kFloat32).contiguous();
  auto gradient_input = torch::empty_like(logits);
  const int64_t width = logits.size(-1);
  const int64_t row_count = logits.numel() / width;
  if (row_count == 0) {
    return gradient_input;
  }
  const auto& properties = device_properties(logits.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().masked_categorical_threads_per_block,
      launch_environment::kMaskedCategoricalThreadsPerBlock, true);
  const int waves_per_block = threads_per_block / properties.warpSize;
  const int64_t required_blocks =
      (row_count + waves_per_block - 1) / waves_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream = c10::hip::getCurrentHIPStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(), "masked_log_sum_exp_backward", [&] {
        hipLaunchKernelGGL(masked_log_sum_exp_backward_kernel<scalar_t>, dim3(blocks),
            dim3(threads_per_block), 0, stream, contiguous_gradient.data_ptr<float>(),
            logits.data_ptr<scalar_t>(), mask.data_ptr<bool>(), normalizers.data_ptr<float>(),
            gradient_input.data_ptr<scalar_t>(), row_count, width);
      });
  check_hip_kernel_launch("masked_log_sum_exp_backward");
  return gradient_input;
}

torch::Tensor scatter_rows_forward(const torch::Tensor& rows,
    const torch::Tensor& row_indices, const torch::Tensor& output) {
  TORCH_CHECK(supports_scatter_rows(rows, row_indices),
      "scatter rows received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(rows.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream(rows.get_device()).stream();
  const size_t output_bytes = static_cast<size_t>(output.numel()) * output.element_size();
  if (output_bytes > 0) {
    const hipError_t memset_error = hipMemsetAsync(output.data_ptr(), 0, output_bytes, stream);
    TORCH_CHECK(memset_error == hipSuccess,
        "scatter rows zero initialization failed: ", hipGetErrorString(memset_error));
  }
  if (rows.numel() == 0) {
    return output;
  }
  const int64_t row_width = rows.numel() / rows.size(0);
  const int64_t row_count = output.size(0);
  const auto& properties = device_properties(rows.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().scatter_rows_threads_per_block,
      launch_environment::kScatterRowsThreadsPerBlock, false);
  const int group_width = properties.warpSize;
  const bool use_grouped_kernel =
      row_width >= group_width && threads_per_block % group_width == 0;
  const int groups_per_block = use_grouped_kernel ? threads_per_block / group_width : 0;
  const int64_t required_blocks = use_grouped_kernel
      ? (rows.size(0) + groups_per_block - 1) / groups_per_block
      : (rows.numel() + threads_per_block - 1) / threads_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, rows.scalar_type(), "scatter_rows", [&] {
        if (use_grouped_kernel) {
          dispatch_scatter_rows_group_width(group_width, [&](auto compiled_group_width) {
            constexpr int launch_group_width = decltype(compiled_group_width)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(scatter_rows_grouped_kernel<launch_group_width, scalar_t>),
                dim3(blocks), dim3(threads_per_block), 0, stream, rows.data_ptr<scalar_t>(),
                row_indices.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), rows.size(0), row_width,
                row_count);
          });
        } else {
          hipLaunchKernelGGL(scatter_rows_kernel<scalar_t>, dim3(blocks),
              dim3(threads_per_block), 0, stream, rows.data_ptr<scalar_t>(),
              row_indices.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), rows.numel(),
              row_width, row_count);
        }
      });
  check_hip_kernel_launch("scatter_rows");
  return output;
}

torch::Tensor scatter_rows_backward(
    const torch::Tensor& gradient_output, const torch::Tensor& row_indices) {
  auto contiguous_gradient = gradient_output.contiguous();
  auto output_shape = contiguous_gradient.sizes().vec();
  output_shape[0] = row_indices.numel();
  auto gradient_rows = torch::empty(output_shape, contiguous_gradient.options());
  if (gradient_rows.numel() == 0) {
    return gradient_rows;
  }
  c10::DeviceGuard device_guard(contiguous_gradient.device());
  const int64_t row_width = gradient_rows.numel() / row_indices.numel();
  const auto& properties = device_properties(contiguous_gradient.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().scatter_rows_threads_per_block,
      launch_environment::kScatterRowsThreadsPerBlock, false);
  const int group_width = properties.warpSize;
  const bool use_grouped_kernel =
      row_width >= group_width && threads_per_block % group_width == 0;
  const int groups_per_block = use_grouped_kernel ? threads_per_block / group_width : 0;
  const int64_t required_blocks = use_grouped_kernel
      ? (row_indices.numel() + groups_per_block - 1) / groups_per_block
      : (gradient_rows.numel() + threads_per_block - 1) / threads_per_block;
  const int blocks = static_cast<int>(
      std::min<int64_t>(required_blocks, static_cast<int64_t>(properties.maxGridSize[0])));
  const hipStream_t stream =
      c10::hip::getCurrentHIPStream(contiguous_gradient.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16,
      contiguous_gradient.scalar_type(), "scatter_rows_backward", [&] {
        if (use_grouped_kernel) {
          dispatch_scatter_rows_group_width(group_width, [&](auto compiled_group_width) {
            constexpr int launch_group_width = decltype(compiled_group_width)::value;
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gather_scattered_row_gradients_grouped_kernel<launch_group_width, scalar_t>),
                dim3(blocks), dim3(threads_per_block), 0, stream, contiguous_gradient.data_ptr<scalar_t>(),
                row_indices.data_ptr<int64_t>(), gradient_rows.data_ptr<scalar_t>(), row_indices.numel(),
                row_width);
          });
        } else {
          hipLaunchKernelGGL(gather_scattered_row_gradients_kernel<scalar_t>, dim3(blocks),
              dim3(threads_per_block), 0, stream, contiguous_gradient.data_ptr<scalar_t>(),
              row_indices.data_ptr<int64_t>(), gradient_rows.data_ptr<scalar_t>(),
              gradient_rows.numel(), row_width);
        }
      });
  check_hip_kernel_launch("scatter_rows_backward");
  return gradient_rows;
}

torch::Tensor padded_batch_gather_forward(
    const torch::Tensor& source, const torch::Tensor& stored_indices) {
  TORCH_CHECK(supports_padded_batch_gather(source, stored_indices),
      "padded batch gather received an unsupported ROCm tensor layout");
  return dispatch_padded_batch_gather<false, false>(
      source, {}, stored_indices, {});
}

torch::Tensor padded_batch_gather_2d_forward(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices) {
  TORCH_CHECK(supports_padded_batch_gather_2d(
          source, outer_stored_indices, inner_stored_indices),
      "two-dimensional padded batch gather received an unsupported ROCm tensor layout");
  return dispatch_padded_batch_gather<true, false>(
      source, {}, outer_stored_indices, inner_stored_indices);
}

torch::Tensor padded_batch_gather_by_batch_indices_forward(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices) {
  TORCH_CHECK(supports_padded_batch_gather_by_batch_indices(
          source, batch_indices, stored_indices),
      "explicit-batch padded gather received an unsupported ROCm tensor layout");
  return dispatch_padded_batch_gather<false, true>(
      source, batch_indices, stored_indices, {});
}

torch::Tensor indexed_relation_bias_forward(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, float scale) {
  TORCH_CHECK(relation_logits.is_cuda(), "indexed relation bias requires a ROCm tensor");
  TORCH_CHECK(is_fused_floating(relation_logits), "relation logits have unsupported dtype");
  TORCH_CHECK(relation_logits.scalar_type() == relation_bias.scalar_type(), "relation logits/bias dtype mismatch");
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt32 || relation_ids.scalar_type() == torch::kInt64,
      "relation IDs must be int32 or int64");
  TORCH_CHECK(relation_logits.dim() == 4, "relation logits must be rank 4");
  const int64_t batch = relation_logits.size(0);
  const int64_t heads = relation_logits.size(1);
  const int64_t query_tokens = relation_logits.size(2);
  const int64_t relation_count = relation_logits.size(3);
  TORCH_CHECK(query_tokens > 0 && relation_count > 0, "relation logits cannot be empty");
  TORCH_CHECK(relation_ids.dim() == 2 || relation_ids.dim() == 3, "relation IDs must be rank 2 or 3");
  TORCH_CHECK(relation_ids.size(-2) == query_tokens, "relation ID query dimension mismatch");
  TORCH_CHECK(relation_ids.dim() == 2 || relation_ids.size(0) == batch, "relation ID batch dimension mismatch");
  const int64_t key_tokens = relation_ids.size(-1);
  TORCH_CHECK(key_tokens > 0, "relation IDs cannot have an empty key dimension");
  TORCH_CHECK(relation_bias.device() == relation_logits.device() && relation_ids.device() == relation_logits.device(),
      "indexed relation bias inputs must be on the same device");

  c10::DeviceGuard device_guard(relation_logits.device());
  torch::Tensor expanded_bias = relation_bias.expand({batch, heads, query_tokens, key_tokens});
  torch::Tensor output = torch::empty({batch, heads, query_tokens, key_tokens},
      relation_logits.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t element_count = output.numel();
  if (element_count == 0) {
    return output;
  }
  const auto& config = rocm_kernel_launch_config();
  const auto launch_plan = indexed_relation_bias_launch_plan(
      relation_logits.get_device(), batch, heads, query_tokens, relation_count,
      config.indexed_relation_bias_forward_waves_per_block,
      config.indexed_relation_bias_forward_queries_per_wave, false);
  TORCH_CHECK(launch_plan.has_value(), "indexed relation bias launch exceeds this device's limits");
  const int64_t id_batch_stride = relation_ids.dim() == 2 ? 0 : relation_ids.stride(0);
  const int64_t id_query_stride = relation_ids.dim() == 2 ? relation_ids.stride(0) : relation_ids.stride(1);
  const int64_t id_key_stride = relation_ids.dim() == 2 ? relation_ids.stride(1) : relation_ids.stride(2);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(relation_logits.get_device()).stream();
  dispatch_compiled_work_size(config.indexed_relation_bias_forward_queries_per_wave,
      launch_environment::kIndexedRelationBiasForwardQueriesPerWave, [&](auto query_work_size) {
        constexpr int queries_per_wave = decltype(query_work_size)::value;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, relation_logits.scalar_type(), "indexed_relation_bias", [&] {
              if (relation_ids.scalar_type() == torch::kInt32) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(indexed_relation_bias_kernel<queries_per_wave, scalar_t, int32_t>),
                    dim3(launch_plan->grid_size), dim3(launch_plan->block_threads), 0, stream,
                    relation_logits.data_ptr<scalar_t>(), expanded_bias.data_ptr<scalar_t>(),
                    relation_ids.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
                    launch_plan->query_chunk_count, launch_plan->work_item_count, heads, query_tokens,
                    key_tokens, relation_count, relation_logits.stride(0), relation_logits.stride(1),
                    relation_logits.stride(2), relation_logits.stride(3), expanded_bias.stride(0),
                    expanded_bias.stride(1), expanded_bias.stride(2), expanded_bias.stride(3),
                    id_batch_stride, id_query_stride, id_key_stride, scale);
              } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(indexed_relation_bias_kernel<queries_per_wave, scalar_t, int64_t>),
                    dim3(launch_plan->grid_size), dim3(launch_plan->block_threads), 0, stream,
                    relation_logits.data_ptr<scalar_t>(), expanded_bias.data_ptr<scalar_t>(),
                    relation_ids.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
                    launch_plan->query_chunk_count, launch_plan->work_item_count, heads, query_tokens,
                    key_tokens, relation_count, relation_logits.stride(0), relation_logits.stride(1),
                    relation_logits.stride(2), relation_logits.stride(3), expanded_bias.stride(0),
                    expanded_bias.stride(1), expanded_bias.stride(2), expanded_bias.stride(3),
                    id_batch_stride, id_query_stride, id_key_stride, scale);
              }
            });
      });
  check_hip_kernel_launch("indexed_relation_bias");
  return output;
}

torch::Tensor indexed_relation_bias_logit_gradient(const torch::Tensor& gradient_output,
    const torch::Tensor& relation_ids, at::IntArrayRef relation_logits_shape, float scale) {
  TORCH_CHECK(relation_logits_shape.size() == 4, "relation logit gradient shape must be rank 4");
  const int64_t batch = relation_logits_shape[0];
  const int64_t heads = relation_logits_shape[1];
  const int64_t query_tokens = relation_logits_shape[2];
  const int64_t relation_count = relation_logits_shape[3];
  const int64_t key_tokens = relation_ids.size(-1);
  c10::DeviceGuard device_guard(gradient_output.device());
  torch::Tensor relation_logits_gradient = torch::empty(
      relation_logits_shape, gradient_output.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t element_count = gradient_output.numel();
  if (element_count == 0) {
    return relation_logits_gradient;
  }
  const auto& config = rocm_kernel_launch_config();
  const auto launch_plan = indexed_relation_bias_launch_plan(
      gradient_output.get_device(), batch, heads, query_tokens, relation_count,
      config.indexed_relation_bias_backward_waves_per_block,
      config.indexed_relation_bias_backward_queries_per_wave, true);
  TORCH_CHECK(launch_plan.has_value(), "indexed relation bias backward launch exceeds this device's limits");
  const int64_t id_batch_stride = relation_ids.dim() == 2 ? 0 : relation_ids.stride(0);
  const int64_t id_query_stride = relation_ids.dim() == 2 ? relation_ids.stride(0) : relation_ids.stride(1);
  const int64_t id_key_stride = relation_ids.dim() == 2 ? relation_ids.stride(1) : relation_ids.stride(2);
  const torch::Tensor contiguous_gradient = gradient_output.contiguous();
  const hipStream_t stream = c10::hip::getCurrentHIPStream(gradient_output.get_device()).stream();
  dispatch_compiled_work_size(config.indexed_relation_bias_backward_queries_per_wave,
      launch_environment::kIndexedRelationBiasBackwardQueriesPerWave, [&](auto query_work_size) {
        constexpr int queries_per_wave = decltype(query_work_size)::value;
        AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16,
            gradient_output.scalar_type(), "indexed_relation_bias_backward", [&] {
              if (relation_ids.scalar_type() == torch::kInt32) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(
                    indexed_relation_bias_logit_gradient_kernel<queries_per_wave, scalar_t, int32_t>),
                    dim3(launch_plan->grid_size), dim3(launch_plan->block_threads),
                    launch_plan->shared_memory_bytes, stream, contiguous_gradient.data_ptr<scalar_t>(),
                    relation_ids.data_ptr<int32_t>(), relation_logits_gradient.data_ptr<scalar_t>(),
                    launch_plan->query_chunk_count, launch_plan->work_item_count, heads, query_tokens,
                    key_tokens, relation_count, id_batch_stride, id_query_stride, id_key_stride, scale);
              } else {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(
                    indexed_relation_bias_logit_gradient_kernel<queries_per_wave, scalar_t, int64_t>),
                    dim3(launch_plan->grid_size), dim3(launch_plan->block_threads),
                    launch_plan->shared_memory_bytes, stream, contiguous_gradient.data_ptr<scalar_t>(),
                    relation_ids.data_ptr<int64_t>(), relation_logits_gradient.data_ptr<scalar_t>(),
                    launch_plan->query_chunk_count, launch_plan->work_item_count, heads, query_tokens,
                    key_tokens, relation_count, id_batch_stride, id_query_stride, id_key_stride, scale);
              }
            });
      });
  check_hip_kernel_launch("indexed_relation_bias_backward");
  return relation_logits_gradient;
}

GroupedIndexedAttentionForwardResult grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, float scale, bool capture_log_sum_exp) {
  TORCH_CHECK(query.is_cuda(), "grouped indexed attention requires a ROCm tensor");
  TORCH_CHECK(query.dim() == 3, "query must be rank 3");
  TORCH_CHECK(shared_key_values.dim() == 3, "shared key/value data must be rank 3");
  TORCH_CHECK(shared_deltas.dim() == 3, "shared deltas must be rank 3");
  TORCH_CHECK(indexed_deltas.dim() == 3, "indexed deltas must be rank 3");
  TORCH_CHECK(indexed_shared_ids.dim() == 2, "indexed shared IDs must be rank 2");
  TORCH_CHECK(queries_per_group > 0, "queries per group must be positive");
  const int64_t query_count = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t group_count = shared_key_values.size(0);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t key_width = heads * key_features;
  TORCH_CHECK(query_count == group_count * queries_per_group, "query/group shape mismatch");
  TORCH_CHECK(heads > 0 && key_features > 0 && shared_tokens > 0, "attention dimensions must be positive");
  TORCH_CHECK(packed_width > key_width && (packed_width - key_width) % heads == 0,
      "packed key/value width is incompatible with the query");
  const int64_t value_features = (packed_width - key_width) / heads;
  TORCH_CHECK(shared_deltas.size(0) == query_count && shared_deltas.size(1) == shared_tokens &&
                  shared_deltas.size(2) == packed_width,
      "shared delta shape mismatch");
  TORCH_CHECK(indexed_deltas.size(0) == query_count && indexed_deltas.size(1) == indexed_tokens &&
                  indexed_deltas.size(2) == packed_width,
      "indexed delta shape mismatch");
  TORCH_CHECK(indexed_shared_ids.size(0) == query_count && indexed_shared_ids.size(1) == indexed_tokens,
      "indexed shared ID shape mismatch");
  TORCH_CHECK(query.scalar_type() == shared_key_values.scalar_type() &&
                  query.scalar_type() == shared_deltas.scalar_type() &&
                  query.scalar_type() == indexed_deltas.scalar_type(),
      "grouped indexed attention dtype mismatch");
  TORCH_CHECK(is_fused_floating(query), "grouped indexed attention has unsupported dtype");
  TORCH_CHECK(indexed_shared_ids.scalar_type() == torch::kInt32 || indexed_shared_ids.scalar_type() == torch::kInt64,
      "indexed shared IDs must be int32 or int64");
  TORCH_CHECK(shared_key_values.device() == query.device() && shared_deltas.device() == query.device() &&
                  indexed_deltas.device() == query.device() && indexed_shared_ids.device() == query.device(),
      "grouped indexed attention inputs must be on the same device");

  c10::DeviceGuard device_guard(query.device());
  torch::Tensor output = torch::empty(
      {query_count, heads, value_features}, query.options().memory_format(torch::MemoryFormat::Contiguous));
  torch::Tensor log_sum_exp = capture_log_sum_exp
                                  ? torch::empty({query_count, heads}, query.options().dtype(torch::kFloat32))
                                  : torch::Tensor();
  if (query_count == 0) {
    return {output, log_sum_exp};
  }
  const int64_t query_head_count = query_count * heads;
  const auto launch_plan = grouped_indexed_attention_launch_plan(
      query, shared_tokens, indexed_tokens, value_features, false);
  TORCH_CHECK(launch_plan.has_value(), "grouped indexed attention launch exceeds this device's limits");
  const GroupedIndexedAttentionLayout layout{queries_per_group,
      heads,
      key_features,
      value_features,
      shared_tokens,
      indexed_tokens,
      packed_width,
      query.stride(0),
      query.stride(1),
      query.stride(2),
      shared_key_values.stride(0),
      shared_key_values.stride(1),
      shared_key_values.stride(2),
      shared_deltas.stride(0),
      shared_deltas.stride(1),
      shared_deltas.stride(2),
      indexed_deltas.stride(0),
      indexed_deltas.stride(1),
      indexed_deltas.stride(2),
      indexed_shared_ids.stride(0),
      indexed_shared_ids.stride(1)};
  const bool contiguous = query.is_contiguous() && shared_key_values.is_contiguous() &&
                          shared_deltas.is_contiguous() && indexed_deltas.is_contiguous() &&
                          indexed_shared_ids.is_contiguous();
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  float* log_sum_exp_data = capture_log_sum_exp ? log_sum_exp.data_ptr<float>() : nullptr;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(), "grouped_indexed_attention", [&] {
        if (indexed_shared_ids.scalar_type() == torch::kInt32) {
          if (contiguous) {
            launch_grouped_indexed_attention<scalar_t, int32_t, true>(query, shared_key_values,
                shared_deltas, indexed_deltas, indexed_shared_ids, output, launch_plan->grid_size,
                launch_plan->block_threads, launch_plan->shared_memory_bytes, stream,
                query_head_count, layout, scale, log_sum_exp_data);
          } else {
            launch_grouped_indexed_attention<scalar_t, int32_t, false>(query, shared_key_values,
                shared_deltas, indexed_deltas, indexed_shared_ids, output, launch_plan->grid_size,
                launch_plan->block_threads, launch_plan->shared_memory_bytes, stream,
                query_head_count, layout, scale, log_sum_exp_data);
          }
        } else {
          if (contiguous) {
            launch_grouped_indexed_attention<scalar_t, int64_t, true>(query, shared_key_values,
                shared_deltas, indexed_deltas, indexed_shared_ids, output, launch_plan->grid_size,
                launch_plan->block_threads, launch_plan->shared_memory_bytes, stream,
                query_head_count, layout, scale, log_sum_exp_data);
          } else {
            launch_grouped_indexed_attention<scalar_t, int64_t, false>(query, shared_key_values,
                shared_deltas, indexed_deltas, indexed_shared_ids, output, launch_plan->grid_size,
                launch_plan->block_threads, launch_plan->shared_memory_bytes, stream,
                query_head_count, layout, scale, log_sum_exp_data);
          }
        }
      });
  check_hip_kernel_launch("grouped_indexed_attention");
  return {output, log_sum_exp};
}

GroupedIndexedAttentionGradients grouped_indexed_attention_backward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, const torch::Tensor& log_sum_exp,
    const torch::Tensor& gradient_output, int64_t queries_per_group, float scale, bool needs_query_gradient,
    bool needs_shared_key_value_gradient, bool needs_shared_delta_gradient,
    bool needs_indexed_delta_gradient) {
  const int64_t query_count = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t group_count = shared_key_values.size(0);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t key_width = heads * key_features;
  const int64_t value_features = (packed_width - key_width) / heads;
  TORCH_CHECK(gradient_output.dim() == 3 && gradient_output.size(0) == query_count &&
                  gradient_output.size(1) == heads && gradient_output.size(2) == value_features,
      "grouped indexed attention output gradient shape mismatch");
  TORCH_CHECK(log_sum_exp.dim() == 2 && log_sum_exp.size(0) == query_count &&
                  log_sum_exp.size(1) == heads &&
                  log_sum_exp.scalar_type() == torch::kFloat32,
      "grouped indexed attention log-sum-exp shape or dtype mismatch");
  TORCH_CHECK(gradient_output.scalar_type() == query.scalar_type(),
      "grouped indexed attention output gradient dtype mismatch");
  TORCH_CHECK(gradient_output.device() == query.device() && log_sum_exp.device() == query.device(),
      "grouped indexed attention backward inputs must be on the same device");

  c10::DeviceGuard device_guard(query.device());
  const auto gradient_options = query.options().memory_format(torch::MemoryFormat::Contiguous);
  torch::Tensor query_gradient =
      needs_query_gradient ? torch::empty({query_count, heads, key_features}, gradient_options) : torch::Tensor();
  const bool needs_shared_delta_workspace =
      needs_shared_key_value_gradient || needs_shared_delta_gradient;
  const bool needs_indexed_delta_workspace =
      needs_shared_key_value_gradient || needs_indexed_delta_gradient;
  torch::Tensor shared_delta_workspace =
      needs_shared_delta_workspace
          ? torch::empty({query_count, shared_tokens, packed_width}, gradient_options)
          : torch::Tensor();
  torch::Tensor indexed_delta_workspace =
      needs_indexed_delta_workspace
          ? torch::empty({query_count, indexed_tokens, packed_width}, gradient_options)
          : torch::Tensor();
  torch::Tensor shared_key_value_gradient =
      needs_shared_key_value_gradient
          ? torch::empty({group_count, shared_tokens, packed_width}, gradient_options)
          : torch::Tensor();
  if (query_count == 0) {
    return {query_gradient, shared_key_value_gradient,
        needs_shared_delta_gradient ? shared_delta_workspace : torch::Tensor(),
        needs_indexed_delta_gradient ? indexed_delta_workspace : torch::Tensor()};
  }

  const GroupedIndexedAttentionLayout layout{queries_per_group,
      heads,
      key_features,
      value_features,
      shared_tokens,
      indexed_tokens,
      packed_width,
      query.stride(0),
      query.stride(1),
      query.stride(2),
      shared_key_values.stride(0),
      shared_key_values.stride(1),
      shared_key_values.stride(2),
      shared_deltas.stride(0),
      shared_deltas.stride(1),
      shared_deltas.stride(2),
      indexed_deltas.stride(0),
      indexed_deltas.stride(1),
      indexed_deltas.stride(2),
      indexed_shared_ids.stride(0),
      indexed_shared_ids.stride(1)};
  const bool contiguous = query.is_contiguous() && shared_key_values.is_contiguous() &&
                          shared_deltas.is_contiguous() && indexed_deltas.is_contiguous() &&
                          indexed_shared_ids.is_contiguous();
  const int64_t query_head_count = query_count * heads;
  const auto query_launch_plan = grouped_indexed_attention_launch_plan(
      query, shared_tokens, indexed_tokens, value_features, true);
  TORCH_CHECK(query_launch_plan.has_value(),
      "grouped indexed attention backward launch exceeds this device's limits");
  std::optional<GroupedIndexedAttentionSharedKeyValueGradientLaunchPlan> shared_launch_plan;
  if (needs_shared_key_value_gradient) {
    shared_launch_plan = grouped_indexed_attention_shared_key_value_gradient_launch_plan(
        query, group_count, shared_tokens, packed_width);
    TORCH_CHECK(shared_launch_plan.has_value(),
        "grouped indexed attention shared key/value gradient launch exceeds this device's limits");
  }

  const torch::Tensor contiguous_log_sum_exp = log_sum_exp.contiguous();
  const torch::Tensor contiguous_gradient_output = gradient_output.contiguous();
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(), "grouped_indexed_attention_backward", [&] {
        const auto launch_query_gradient = [&](auto index_tag) {
          using index_t = decltype(index_tag);
          if (contiguous) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(grouped_indexed_attention_backward_query_kernel<scalar_t, index_t, true>),
                dim3(query_launch_plan->grid_size), dim3(query_launch_plan->block_threads),
                query_launch_plan->shared_memory_bytes, stream,
                query.data_ptr<scalar_t>(), shared_key_values.data_ptr<scalar_t>(),
                shared_deltas.data_ptr<scalar_t>(), indexed_deltas.data_ptr<scalar_t>(),
                indexed_shared_ids.data_ptr<index_t>(), contiguous_log_sum_exp.data_ptr<float>(),
                contiguous_gradient_output.data_ptr<scalar_t>(),
                needs_query_gradient ? query_gradient.data_ptr<scalar_t>() : nullptr,
                needs_shared_delta_workspace ? shared_delta_workspace.data_ptr<scalar_t>() : nullptr,
                needs_indexed_delta_workspace ? indexed_delta_workspace.data_ptr<scalar_t>() : nullptr,
                query_head_count, layout, scale);
          } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(grouped_indexed_attention_backward_query_kernel<scalar_t, index_t, false>),
                dim3(query_launch_plan->grid_size), dim3(query_launch_plan->block_threads),
                query_launch_plan->shared_memory_bytes, stream,
                query.data_ptr<scalar_t>(), shared_key_values.data_ptr<scalar_t>(),
                shared_deltas.data_ptr<scalar_t>(), indexed_deltas.data_ptr<scalar_t>(),
                indexed_shared_ids.data_ptr<index_t>(), contiguous_log_sum_exp.data_ptr<float>(),
                contiguous_gradient_output.data_ptr<scalar_t>(),
                needs_query_gradient ? query_gradient.data_ptr<scalar_t>() : nullptr,
                needs_shared_delta_workspace ? shared_delta_workspace.data_ptr<scalar_t>() : nullptr,
                needs_indexed_delta_workspace ? indexed_delta_workspace.data_ptr<scalar_t>() : nullptr,
                query_head_count, layout, scale);
          }
          if (needs_shared_key_value_gradient) {
            const auto& config = rocm_kernel_launch_config();
            dispatch_compiled_work_size(
                config.grouped_indexed_attention_shared_key_value_gradient_features_per_block,
                launch_environment::kGroupedIndexedAttentionSharedKeyValueGradientFeaturesPerBlock,
                [&](auto feature_work_size) {
                  constexpr int features_per_block = decltype(feature_work_size)::value;
                  if (indexed_shared_ids.is_contiguous()) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                                           grouped_indexed_attention_shared_key_value_gradient_kernel<
                                               features_per_block, scalar_t, index_t, true>),
                        dim3(shared_launch_plan->grid_size), dim3(shared_launch_plan->threads_per_block),
                        shared_launch_plan->shared_memory_bytes, stream,
                        shared_delta_workspace.data_ptr<scalar_t>(),
                        indexed_delta_workspace.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<index_t>(),
                        shared_key_value_gradient.data_ptr<scalar_t>(),
                        shared_launch_plan->feature_block_count_per_group,
                        layout);
                  } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                                           grouped_indexed_attention_shared_key_value_gradient_kernel<
                                               features_per_block, scalar_t, index_t, false>),
                        dim3(shared_launch_plan->grid_size), dim3(shared_launch_plan->threads_per_block),
                        shared_launch_plan->shared_memory_bytes, stream,
                        shared_delta_workspace.data_ptr<scalar_t>(),
                        indexed_delta_workspace.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<index_t>(),
                        shared_key_value_gradient.data_ptr<scalar_t>(),
                        shared_launch_plan->feature_block_count_per_group,
                        layout);
                  }
                });
          }
        };
        if (indexed_shared_ids.scalar_type() == torch::kInt32) {
          launch_query_gradient(int32_t{});
        } else {
          launch_query_gradient(int64_t{});
        }
      });
  check_hip_kernel_launch("grouped_indexed_attention_backward");
  return {query_gradient, shared_key_value_gradient,
      needs_shared_delta_gradient ? shared_delta_workspace : torch::Tensor(),
      needs_indexed_delta_gradient ? indexed_delta_workspace : torch::Tensor()};
}

torch::Tensor add_to_owned_residual_and_layer_norm(torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon) {
  TORCH_CHECK(residual.is_cuda(), "residual add LayerNorm requires a ROCm tensor");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(update.is_contiguous(), "residual update must be contiguous");
  TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "LayerNorm parameters must be contiguous");
  TORCH_CHECK(residual.sizes() == update.sizes(), "residual/update shape mismatch");
  TORCH_CHECK(is_fused_floating(residual) && is_fused_floating(update), "residual/update dtype is unsupported");
  TORCH_CHECK(residual.dim() >= 1 && residual.size(-1) > 0, "residual must have a non-empty trailing dimension");
  const int64_t width = residual.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == width, "LayerNorm weight shape mismatch");
  TORCH_CHECK(bias.sizes() == weight.sizes(), "LayerNorm bias shape mismatch");
  TORCH_CHECK(is_fused_floating(weight), "LayerNorm parameter dtype is unsupported");
  TORCH_CHECK(weight.scalar_type() == bias.scalar_type(), "LayerNorm parameter dtype mismatch");
  TORCH_CHECK(supports_owned_residual_layer_norm_types(
                  residual.scalar_type(), update.scalar_type(), weight.scalar_type()),
      "residual add LayerNorm dtype combination is unsupported");
  TORCH_CHECK(update.device() == residual.device() && weight.device() == residual.device() &&
                  bias.device() == residual.device(),
      "residual add LayerNorm inputs must be on the same device");

  c10::DeviceGuard device_guard(residual.device());
  torch::Tensor output = torch::empty_like(residual, residual.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t row_count = residual.numel() / width;
  if (row_count == 0) {
    return output;
  }
  const auto& properties = device_properties(residual.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().residual_add_layer_norm_threads_per_block,
      launch_environment::kResidualAddLayerNormThreadsPerBlock, true);
  const int waves_per_block = threads_per_block / properties.warpSize;
  const int cached_values_per_lane =
      static_cast<int>((width + properties.warpSize - 1) / properties.warpSize);
  const bool use_cached_kernel =
      cached_values_per_lane <= kResidualCachedValuesPerLaneLimit &&
      residual.scalar_type() == update.scalar_type() && weight.scalar_type() == torch::kFloat32;
  const dim3 grid((row_count + waves_per_block - 1) / waves_per_block);
  const dim3 block(threads_per_block);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(residual.get_device()).stream();
  if (use_cached_kernel) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf, torch::kBFloat16, residual.scalar_type(), "residual_add_layer_norm_cached", [&] {
          using residual_t = scalar_t;
          dispatch_residual_cached_values_per_lane(cached_values_per_lane, [&](auto cached_values) {
            constexpr int compiled_cached_values = decltype(cached_values)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(residual_add_layer_norm_cached_in_place_kernel<
                                   compiled_cached_values, residual_t, residual_t, float>),
                grid, block, 0, stream, residual.data_ptr<residual_t>(), update.data_ptr<residual_t>(),
                weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<residual_t>(), row_count,
                width, epsilon);
          });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf, torch::kBFloat16, residual.scalar_type(), "residual_add_layer_norm", [&] {
          using residual_t = scalar_t;
          AT_DISPATCH_FLOATING_TYPES_AND2(
              torch::kHalf, torch::kBFloat16, update.scalar_type(), "residual_add_layer_norm_update", [&] {
                using update_t = scalar_t;
                AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16,
                    weight.scalar_type(), "residual_add_layer_norm_parameter", [&] {
                  using parameter_t = scalar_t;
                  hipLaunchKernelGGL(
                      HIP_KERNEL_NAME(residual_add_layer_norm_in_place_kernel<residual_t, update_t, parameter_t>),
                      grid, block, 0, stream, residual.data_ptr<residual_t>(), update.data_ptr<update_t>(),
                      weight.data_ptr<parameter_t>(), bias.data_ptr<parameter_t>(), output.data_ptr<residual_t>(),
                      row_count, width, epsilon);
                });
              });
        });
  }
  check_hip_kernel_launch("residual_add_layer_norm");
  return output;
}

torch::Tensor autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, float epsilon) {
  const int64_t width = input.size(-1);
  TORCH_CHECK(supports_autocast_layer_norm(input, weight, bias, at::IntArrayRef(&width, 1)),
      "autocast LayerNorm received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(input.device());
  torch::Tensor output = torch::empty(input.sizes(),
      input.options().dtype(torch::kFloat32).memory_format(torch::MemoryFormat::Contiguous));
  if (input.numel() == 0) {
    return output;
  }

  const int64_t row_count = input.numel() / width;
  const auto& properties = device_properties(input.get_device());
  const int threads_per_block = configured_threads_per_block(properties,
      rocm_kernel_launch_config().residual_add_layer_norm_threads_per_block,
      launch_environment::kResidualAddLayerNormThreadsPerBlock, true);
  const int waves_per_block = threads_per_block / properties.warpSize;
  const int cached_values_per_lane =
      static_cast<int>((width + properties.warpSize - 1) / properties.warpSize);
  const dim3 grid((row_count + waves_per_block - 1) / waves_per_block);
  const dim3 block(threads_per_block);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(input.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, input.scalar_type(), "autocast_layer_norm_input", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16,
            weight.scalar_type(), "autocast_layer_norm_parameter", [&] {
              using parameter_t = scalar_t;
              dispatch_autocast_layer_norm_cached_values_per_lane(
                  cached_values_per_lane, [&](auto cached_values) {
                    constexpr int compiled_cached_values = decltype(cached_values)::value;
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(autocast_layer_norm_kernel<
                                           compiled_cached_values, input_t, parameter_t>),
                        grid, block, 0, stream, input.data_ptr<input_t>(),
                        weight.data_ptr<parameter_t>(), bias.data_ptr<parameter_t>(),
                        output.data_ptr<float>(), row_count, width, epsilon);
                  });
            });
      });
  check_hip_kernel_launch("autocast_layer_norm");
  return output;
}

}  // namespace djl::pytorch::rocm
