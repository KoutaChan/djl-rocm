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

#include "djl_pytorch_masked_categorical_cuda.h"

#include "djl_pytorch_masked_categorical.h"

#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace djl::pytorch::cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kMaximumIndexedPoolChoices = 16;

// Kernel arguments carry these small, immutable indices without a device allocation or H2D copy.
struct IndexedPoolChoices {
  int count;
  int64_t columns[kMaximumIndexedPoolChoices];
};

bool is_fused_floating(const torch::Tensor& tensor) {
  const auto type = tensor.scalar_type();
  return type == torch::kFloat32 || type == torch::kFloat16 || type == torch::kBFloat16;
}

int row_blocks(int64_t rows, int rows_per_block = 1) {
  return static_cast<int>(std::min<int64_t>((rows + rows_per_block - 1) / rows_per_block, 65535));
}

IndexedPoolChoices pool_choices(at::IntArrayRef indices) {
  IndexedPoolChoices selected{};
  selected.count = static_cast<int>(indices.size());
  std::copy(indices.begin(), indices.end(), selected.columns);
  return selected;
}

__device__ float maximum_propagating_nan(float first, float second) {
  return isnan(first) || first > second ? first : second;
}

template <typename scalar_t, bool write_probabilities>
__global__ void masked_categorical_forward_kernel(const scalar_t* logits, const bool* mask,
    float* output, int64_t row_count, int64_t width, float* normalization) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
       row < row_count; row += static_cast<int64_t>(gridDim.x) * warps_per_block) {
    const int64_t row_offset = row * width;
    float maximum = -INFINITY;
    int selected_count = 0;
    for (int64_t column = lane; column < width; column += kWarpSize) {
      if (mask[row_offset + column]) {
        maximum = maximum_propagating_nan(maximum, static_cast<float>(logits[row_offset + column]));
        ++selected_count;
      }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      maximum = maximum_propagating_nan(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
      selected_count += __shfl_down_sync(0xffffffff, selected_count, offset);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    selected_count = __shfl_sync(0xffffffff, selected_count, 0);
    float denominator = 0.0f;
    for (int64_t column = lane; column < width; column += kWarpSize) {
      if (mask[row_offset + column]) {
        denominator += expf(static_cast<float>(logits[row_offset + column]) - maximum);
      }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      denominator += __shfl_down_sync(0xffffffff, denominator, offset);
    }
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    if constexpr (write_probabilities) {
      for (int64_t column = lane; column < width; column += kWarpSize) {
        output[row_offset + column] = mask[row_offset + column]
            ? expf(static_cast<float>(logits[row_offset + column]) - maximum) / denominator
            : 0.0f;
      }
    } else if (lane == 0) {
      output[row] = selected_count > 0 ? logf(denominator) + maximum : 0.0f;
      if (normalization != nullptr) {
        normalization[row * 2] = maximum;
        normalization[row * 2 + 1] = denominator;
      }
    }
  }
}

template <typename scalar_t>
__global__ void masked_softmax_backward_kernel(const float* gradient, const float* probabilities,
    const bool* mask, scalar_t* output, int64_t row_count, int64_t width) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
       row < row_count; row += static_cast<int64_t>(gridDim.x) * warps_per_block) {
    const int64_t row_offset = row * width;
    float product_sum = 0.0f;
    for (int64_t column = lane; column < width; column += kWarpSize) {
      const int64_t index = row_offset + column;
      if (mask[index]) {
        product_sum += gradient[index] * probabilities[index];
      }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      product_sum += __shfl_down_sync(0xffffffff, product_sum, offset);
    }
    product_sum = __shfl_sync(0xffffffff, product_sum, 0);
    for (int64_t column = lane; column < width; column += kWarpSize) {
      const int64_t index = row_offset + column;
      output[index] = static_cast<scalar_t>(
          mask[index] ? probabilities[index] * (gradient[index] - product_sum) : 0.0f);
    }
  }
}

template <typename scalar_t>
__global__ void masked_log_sum_exp_backward_kernel(const float* gradient, const scalar_t* logits,
    const bool* mask, const float* normalization, scalar_t* output, int64_t elements, int64_t width) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    output[index] = static_cast<scalar_t>(mask[index]
        ? gradient[row] * expf(static_cast<float>(logits[index]) - normalization[row * 2]) /
              normalization[row * 2 + 1]
        : 0.0f);
  }
}

template <typename logit_t>
__device__ void indexed_pool_probabilities(const logit_t* logits, const bool* mask,
    float* probabilities, int64_t row_offset, IndexedPoolChoices selected) {
  float maximum = -INFINITY;
  for (int index = 0; index < selected.count; ++index) {
    const int64_t offset = row_offset + selected.columns[index];
    if (mask[offset]) {
      maximum = maximum_propagating_nan(maximum, static_cast<float>(logits[offset]));
    }
  }
  float denominator = 0.0f;
  for (int index = 0; index < selected.count; ++index) {
    const int64_t offset = row_offset + selected.columns[index];
    probabilities[index] = mask[offset] ? expf(static_cast<float>(logits[offset]) - maximum) : 0.0f;
    denominator += probabilities[index];
  }
  for (int index = 0; index < selected.count; ++index) {
    if (mask[row_offset + selected.columns[index]]) {
      probabilities[index] /= denominator;
    }
  }
}

template <typename logit_t, typename value_t, bool backward>
__global__ void indexed_masked_softmax_pool_kernel(const logit_t* logits, const bool* mask,
    const value_t* values, const float* gradient, float* output, value_t* gradient_values,
    int64_t row_count, int64_t choice_count, int64_t feature_count, IndexedPoolChoices selected) {
  __shared__ float probabilities[kMaximumIndexedPoolChoices];
  for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const int64_t row_offset = row * choice_count;
    if (threadIdx.x == 0) {
      indexed_pool_probabilities(logits, mask, probabilities, row_offset, selected);
    }
    __syncthreads();
    if constexpr (backward) {
      const int64_t value_offset = row_offset * feature_count;
      for (int64_t index = threadIdx.x; index < choice_count * feature_count; index += blockDim.x) {
        gradient_values[value_offset + index] = static_cast<value_t>(0.0f);
      }
      __syncthreads();
      for (int64_t index = threadIdx.x;
           index < static_cast<int64_t>(selected.count) * feature_count; index += blockDim.x) {
        const int selected_index = static_cast<int>(index / feature_count);
        const int64_t feature = index % feature_count;
        const int64_t choice = selected.columns[selected_index];
        gradient_values[value_offset + choice * feature_count + feature] =
            static_cast<value_t>(mask[row_offset + choice]
                ? probabilities[selected_index] * gradient[row * feature_count + feature] : 0.0f);
      }
    } else {
      for (int64_t feature = threadIdx.x; feature < feature_count; feature += blockDim.x) {
        float pooled = 0.0f;
        for (int index = 0; index < selected.count; ++index) {
          const int64_t choice_offset = row_offset + selected.columns[index];
          if (mask[choice_offset]) {
            pooled += probabilities[index] * static_cast<float>(values[choice_offset * feature_count + feature]);
          }
        }
        output[row * feature_count + feature] = pooled;
      }
    }
    __syncthreads();
  }
}

template <bool write_probabilities>
torch::Tensor masked_categorical_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, torch::Tensor* normalization) {
  TORCH_CHECK(supports_masked_categorical(logits, mask, logits.dim() - 1),
      "masked categorical received an unsupported CUDA tensor layout");
  c10::DeviceGuard device_guard(logits.device());
  const int64_t width = logits.size(-1);
  const int64_t row_count = logits.numel() / width;
  auto output_shape = logits.sizes().vec();
  if constexpr (!write_probabilities) {
    output_shape.back() = 1;
  }
  auto output = torch::empty(output_shape, logits.options().dtype(torch::kFloat32));
  if (normalization != nullptr) {
    *normalization = torch::empty({row_count, 2}, logits.options().dtype(torch::kFloat32));
  }
  if (row_count == 0) {
    return output;
  }
  float* normalization_data = normalization == nullptr ? nullptr : normalization->data_ptr<float>();
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  const int blocks = row_blocks(row_count, kThreadsPerBlock / kWarpSize);
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "masked_categorical_cuda", [&] {
        masked_categorical_forward_kernel<scalar_t, write_probabilities>
            <<<blocks, kThreadsPerBlock, 0, stream>>>(logits.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
                output.data_ptr<float>(), row_count, width, normalization_data);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <bool backward>
void launch_indexed_pool(const torch::Tensor& logits, const torch::Tensor& mask,
    const torch::Tensor& values, const torch::Tensor& gradient, torch::Tensor& output,
    at::IntArrayRef choice_indices) {
  const int64_t choice_count = logits.size(-1);
  const int64_t row_count = logits.numel() / choice_count;
  if (row_count == 0) {
    return;
  }
  const int64_t feature_count = backward ? output.size(-1) : values.size(-1);
  const auto value_type = backward ? output.scalar_type() : values.scalar_type();
  const auto selected = pool_choices(choice_indices);
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "indexed_masked_softmax_pool_cuda_logits", [&] {
        using logit_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, value_type,
            "indexed_masked_softmax_pool_cuda_values", [&] {
              indexed_masked_softmax_pool_kernel<logit_t, scalar_t, backward>
                  <<<row_blocks(row_count), kThreadsPerBlock, 0, stream>>>(
                      logits.data_ptr<logit_t>(), mask.data_ptr<bool>(),
                      backward ? nullptr : values.data_ptr<scalar_t>(),
                      backward ? gradient.data_ptr<float>() : nullptr,
                      backward ? nullptr : output.data_ptr<float>(),
                      backward ? output.data_ptr<scalar_t>() : nullptr,
                      row_count, choice_count, feature_count, selected);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool supports_masked_categorical(const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  return detail::is_masked_categorical_layout_supported(logits, mask, axis);
}

bool supports_indexed_masked_softmax_pool(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values, at::IntArrayRef choice_indices) {
  return supports_masked_categorical(logits, mask, logits.dim() - 1) &&
      values.device() == logits.device() && values.is_contiguous() && is_fused_floating(values) &&
      values.dim() == logits.dim() + 1 && values.size(-1) > 0 &&
      values.sizes().slice(0, logits.dim()) == logits.sizes() &&
      !choice_indices.empty() && choice_indices.size() <= kMaximumIndexedPoolChoices;
}

torch::Tensor masked_softmax_forward(const torch::Tensor& logits, const torch::Tensor& mask) {
  return masked_categorical_forward<true>(logits, mask, nullptr);
}

torch::Tensor masked_softmax_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& probabilities, const torch::Tensor& mask, torch::ScalarType input_type) {
  c10::DeviceGuard device_guard(probabilities.device());
  auto gradient = gradient_output.to(torch::kFloat32).contiguous();
  auto output = torch::empty(probabilities.sizes(), probabilities.options().dtype(input_type));
  const int64_t width = probabilities.size(-1);
  const int64_t row_count = probabilities.numel() / width;
  if (row_count == 0) {
    return output;
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(probabilities.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, input_type,
      "masked_softmax_backward_cuda", [&] {
        masked_softmax_backward_kernel<scalar_t>
            <<<row_blocks(row_count, kThreadsPerBlock / kWarpSize), kThreadsPerBlock, 0, stream>>>(
                gradient.data_ptr<float>(), probabilities.data_ptr<float>(), mask.data_ptr<bool>(),
                output.data_ptr<scalar_t>(), row_count, width);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor masked_log_sum_exp_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, torch::Tensor* normalization) {
  return masked_categorical_forward<false>(logits, mask, normalization);
}

torch::Tensor masked_log_sum_exp_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& normalization) {
  c10::DeviceGuard device_guard(logits.device());
  auto gradient = gradient_output.to(torch::kFloat32).contiguous();
  auto output = torch::empty_like(logits);
  if (logits.numel() == 0) {
    return output;
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "masked_log_sum_exp_backward_cuda", [&] {
        masked_log_sum_exp_backward_kernel<scalar_t>
            <<<row_blocks(logits.numel(), kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                gradient.data_ptr<float>(), logits.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
                normalization.data_ptr<float>(), output.data_ptr<scalar_t>(), logits.numel(), logits.size(-1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor indexed_masked_softmax_pool_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values, at::IntArrayRef choice_indices) {
  TORCH_CHECK(supports_indexed_masked_softmax_pool(logits, mask, values, choice_indices),
      "indexed masked softmax pool received an unsupported CUDA tensor layout");
  c10::DeviceGuard device_guard(logits.device());
  auto output_shape = logits.sizes().vec();
  output_shape.back() = values.size(-1);
  auto output = torch::empty(output_shape, logits.options().dtype(torch::kFloat32));
  launch_indexed_pool<false>(logits, mask, values, torch::Tensor(), output, choice_indices);
  return output;
}

torch::Tensor indexed_masked_softmax_pool_value_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, at::IntArrayRef value_shape,
    torch::ScalarType value_type, at::IntArrayRef choice_indices) {
  c10::DeviceGuard device_guard(logits.device());
  auto gradient = gradient_output.to(torch::kFloat32).contiguous();
  auto output = torch::empty(value_shape, logits.options().dtype(value_type));
  launch_indexed_pool<true>(logits, mask, torch::Tensor(), gradient, output, choice_indices);
  return output;
}

}  // namespace djl::pytorch::cuda
