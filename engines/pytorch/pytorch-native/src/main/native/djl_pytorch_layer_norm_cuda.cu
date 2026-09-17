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

#include "djl_pytorch_layer_norm_cuda.h"

#include "djl_pytorch_layer_norm.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "djl_pytorch_layer_norm_kernels.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace djl::pytorch::cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kMaximumCachedValuesPerLane = 16;

// Accumulate both statistics and affine results in FP32 under autocast.
template <int cached_values_per_lane, typename input_t, typename parameter_t,
    typename converted_t>
__global__ void autocast_layer_norm_and_cast_kernel(const input_t* input,
    const parameter_t* weight, const parameter_t* bias, float* normalized_output,
    converted_t* converted_output, int64_t row_count, int64_t width, float epsilon) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (row >= row_count) {
    return;
  }

  float cached_values[cached_values_per_lane];
  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * width;
#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; ++cached_index) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * kWarpSize;
    if (feature < width) {
      const float value = static_cast<float>(input[row_offset + feature]);
      cached_values[cached_index] = value;
      ++count;
      const float delta = value - mean;
      mean += delta / count;
      moment += delta * (value - mean);
    }
  }

  detail::reduce_layer_norm_statistics(mean, moment, count);
  mean = __shfl_sync(0xffffffff, mean, 0);
  const float inverse_standard_deviation =
      rsqrtf(__shfl_sync(0xffffffff, moment, 0) / static_cast<float>(width) + epsilon);

#pragma unroll
  for (int cached_index = 0; cached_index < cached_values_per_lane; ++cached_index) {
    const int64_t feature = lane + static_cast<int64_t>(cached_index) * kWarpSize;
    if (feature < width) {
      const float normalized =
          (cached_values[cached_index] - mean) * inverse_standard_deviation;
      const float affine = normalized * static_cast<float>(weight[feature]) +
                           static_cast<float>(bias[feature]);
      normalized_output[row_offset + feature] = affine;
      converted_output[row_offset + feature] = static_cast<converted_t>(affine);
    }
  }
}

template <typename input_t, typename parameter_t, typename converted_t>
void launch_typed(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, torch::Tensor& normalized, torch::Tensor& converted,
    float epsilon) {
  const int64_t width = weight.numel();
  const int64_t row_count = input.numel() / width;
  constexpr int warps_per_block = kThreadsPerBlock / kWarpSize;
  const dim3 grid((row_count + warps_per_block - 1) / warps_per_block);
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  if (width > kWarpSize * kMaximumCachedValuesPerLane) {
    const auto launch_wide = [&](auto cached_values) {
      constexpr int compiled_cached_values = decltype(cached_values)::value;
      detail::wide_autocast_layer_norm_kernel<compiled_cached_values, input_t,
          parameter_t, converted_t, true><<<row_count, detail::kLayerNormWideThreadsPerBlock, 0, stream>>>(
          input.data_ptr<input_t>(), weight.data_ptr<parameter_t>(), bias.data_ptr<parameter_t>(),
          normalized.data_ptr<float>(), converted.data_ptr<converted_t>(), nullptr, nullptr, width, epsilon);
    };
    if (width <= detail::kLayerNormWideThreadsPerBlock * 4) {
      launch_wide(std::integral_constant<int, 4>{});
    } else if (width <= detail::kLayerNormWideThreadsPerBlock * 8) {
      launch_wide(std::integral_constant<int, 8>{});
    } else if (width <= detail::kLayerNormWideThreadsPerBlock * kMaximumCachedValuesPerLane) {
      launch_wide(std::integral_constant<int, kMaximumCachedValuesPerLane>{});
    } else {
      launch_wide(std::integral_constant<int, 0>{});
    }
    return;
  }
  const auto launch = [&](auto cached_values) {
    constexpr int compiled_cached_values = decltype(cached_values)::value;
    autocast_layer_norm_and_cast_kernel<compiled_cached_values, input_t,
        parameter_t, converted_t><<<grid, kThreadsPerBlock, 0, stream>>>(
        input.data_ptr<input_t>(), weight.data_ptr<parameter_t>(), bias.data_ptr<parameter_t>(),
        normalized.data_ptr<float>(), converted.data_ptr<converted_t>(), row_count, width, epsilon);
  };
  if (width <= kWarpSize) {
    launch(std::integral_constant<int, 1>{});
  } else if (width <= kWarpSize * 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (width <= kWarpSize * 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (width <= kWarpSize * 8) {
    launch(std::integral_constant<int, 8>{});
  } else {
    launch(std::integral_constant<int, kMaximumCachedValuesPerLane>{});
  }
}

template <typename input_t, typename parameter_t>
void dispatch_converted_type(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, torch::Tensor& normalized, torch::Tensor& converted,
    float epsilon) {
  if (converted.scalar_type() == torch::kFloat16) {
    launch_typed<input_t, parameter_t, c10::Half>(
        input, weight, bias, normalized, converted, epsilon);
  } else {
    launch_typed<input_t, parameter_t, c10::BFloat16>(
        input, weight, bias, normalized, converted, epsilon);
  }
}

template <typename input_t>
void dispatch_parameter_type(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, torch::Tensor& normalized, torch::Tensor& converted,
    float epsilon) {
  switch (weight.scalar_type()) {
    case torch::kFloat32:
      dispatch_converted_type<input_t, float>(input, weight, bias, normalized, converted, epsilon);
      break;
    case torch::kFloat16:
      dispatch_converted_type<input_t, c10::Half>(input, weight, bias, normalized, converted, epsilon);
      break;
    case torch::kBFloat16:
      dispatch_converted_type<input_t, c10::BFloat16>(input, weight, bias, normalized, converted, epsilon);
      break;
    default:
      TORCH_CHECK(false, "unsupported CUDA LayerNorm affine type");
  }
}

}  // namespace

bool supports_autocast_layer_norm_and_cast(const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias,
    at::IntArrayRef normalized_shape, torch::ScalarType converted_type) {
  if (!detail::is_autocast_layer_norm_layout_supported(input, weight, bias, normalized_shape) ||
      weight.numel() > std::numeric_limits<int>::max() ||
      (converted_type != torch::kFloat16 && converted_type != torch::kBFloat16)) {
    return false;
  }
  constexpr int warps_per_block = kThreadsPerBlock / kWarpSize;
  const int64_t width = weight.numel();
  const int64_t row_count = input.numel() / width;
  const int64_t grid_size = width <= kWarpSize * kMaximumCachedValuesPerLane
      ? (row_count + warps_per_block - 1) / warps_per_block : row_count;
  return grid_size <= std::numeric_limits<int>::max();
}

std::vector<torch::Tensor> autocast_layer_norm_and_cast(const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon,
    torch::ScalarType converted_type) {
  TORCH_CHECK(supports_autocast_layer_norm_and_cast(
                  input, weight, bias, weight.sizes(), converted_type),
      "autocast LayerNorm and cast received an unsupported CUDA tensor layout");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto normalized = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32)
      .memory_format(torch::MemoryFormat::Contiguous));
  auto converted = torch::empty(input.sizes(), input.options().dtype(converted_type)
      .memory_format(torch::MemoryFormat::Contiguous));
  if (input.numel() != 0) {
    if (input.scalar_type() == torch::kFloat16) {
      dispatch_parameter_type<c10::Half>(input, weight, bias, normalized, converted, epsilon);
    } else {
      dispatch_parameter_type<c10::BFloat16>(input, weight, bias, normalized, converted, epsilon);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {std::move(normalized), std::move(converted)};
}

}  // namespace djl::pytorch::cuda
