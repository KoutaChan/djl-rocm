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

#ifndef DJL_PYTORCH_LAYER_NORM_KERNELS_H
#define DJL_PYTORCH_LAYER_NORM_KERNELS_H

#include <cstdint>

namespace djl::pytorch::detail {

constexpr int kLayerNormWideThreadsPerBlock = 256;

__device__ inline void merge_layer_norm_statistics(
    float& mean, float& moment, int& count, float other_mean, float other_moment, int other_count) {
  if (other_count > 0) {
    const int combined_count = count + other_count;
    const float delta = other_mean - mean;
    moment += other_moment + delta * delta * static_cast<float>(count) * static_cast<float>(other_count) /
                                 static_cast<float>(combined_count);
    mean += delta * static_cast<float>(other_count) / static_cast<float>(combined_count);
    count = combined_count;
  }
}

__device__ inline void reduce_layer_norm_statistics(float& mean, float& moment, int& count) {
  const int lane = threadIdx.x % warpSize;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
#if defined(__HIPCC__)
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int other_count = __shfl_down(count, offset);
#else
    const float other_mean = __shfl_down_sync(0xffffffff, mean, offset);
    const float other_moment = __shfl_down_sync(0xffffffff, moment, offset);
    const int other_count = __shfl_down_sync(0xffffffff, count, offset);
#endif
    if (lane + offset < warpSize) {
      merge_layer_norm_statistics(mean, moment, count, other_mean, other_moment, other_count);
    }
  }
}

// Wide rows use a whole block. Cache a bounded number of values per thread;
// larger rows are reread after reduction instead of consuming unbounded registers.
template <int cached_values_per_thread, typename input_t, typename parameter_t, typename converted_t,
    bool write_converted>
__global__ void wide_autocast_layer_norm_kernel(const input_t* input, const parameter_t* weight,
    const parameter_t* bias, float* normalized_output, converted_t* converted_output, float* mean_output,
    float* reciprocal_standard_deviation_output, int64_t width, float epsilon) {
  constexpr int maximum_warps_per_block = kLayerNormWideThreadsPerBlock / 32;
  const int warps_per_block = kLayerNormWideThreadsPerBlock / warpSize;
  __shared__ float means[maximum_warps_per_block];
  __shared__ float moments[maximum_warps_per_block];
  __shared__ int counts[maximum_warps_per_block];
  float cached_values[cached_values_per_thread > 0 ? cached_values_per_thread : 1];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * width;
  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  if constexpr (cached_values_per_thread > 0) {
#pragma unroll
    for (int index = 0; index < cached_values_per_thread; ++index) {
      const int64_t feature = threadIdx.x + static_cast<int64_t>(index) * kLayerNormWideThreadsPerBlock;
      if (feature < width) {
        const float value = static_cast<float>(input[row_offset + feature]);
        cached_values[index] = value;
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
  } else {
    for (int64_t feature = threadIdx.x; feature < width; feature += kLayerNormWideThreadsPerBlock) {
      const float value = static_cast<float>(input[row_offset + feature]);
      ++count;
      const float delta = value - mean;
      mean += delta / count;
      moment += delta * (value - mean);
    }
  }
  reduce_layer_norm_statistics(mean, moment, count);
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    means[warp] = mean;
    moments[warp] = moment;
    counts[warp] = count;
  }
  __syncthreads();
  if (warp == 0) {
    mean = lane < warps_per_block ? means[lane] : 0.0f;
    moment = lane < warps_per_block ? moments[lane] : 0.0f;
    count = lane < warps_per_block ? counts[lane] : 0;
    reduce_layer_norm_statistics(mean, moment, count);
    if (lane == 0) {
      means[0] = mean;
      moments[0] = rsqrtf(moment / static_cast<float>(width) + epsilon);
      if (mean_output != nullptr) {
        mean_output[blockIdx.x] = mean;
        reciprocal_standard_deviation_output[blockIdx.x] = moments[0];
      }
    }
  }
  __syncthreads();
  mean = means[0];
  const float inverse_standard_deviation = moments[0];
  if constexpr (cached_values_per_thread > 0) {
#pragma unroll
    for (int index = 0; index < cached_values_per_thread; ++index) {
      const int64_t feature = threadIdx.x + static_cast<int64_t>(index) * kLayerNormWideThreadsPerBlock;
      if (feature < width) {
        const float normalized = (cached_values[index] - mean) * inverse_standard_deviation;
        const float affine = normalized * static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]);
        normalized_output[row_offset + feature] = affine;
        if constexpr (write_converted) {
          converted_output[row_offset + feature] = static_cast<converted_t>(affine);
        }
      }
    }
  } else {
    for (int64_t feature = threadIdx.x; feature < width; feature += kLayerNormWideThreadsPerBlock) {
      const float value = static_cast<float>(input[row_offset + feature]);
      const float normalized = (value - mean) * inverse_standard_deviation;
      const float affine = normalized * static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]);
      normalized_output[row_offset + feature] = affine;
      if constexpr (write_converted) {
        converted_output[row_offset + feature] = static_cast<converted_t>(affine);
      }
    }
  }
}

}  // namespace djl::pytorch::detail

#endif  // DJL_PYTORCH_LAYER_NORM_KERNELS_H
