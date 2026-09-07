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
#ifndef DJL_TORCH_DJL_PYTORCH_FUSION_BACKEND_H
#define DJL_TORCH_DJL_PYTORCH_FUSION_BACKEND_H

#include <c10/util/Exception.h>

#include <cstddef>

#if defined(DJL_USE_ROCM_KERNELS)
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#elif defined(DJL_USE_CUDA_FUSION_KERNELS)
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#else
#error "A CUDA or ROCm fusion backend must be selected"
#endif

namespace djl::pytorch::fusion::backend {

#if defined(DJL_USE_ROCM_KERNELS)

using Stream = hipStream_t;

inline Stream GetCurrentStream(int device) {
  return at::hip::getCurrentHIPStream(device).stream();
}

inline void CheckLastLaunch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ",
      hipGetErrorString(error));
}

inline void MemsetAsync(
    void* destination, int value, std::size_t bytes, Stream stream,
    const char* operation) {
  const hipError_t error = hipMemsetAsync(destination, value, bytes, stream);
  TORCH_CHECK(error == hipSuccess, operation, " failed: ",
      hipGetErrorString(error));
}

#define DJL_FUSION_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
  hipLaunchKernelGGL((kernel), grid, block, shared, stream, __VA_ARGS__)

#else

using Stream = cudaStream_t;

inline Stream GetCurrentStream(int device) {
  return at::cuda::getCurrentCUDAStream(device).stream();
}

inline void CheckLastLaunch(const char* kernel_name) {
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, kernel_name, " launch failed: ",
      cudaGetErrorString(error));
}

inline void MemsetAsync(
    void* destination, int value, std::size_t bytes, Stream stream,
    const char* operation) {
  const cudaError_t error = cudaMemsetAsync(destination, value, bytes, stream);
  TORCH_CHECK(error == cudaSuccess, operation, " failed: ",
      cudaGetErrorString(error));
}

#if defined(__CUDACC__)

template <typename T>
__device__ inline T Shuffle(T value, int source_lane) {
  return __shfl_sync(__activemask(), value, source_lane);
}

template <typename T>
__device__ inline T ShuffleDown(T value, unsigned int delta) {
  return __shfl_down_sync(__activemask(), value, delta);
}

template <typename T>
__device__ inline T ShuffleDown(T value, unsigned int delta, int width) {
  return __shfl_down_sync(__activemask(), value, delta, width);
}

#define DJL_FUSION_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
  (kernel)<<<grid, block, shared, stream>>>(__VA_ARGS__)
#define __shfl ::djl::pytorch::fusion::backend::Shuffle
#define __shfl_down ::djl::pytorch::fusion::backend::ShuffleDown

#endif  // defined(__CUDACC__)

#endif  // defined(DJL_USE_ROCM_KERNELS)

}  // namespace djl::pytorch::fusion::backend

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_BACKEND_H
