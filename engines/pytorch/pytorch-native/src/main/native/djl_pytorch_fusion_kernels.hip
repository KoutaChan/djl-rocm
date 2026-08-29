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

namespace djl::pytorch::fusion {
namespace {

constexpr int kThreadsPerBlock = 256;

struct OutputPackArguments {
  const void* sources[kMaximumOutputPackSources];
  int64_t widths[kMaximumOutputPackSources];
  int64_t destination_offsets[kMaximumOutputPackSources];
  int32_t data_types[kMaximumOutputPackSources];
  float* output;
  int64_t row_count;
  int64_t output_width;
};

template <typename scalar_t>
__device__ void pack_source(const OutputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const scalar_t*>(arguments.sources[source_index]);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    arguments.output[row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<float>(source[index]);
  }
}

__global__ void output_pack_kernel(OutputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_source<c10::Half>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_source<c10::BFloat16>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_source<float>(arguments, source_index);
      break;
    default:
      break;
  }
}

void CheckHipLaunch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ", hipGetErrorString(error));
}

}  // namespace

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  OutputPackArguments arguments{};
  arguments.output = output.data_ptr<float>();
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

  const int64_t maximum_elements = row_count * maximum_width;
  const int64_t required_blocks =
      (maximum_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  hipLaunchKernelGGL(output_pack_kernel, dim3(block_count, source_count),
      dim3(kThreadsPerBlock), 0, stream.stream(), arguments);
  CheckHipLaunch("fusion output pack");
}

}  // namespace djl::pytorch::fusion
