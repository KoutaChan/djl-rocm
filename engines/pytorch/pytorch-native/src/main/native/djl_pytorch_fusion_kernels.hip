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
  float* output;
  int64_t row_count;
  int64_t output_width;
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

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion output-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = at::hip::getCurrentHIPStream(output.get_device());
  hipLaunchKernelGGL(output_pack_kernel, dim3(block_count, source_count),
      dim3(kThreadsPerBlock), 0, stream.stream(), arguments);
  CheckHipLaunch("fusion output pack");
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

}  // namespace djl::pytorch::fusion
