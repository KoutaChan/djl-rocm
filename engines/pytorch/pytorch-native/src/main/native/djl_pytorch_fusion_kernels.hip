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

}  // namespace djl::pytorch::fusion
