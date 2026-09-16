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

#include "djl_pytorch_row_ops_cuda.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace djl::pytorch::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaximumIndexRank = 8;

// Sizes and strides travel as launch arguments; index tensors stay on their current device.
struct IndexLayout {
  const void* data;
  int rank;
  torch::ScalarType type;
  int64_t sizes[kMaximumIndexRank];
  int64_t strides[kMaximumIndexRank];
};

bool supports_index_layout(const torch::Tensor& indices, const torch::Device& device) {
  if (!indices.is_cuda() || indices.device() != device || indices.dim() > kMaximumIndexRank ||
      (indices.scalar_type() != torch::kInt16 && indices.scalar_type() != torch::kInt32 &&
          indices.scalar_type() != torch::kInt64)) {
    return false;
  }
  for (int64_t axis = 0; axis < indices.dim(); ++axis) {
    if (indices.stride(axis) < 0) {
      return false;
    }
  }
  return true;
}

IndexLayout index_layout(const torch::Tensor& indices) {
  IndexLayout layout{};
  if (indices.defined()) {
    layout.data = indices.data_ptr();
    layout.rank = static_cast<int>(indices.dim());
    layout.type = indices.scalar_type();
    std::copy(indices.sizes().begin(), indices.sizes().end(), layout.sizes);
    std::copy(indices.strides().begin(), indices.strides().end(), layout.strides);
  }
  return layout;
}

__device__ int64_t index_value(int64_t logical_index, const IndexLayout& layout) {
  int64_t offset = 0;
  for (int axis = layout.rank - 1; axis >= 0; --axis) {
    const int64_t coordinate = logical_index % layout.sizes[axis];
    logical_index /= layout.sizes[axis];
    offset += coordinate * layout.strides[axis];
  }
  switch (layout.type) {
    case torch::kInt16:
      return static_cast<const int16_t*>(layout.data)[offset];
    case torch::kInt32:
      return static_cast<const int32_t*>(layout.data)[offset];
    case torch::kInt64:
      return static_cast<const int64_t*>(layout.data)[offset];
    default:
      return 0;
  }
}

__device__ int64_t clamp_index(int64_t value, int64_t minimum, int64_t maximum) {
  return value < minimum ? minimum : (value > maximum ? maximum : value);
}

template <int group_width, typename scalar_t, bool has_inner, bool has_batch>
__global__ void padded_batch_gather_kernel(const scalar_t* source, scalar_t* output,
    IndexLayout batch_layout, IndexLayout outer_layout, IndexLayout inner_layout,
    int64_t output_rows, int64_t indices_per_batch, int64_t batch_count,
    int64_t outer_entries, int64_t inner_entries, int64_t row_width,
    int64_t batch_stride, int64_t outer_stride, int64_t inner_stride) {
  const int lane = threadIdx.x % group_width;
  const int group = threadIdx.x / group_width;
  const int groups_per_block = blockDim.x / group_width;
  for (int64_t first_row = static_cast<int64_t>(blockIdx.x) * groups_per_block;
       first_row < output_rows; first_row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t row = first_row + group;
    long long source_offset = 0;
    int valid = 0;
    if (lane == 0 && row < output_rows) {
      const int64_t outer = index_value(row, outer_layout);
      const int64_t inner = has_inner ? index_value(row, inner_layout) : 1;
      const int64_t batch = has_batch ? index_value(row, batch_layout) : row / indices_per_batch;
      valid = batch >= 0 && batch < batch_count && outer > 0 && outer <= outer_entries &&
          inner > 0 && inner <= inner_entries;
      source_offset = clamp_index(batch, 0, batch_count - 1) * batch_stride +
          (clamp_index(outer, 1, outer_entries) - 1) * outer_stride +
          (clamp_index(inner, 1, inner_entries) - 1) * inner_stride;
    }
    source_offset = __shfl_sync(0xffffffff, source_offset, 0, group_width);
    valid = __shfl_sync(0xffffffff, valid, 0, group_width);
    if (row < output_rows) {
      for (int64_t feature = lane; feature < row_width; feature += group_width) {
        // Match reference clamp/gather/multiply semantics for NaN, infinity, and signed zero.
        output[row * row_width + feature] = static_cast<scalar_t>(
            static_cast<float>(source[source_offset + feature]) * static_cast<float>(valid));
      }
    }
  }
}

template <bool has_inner, bool has_batch>
torch::Tensor launch_padded_batch_gather(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  constexpr int64_t indexed_dimensions = has_inner ? 3 : 2;
  c10::cuda::CUDAGuard device_guard(source.device());
  auto shape = outer_stored_indices.sizes().vec();
  shape.insert(shape.end(), source.sizes().begin() + indexed_dimensions, source.sizes().end());
  auto output = torch::empty(shape, source.options());
  if (output.numel() == 0) {
    return output;
  }
  const int64_t output_rows = outer_stored_indices.numel();
  const int64_t row_width = output.numel() / output_rows;
  const int64_t indices_per_batch = has_batch ? 1 : output_rows / source.size(0);
  const int64_t inner_entries = has_inner ? source.size(2) : 1;
  const int64_t inner_stride = has_inner ? source.stride(2) : 0;
  const auto batch_layout = index_layout(batch_indices);
  const auto outer_layout = index_layout(outer_stored_indices);
  const auto inner_layout = index_layout(inner_stored_indices);
  const auto stream = c10::cuda::getCurrentCUDAStream(source.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, source.scalar_type(),
      "padded_batch_gather_cuda", [&] {
        const auto launch = [&](auto group_size) {
          constexpr int group_width = decltype(group_size)::value;
          constexpr int groups_per_block = kThreadsPerBlock / group_width;
          const int blocks = static_cast<int>(std::min<int64_t>(
              (output_rows + groups_per_block - 1) / groups_per_block, 65535));
          padded_batch_gather_kernel<group_width, scalar_t, has_inner, has_batch>
              <<<blocks, kThreadsPerBlock, 0, stream>>>(source.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                  batch_layout, outer_layout, inner_layout, output_rows, indices_per_batch, source.size(0),
                  source.size(1), inner_entries, row_width, source.stride(0), source.stride(1), inner_stride);
        };
        if (row_width < 8) {
          launch(std::integral_constant<int, 1>{});
        } else if (row_width < 32) {
          launch(std::integral_constant<int, 8>{});
        } else {
          launch(std::integral_constant<int, 32>{});
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

bool supports_padded_batch_gather(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  const int64_t indexed_dimensions = inner_stored_indices.defined() ? 3 : 2;
  const auto type = source.scalar_type();
  if (!source.is_cuda() || source.dim() < indexed_dimensions ||
      (batch_indices.defined() && inner_stored_indices.defined()) ||
      (type != torch::kFloat32 && type != torch::kFloat16 && type != torch::kBFloat16) ||
      !supports_index_layout(outer_stored_indices, source.device())) {
    return false;
  }
  if (batch_indices.defined() &&
      (!supports_index_layout(batch_indices, source.device()) ||
          batch_indices.sizes() != outer_stored_indices.sizes())) {
    return false;
  }
  if (inner_stored_indices.defined() &&
      (!supports_index_layout(inner_stored_indices, source.device()) ||
          inner_stored_indices.sizes() != outer_stored_indices.sizes())) {
    return false;
  }
  int64_t expected_stride = 1;
  for (int64_t axis = source.dim() - 1; axis >= indexed_dimensions; --axis) {
    if (source.size(axis) > 1 && source.stride(axis) != expected_stride) {
      return false;
    }
    expected_stride *= source.size(axis);
  }
  for (int64_t axis = 0; axis < indexed_dimensions; ++axis) {
    if (source.size(axis) <= 0 || source.stride(axis) < 0) {
      return false;
    }
  }
  return batch_indices.defined() ||
      (outer_stored_indices.dim() > 0 && outer_stored_indices.size(0) == source.size(0));
}

torch::Tensor padded_batch_gather_forward(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  TORCH_CHECK(supports_padded_batch_gather(source, batch_indices, outer_stored_indices, inner_stored_indices),
      "padded batch gather received an unsupported CUDA tensor layout");
  if (inner_stored_indices.defined()) {
    return launch_padded_batch_gather<true, false>(source, {}, outer_stored_indices, inner_stored_indices);
  }
  if (batch_indices.defined()) {
    return launch_padded_batch_gather<false, true>(source, batch_indices, outer_stored_indices, {});
  }
  return launch_padded_batch_gather<false, false>(source, {}, outer_stored_indices, {});
}

}  // namespace djl::pytorch::cuda
