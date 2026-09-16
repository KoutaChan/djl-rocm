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

#include "djl_pytorch_nn_functional_cuda.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace djl::pytorch::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kEmbeddingGroupWidth = 32;
constexpr int kMaximumIndices = 2;

bool is_supported_floating(const torch::Tensor& tensor) {
  return !tensor.is_neg() && !tensor.is_conj() &&
      (tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat16 || tensor.scalar_type() == torch::kBFloat16);
}

struct IndexLayout {
  const void* data;
  int64_t row_stride;
  int64_t token_stride;
  torch::ScalarType type;
};

bool supports_index_layout(const torch::Tensor& indices, const torch::Tensor& tokens) {
  return indices.is_cuda() && indices.device() == tokens.device() && indices.dim() == 2 &&
      indices.size(0) == tokens.size(0) && indices.size(1) == tokens.size(1) &&
      indices.stride(0) >= 0 && indices.stride(1) >= 0 && !tokens.is_alias_of(indices) &&
      !indices.is_neg() && !indices.is_conj() &&
      (indices.scalar_type() == torch::kInt16 || indices.scalar_type() == torch::kInt32 ||
          indices.scalar_type() == torch::kInt64);
}

IndexLayout index_layout(const torch::Tensor& indices) {
  return {indices.data_ptr(), indices.stride(0), indices.stride(1), indices.scalar_type()};
}

__device__ int64_t index_value(const IndexLayout& layout, int64_t batch, int64_t token) {
  const int64_t offset = batch * layout.row_stride + token * layout.token_stride;
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

template <typename scalar_t>
__device__ float rounded(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

struct MaskedEmbeddingArguments {
  IndexLayout indices[kMaximumIndices];
  IndexLayout mask;
  int64_t rows;
  int64_t tokens_per_batch;
  int64_t width;
  int64_t vocabulary_size;
  int64_t padding_index;
  int index_count;
  bool mean_valid;
};

template <typename scalar_t>
__global__ void masked_embedding_residual_kernel(scalar_t* tokens,
    const scalar_t* table, scalar_t* converted_mask, MaskedEmbeddingArguments arguments) {
  constexpr int groups_per_block = kThreadsPerBlock / kEmbeddingGroupWidth;
  const int lane = threadIdx.x % kEmbeddingGroupWidth;
  const int group = threadIdx.x / kEmbeddingGroupWidth;
  for (int64_t first_row = static_cast<int64_t>(blockIdx.x) * groups_per_block;
       first_row < arguments.rows;
       first_row += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t row = first_row + group;
    if (row >= arguments.rows) {
      continue;
    }
    const int64_t batch = row / arguments.tokens_per_batch;
    const int64_t token = row % arguments.tokens_per_batch;
    int64_t table_rows[kMaximumIndices];
    int valid_count = 0;
    for (int index = 0; index < arguments.index_count; ++index) {
      table_rows[index] = index_value(arguments.indices[index], batch, token);
      // CUDA embedding validates indices even when the final token mask is zero.
      if (table_rows[index] < 0 || table_rows[index] >= arguments.vocabulary_size) {
        CUDA_KERNEL_ASSERT(false && "embedding index is out of bounds");
        return;
      }
      valid_count += table_rows[index] != arguments.padding_index;
    }
    const scalar_t mask = static_cast<scalar_t>(
        static_cast<float>(index_value(arguments.mask, batch, token)));
    if (lane == 0) {
      converted_mask[row] = mask;
    }
    for (int64_t column = lane; column < arguments.width; column += kEmbeddingGroupWidth) {
      // Keep padding loads and multiply by zero: NaN/Inf must remain observable.
      float identity = rounded<scalar_t>(__fmul_rn(
          static_cast<float>(table[table_rows[0] * arguments.width + column]),
          table_rows[0] != arguments.padding_index ? 1.0f : 0.0f));
      if (arguments.index_count == 2) {
        const float second = rounded<scalar_t>(__fmul_rn(
            static_cast<float>(table[table_rows[1] * arguments.width + column]),
            table_rows[1] != arguments.padding_index ? 1.0f : 0.0f));
        identity = rounded<scalar_t>(__fadd_rn(identity, second));
      }
      if (arguments.mean_valid) {
        identity = rounded<scalar_t>(identity / static_cast<float>(valid_count > 0 ? valid_count : 1));
      }
      const int64_t offset = row * arguments.width + column;
      const float updated = rounded<scalar_t>(
          __fadd_rn(static_cast<float>(tokens[offset]), identity));
      tokens[offset] = static_cast<scalar_t>(__fmul_rn(updated, static_cast<float>(mask)));
    }
  }
}

template <typename scalar_t>
__global__ void broadcast_residual_silu_kernel(scalar_t* values,
    const scalar_t* bias, const scalar_t* residual, const scalar_t* mask,
    int64_t elements, int64_t items_per_batch, int64_t width) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index % width;
    const int64_t residual_offset = row / items_per_batch * width + column;
    float value = static_cast<float>(values[index]);
    if (bias != nullptr) {
      value = rounded<scalar_t>(__fadd_rn(value, static_cast<float>(bias[column])));
    }
    value = rounded<scalar_t>(__fadd_rn(value, static_cast<float>(residual[residual_offset])));
    // The portable path materializes sigmoid before multiplying in the values dtype.
    const float sigmoid = rounded<scalar_t>(1.0f / (1.0f + expf(-value)));
    float activated = rounded<scalar_t>(__fmul_rn(value, sigmoid));
    if (mask != nullptr) {
      activated = __fmul_rn(activated, static_cast<float>(mask[row]));
    }
    values[index] = static_cast<scalar_t>(activated);
  }
}

void launch_broadcast_residual_silu(torch::Tensor& values, const torch::Tensor* bias,
    const torch::Tensor& residual, const torch::Tensor* mask) {
  const int64_t elements = values.numel();
  if (elements == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  const uint32_t blocks = static_cast<uint32_t>(
      std::min<int64_t>((elements - 1) / kThreadsPerBlock + 1, 65535));
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, values.scalar_type(),
      "owned_broadcast_residual_silu", [&] {
        broadcast_residual_silu_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
            values.data_ptr<scalar_t>(), bias == nullptr ? nullptr : bias->data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(), mask == nullptr ? nullptr : mask->data_ptr<scalar_t>(),
            elements, values.size(1), values.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool supports_masked_embedding_residual_to_owned_tokens(const torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask) {
  if (!tokens.is_cuda() || !tokens.is_contiguous() || tokens.dim() != 3 || tokens.size(2) <= 0 ||
      !is_supported_floating(tokens) || !embedding_table.is_cuda() || !embedding_table.is_contiguous() ||
      embedding_table.dim() != 2 || embedding_table.size(1) != tokens.size(2) ||
      embedding_table.scalar_type() != tokens.scalar_type() || embedding_table.device() != tokens.device() ||
      embedding_table.is_neg() || embedding_table.is_conj() ||
      tokens.is_alias_of(embedding_table) || stored_indices.empty() || stored_indices.size() > kMaximumIndices ||
      !supports_index_layout(valid_mask, tokens)) {
    return false;
  }
  return std::all_of(stored_indices.begin(), stored_indices.end(), [&](const torch::Tensor& indices) {
    return supports_index_layout(indices, tokens);
  });
}

torch::Tensor add_masked_embedding_residual_to_owned_tokens(torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask, int64_t padding_index, bool mean_valid) {
  TORCH_CHECK(supports_masked_embedding_residual_to_owned_tokens(
          tokens, stored_indices, embedding_table, valid_mask),
      "masked embedding residual received an unsupported CUDA tensor layout");
  const c10::cuda::CUDAGuard guard(tokens.device());
  torch::Tensor converted_mask = torch::empty(valid_mask.sizes(),
      tokens.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t rows = tokens.numel() / tokens.size(2);
  if (rows == 0) {
    return converted_mask;
  }
  MaskedEmbeddingArguments arguments{};
  arguments.mask = index_layout(valid_mask);
  arguments.rows = rows;
  arguments.tokens_per_batch = tokens.size(1);
  arguments.width = tokens.size(2);
  arguments.vocabulary_size = embedding_table.size(0);
  arguments.padding_index = padding_index;
  arguments.index_count = static_cast<int>(stored_indices.size());
  arguments.mean_valid = mean_valid;
  for (int index = 0; index < arguments.index_count; ++index) {
    arguments.indices[index] = index_layout(stored_indices[index]);
  }
  constexpr int groups_per_block = kThreadsPerBlock / kEmbeddingGroupWidth;
  const uint32_t blocks = static_cast<uint32_t>(
      std::min<int64_t>((rows - 1) / groups_per_block + 1, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(tokens.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, tokens.scalar_type(),
      "owned_masked_embedding_residual", [&] {
        masked_embedding_residual_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
            tokens.data_ptr<scalar_t>(), embedding_table.data_ptr<scalar_t>(),
            converted_mask.data_ptr<scalar_t>(), arguments);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return converted_mask;
}

bool supports_broadcast_residual_to_owned_silu(const torch::Tensor& values,
    const torch::Tensor& residual, const torch::Tensor* mask) {
  if (!values.is_cuda() || !values.is_contiguous() || values.dim() != 3 || values.size(2) <= 0 ||
      !is_supported_floating(values) || !residual.is_cuda() || !residual.is_contiguous() ||
      residual.dim() != 3 || residual.size(0) != values.size(0) || residual.size(1) != 1 ||
      residual.size(2) != values.size(2) || residual.scalar_type() != values.scalar_type() ||
      residual.device() != values.device() || values.is_alias_of(residual) ||
      residual.is_neg() || residual.is_conj()) {
    return false;
  }
  return mask == nullptr || (mask->is_cuda() && mask->is_contiguous() && mask->dim() == 2 &&
      mask->size(0) == values.size(0) && mask->size(1) == values.size(1) &&
      mask->scalar_type() == values.scalar_type() && mask->device() == values.device() &&
      !values.is_alias_of(*mask) && !mask->is_neg() && !mask->is_conj());
}

bool supports_bias_and_broadcast_residual_to_owned_silu(const torch::Tensor& values,
    const torch::Tensor& bias, const torch::Tensor& residual, const torch::Tensor* mask) {
  return supports_broadcast_residual_to_owned_silu(values, residual, mask) && bias.is_cuda() &&
      bias.is_contiguous() && bias.dim() == 1 && bias.size(0) == values.size(2) &&
      bias.scalar_type() == values.scalar_type() && bias.device() == values.device() &&
      !values.is_alias_of(bias) && !bias.is_neg() && !bias.is_conj();
}

void add_broadcast_residual_to_owned_and_silu(
    torch::Tensor& values, const torch::Tensor& residual, const torch::Tensor* mask) {
  TORCH_CHECK(supports_broadcast_residual_to_owned_silu(values, residual, mask),
      "broadcast residual SiLU received an unsupported CUDA tensor layout");
  launch_broadcast_residual_silu(values, nullptr, residual, mask);
}

void add_bias_and_broadcast_residual_to_owned_and_silu(torch::Tensor& values,
    const torch::Tensor& bias, const torch::Tensor& residual, const torch::Tensor* mask) {
  TORCH_CHECK(supports_bias_and_broadcast_residual_to_owned_silu(values, bias, residual, mask),
      "bias and broadcast residual SiLU received an unsupported CUDA tensor layout");
  launch_broadcast_residual_silu(values, &bias, residual, mask);
}

}  // namespace djl::pytorch::cuda
