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
#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "djl_pytorch_single_query_attention.h"

namespace djl::pytorch::rocm {
namespace {

constexpr int kSingleQueryAttentionThreads = 256;
constexpr int kSingleQueryAttentionMaximumKeys = 256;
constexpr int kSingleQueryAttentionWideHead = 32;
constexpr int kSingleQueryAttentionLongSequence = 64;

void check_single_query_attention_launch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ", hipGetErrorString(error));
}

bool supports_attention_mask(const torch::Tensor& mask, const torch::Tensor& query, int64_t key_tokens) {
  if (!mask.is_cuda() || mask.device() != query.device() || !mask.is_contiguous() || mask.dim() < 1 || mask.dim() > 4 ||
      (mask.scalar_type() != torch::kFloat32 && mask.scalar_type() != torch::kBFloat16)) {
    return false;
  }
  const int64_t expected_sizes[] = {query.size(0), query.size(1), 1, key_tokens};
  const int64_t leading_dimensions = 4 - mask.dim();
  for (int64_t dimension = 0; dimension < mask.dim(); ++dimension) {
    const int64_t expected = expected_sizes[leading_dimensions + dimension];
    if (mask.size(dimension) != 1 && mask.size(dimension) != expected) {
      return false;
    }
  }
  return true;
}

struct AttentionMaskStrides {
  int64_t batch;
  int64_t head;
  int64_t key;
};

AttentionMaskStrides attention_mask_strides(const torch::Tensor& mask) {
  const int64_t leading_dimensions = 4 - mask.dim();
  const auto broadcast_stride = [&](int64_t attention_dimension) {
    const int64_t mask_dimension = attention_dimension - leading_dimensions;
    return mask_dimension >= 0 && mask.size(mask_dimension) > 1 ? mask.stride(mask_dimension) : int64_t{0};
  };
  return {broadcast_stride(0), broadcast_stride(1), broadcast_stride(3)};
}

__device__ __forceinline__ float maximum_propagating_nan(float left, float right) {
  return isnan(left) || isnan(right) ? NAN : fmaxf(left, right);
}

template <typename mask_t, bool has_mask>
__global__ void single_query_attention_forward_kernel(const c10::BFloat16* query, const c10::BFloat16* key,
    const c10::BFloat16* value, const mask_t* mask, c10::BFloat16* output, float* probabilities, int heads,
    int key_tokens, int width, int64_t mask_batch_stride, int64_t mask_head_stride, int64_t mask_key_stride,
    float scale) {
  __shared__ float reduction[kSingleQueryAttentionMaximumKeys];
  const int key_token = threadIdx.x;
  const int64_t matrix = blockIdx.x;
  const int head = static_cast<int>(matrix % heads);
  const int64_t batch = matrix / heads;
  const int64_t query_offset = matrix * width;
  const int64_t key_value_offset = matrix * key_tokens * width;

  float score = -std::numeric_limits<float>::infinity();
  if (key_token < key_tokens) {
    score = 0.0f;
    const int64_t token_offset = key_value_offset + key_token * width;
    for (int feature = 0; feature < width; ++feature) {
      score += static_cast<float>(query[query_offset + feature]) * static_cast<float>(key[token_offset + feature]);
    }
    score *= scale;
    if constexpr (has_mask) {
      const int64_t mask_offset = batch * mask_batch_stride + static_cast<int64_t>(head) * mask_head_stride +
                                  static_cast<int64_t>(key_token) * mask_key_stride;
      score += static_cast<float>(mask[mask_offset]);
    }
  }
  reduction[key_token] = score;
  __syncthreads();

  for (int stride = kSingleQueryAttentionThreads / 2; stride > 0; stride /= 2) {
    if (key_token < stride) {
      reduction[key_token] = maximum_propagating_nan(reduction[key_token], reduction[key_token + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  const bool all_masked = isinf(maximum) && maximum < 0.0f;
  const float weight = key_token < key_tokens && !all_masked ? expf(score - maximum) : 0.0f;
  reduction[key_token] = weight;
  __syncthreads();
  for (int stride = kSingleQueryAttentionThreads / 2; stride > 0; stride /= 2) {
    if (key_token < stride) {
      reduction[key_token] += reduction[key_token + stride];
    }
    __syncthreads();
  }
  const float probability = key_token < key_tokens && !all_masked ? weight / reduction[0] : 0.0f;
  reduction[key_token] = probability;
  if (probabilities != nullptr && key_token < key_tokens) {
    probabilities[matrix * key_tokens + key_token] = probability;
  }
  __syncthreads();

  if (key_token < width) {
    float context = 0.0f;
    for (int token = 0; token < key_tokens; ++token) {
      context += reduction[token] * static_cast<float>(value[key_value_offset + token * width + key_token]);
    }
    output[query_offset + key_token] = static_cast<c10::BFloat16>(context);
  }
}

__global__ void single_query_attention_backward_kernel(const c10::BFloat16* query, const c10::BFloat16* key,
    const c10::BFloat16* value, const float* probabilities, const c10::BFloat16* gradient_output,
    c10::BFloat16* query_gradient, c10::BFloat16* key_gradient, c10::BFloat16* value_gradient, int key_tokens,
    int width, float scale) {
  __shared__ float probability_gradient[kSingleQueryAttentionMaximumKeys];
  __shared__ float reduction[kSingleQueryAttentionMaximumKeys];
  const int key_token = threadIdx.x;
  const int64_t matrix = blockIdx.x;
  const int64_t query_offset = matrix * width;
  const int64_t key_value_offset = matrix * key_tokens * width;

  float gradient = 0.0f;
  float probability = 0.0f;
  if (key_token < key_tokens) {
    probability = probabilities[matrix * key_tokens + key_token];
    const int64_t token_offset = key_value_offset + key_token * width;
    for (int feature = 0; feature < width; ++feature) {
      gradient += static_cast<float>(gradient_output[query_offset + feature]) *
                  static_cast<float>(value[token_offset + feature]);
    }
  }
  probability_gradient[key_token] = gradient;
  reduction[key_token] = probability * gradient;
  __syncthreads();
  for (int stride = kSingleQueryAttentionThreads / 2; stride > 0; stride /= 2) {
    if (key_token < stride) {
      reduction[key_token] += reduction[key_token + stride];
    }
    __syncthreads();
  }
  const float score_gradient =
      key_token < key_tokens ? probability * (probability_gradient[key_token] - reduction[0]) : 0.0f;
  reduction[key_token] = score_gradient;
  __syncthreads();

  if (key_token < key_tokens && (key_gradient != nullptr || value_gradient != nullptr)) {
    const int64_t token_offset = key_value_offset + key_token * width;
    for (int feature = 0; feature < width; ++feature) {
      if (key_gradient != nullptr) {
        key_gradient[token_offset + feature] =
            static_cast<c10::BFloat16>(score_gradient * scale * static_cast<float>(query[query_offset + feature]));
      }
      if (value_gradient != nullptr) {
        value_gradient[token_offset + feature] =
            static_cast<c10::BFloat16>(probability * static_cast<float>(gradient_output[query_offset + feature]));
      }
    }
  }
  if (query_gradient != nullptr && key_token < width) {
    float query_value_gradient = 0.0f;
    for (int token = 0; token < key_tokens; ++token) {
      query_value_gradient += reduction[token] * static_cast<float>(key[key_value_offset + token * width + key_token]);
    }
    query_gradient[query_offset + key_token] = static_cast<c10::BFloat16>(query_value_gradient * scale);
  }
}

template <typename mask_t, bool has_mask>
void launch_single_query_attention_forward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor* mask, torch::Tensor& output, torch::Tensor& probabilities,
    AttentionMaskStrides mask_strides, float scale, hipStream_t stream) {
  const int64_t matrices = query.size(0) * query.size(1);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(single_query_attention_forward_kernel<mask_t, has_mask>),
      dim3(static_cast<unsigned int>(matrices)), dim3(kSingleQueryAttentionThreads), 0, stream,
      query.data_ptr<c10::BFloat16>(), key.data_ptr<c10::BFloat16>(), value.data_ptr<c10::BFloat16>(),
      has_mask ? mask->data_ptr<mask_t>() : nullptr, output.data_ptr<c10::BFloat16>(),
      probabilities.defined() ? probabilities.data_ptr<float>() : nullptr, static_cast<int>(query.size(1)),
      static_cast<int>(key.size(2)), static_cast<int>(query.size(3)), mask_strides.batch, mask_strides.head,
      mask_strides.key, scale);
}

}  // namespace

bool supports_single_query_attention(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const std::optional<torch::Tensor>& mask) {
  if (!query.is_cuda() || query.scalar_type() != torch::kBFloat16 || !query.is_contiguous() || !key.is_contiguous() ||
      !value.is_contiguous() || query.dim() != 4 || key.dim() != 4 || value.dim() != 4 || query.size(0) <= 0 ||
      query.size(1) <= 0 || query.size(2) != 1 || (query.size(3) != 16 && query.size(3) != 32) ||
      key.scalar_type() != query.scalar_type() || value.scalar_type() != query.scalar_type() ||
      key.device() != query.device() || value.device() != query.device() || key.size(0) != query.size(0) ||
      key.size(1) != query.size(1) || value.sizes() != key.sizes() || key.size(2) <= 0 ||
      key.size(2) > kSingleQueryAttentionMaximumKeys || key.size(3) != query.size(3) ||
      query.size(0) > std::numeric_limits<unsigned int>::max() / query.size(1)) {
    return false;
  }
  return !mask.has_value() || supports_attention_mask(*mask, query, key.size(2));
}

bool supports_single_query_attention_backward(const torch::Tensor& query, const torch::Tensor& key) {
  // PyTorch's tiled backend is faster once both dimensions require multiple
  // waves. The custom backward remains useful for narrow heads and short keys.
  return query.size(3) < kSingleQueryAttentionWideHead || key.size(2) < kSingleQueryAttentionLongSequence;
}

SingleQueryAttentionForwardResult single_query_attention_forward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const std::optional<torch::Tensor>& mask, float scale, bool capture_probabilities) {
  TORCH_CHECK(supports_single_query_attention(query, key, value, mask),
      "single-query attention received an unsupported ROCm tensor layout");
  c10::DeviceGuard device_guard(query.device());
  const auto output_options = query.options().memory_format(torch::MemoryFormat::Contiguous);
  torch::Tensor output = torch::empty(query.sizes(), output_options);
  torch::Tensor probabilities = capture_probabilities ? torch::empty({query.size(0), query.size(1), key.size(2)},
                                                            output_options.dtype(torch::kFloat32))
                                                      : torch::Tensor();
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  if (!mask.has_value()) {
    launch_single_query_attention_forward<float, false>(
        query, key, value, nullptr, output, probabilities, {0, 0, 0}, scale, stream);
  } else {
    const AttentionMaskStrides strides = attention_mask_strides(*mask);
    if (mask->scalar_type() == torch::kFloat32) {
      launch_single_query_attention_forward<float, true>(
          query, key, value, &*mask, output, probabilities, strides, scale, stream);
    } else {
      launch_single_query_attention_forward<c10::BFloat16, true>(
          query, key, value, &*mask, output, probabilities, strides, scale, stream);
    }
  }
  check_single_query_attention_launch("single_query_attention_forward");
  return {std::move(output), std::move(probabilities)};
}

SingleQueryAttentionGradients single_query_attention_backward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& probabilities, const torch::Tensor& gradient_output, float scale,
    bool needs_query_gradient, bool needs_key_gradient, bool needs_value_gradient) {
  TORCH_CHECK(query.is_cuda() && query.scalar_type() == torch::kBFloat16 && key.device() == query.device() &&
                  value.device() == query.device() && probabilities.device() == query.device() &&
                  gradient_output.device() == query.device(),
      "single-query attention backward inputs must be on the same ROCm device");
  TORCH_CHECK(probabilities.scalar_type() == torch::kFloat32 && probabilities.is_contiguous() &&
                  probabilities.dim() == 3 && probabilities.size(0) == query.size(0) &&
                  probabilities.size(1) == query.size(1) && probabilities.size(2) == key.size(2),
      "single-query attention probability shape or dtype mismatch");
  TORCH_CHECK(gradient_output.scalar_type() == query.scalar_type() && gradient_output.sizes() == query.sizes(),
      "single-query attention output gradient shape or dtype mismatch");
  c10::DeviceGuard device_guard(query.device());
  const auto options = query.options().memory_format(torch::MemoryFormat::Contiguous);
  torch::Tensor query_gradient = needs_query_gradient ? torch::empty(query.sizes(), options) : torch::Tensor();
  torch::Tensor key_gradient = needs_key_gradient ? torch::empty(key.sizes(), options) : torch::Tensor();
  torch::Tensor value_gradient = needs_value_gradient ? torch::empty(value.sizes(), options) : torch::Tensor();
  const torch::Tensor contiguous_gradient = gradient_output.contiguous();
  const int64_t matrices = query.size(0) * query.size(1);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  hipLaunchKernelGGL(single_query_attention_backward_kernel, dim3(static_cast<unsigned int>(matrices)),
      dim3(kSingleQueryAttentionThreads), 0, stream, query.data_ptr<c10::BFloat16>(), key.data_ptr<c10::BFloat16>(),
      value.data_ptr<c10::BFloat16>(), probabilities.data_ptr<float>(), contiguous_gradient.data_ptr<c10::BFloat16>(),
      query_gradient.defined() ? query_gradient.data_ptr<c10::BFloat16>() : nullptr,
      key_gradient.defined() ? key_gradient.data_ptr<c10::BFloat16>() : nullptr,
      value_gradient.defined() ? value_gradient.data_ptr<c10::BFloat16>() : nullptr, static_cast<int>(key.size(2)),
      static_cast<int>(query.size(3)), scale);
  check_single_query_attention_launch("single_query_attention_backward");
  return {std::move(query_gradient), std::move(key_gradient), std::move(value_gradient)};
}

}  // namespace djl::pytorch::rocm
