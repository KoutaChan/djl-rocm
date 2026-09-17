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
#include "djl_pytorch_short_attention.h"

#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace djl::pytorch::rocm {
namespace {

constexpr int kGroupWidth = 32;
constexpr int kThreads = 256;

struct TensorStrides {
  int64_t batch;
  int64_t head;
  int64_t token;
  int64_t feature;
};

TensorStrides strides(const torch::Tensor& tensor) {
  return {tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)};
}

TensorStrides mask_strides(const torch::Tensor& mask) {
  int64_t values[4]{};
  if (mask.defined()) {
    for (int64_t axis = 0; axis < mask.dim(); ++axis) {
      values[4 - mask.dim() + axis] = mask.size(axis) == 1 ? 0 : mask.stride(axis);
    }
  }
  return {values[0], values[1], values[2], values[3]};
}

__device__ int64_t row_offset(TensorStrides layout, int64_t batch, int64_t head, int64_t token) {
  return batch * layout.batch + head * layout.head + token * layout.token;
}

__device__ float maximum_propagating_nan(float left, float right) {
  return isnan(left) || isnan(right) ? NAN : fmaxf(left, right);
}

__device__ float group_maximum(float value) {
  for (int offset = kGroupWidth / 2; offset > 0; offset /= 2) {
    value = maximum_propagating_nan(value, __shfl_down(value, offset, kGroupWidth));
  }
  return __shfl(value, 0, kGroupWidth);
}

__device__ float group_sum(float value) {
  for (int offset = kGroupWidth / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset, kGroupWidth);
  }
  return __shfl(value, 0, kGroupWidth);
}

template <typename mask_t>
__device__ float mask_value(const mask_t* mask, int64_t index) {
  if (mask == nullptr) {
    return 0.0f;
  }
  if constexpr (std::is_same_v<mask_t, bool>) {
    return mask[index] ? 0.0f : -INFINITY;
  } else {
    return static_cast<float>(mask[index]);
  }
}

template <typename scalar_t, int dimension, bool transpose = false>
__device__ void load_features(const scalar_t* source, float* destination, int64_t tokens,
    TensorStrides layout, int64_t batch, int64_t head) {
  for (int64_t index = threadIdx.x; index < tokens * dimension; index += kThreads) {
    const int64_t token = index / dimension;
    const int feature = index % dimension;
    const int64_t target = transpose ? feature * tokens + token : index;
    destination[target] = static_cast<float>(source[
        row_offset(layout, batch, head, token) + feature * layout.feature]);
  }
}

// A block reuses one batch/head's K/V across every query. Each complete 32-lane
// group owns a query; feature shuffles remain unconditional for padding lanes.
template <typename scalar_t, typename mask_t, int dimension, int key_groups>
__global__ void short_attention_forward_kernel(const scalar_t* query, const scalar_t* key,
    const scalar_t* value, const mask_t* mask, scalar_t* output, float* probabilities,
    int64_t batch_heads, int64_t heads, int64_t queries, int64_t keys, TensorStrides query_layout,
    TensorStrides key_layout, TensorStrides value_layout, TensorStrides mask_layout, float scale) {
  extern __shared__ float memory[];
  float* shared_key = memory;
  float* shared_value = shared_key + keys * dimension;
  const int lane = threadIdx.x % kGroupWidth;
  const int group = threadIdx.x / kGroupWidth;
  for (int64_t batch_head = blockIdx.x; batch_head < batch_heads; batch_head += gridDim.x) {
    const int64_t head = batch_head % heads;
    const int64_t batch = batch_head / heads;
    load_features<scalar_t, dimension, true>(key, shared_key, keys, key_layout, batch, head);
    load_features<scalar_t, dimension>(value, shared_value, keys, value_layout, batch, head);
    __syncthreads();
    for (int64_t query_token = group; query_token < queries; query_token += kThreads / kGroupWidth) {
      const int64_t row = batch_head * queries + query_token;
      const int64_t query_offset = row_offset(query_layout, batch, head, query_token);
      const int64_t mask_offset = row_offset(mask_layout, batch, head, query_token);
      const float query_feature = lane < dimension
          ? static_cast<float>(query[query_offset + lane * query_layout.feature]) : 0.0f;
      float weights[key_groups];
      float maximum = -INFINITY;
      #pragma unroll
      for (int part = 0; part < key_groups; ++part) {
        const int key_token = part * kGroupWidth + lane;
        float score = 0.0f;
        #pragma unroll
        for (int feature = 0; feature < dimension; ++feature) {
          const float q = __shfl(query_feature, feature, kGroupWidth);
          const float k = key_token < keys ? shared_key[feature * keys + key_token] : 0.0f;
          score = fmaf(q, k, score);
        }
        const float bias = key_token < keys
            ? mask_value(mask, mask_offset + key_token * mask_layout.feature) : -INFINITY;
        weights[part] = bias == -INFINITY ? -INFINITY : score * scale + bias;
        maximum = maximum_propagating_nan(maximum, weights[part]);
      }
      maximum = group_maximum(maximum);
      float denominator = 0.0f;
      #pragma unroll
      for (int part = 0; part < key_groups; ++part) {
        weights[part] = maximum == -INFINITY ? 0.0f : expf(weights[part] - maximum);
        denominator += weights[part];
      }
      denominator = group_sum(denominator);
      #pragma unroll
      for (int part = 0; part < key_groups; ++part) {
        weights[part] = maximum == -INFINITY ? 0.0f : weights[part] / denominator;
        const int key_token = part * kGroupWidth + lane;
        if (key_token < keys) probabilities[row * keys + key_token] = weights[part];
      }
      float result = 0.0f;
      for (int key_token = 0; key_token < keys; ++key_token) {
        const float probability = __shfl(
            weights[key_token / kGroupWidth], key_token % kGroupWidth, kGroupWidth);
        if (lane < dimension && probability != 0.0f) {
          result = fmaf(probability, shared_value[key_token * dimension + lane], result);
        }
      }
      if (lane < dimension) {
        output[((batch * queries + query_token) * heads + head) * dimension + lane] =
            static_cast<scalar_t>(result);
      }
    }
    if (batch_head + gridDim.x < batch_heads) __syncthreads();
  }
}

// Keep P/dS and feature matrices local to the batch/head. Query and key phases
// retain their FP32 reduction order and each gradient element has one writer.
template <typename scalar_t, int dimension, int key_groups>
__global__ void short_attention_backward_kernel(const scalar_t* query, const scalar_t* key, const scalar_t* value,
    const scalar_t* gradient_output, const float* probabilities, scalar_t* gradient_query,
    scalar_t* gradient_key, scalar_t* gradient_value, float* gradient_scores, int64_t batch_heads,
    int64_t heads, int64_t queries, int64_t keys, TensorStrides query_layout, TensorStrides key_layout,
    TensorStrides value_layout, TensorStrides output_layout, float scale) {
  extern __shared__ float memory[];
  float* shared_query = memory;
  float* shared_key = shared_query + queries * dimension;
  float* shared_value = shared_key + keys * dimension;
  float* shared_output_gradient = shared_value + keys * dimension;
  float* shared_probability = shared_output_gradient + queries * dimension;
  float* shared_score_gradient = shared_probability + queries * keys;
  const bool query_phase = gradient_query != nullptr || gradient_key != nullptr || gradient_scores != nullptr;
  const int lane = threadIdx.x % kGroupWidth;
  const int group = threadIdx.x / kGroupWidth;
  for (int64_t batch_head = blockIdx.x; batch_head < batch_heads; batch_head += gridDim.x) {
    const int64_t head = batch_head % heads;
    const int64_t batch = batch_head / heads;
    if (gradient_key != nullptr) {
      load_features<scalar_t, dimension>(query, shared_query, queries, query_layout, batch, head);
    }
    if (gradient_query != nullptr) {
      load_features<scalar_t, dimension>(key, shared_key, keys, key_layout, batch, head);
    }
    if (query_phase) {
      load_features<scalar_t, dimension, true>(value, shared_value, keys, value_layout, batch, head);
    }
    load_features<scalar_t, dimension>(gradient_output, shared_output_gradient, queries, output_layout, batch, head);
    for (int64_t index = threadIdx.x; index < queries * keys; index += kThreads) {
      shared_probability[index] = probabilities[batch_head * queries * keys + index];
    }
    __syncthreads();
    if (query_phase) {
      for (int64_t token = group; token < queries; token += kThreads / kGroupWidth) {
        const int64_t row = batch_head * queries + token;
        const float gradient_feature = lane < dimension ? shared_output_gradient[token * dimension + lane] : 0.0f;
        float probability[key_groups];
        float gradient[key_groups];
        float dot = 0.0f;
        #pragma unroll
        for (int part = 0; part < key_groups; ++part) {
          const int key_token = part * kGroupWidth + lane;
          probability[part] = key_token < keys ? shared_probability[token * keys + key_token] : 0.0f;
          gradient[part] = 0.0f;
          #pragma unroll
          for (int feature = 0; feature < dimension; ++feature) {
            const float g = __shfl(gradient_feature, feature, kGroupWidth);
            const float v = probability[part] != 0.0f ? shared_value[feature * keys + key_token] : 0.0f;
            gradient[part] = fmaf(g, v, gradient[part]);
          }
          if (probability[part] != 0.0f) dot = fmaf(probability[part], gradient[part], dot);
        }
        dot = group_sum(dot);
        #pragma unroll
        for (int part = 0; part < key_groups; ++part) {
          gradient[part] = probability[part] == 0.0f ? 0.0f : probability[part] * (gradient[part] - dot);
          const int key_token = part * kGroupWidth + lane;
          if (key_token < keys) {
            shared_score_gradient[token * keys + key_token] = gradient[part];
            if (gradient_scores != nullptr) gradient_scores[row * keys + key_token] = gradient[part];
          }
        }
        if (gradient_query != nullptr) {
          float result = 0.0f;
          for (int key_token = 0; key_token < keys; ++key_token) {
            const float score_gradient = __shfl(
                gradient[key_token / kGroupWidth], key_token % kGroupWidth, kGroupWidth);
            if (lane < dimension && score_gradient != 0.0f) {
              result = fmaf(score_gradient, shared_key[key_token * dimension + lane], result);
            }
          }
          if (lane < dimension) gradient_query[row * dimension + lane] = static_cast<scalar_t>(result * scale);
        }
      }
    }
    __syncthreads();
    if (lane < dimension && (gradient_key != nullptr || gradient_value != nullptr)) {
      for (int64_t key_token = group; key_token < keys; key_token += kThreads / kGroupWidth) {
        float key_gradient = 0.0f;
        float value_gradient = 0.0f;
        for (int64_t token = 0; token < queries; ++token) {
          const int64_t score_offset = token * keys + key_token;
          if (gradient_key != nullptr) {
            const float gradient = shared_score_gradient[score_offset];
            if (gradient != 0.0f) {
              key_gradient = fmaf(gradient, shared_query[token * dimension + lane], key_gradient);
            }
          }
          if (gradient_value != nullptr) {
            const float probability = shared_probability[score_offset];
            if (probability != 0.0f) {
              value_gradient = fmaf(probability, shared_output_gradient[token * dimension + lane], value_gradient);
            }
          }
        }
        const int64_t index = (batch_head * keys + key_token) * dimension + lane;
        if (gradient_key != nullptr) gradient_key[index] = static_cast<scalar_t>(key_gradient * scale);
        if (gradient_value != nullptr) gradient_value[index] = static_cast<scalar_t>(value_gradient);
      }
    }
    if (batch_head + gridDim.x < batch_heads) __syncthreads();
  }
}

int blocks(int64_t batch_heads) {
  return static_cast<int>(std::min<int64_t>(batch_heads, 65535));
}

void check_launch() {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "short attention launch failed: ", hipGetErrorString(error));
}

template <typename scalar_t, int dimension, int key_groups>
ShortAttentionForwardResult forward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& mask, float scale) {
  // Preserve the token-major backing layout used by fused SDPA, so the usual
  // transpose-and-merge-heads operation is a view rather than a tensor copy.
  auto output = torch::empty_strided(query.sizes(),
      {query.size(2) * query.size(1) * dimension, dimension, query.size(1) * dimension, 1}, query.options());
  auto probabilities = torch::empty({query.size(0), query.size(1), query.size(2), key.size(2)},
      query.options().dtype(torch::kFloat32));
  const int64_t batch_heads = query.size(0) * query.size(1);
  const size_t shared_bytes = 2 * key.size(2) * dimension * sizeof(float);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  auto launch = [&](auto mask_pointer) {
    using mask_t = std::remove_cv_t<std::remove_pointer_t<decltype(mask_pointer)>>;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(short_attention_forward_kernel<scalar_t, mask_t, dimension, key_groups>),
        dim3(blocks(batch_heads)), dim3(kThreads), shared_bytes, stream, query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(), mask_pointer, output.data_ptr<scalar_t>(), probabilities.data_ptr<float>(),
        batch_heads, query.size(1), query.size(2), key.size(2), strides(query), strides(key), strides(value),
        mask_strides(mask), scale);
  };
  if (!mask.defined()) launch(static_cast<const float*>(nullptr));
  else if (mask.scalar_type() == torch::kBool) launch(mask.data_ptr<bool>());
  else if (mask.scalar_type() == torch::kFloat32) launch(mask.data_ptr<float>());
  else launch(mask.data_ptr<c10::BFloat16>());
  check_launch();
  return {output, probabilities};
}

template <typename scalar_t, int dimension, int key_groups>
ShortAttentionGradients backward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& mask, const torch::Tensor& probabilities,
    const torch::Tensor& gradient_output, float scale, bool needs_query_gradient,
    bool needs_key_gradient, bool needs_value_gradient, bool needs_mask_gradient) {
  ShortAttentionGradients gradients;
  if (needs_query_gradient) gradients.query = torch::empty(query.sizes(), query.options());
  if (needs_key_gradient) gradients.key = torch::empty(key.sizes(), key.options());
  if (needs_value_gradient) gradients.value = torch::empty(value.sizes(), value.options());
  auto score_gradient = needs_mask_gradient ? torch::empty_like(probabilities) : torch::Tensor();
  const int64_t batch_heads = query.size(0) * query.size(1);
  const int64_t queries = query.size(2);
  const int64_t keys = key.size(2);
  const size_t shared_bytes = (2 * (queries + keys) * dimension + 2 * queries * keys) * sizeof(float);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(short_attention_backward_kernel<scalar_t, dimension, key_groups>),
      dim3(blocks(batch_heads)), dim3(kThreads), shared_bytes, stream, query.data_ptr<scalar_t>(),
      key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), gradient_output.data_ptr<scalar_t>(),
      probabilities.data_ptr<float>(), needs_query_gradient ? gradients.query.data_ptr<scalar_t>() : nullptr,
      needs_key_gradient ? gradients.key.data_ptr<scalar_t>() : nullptr,
      needs_value_gradient ? gradients.value.data_ptr<scalar_t>() : nullptr,
      score_gradient.defined() ? score_gradient.data_ptr<float>() : nullptr,
      batch_heads, query.size(1), queries, keys, strides(query), strides(key), strides(value), strides(gradient_output), scale);
  check_launch();
  if (needs_mask_gradient) gradients.mask = score_gradient.sum_to_size(mask.sizes()).to(mask.scalar_type());
  return gradients;
}

template <typename Callback>
auto dispatch(const torch::Tensor& query, const torch::Tensor& key, Callback&& callback) {
  if (query.scalar_type() == torch::kFloat32) {
    if (query.size(3) == 16) {
      if (key.size(2) <= 32) return callback(float{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
      return callback(float{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
    }
    if (key.size(2) <= 32) return callback(float{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
    return callback(float{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
  }
  if (query.size(3) == 16) {
    if (key.size(2) <= 32) return callback(c10::BFloat16{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
    return callback(c10::BFloat16{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
  }
  if (key.size(2) <= 32) return callback(c10::BFloat16{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
  return callback(c10::BFloat16{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
}

}  // namespace

bool supports_short_attention(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& mask) {
  if (!query.is_cuda() || query.dim() != 4 || key.dim() != 4 || value.dim() != 4 ||
      (query.scalar_type() != torch::kFloat32 && query.scalar_type() != torch::kBFloat16) ||
      key.scalar_type() != query.scalar_type() || value.scalar_type() != query.scalar_type() ||
      key.device() != query.device() || value.device() != query.device() ||
      query.size(0) <= 0 || query.size(1) <= 0 || query.size(2) <= 1 || query.size(2) > 64 ||
      key.size(2) <= 0 || key.size(2) > 64 || (query.size(3) != 16 && query.size(3) != 32) ||
      key.size(0) != query.size(0) || key.size(1) != query.size(1) || key.size(3) != query.size(3) ||
      value.sizes() != key.sizes()) return false;
  if (!mask.defined()) return true;
  if (mask.device() != query.device() || mask.dim() > 4 ||
      (mask.scalar_type() != torch::kBool && mask.scalar_type() != torch::kFloat32 &&
          mask.scalar_type() != torch::kBFloat16)) return false;
  const int64_t shape[] = {query.size(0), query.size(1), query.size(2), key.size(2)};
  for (int64_t axis = 0; axis < mask.dim(); ++axis) {
    if (mask.size(axis) != 1 && mask.size(axis) != shape[4 - mask.dim() + axis]) return false;
  }
  return true;
}

ShortAttentionForwardResult short_attention_forward(const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value, const torch::Tensor& mask, float scale) {
  c10::DeviceGuard guard(query.device());
  return dispatch(query, key, [&](auto scalar, auto dimension, auto key_groups) {
    return forward<decltype(scalar), decltype(dimension)::value, decltype(key_groups)::value>(query, key, value, mask, scale);
  });
}

ShortAttentionGradients short_attention_backward(const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value, const torch::Tensor& mask,
    const torch::Tensor& probabilities, const torch::Tensor& gradient_output, float scale,
    bool needs_query_gradient, bool needs_key_gradient, bool needs_value_gradient, bool needs_mask_gradient) {
  c10::DeviceGuard guard(query.device());
  return dispatch(query, key, [&](auto scalar, auto dimension, auto key_groups) {
    return backward<decltype(scalar), decltype(dimension)::value, decltype(key_groups)::value>(query, key, value,
        mask, probabilities, gradient_output, scale, needs_query_gradient, needs_key_gradient,
        needs_value_gradient, needs_mask_gradient);
  });
}

}  // namespace djl::pytorch::rocm
