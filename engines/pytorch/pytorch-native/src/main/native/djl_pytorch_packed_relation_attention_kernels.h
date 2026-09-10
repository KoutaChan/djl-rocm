/* Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace djl::pytorch::rocm::packed_relation_detail {
constexpr int kThreads = 64;

struct Layout {
  int64_t queries;
  int64_t keys;
  int64_t width;
  int64_t heads;
  int64_t features;
  int64_t rows;
  int64_t entries;
  int64_t segments;
  int64_t words;
};

__device__ int relation_index(const int16_t* codes, int64_t pair, int segment,
                              const Layout& l) {
  const auto word = static_cast<uint16_t>(codes[pair * l.words + segment / 4]);
  const int code = (word >> ((segment % 4) * 4)) & 15;
  const int index = segment * l.entries + code;
  if (code >= l.entries || index >= l.rows) {
    // Keep bounds enforcement in Release (-DNDEBUG) without synchronizing
    // device codes back to the host before every attention operation.
    __builtin_trap();
  }
  return index;
}

__device__ float reduce_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ float reduce_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t>
__global__ void forward_kernel(const scalar_t* q, const scalar_t* kv,
                               const bool* mask, const int16_t* codes,
                               const float* table, scalar_t* output,
                               float* saved_output, float* saved_probabilities,
                               Layout l, float scale) {
  const int64_t head = blockIdx.x % l.heads;
  const int64_t query_row = blockIdx.x / l.heads;
  const int64_t batch = query_row / l.queries;
  const int64_t q_offset = query_row * l.width + head * l.features;
  extern __shared__ float shared[];
  float* probabilities = shared;
  float* scratch = shared + l.keys;
  float local_max = -INFINITY;
  for (int64_t key = threadIdx.x; key < l.keys; key += kThreads) {
    float score = -INFINITY;
    if (mask[batch * l.keys + key]) {
      score = 0.0f;
      const int64_t kv_offset =
          (batch * l.keys + key) * 2 * l.width + head * l.features;
      for (int64_t d = 0; d < l.features; ++d) {
        score += static_cast<float>(q[q_offset + d]) *
                 static_cast<float>(kv[kv_offset + d]);
      }
      score *= scale;
      const int64_t pair = query_row * l.keys + key;
      for (int segment = 0; segment < l.segments; ++segment) {
        score += table[relation_index(codes, pair, segment, l) *
                           (l.heads + l.width) +
                       head];
      }
    }
    probabilities[key] = score;
    local_max = fmaxf(local_max, score);
  }
  const float maximum = reduce_max(local_max, scratch);
  __syncthreads();
  float local_sum = 0.0f;
  for (int64_t key = threadIdx.x; key < l.keys; key += kThreads) {
    const float probability =
        mask[batch * l.keys + key] ? expf(probabilities[key] - maximum) : 0.0f;
    probabilities[key] = probability;
    local_sum += probability;
  }
  const float total = reduce_sum(local_sum, scratch);
  __syncthreads();
  for (int64_t key = threadIdx.x; key < l.keys; key += kThreads) {
    probabilities[key] =
        mask[batch * l.keys + key] ? probabilities[key] / total : 0.0f;
    if (saved_probabilities != nullptr) {
      saved_probabilities[(query_row * l.heads + head) * l.keys + key] =
          probabilities[key];
    }
  }
  __syncthreads();
  for (int64_t d = threadIdx.x; d < l.features; d += kThreads) {
    float value = 0.0f;
    for (int64_t key = 0; key < l.keys; ++key) {
      if (!mask[batch * l.keys + key]) {
        continue;
      }
      const int64_t kv_offset = (batch * l.keys + key) * 2 * l.width + l.width +
                                head * l.features + d;
      float memory_value = static_cast<float>(kv[kv_offset]);
      const int64_t pair = query_row * l.keys + key;
      for (int segment = 0; segment < l.segments; ++segment) {
        memory_value += table[relation_index(codes, pair, segment, l) *
                                  (l.heads + l.width) +
                              l.heads + head * l.features + d];
      }
      value += probabilities[key] * memory_value;
    }
    output[q_offset + d] = static_cast<scalar_t>(value);
    if (saved_output != nullptr) {
      saved_output[q_offset + d] = value;
    }
  }
}

template <typename scalar_t>
__global__ void backward_kernel(const scalar_t* q, const scalar_t* kv,
                                const bool* mask, const int16_t* codes,
                                const float* table, const float* probabilities,
                                const float* output,
                                const scalar_t* grad_output, float* grad_q,
                                float* grad_kv, float* grad_table, Layout l,
                                float scale) {
  const int64_t head = blockIdx.x % l.heads;
  const int64_t query_row = blockIdx.x / l.heads;
  const int64_t batch = query_row / l.queries;
  const int64_t q_offset = query_row * l.width + head * l.features;
  const float* p = probabilities + (query_row * l.heads + head) * l.keys;
  extern __shared__ float shared[];
  float* ds = shared;
  float* scratch = shared + l.keys;
  float partial = 0.0f;
  for (int64_t d = threadIdx.x; d < l.features; d += kThreads) {
    partial +=
        static_cast<float>(grad_output[q_offset + d]) * output[q_offset + d];
  }
  const float delta = reduce_sum(partial, scratch);
  __syncthreads();
  for (int64_t key = threadIdx.x; key < l.keys; key += kThreads) {
    float dp = 0.0f;
    if (mask[batch * l.keys + key]) {
      const int64_t offset =
          (batch * l.keys + key) * 2 * l.width + l.width + head * l.features;
      const int64_t pair = query_row * l.keys + key;
      for (int64_t d = 0; d < l.features; ++d) {
        float value = static_cast<float>(kv[offset + d]);
        for (int segment = 0; segment < l.segments; ++segment) {
          value += table[relation_index(codes, pair, segment, l) *
                             (l.heads + l.width) +
                         l.heads + head * l.features + d];
        }
        dp += static_cast<float>(grad_output[q_offset + d]) * value;
      }
    }
    ds[key] = mask[batch * l.keys + key] ? p[key] * (dp - delta) : 0.0f;
  }
  __syncthreads();
  for (int64_t d = threadIdx.x; d < l.features; d += kThreads) {
    float gradient = 0.0f;
    for (int64_t key = 0; key < l.keys; ++key) {
      if (mask[batch * l.keys + key]) {
        gradient += ds[key] *
                    static_cast<float>(kv[(batch * l.keys + key) * 2 * l.width +
                                          head * l.features + d]) *
                    scale;
      }
    }
    grad_q[q_offset + d] = gradient;
  }
  for (int64_t key = threadIdx.x; key < l.keys; key += kThreads) {
    if (!mask[batch * l.keys + key]) {
      continue;
    }
    const int64_t offset =
        (batch * l.keys + key) * 2 * l.width + head * l.features;
    for (int64_t d = 0; d < l.features; ++d) {
      atomicAdd(grad_kv + offset + d,
                ds[key] * static_cast<float>(q[q_offset + d]) * scale);
      atomicAdd(grad_kv + offset + l.width + d,
                p[key] * static_cast<float>(grad_output[q_offset + d]));
    }
  }
  // Aggregate each relation once per query/head, instead of atomically updating
  // its entire value vector once for every key that uses it.
  for (int64_t row = threadIdx.x; row < l.rows; row += kThreads) {
    const int segment = row / l.entries;
    float probability_mass = 0.0f;
    float bias_gradient = 0.0f;
    for (int64_t key = 0; key < l.keys; ++key) {
      if (mask[batch * l.keys + key] &&
          relation_index(codes, query_row * l.keys + key, segment, l) == row) {
        probability_mass += p[key];
        bias_gradient += ds[key];
      }
    }
    const int64_t offset = row * (l.heads + l.width);
    atomicAdd(grad_table + offset + head, bias_gradient);
    for (int64_t d = 0; d < l.features; ++d) {
      atomicAdd(
          grad_table + offset + l.heads + head * l.features + d,
          probability_mass * static_cast<float>(grad_output[q_offset + d]));
    }
  }
}

}  // namespace djl::pytorch::rocm::packed_relation_detail
