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
#ifndef DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
#define DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H

#include <torch/torch.h>

#include <cstdint>

namespace djl::pytorch::fusion {

inline constexpr int32_t kMaximumOutputPackSources = 32;
inline constexpr int32_t kMaximumAffineTerms = 32;
inline constexpr int32_t kMaximumAffineGroups = 32;
inline constexpr int32_t kMaximumAffinePrefixRank = 8;
inline constexpr int32_t kMaximumIndexedAffineSources = 32;
inline constexpr int32_t kMaximumIndexedAffineOutputWidth = 32;

struct OutputPackSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
};

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width);

void LaunchBinaryBranchBlend(const torch::Tensor& baseline_context,
    const torch::Tensor& selected_context,
    const torch::Tensor& selected_logit,
    const torch::Tensor& baseline_presence,
    const torch::Tensor& selected_presence, torch::Tensor& output,
    int64_t row_count, int64_t width);

struct AffineInputPackSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
};

void LaunchAffineInputPack(const AffineInputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width);

enum class AffineActivation : int32_t {
  kNone = 0,
  kSilu = 1,
};

struct IndexedAffineSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
  int64_t index_divisor;
  int64_t row_count;
};

void LaunchIndexedAffineInputPack(const IndexedAffineSource* sources,
    int32_t source_count, const void* indices, torch::ScalarType index_data_type,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t output_width);

void LaunchIndexedAffineFinalize(const void* indices,
    torch::ScalarType index_data_type, const torch::Tensor& hidden,
    const void* hidden_bias, const void* output_weight, const void* output_bias,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t hidden_width, int64_t output_width, AffineActivation activation);

struct AffineSumSource {
  const void* data;
  int64_t rows_per_batch;
  int64_t output_strides[kMaximumAffinePrefixRank];
};

void LaunchAffineFinalize(const AffineSumSource* sources, int32_t source_count,
    const void* bias, torch::Tensor& output, int64_t batch_count,
    const int64_t* output_prefix, int32_t output_prefix_rank,
    int64_t output_width, AffineActivation activation);

void LaunchTransformerCopyAndLayerNorm(const torch::Tensor& input,
    torch::Tensor& state, torch::Tensor& normalized,
    const torch::Tensor& weight, const torch::Tensor& bias,
    int64_t batch_count, int64_t token_count, int64_t hidden_width,
    float epsilon, bool copy_input);

void LaunchTransformerAttention(const torch::Tensor& query_key_value,
    torch::Tensor& context, int64_t batch_count, int64_t token_count,
    int64_t attention_heads, int64_t attention_width, int64_t hidden_width);

void LaunchTransformerBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t active_rows, int64_t width);

void LaunchTransformerResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, int64_t hidden_width,
    float epsilon);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
