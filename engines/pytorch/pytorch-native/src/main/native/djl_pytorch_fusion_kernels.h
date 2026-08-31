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
inline constexpr int32_t kMaximumIndexedLocalTransformerSegments = 8;

struct OutputPackSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
};

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width);

struct SegmentedOutputPackSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t source_prefix_count;
  int64_t source_token_count;
  int64_t token_offset;
  int64_t token_count;
  int64_t hidden_width;
  int64_t destination_offset;
};

void LaunchSegmentedOutputPack(const SegmentedOutputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width);

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

void LaunchIndexedRelationAttentionBias(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_ids,
    const torch::Tensor& pair_bias, torch::Tensor& attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale);

void LaunchTransformerBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t active_rows, int64_t width,
    bool round_bias_before_activation = false);

void LaunchTransformerResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, int64_t hidden_width,
    float epsilon, bool round_update_before_residual = false);

void LaunchSingleQueryReadoutSeedInput(const torch::Tensor& memory,
    const torch::Tensor& query_source, const torch::Tensor& valid_mask,
    torch::Tensor& seed_input, int64_t batch_count, int64_t token_count,
    int64_t hidden_width, int64_t query_token_count, int64_t query_index);

void LaunchSingleQueryReadoutAttention(const torch::Tensor& seed_products,
    const torch::Tensor& seed_bias, const torch::Tensor& memory,
    const torch::Tensor& valid_mask, const torch::Tensor& query_weight,
    const torch::Tensor& query_bias, const torch::Tensor& key_value_weight,
    torch::Tensor& state, torch::Tensor& context, int64_t batch_count,
    int64_t readout_count, int64_t token_count, int64_t hidden_width,
    int64_t attention_heads, int64_t attention_width);

void LaunchSingleQueryReadoutResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor* state_bias,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output,
    int64_t batch_count, int64_t readout_count, int64_t hidden_width,
    int64_t output_batch_stride, float epsilon, bool batch_major_output);

void LaunchSingleQueryReadoutLayerNorm(const torch::Tensor& input,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    int64_t hidden_width, float epsilon);

void LaunchSingleQueryReadoutBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t batch_count, int64_t readout_count,
    int64_t width);

void LaunchIndexedLocalTransformerClear(torch::Tensor& output, int64_t dense_rows, int64_t hidden_width);

void LaunchIndexedLocalTransformerGatherNormalize(const torch::Tensor& input, const torch::Tensor& indices,
    torch::ScalarType index_data_type, const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t hidden_width,
    float epsilon);

struct IndexedLocalTransformerInputSegment {
  const void* data;
  int64_t token_offset;
  int64_t token_count;
};

void LaunchIndexedLocalTransformerGatherNormalizeSegments(
    const IndexedLocalTransformerInputSegment* input_segments, int32_t input_segment_count,
    torch::ScalarType input_data_type, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t group_count,
    int64_t token_count, int64_t hidden_width, float epsilon);

void LaunchIndexedLocalTransformerAttention(torch::Tensor& query_key_value, const torch::Tensor& indices,
    torch::ScalarType index_data_type, int64_t active_rows, int64_t dense_rows, int64_t token_count,
    int64_t attention_heads, int64_t attention_width);

void LaunchIndexedLocalTransformerResidualLayerNorm(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, torch::Tensor& normalized, int64_t active_offset,
    int64_t active_rows, int64_t dense_rows, int64_t hidden_width, float epsilon);

void LaunchIndexedLocalTransformerBiasSilu(
    torch::Tensor& values, const torch::Tensor& bias, int64_t active_rows, int64_t width);

void LaunchIndexedLocalTransformerFinalize(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& indices, torch::ScalarType index_data_type, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, int64_t active_offset, int64_t active_rows,
    int64_t dense_rows, int64_t hidden_width, float epsilon);

void LaunchMappedGroupedMaskedSoftmaxPool(const torch::Tensor& scores,
    const torch::Tensor& masks, const torch::Tensor& values,
    const torch::Tensor& destination_metadata, torch::Tensor& contexts,
    torch::Tensor& presence, int64_t batch_count, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
