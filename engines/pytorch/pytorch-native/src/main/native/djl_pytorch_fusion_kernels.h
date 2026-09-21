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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace djl::pytorch::fusion {

void LaunchWeightedCompactReduce(const torch::Tensor& rows, const torch::Tensor& weights,
    const torch::Tensor& indices, torch::Tensor& output, int64_t capacity);

inline constexpr int32_t kMaximumOutputPackSources = 32;
inline constexpr int32_t kMaximumAffineTerms = 32;
inline constexpr int32_t kMaximumAffineGroups = 32;
inline constexpr int32_t kMaximumAffinePrefixRank = 8;
inline constexpr int32_t kMaximumIndexedAffineSources = 32;
inline constexpr int32_t kMaximumIndexedAffineOutputWidth = 32;
inline constexpr int32_t kMaximumIndexedLocalTransformerSegments = 8;
// Portable per-block budgets avoid requiring backend-specific shared-memory opt-in.
inline constexpr int64_t kFusionAttentionSharedMemoryBytes = 48 * 1024;
inline constexpr int64_t kSingleQueryReadoutSharedMemoryReserve = 32;

inline bool UsesIndexedLocalAttentionInPlace(int64_t token_count, int64_t attention_heads, int64_t attention_width) {
  return token_count > 0 && token_count <= 32 && attention_heads == 4 && attention_width == 64;
}

class FusionMatmulPlan;
class FusionMatmulWorkspace;

/** Returns whether this build can execute Fusion matrix products directly. */
bool IsFusionMatmulSupported();

/** Returns whether this build can execute the hipBLASLt linear bias-SiLU path. */
bool IsLinearBiasSiluSupported();

/** Returns the configured per-stream upper bound for hipBLASLt scratch storage. */
std::size_t GetFusionMatmulWorkspaceUpperBoundBytes();

/**
 * Pins the current ROCm stream's matrix-product context for a graph lifetime.
 *
 * <p>Accelerator graphs call this immediately before capture. The call reserves
 * the configured hipBLASLt workspace upper bound so captured nodes retain one
 * stable workspace address until the matching unpin.
 */
void PinCurrentRocmMatmulStreamContext(c10::Device device);

/** Releases one graph-lifetime pin on the current ROCm stream context. */
void UnpinCurrentRocmMatmulStreamContext(c10::Device device);

/**
 * Creates a reusable accelerator matrix-product plan.
 *
 * <p>The ROCm backend resolves shape- and stride-specific hipBLASLt descriptors
 * and algorithms through the process-lifetime raw-stream context. Other
 * backends may use the ordinary PyTorch path. This object preserves the Fusion
 * plan API without owning tensors or accelerator scratch storage.
 */
std::shared_ptr<FusionMatmulPlan> CreateFusionMatmulPlan();

/** Creates a lightweight binding to a process-lifetime device-stream context. */
std::shared_ptr<FusionMatmulWorkspace> CreateFusionMatmulWorkspace();

/**
 * Tries an inference matrix product on the current ROCm stream.
 *
 * <p>On success, creates {@code output = left * right + bias}. Unsupported
 * data types, layouts, devices, modes, and shapes return {@code false} without
 * modifying {@code output}, so callers can preserve the ordinary ATen path.
 * Matrix operands may carry broadcast-compatible leading batch dimensions.
 */
bool TryExecuteRocmInferenceMatmul(torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor* bias);

/**
 * Tries {@code linear(input, weight, bias)} for rank-two or higher input on
 * the current ROCm stream. Rank-one input retains PyTorch's GEMV path.
 * Autocast converts the weight before forming its transpose so that the
 * transpose remains a metadata-only view.
 */
bool TryExecuteRocmInferenceLinear(torch::Tensor& output,
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor* bias);

/**
 * Executes a two-dimensional or strided-batched matrix product.
 *
 * <p>The ROCm implementation invokes hipBLASLt directly and therefore does not enter PyTorch's
 * process-global, thread-handle-keyed workspace cache. It accepts row-major tensors, including
 * interleaved read-only batches, and a right operand represented by a transpose view whose final
 * two strides are {@code [1, K]}. A supported backend reports invalid layouts and unavailable
 * algorithms as errors; {@code false} is reserved for builds without the direct backend.
 */
bool ExecuteFusionMatmul(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right);

/** Executes {@code output = left * right + bias} using a plain bias epilogue. */
bool ExecuteFusionMatmulBias(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor& bias);

/** Executes {@code output += left * right} using hipBLASLt beta-one accumulation. */
bool ExecuteFusionMatmulAccumulate(
    const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right);

/**
 * Executes {@code output = silu(input * weight + bias)} with a backend epilogue when supported.
 *
 * <p>All tensors are two-dimensional row-major matrices except the one-dimensional bias. The
 * weight is laid out as {@code [inputWidth, outputWidth]}. The method returns {@code false} for a
 * data type or shape that must use the ordinary PyTorch path.
 */
bool ExecuteLinearBiasSilu(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace,
    torch::Tensor& output, const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias);

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

void LaunchProjectedResidualMlpPrepare(torch::Tensor& combined,
    torch::Tensor& activated, torch::Tensor& residual,
    int64_t row_count, int64_t output_width, int64_t hidden_width);

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
    int64_t attention_heads, int64_t attention_width, torch::Tensor* context);

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
    torch::Tensor& presence, int64_t batch_count,
    int64_t output_batch_capacity, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
