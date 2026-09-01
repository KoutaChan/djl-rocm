#ifndef DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H
#define DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H

namespace djl::pytorch::rocm {

namespace launch_environment {

inline constexpr char kIndexedRelationBiasForwardWavesPerBlock[] =
    "DJL_ROCM_INDEXED_RELATION_BIAS_FORWARD_WAVES_PER_BLOCK";
inline constexpr char kIndexedRelationBiasBackwardWavesPerBlock[] =
    "DJL_ROCM_INDEXED_RELATION_BIAS_BACKWARD_WAVES_PER_BLOCK";
inline constexpr char kIndexedRelationBiasForwardQueriesPerWave[] =
    "DJL_ROCM_INDEXED_RELATION_BIAS_FORWARD_QUERIES_PER_WAVE";
inline constexpr char kIndexedRelationBiasBackwardQueriesPerWave[] =
    "DJL_ROCM_INDEXED_RELATION_BIAS_BACKWARD_QUERIES_PER_WAVE";
inline constexpr char kGroupedIndexedAttentionForwardMaxThreadsPerBlock[] =
    "DJL_ROCM_GROUPED_INDEXED_ATTENTION_FORWARD_MAX_THREADS_PER_BLOCK";
inline constexpr char kGroupedIndexedAttentionBackwardMaxThreadsPerBlock[] =
    "DJL_ROCM_GROUPED_INDEXED_ATTENTION_BACKWARD_MAX_THREADS_PER_BLOCK";
inline constexpr char kGroupedIndexedAttentionSharedKeyValueGradientThreadsPerBlock[] =
    "DJL_ROCM_GROUPED_INDEXED_ATTENTION_SHARED_KEY_VALUE_GRADIENT_THREADS_PER_BLOCK";
inline constexpr char kGroupedIndexedAttentionSharedKeyValueGradientFeaturesPerBlock[] =
    "DJL_ROCM_GROUPED_INDEXED_ATTENTION_SHARED_KEY_VALUE_GRADIENT_FEATURES_PER_BLOCK";
inline constexpr char kMappedGroupedIndexedAttentionPackedGradientThreadsPerBlock[] =
    "DJL_ROCM_MAPPED_GROUPED_INDEXED_ATTENTION_PACKED_GRADIENT_THREADS_PER_BLOCK";
inline constexpr char kResidualAddLayerNormThreadsPerBlock[] =
    "DJL_ROCM_RESIDUAL_ADD_LAYER_NORM_THREADS_PER_BLOCK";
inline constexpr char kFusedAdamUpdateThreadsPerBlock[] =
    "DJL_ROCM_FUSED_ADAM_UPDATE_THREADS_PER_BLOCK";
inline constexpr char kMaskedCategoricalThreadsPerBlock[] =
    "DJL_ROCM_MASKED_CATEGORICAL_THREADS_PER_BLOCK";
inline constexpr char kScatterRowsThreadsPerBlock[] =
    "DJL_ROCM_SCATTER_ROWS_THREADS_PER_BLOCK";
inline constexpr char kPaddedBatchGatherThreadsPerBlock[] =
    "DJL_ROCM_PADDED_BATCH_GATHER_THREADS_PER_BLOCK";

}  // namespace launch_environment

/** Process-wide launch tuning for DJL's generic native ROCm operators. */
struct RocmKernelLaunchConfig {
  int indexed_relation_bias_forward_waves_per_block;
  int indexed_relation_bias_backward_waves_per_block;
  int indexed_relation_bias_forward_queries_per_wave;
  int indexed_relation_bias_backward_queries_per_wave;
  int grouped_indexed_attention_forward_max_threads_per_block;
  int grouped_indexed_attention_backward_max_threads_per_block;
  int grouped_indexed_attention_shared_key_value_gradient_threads_per_block;
  int grouped_indexed_attention_shared_key_value_gradient_features_per_block;
  int mapped_grouped_indexed_attention_packed_gradient_threads_per_block;
  int residual_add_layer_norm_threads_per_block;
  int fused_adam_update_threads_per_block;
  int masked_categorical_threads_per_block;
  int scatter_rows_threads_per_block;
  int padded_batch_gather_threads_per_block;
};

/** Returns the immutable launch configuration loaded from the process environment. */
const RocmKernelLaunchConfig& rocm_kernel_launch_config();

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H
