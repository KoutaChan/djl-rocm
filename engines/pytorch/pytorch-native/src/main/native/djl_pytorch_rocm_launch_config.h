#ifndef DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H
#define DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H

namespace djl::pytorch::rocm {

/** Process-wide launch tuning for DJL's native ROCm kernels. */
struct RocmKernelLaunchConfig {
  int relation_forward_waves_per_block;
  int relation_backward_waves_per_block;
  int relation_forward_queries_per_wave;
  int relation_backward_queries_per_wave;
  int grouped_attention_threads_per_block;
  int grouped_attention_backward_threads_per_block;
  int shared_gradient_threads_per_block;
  int shared_gradient_feature_tile;
  int residual_norm_threads_per_block;
  int optimizer_threads_per_block;
  int masked_categorical_threads_per_block;
  int row_operation_threads_per_block;
};

/** Returns the immutable launch configuration loaded from the process environment. */
const RocmKernelLaunchConfig& rocm_kernel_launch_config();

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_LAUNCH_CONFIG_H
