#ifndef DJL_PYTORCH_ROCM_KERNELS_H
#define DJL_PYTORCH_ROCM_KERNELS_H

#include <torch/torch.h>

namespace djl::pytorch::rocm {

torch::Tensor tile_relation_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& relation_key,
    const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids,
    float scale);

torch::Tensor tile_relation_mask(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids,
    float scale);

torch::Tensor transition_tile_attention(
    const torch::Tensor& query,
    const torch::Tensor& tile_key_value,
    const torch::Tensor& relation_key_value,
    const torch::Tensor& wait_key_value,
    const torch::Tensor& wait_tile_ids,
    int64_t candidates_per_state,
    float scale);

torch::Tensor residual_layer_norm_in_place(
    torch::Tensor& residual,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float epsilon);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_KERNELS_H
