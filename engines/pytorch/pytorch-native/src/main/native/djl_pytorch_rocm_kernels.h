#ifndef DJL_PYTORCH_ROCM_KERNELS_H
#define DJL_PYTORCH_ROCM_KERNELS_H

#include <torch/torch.h>

namespace djl::pytorch::rocm {

bool can_use_indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids);

torch::Tensor indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, float scale);

bool can_use_grouped_indexed_scaled_dot_product_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group);

torch::Tensor grouped_indexed_scaled_dot_product_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, float scale);

bool can_use_owned_residual_layer_norm(const torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias);

torch::Tensor add_to_owned_residual_and_layer_norm(torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_KERNELS_H
