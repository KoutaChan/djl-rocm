#ifndef DJL_PYTORCH_ROCM_KERNELS_H
#define DJL_PYTORCH_ROCM_KERNELS_H

#include <torch/torch.h>

namespace djl::pytorch::rocm {

bool supports_indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids);

torch::Tensor indexed_relation_bias_forward(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, float scale);

torch::Tensor indexed_relation_bias_logit_gradient(const torch::Tensor& gradient_output,
    const torch::Tensor& relation_ids, at::IntArrayRef relation_logits_shape, float scale);

bool supports_grouped_indexed_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group);

struct GroupedIndexedAttentionForwardResult {
  torch::Tensor output;
  torch::Tensor log_sum_exp;
};

GroupedIndexedAttentionForwardResult grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, float scale, bool capture_log_sum_exp);

struct GroupedIndexedAttentionGradients {
  torch::Tensor query;
  torch::Tensor shared_key_values;
  torch::Tensor shared_deltas;
  torch::Tensor indexed_deltas;
};

GroupedIndexedAttentionGradients grouped_indexed_attention_backward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, const torch::Tensor& log_sum_exp,
    const torch::Tensor& gradient_output, int64_t queries_per_group, float scale, bool needs_query_gradient,
    bool needs_shared_gradient, bool needs_shared_delta_gradient, bool needs_indexed_delta_gradient);

bool supports_owned_residual_layer_norm(const torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias);

torch::Tensor add_to_owned_residual_and_layer_norm(torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_KERNELS_H
