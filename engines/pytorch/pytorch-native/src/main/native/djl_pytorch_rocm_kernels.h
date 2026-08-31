#ifndef DJL_PYTORCH_ROCM_KERNELS_H
#define DJL_PYTORCH_ROCM_KERNELS_H

#include <torch/torch.h>

namespace djl::pytorch::rocm {

bool supports_fused_adam_update(const torch::Tensor& weight, const torch::Tensor& gradient,
    const torch::Tensor& mean, const torch::Tensor& variance);

void fused_adam_update(torch::Tensor& weight, const torch::Tensor& gradient, torch::Tensor& mean,
    torch::Tensor& variance, float learning_rate, float learning_rate_bias_correction, float weight_decay,
    float rescale_gradient, float clip_gradient, float beta1, float beta2, float epsilon, bool adamw);

bool supports_masked_categorical(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

bool supports_scatter_rows(const torch::Tensor& rows, const torch::Tensor& row_indices);

bool supports_segmented_lookup_sum(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices);

bool supports_padded_batch_gather(
    const torch::Tensor& source, const torch::Tensor& stored_indices);

bool supports_padded_batch_gather_2d(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices);

bool supports_padded_batch_gather_by_batch_indices(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices);

torch::Tensor masked_softmax_forward(const torch::Tensor& logits, const torch::Tensor& mask);

torch::Tensor masked_softmax_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& probabilities, const torch::Tensor& mask,
    torch::ScalarType input_type);

bool supports_grouped_masked_softmax_pool(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values);

torch::Tensor grouped_masked_softmax_pool_forward(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values);

torch::Tensor masked_log_sum_exp_forward(const torch::Tensor& logits, const torch::Tensor& mask);

torch::Tensor masked_log_sum_exp_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& normalizers);

torch::Tensor scatter_rows_forward(const torch::Tensor& rows,
    const torch::Tensor& row_indices, const torch::Tensor& output);

torch::Tensor scatter_rows_backward(
    const torch::Tensor& gradient_output, const torch::Tensor& row_indices);

torch::Tensor segmented_lookup_sum_forward(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices);

torch::Tensor padded_batch_gather_forward(
    const torch::Tensor& source, const torch::Tensor& stored_indices);

torch::Tensor padded_batch_gather_2d_forward(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices);

torch::Tensor padded_batch_gather_by_batch_indices_forward(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices);

bool supports_indexed_relation_bias_forward(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids);

bool supports_indexed_relation_bias_logit_gradient(const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias, const torch::Tensor& relation_ids);

torch::Tensor indexed_relation_bias_forward(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, float scale);

torch::Tensor indexed_relation_bias_logit_gradient(const torch::Tensor& gradient_output,
    const torch::Tensor& relation_ids, at::IntArrayRef relation_logits_shape, float scale);

bool supports_grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group);

bool supports_grouped_indexed_attention_backward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group,
    bool needs_shared_key_value_gradient);

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
    bool needs_shared_key_value_gradient, bool needs_shared_delta_gradient,
    bool needs_indexed_delta_gradient);

bool supports_owned_residual_layer_norm(const torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias);

torch::Tensor add_to_owned_residual_and_layer_norm(torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon);

bool supports_autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, at::IntArrayRef normalized_shape);

torch::Tensor autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, float epsilon);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_KERNELS_H
