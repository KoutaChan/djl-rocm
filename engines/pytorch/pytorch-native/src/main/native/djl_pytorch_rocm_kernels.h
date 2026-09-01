#ifndef DJL_PYTORCH_ROCM_KERNELS_H
#define DJL_PYTORCH_ROCM_KERNELS_H

#include <torch/torch.h>

#include <vector>

namespace djl::pytorch::rocm {

bool supports_fused_adam_update(const torch::Tensor& weight, const torch::Tensor& gradient,
    const torch::Tensor& mean, const torch::Tensor& variance);

void fused_adam_update(torch::Tensor& weight, const torch::Tensor& gradient, torch::Tensor& mean,
    torch::Tensor& variance, float learning_rate, float learning_rate_bias_correction, float weight_decay,
    float rescale_gradient, float clip_gradient, float beta1, float beta2, float epsilon, bool adamw);

bool supports_masked_categorical(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

bool supports_embedding_with_offsets(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table);

torch::Tensor embedding_with_offsets_forward(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table);

bool supports_embedding_feature_pack(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features);

torch::Tensor embedding_feature_pack_forward(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features);

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

bool supports_indexed_masked_softmax_pool(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices);

torch::Tensor indexed_masked_softmax_pool_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices);

torch::Tensor masked_log_sum_exp_forward(const torch::Tensor& logits, const torch::Tensor& mask);

torch::Tensor masked_log_sum_exp_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& normalizers);

torch::Tensor scatter_rows_forward(const torch::Tensor& rows,
    const torch::Tensor& row_indices, const torch::Tensor& output);

torch::Tensor scatter_rows_backward(
    const torch::Tensor& gradient_output, const torch::Tensor& row_indices);

bool supports_categorical_masks(
    const torch::Tensor& categories, const torch::Tensor& mask, size_t rule_count);

torch::Tensor categorical_masks(const torch::Tensor& categories, const torch::Tensor& mask,
    const std::vector<int64_t>& field_indices, const std::vector<uint64_t>& category_sets);

bool supports_binary_choice_masks(const torch::Tensor& routes, const torch::Tensor& first_mask,
    const torch::Tensor& second_mask);

torch::Tensor binary_choice_masks(const torch::Tensor& routes, const torch::Tensor& first_mask,
    const torch::Tensor& second_mask, int64_t representative_field, int64_t first_route_field,
    int64_t second_route_field, int64_t padding_value);

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

bool supports_mapped_grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids);

torch::Tensor mapped_grouped_indexed_attention_forward(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, float scale);

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

bool supports_masked_embedding_residual_to_owned_tokens(const torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask);

torch::Tensor add_masked_embedding_residual_to_owned_tokens(torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask, int64_t padding_index, bool mean_valid);

bool supports_broadcast_residual_to_owned_silu(const torch::Tensor& values,
    const torch::Tensor& residual, const torch::Tensor* mask);

void add_broadcast_residual_to_owned_and_silu(
    torch::Tensor& values, const torch::Tensor& residual, const torch::Tensor* mask);

bool supports_bias_and_broadcast_residual_to_owned_silu(
    const torch::Tensor& values, const torch::Tensor& bias,
    const torch::Tensor& residual, const torch::Tensor* mask);

void add_bias_and_broadcast_residual_to_owned_and_silu(
    torch::Tensor& values, const torch::Tensor& bias,
    const torch::Tensor& residual, const torch::Tensor* mask);

bool supports_autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, at::IntArrayRef normalized_shape);

torch::Tensor autocast_layer_norm(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, float epsilon);

struct AutocastLayerNormAndCastResult {
  torch::Tensor normalized;
  torch::Tensor converted;
};

bool supports_autocast_layer_norm_and_cast(const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, at::IntArrayRef normalized_shape, torch::ScalarType converted_type);

AutocastLayerNormAndCastResult autocast_layer_norm_and_cast(const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon,
    torch::ScalarType converted_type);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_ROCM_KERNELS_H
