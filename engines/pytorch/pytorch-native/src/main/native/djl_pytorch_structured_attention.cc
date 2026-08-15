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

#include "djl_pytorch_structured_attention.h"

#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <initializer_list>

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_rocm_kernels.h"
#endif

namespace djl::pytorch {
namespace {

bool requires_autograd(std::initializer_list<const torch::Tensor*> tensors) {
  if (!at::GradMode::is_enabled()) {
    return false;
  }
  return std::any_of(tensors.begin(), tensors.end(), [](const torch::Tensor* tensor) {
    return tensor != nullptr && tensor->requires_grad();
  });
}

torch::Tensor indexed_relation_bias_reference(const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias, const torch::Tensor& relation_ids, double scale) {
  const auto batch = relation_logits.size(0);
  const auto heads = relation_logits.size(1);
  const auto query_tokens = relation_logits.size(2);
  const auto key_tokens = relation_ids.size(-1);
  auto stored_ids = relation_ids.to(torch::kLong);
  auto relation_indices =
      relation_ids.dim() == 2
          ? stored_ids.reshape({1, 1, query_tokens, key_tokens}).expand({batch, heads, query_tokens, key_tokens})
          : stored_ids.reshape({batch, 1, query_tokens, key_tokens}).expand({batch, heads, query_tokens, key_tokens});
  return relation_logits.gather(3, relation_indices).mul(scale).add(relation_bias);
}

torch::Tensor grouped_indexed_attention_reference(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, int64_t queries_per_group,
    double scale) {
  const auto query_count = query.size(0);
  const auto heads = query.size(1);
  const auto key_size = query.size(2);
  const auto key_width = heads * key_size;
  const auto packed_width = shared_key_values.size(2);
  const auto value_size = (packed_width - key_width) / heads;
  const auto shared_tokens = shared_key_values.size(1);
  const auto indexed_tokens = indexed_deltas.size(1);

  auto group_indices = torch::arange(
      shared_key_values.size(0), torch::TensorOptions().device(query.device()).dtype(torch::kLong))
                           .repeat_interleave(queries_per_group);
  auto query_shared_key_values = shared_key_values.index_select(0, group_indices);
  auto stored_ids = indexed_shared_ids.to(torch::kLong);
  auto gather_indices = stored_ids.sub(1).clamp_min(0).unsqueeze(2).expand({query_count, indexed_tokens, packed_width});
  auto indexed_shared_key_values = query_shared_key_values.gather(1, gather_indices);

  auto shared_keys = query_shared_key_values.slice(2, 0, key_width)
                         .reshape({query_count, shared_tokens, heads, key_size})
                         .permute({0, 2, 1, 3});
  auto shared_delta_keys = shared_deltas.slice(2, 0, key_width)
                               .reshape({query_count, shared_tokens, heads, key_size})
                               .permute({0, 2, 1, 3});
  auto indexed_keys = indexed_shared_key_values.slice(2, 0, key_width)
                          .reshape({query_count, indexed_tokens, heads, key_size})
                          .permute({0, 2, 1, 3});
  auto indexed_delta_keys = indexed_deltas.slice(2, 0, key_width)
                                .reshape({query_count, indexed_tokens, heads, key_size})
                                .permute({0, 2, 1, 3});
  auto keys = torch::cat({shared_keys.add(shared_delta_keys), indexed_keys.add(indexed_delta_keys)}, 2);
  auto scores = query.unsqueeze(2).mul(keys).sum(3).mul(scale);
  auto indexed_present = stored_ids.ne(0).unsqueeze(1).expand({query_count, heads, indexed_tokens});
  auto indexed_scores = scores.slice(2, shared_tokens).masked_fill(indexed_present.logical_not(), -1.0e9);
  scores = torch::cat({scores.slice(2, 0, shared_tokens), indexed_scores}, 2);
  auto weights = scores.softmax(2);

  auto shared_values = query_shared_key_values.slice(2, key_width)
                           .reshape({query_count, shared_tokens, heads, value_size})
                           .permute({0, 2, 1, 3});
  auto shared_delta_values = shared_deltas.slice(2, key_width)
                                 .reshape({query_count, shared_tokens, heads, value_size})
                                 .permute({0, 2, 1, 3});
  auto indexed_values = indexed_shared_key_values.slice(2, key_width)
                            .reshape({query_count, indexed_tokens, heads, value_size})
                            .permute({0, 2, 1, 3});
  auto indexed_delta_values = indexed_deltas.slice(2, key_width)
                                  .reshape({query_count, indexed_tokens, heads, value_size})
                                  .permute({0, 2, 1, 3});
  auto values = torch::cat({shared_values.add(shared_delta_values), indexed_values.add(indexed_delta_values)}, 2);
  return weights.unsqueeze(3).mul(values).sum(2);
}

#if defined(DJL_USE_ROCM_KERNELS)

class IndexedRelationBiasFunction : public torch::autograd::Function<IndexedRelationBiasFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& relation_logits,
      const torch::Tensor& relation_bias, const torch::Tensor& relation_ids, double scale) {
    context->save_for_backward({relation_ids});
    context->saved_data["relation_logits_shape"] = relation_logits.sizes().vec();
    context->saved_data["relation_bias_shape"] = relation_bias.sizes().vec();
    context->saved_data["scale"] = scale;
    return rocm::indexed_relation_bias_forward(
        relation_logits, relation_bias, relation_ids, static_cast<float>(scale));
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    const auto relation_ids = context->get_saved_variables().at(0);
    const auto relation_logits_shape = context->saved_data["relation_logits_shape"].toIntVector();
    const auto relation_bias_shape = context->saved_data["relation_bias_shape"].toIntVector();
    const auto scale = context->saved_data["scale"].toDouble();
    const auto& gradient_output = gradient_outputs.at(0);
    auto relation_logits_gradient =
        context->needs_input_grad(0)
            ? rocm::indexed_relation_bias_logit_gradient(
                  gradient_output, relation_ids, relation_logits_shape, static_cast<float>(scale))
            : torch::Tensor();
    auto relation_bias_gradient =
        context->needs_input_grad(1) ? gradient_output.sum_to_size(relation_bias_shape) : torch::Tensor();
    return {relation_logits_gradient, relation_bias_gradient, torch::Tensor(), torch::Tensor()};
  }
};

class GroupedIndexedAttentionFunction : public torch::autograd::Function<GroupedIndexedAttentionFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& query,
      const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas,
      const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, int64_t queries_per_group,
      double scale) {
    auto result = rocm::grouped_indexed_attention_forward(query, shared_key_values, shared_deltas, indexed_deltas,
        indexed_shared_ids, queries_per_group, static_cast<float>(scale), true);
    context->save_for_backward(
        {query, shared_key_values, shared_deltas, indexed_deltas, indexed_shared_ids, result.log_sum_exp});
    context->saved_data["queries_per_group"] = queries_per_group;
    context->saved_data["scale"] = scale;
    return result.output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto queries_per_group = context->saved_data["queries_per_group"].toInt();
    const auto scale = context->saved_data["scale"].toDouble();
    auto gradients = rocm::grouped_indexed_attention_backward(saved.at(0), saved.at(1), saved.at(2), saved.at(3),
        saved.at(4), saved.at(5), gradient_outputs.at(0), queries_per_group, static_cast<float>(scale),
        context->needs_input_grad(0), context->needs_input_grad(1), context->needs_input_grad(2),
        context->needs_input_grad(3));
    return {gradients.query, gradients.shared_key_values, gradients.shared_deltas, gradients.indexed_deltas,
        torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

#endif

}  // namespace

torch::Tensor indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, double scale) {
#if defined(DJL_USE_ROCM_KERNELS)
  const bool needs_autograd = requires_autograd({&relation_logits, &relation_bias});
  const bool needs_logit_gradient = needs_autograd && relation_logits.requires_grad();
  if (rocm::supports_indexed_relation_bias_forward(relation_logits, relation_bias, relation_ids) &&
      (!needs_logit_gradient ||
          rocm::supports_indexed_relation_bias_logit_gradient(relation_logits, relation_bias, relation_ids))) {
    if (needs_autograd) {
      return IndexedRelationBiasFunction::apply(relation_logits, relation_bias, relation_ids, scale);
    }
    return rocm::indexed_relation_bias_forward(
        relation_logits, relation_bias, relation_ids, static_cast<float>(scale));
  }
#endif
  return indexed_relation_bias_reference(relation_logits, relation_bias, relation_ids, scale);
}

torch::Tensor grouped_indexed_attention(const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, double scale) {
#if defined(DJL_USE_ROCM_KERNELS)
  const bool needs_autograd = requires_autograd({&query, &shared_key_values, &shared_deltas, &indexed_deltas});
  if (rocm::supports_grouped_indexed_attention_forward(
          query, shared_key_values, shared_deltas, indexed_deltas, indexed_shared_ids, queries_per_group) &&
      (!needs_autograd || rocm::supports_grouped_indexed_attention_backward(query, shared_key_values,
          shared_deltas, indexed_deltas, indexed_shared_ids, queries_per_group,
          shared_key_values.requires_grad()))) {
    if (needs_autograd) {
      return GroupedIndexedAttentionFunction::apply(query, shared_key_values, shared_deltas, indexed_deltas,
          indexed_shared_ids, queries_per_group, scale);
    }
    return rocm::grouped_indexed_attention_forward(query, shared_key_values, shared_deltas, indexed_deltas,
        indexed_shared_ids, queries_per_group, static_cast<float>(scale), false)
        .output;
  }
#endif
  return grouped_indexed_attention_reference(
      query, shared_key_values, shared_deltas, indexed_deltas, indexed_shared_ids, queries_per_group, scale);
}

}  // namespace djl::pytorch
