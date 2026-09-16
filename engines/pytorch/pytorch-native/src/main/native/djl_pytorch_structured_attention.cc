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

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <optional>
#include <vector>

#include "djl_pytorch_kernel_backend.h"

namespace djl::pytorch {
namespace detail {

bool is_mapped_grouped_indexed_attention_layout_supported(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids) {
  if (!query.is_cuda() || query.dim() != 3 || shared_key_values.dim() != 3 ||
      shared_group_indices.dim() != 1 || shared_delta_table.dim() != 2 ||
      shared_delta_indices.dim() != 2 || indexed_deltas.dim() != 3 ||
      indexed_shared_ids.dim() != 2 || !query.is_contiguous() ||
      !shared_key_values.is_contiguous() || !shared_group_indices.is_contiguous() ||
      !shared_delta_table.is_contiguous() || !shared_delta_indices.is_contiguous() ||
      !indexed_deltas.is_contiguous() || !indexed_shared_ids.is_contiguous() ||
      (query.scalar_type() != torch::kFloat32 && query.scalar_type() != torch::kFloat16 &&
          query.scalar_type() != torch::kBFloat16) || query.scalar_type() != shared_key_values.scalar_type() ||
      query.scalar_type() != shared_delta_table.scalar_type() ||
      query.scalar_type() != indexed_deltas.scalar_type() ||
      shared_group_indices.scalar_type() != torch::kInt32 ||
      shared_delta_indices.scalar_type() != torch::kInt32 ||
      indexed_shared_ids.scalar_type() != torch::kInt32) {
    return false;
  }
  const int64_t query_count = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t group_count = shared_key_values.size(0);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t key_width = heads * key_features;
  return query_count > 0 && heads > 0 && key_features > 0 && group_count > 0 &&
         shared_tokens > 0 && shared_delta_table.size(0) > 0 && packed_width > key_width &&
         (packed_width - key_width) % heads == 0 &&
         shared_group_indices.size(0) == query_count &&
         shared_delta_table.size(1) == packed_width &&
         shared_delta_indices.size(0) == query_count &&
         shared_delta_indices.size(1) == shared_tokens &&
         indexed_deltas.size(0) == query_count && indexed_deltas.size(2) == packed_width &&
         indexed_shared_ids.size(0) == query_count &&
         indexed_shared_ids.size(1) == indexed_tokens &&
         shared_key_values.device() == query.device() &&
         shared_group_indices.device() == query.device() &&
         shared_delta_table.device() == query.device() &&
         shared_delta_indices.device() == query.device() && indexed_deltas.device() == query.device() &&
         indexed_shared_ids.device() == query.device();
}

}  // namespace detail
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

torch::Tensor grouped_packed_attention_reference(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    int64_t heads, double scale) {
  const int64_t batch = query.size(0);
  const int64_t query_tokens = query.size(1);
  const int64_t query_width = query.size(2);
  const int64_t groups = packed_key_value.size(1);
  const int64_t key_tokens = packed_key_value.size(2);
  const int64_t packed_width = packed_key_value.size(3);
  const int64_t key_features = query_width / heads;
  const int64_t value_width = packed_width - query_width;
  const int64_t value_features = value_width / heads;
  auto queries = query.reshape({batch, query_tokens, heads, key_features})
                     .transpose(1, 2)
                     .unsqueeze(1)
                     .expand({batch, groups, heads, query_tokens, key_features});
  auto keys = packed_key_value.slice(3, 0, query_width)
                  .reshape({batch, groups, key_tokens, heads, key_features})
                  .transpose(2, 3);
  auto values = packed_key_value.slice(3, query_width, packed_width)
                    .reshape({batch, groups, key_tokens, heads, value_features})
                    .transpose(2, 3);
  auto token_valid = mask.ne(0);
  auto key_value_valid = token_valid.reshape({batch, groups, 1, key_tokens, 1});
  keys = torch::where(key_value_valid, keys, torch::zeros_like(keys));
  values = torch::where(key_value_valid, values, torch::zeros_like(values));
  auto valid = token_valid.reshape({batch, groups, 1, 1, key_tokens});
  auto scores = queries.matmul(keys.transpose(3, 4)).mul(scale);
  auto safe_scores = torch::where(valid, scores, torch::zeros_like(scores));
  auto attention_valid = valid.logical_or(valid.any(4, true).logical_not());
  auto probabilities = torch::where(valid,
      torch::where(attention_valid, safe_scores,
          torch::full_like(scores, -std::numeric_limits<double>::infinity())).softmax(4),
      torch::zeros_like(scores));
  return probabilities.matmul(values)
      .transpose(2, 3)
      .contiguous()
      .view({batch, groups, query_tokens, value_width});
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
  auto indexed_present = stored_ids.ne(0).unsqueeze(1).expand({query_count, heads, indexed_tokens});
  auto combined_indexed_keys = indexed_keys.add(indexed_delta_keys);
  auto participating_indexed_keys = torch::where(
      indexed_present.unsqueeze(3), combined_indexed_keys, torch::zeros_like(combined_indexed_keys));
  auto keys = torch::cat({shared_keys.add(shared_delta_keys), participating_indexed_keys}, 2);
  auto scores = query.unsqueeze(2).mul(keys).sum(3).mul(scale);
  auto indexed_scores = scores.slice(2, shared_tokens)
                            .masked_fill(indexed_present.logical_not(),
                                -std::numeric_limits<float>::infinity());
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
  auto combined_indexed_values = indexed_values.add(indexed_delta_values);
  auto participating_indexed_values = torch::where(
      indexed_present.unsqueeze(3), combined_indexed_values, torch::zeros_like(combined_indexed_values));
  auto values = torch::cat({shared_values.add(shared_delta_values), participating_indexed_values}, 2);
  return weights.unsqueeze(3).mul(values).sum(2);
}

torch::Tensor index_select_with_float32_gradients(
    const torch::Tensor& source, const torch::Tensor& indices) {
  if (requires_autograd({&source}) &&
      (source.scalar_type() == torch::kFloat16 || source.scalar_type() == torch::kBFloat16)) {
    // Repeated mapped rows accumulate their gradients in FP32. Cast selected
    // rows back before attention to preserve the original forward rounding.
    return source.to(torch::kFloat32).index_select(0, indices).to(source.scalar_type());
  }
  return source.index_select(0, indices);
}

torch::Tensor mapped_grouped_indexed_attention_reference(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, double scale) {
  const auto query_count = query.size(0);
  const auto shared_tokens = shared_key_values.size(1);
  const auto packed_width = shared_key_values.size(2);
  auto mapped_shared_key_values = index_select_with_float32_gradients(
      shared_key_values, shared_group_indices.to(torch::kLong));
  auto mapped_shared_deltas = index_select_with_float32_gradients(
      shared_delta_table, shared_delta_indices.to(torch::kLong).reshape(-1))
      .reshape({query_count, shared_tokens, packed_width});
  return grouped_indexed_attention_reference(query, mapped_shared_key_values, mapped_shared_deltas,
      indexed_deltas, indexed_shared_ids, 1, scale);
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

class GroupedPackedAttentionFunction
    : public torch::autograd::Function<GroupedPackedAttentionFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& query, const torch::Tensor& packed_key_value,
      const torch::Tensor& mask, int64_t heads, double scale) {
    auto boolean_mask = mask.scalar_type() == torch::kBool
        ? mask.contiguous()
        : mask.ne(0).contiguous();
    auto result = rocm::grouped_packed_attention_forward(query, packed_key_value,
        boolean_mask, heads, static_cast<float>(scale), true);
    context->save_for_backward(
        {query, packed_key_value, boolean_mask, result.probabilities});
    context->saved_data["heads"] = heads;
    context->saved_data["scale"] = scale;
    return result.output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto heads = context->saved_data["heads"].toInt();
    const auto scale = context->saved_data["scale"].toDouble();
    auto gradients = rocm::grouped_packed_attention_backward(
        saved.at(0), saved.at(1), saved.at(2), saved.at(3),
        gradient_outputs.at(0), heads, static_cast<float>(scale),
        context->needs_input_grad(0), context->needs_input_grad(1));
    return {gradients.query, gradients.packed_key_value, torch::Tensor(),
        torch::Tensor(), torch::Tensor()};
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
        {query, shared_key_values, shared_deltas, indexed_deltas, indexed_shared_ids, result.probabilities});
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

#if defined(DJL_USE_ACCELERATOR_KERNELS)

class MappedGroupedIndexedAttentionFunction
    : public torch::autograd::Function<MappedGroupedIndexedAttentionFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& query, const torch::Tensor& shared_key_values,
      const torch::Tensor& shared_group_indices, const torch::Tensor& shared_delta_table,
      const torch::Tensor& shared_delta_indices, const torch::Tensor& indexed_deltas,
      const torch::Tensor& indexed_shared_ids, double scale) {
    auto result = kernel_backend::mapped_grouped_indexed_attention_forward(query, shared_key_values,
        shared_group_indices, shared_delta_table, shared_delta_indices, indexed_deltas,
        indexed_shared_ids, static_cast<float>(scale), true);
    context->save_for_backward({query, shared_key_values, shared_group_indices,
        shared_delta_table, shared_delta_indices, indexed_deltas, indexed_shared_ids,
        result.probabilities});
    context->saved_data["scale"] = scale;
#if defined(DJL_USE_CUDA_KERNELS)
    context->saved_data["autocast_enabled"] = at::autocast::is_autocast_enabled(at::DeviceType::CUDA);
#endif
    return result.output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto scale = context->saved_data["scale"].toDouble();
#if defined(DJL_USE_CUDA_KERNELS)
    if (at::GradMode::is_enabled()) {
      // Recompute a differentiable reference graph only for higher-order derivatives.
      c10::impl::ForceDispatchKeyGuard restore_dispatch;
      at::autocast::set_autocast_enabled(
          at::DeviceType::CUDA, context->saved_data["autocast_enabled"].toBool());
      auto output = mapped_grouped_indexed_attention_reference(saved.at(0), saved.at(1),
          saved.at(2), saved.at(3), saved.at(4), saved.at(5), saved.at(6), scale);
      torch::autograd::variable_list inputs;
      std::vector<size_t> input_indices;
      for (size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{5}}) {
        if (context->needs_input_grad(index)) {
          inputs.push_back(saved.at(index));
          input_indices.push_back(index);
        }
      }
      auto gradients = torch::autograd::grad(
          {output}, inputs, {gradient_outputs.at(0)}, std::nullopt, true, true);
      torch::autograd::variable_list result(8);
      for (size_t index = 0; index < gradients.size(); ++index) {
        result[input_indices[index]] = gradients[index];
      }
      return result;
    }
#endif
    auto gradients = kernel_backend::mapped_grouped_indexed_attention_backward(saved.at(0),
        saved.at(1), saved.at(2), saved.at(3), saved.at(4), saved.at(5), saved.at(6),
        saved.at(7), gradient_outputs.at(0), static_cast<float>(scale),
        context->needs_input_grad(0), context->needs_input_grad(1),
        context->needs_input_grad(3), context->needs_input_grad(5));
    return {gradients.query, gradients.shared_key_values, torch::Tensor(),
        gradients.shared_delta_table, torch::Tensor(), gradients.indexed_deltas,
        torch::Tensor(), torch::Tensor()};
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

torch::Tensor grouped_packed_attention(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    int64_t heads, double scale) {
#if defined(DJL_USE_ROCM_KERNELS)
  const bool needs_autograd = requires_autograd({&query, &packed_key_value});
  if (rocm::supports_grouped_packed_attention_forward(
          query, packed_key_value, mask, heads) &&
      (!needs_autograd || rocm::supports_grouped_packed_attention_backward(
          query, packed_key_value, mask, heads))) {
    if (needs_autograd) {
      return GroupedPackedAttentionFunction::apply(
          query, packed_key_value, mask, heads, scale);
    }
    return rocm::grouped_packed_attention_forward(query, packed_key_value,
        mask, heads, static_cast<float>(scale), false)
        .output;
  }
#endif
  return grouped_packed_attention_reference(
      query, packed_key_value, mask, heads, scale);
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

torch::Tensor mapped_grouped_indexed_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, double scale) {
#if defined(DJL_USE_ACCELERATOR_KERNELS)
  const bool needs_autograd = requires_autograd(
      {&query, &shared_key_values, &shared_delta_table, &indexed_deltas});
  const bool needs_atomic_table_gradient = needs_autograd &&
      (shared_key_values.requires_grad() || shared_delta_table.requires_grad());
  const bool deterministic_fallback =
      needs_atomic_table_gradient && at::globalContext().deterministicAlgorithms();
  if (!deterministic_fallback &&
#if defined(DJL_USE_ROCM_KERNELS)
      rocm::supports_mapped_grouped_indexed_attention_forward(query, shared_key_values,
          shared_group_indices, shared_delta_table, shared_delta_indices, indexed_deltas,
          indexed_shared_ids) &&
      (!needs_autograd || rocm::supports_mapped_grouped_indexed_attention_backward(query,
          shared_key_values, shared_group_indices, shared_delta_table,
          shared_delta_indices, indexed_deltas, indexed_shared_ids))) {
#else
      cuda::supports_mapped_grouped_indexed_attention(query, shared_key_values,
          shared_group_indices, shared_delta_table, shared_delta_indices,
          indexed_deltas, indexed_shared_ids, needs_autograd)) {
#endif
    if (needs_autograd) {
      return MappedGroupedIndexedAttentionFunction::apply(query, shared_key_values,
          shared_group_indices, shared_delta_table, shared_delta_indices, indexed_deltas,
          indexed_shared_ids, scale);
    }
    return kernel_backend::mapped_grouped_indexed_attention_forward(query, shared_key_values,
        shared_group_indices, shared_delta_table, shared_delta_indices, indexed_deltas,
        indexed_shared_ids, static_cast<float>(scale), false)
        .output;
  }
#endif
  return mapped_grouped_indexed_attention_reference(query, shared_key_values, shared_group_indices,
      shared_delta_table, shared_delta_indices, indexed_deltas, indexed_shared_ids, scale);
}

}  // namespace djl::pytorch
