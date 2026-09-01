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
#include "djl_pytorch_attention.h"

#include <torch/csrc/autograd/custom_function.h>

#include <cmath>

#include "djl_pytorch_rocm_attention.h"

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_single_query_attention.h"
#endif

namespace djl::pytorch {
namespace {

bool requires_autograd(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value) {
  return at::GradMode::is_enabled() && (query.requires_grad() || key.requires_grad() || value.requires_grad());
}

#if defined(DJL_USE_ROCM_KERNELS)

torch::Tensor single_query_attention_forward(torch::autograd::AutogradContext* context, const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value, const std::optional<torch::Tensor>& mask, double scale) {
  auto result = rocm::single_query_attention_forward(query, key, value, mask, static_cast<float>(scale), true);
  context->save_for_backward({query, key, value, result.probabilities});
  context->saved_data["scale"] = scale;
  return result.output;
}

rocm::SingleQueryAttentionGradients single_query_attention_backward(
    torch::autograd::AutogradContext* context, const torch::Tensor& gradient_output) {
  const auto saved = context->get_saved_variables();
  const double scale = context->saved_data["scale"].toDouble();
  return rocm::single_query_attention_backward(saved.at(0), saved.at(1), saved.at(2), saved.at(3), gradient_output,
      static_cast<float>(scale), context->needs_input_grad(0), context->needs_input_grad(1),
      context->needs_input_grad(2));
}

class SingleQueryAttentionFunction : public torch::autograd::Function<SingleQueryAttentionFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& query,
      const torch::Tensor& key, const torch::Tensor& value, const torch::Tensor& mask, double scale) {
    return single_query_attention_forward(context, query, key, value, mask, scale);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    auto gradients = single_query_attention_backward(context, gradient_outputs.at(0));
    return {gradients.query, gradients.key, gradients.value, torch::Tensor(), torch::Tensor()};
  }
};

class SingleQueryAttentionWithoutMaskFunction
    : public torch::autograd::Function<SingleQueryAttentionWithoutMaskFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& query,
      const torch::Tensor& key, const torch::Tensor& value, double scale) {
    return single_query_attention_forward(context, query, key, value, std::nullopt, scale);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    auto gradients = single_query_attention_backward(context, gradient_outputs.at(0));
    return {gradients.query, gradients.key, gradients.value, torch::Tensor()};
  }
};

#endif

}  // namespace

torch::Tensor scaled_dot_product_attention(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const std::optional<torch::Tensor>& mask, double dropout, bool causal,
    const std::optional<double>& scale) {
#if defined(DJL_USE_ROCM_KERNELS)
  prepare_rocm_attention_backend();
  if (query.dim() == 4 && query.size(2) == 1 && dropout == 0.0 && !causal &&
      (!mask.has_value() || !mask->requires_grad()) && rocm::supports_single_query_attention(query, key, value, mask)) {
    const bool needs_autograd = requires_autograd(query, key, value);
    if (!needs_autograd || rocm::supports_single_query_attention_backward(query, key)) {
      const double attention_scale = scale.value_or(1.0 / std::sqrt(static_cast<double>(query.size(3))));
      if (needs_autograd) {
        if (mask.has_value()) {
          return SingleQueryAttentionFunction::apply(query, key, value, *mask, attention_scale);
        }
        return SingleQueryAttentionWithoutMaskFunction::apply(query, key, value, attention_scale);
      }
      return rocm::single_query_attention_forward(query, key, value, mask, static_cast<float>(attention_scale), false)
          .output;
    }
  }
  if (at::GradMode::is_enabled() && mask.has_value() && mask->requires_grad() && !query.requires_grad() &&
      !key.requires_grad() && !value.requires_grad()) {
    return std::get<0>(
        at::_scaled_dot_product_attention_math(query, key, value, mask, dropout, causal, std::nullopt, scale, false));
  }
#endif
  return at::scaled_dot_product_attention(query, key, value, mask, dropout, causal, scale);
}

}  // namespace djl::pytorch
