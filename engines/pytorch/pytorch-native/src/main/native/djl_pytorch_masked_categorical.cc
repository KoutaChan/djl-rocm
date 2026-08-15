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

#include "djl_pytorch_masked_categorical.h"

#include <torch/csrc/autograd/custom_function.h>

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_rocm_kernels.h"
#endif

namespace djl::pytorch {
namespace {

int64_t normalize_axis(int64_t axis, int64_t dimensions) {
  TORCH_CHECK(dimensions > 0, "masked categorical logits must have at least one dimension");
  const int64_t normalized = axis < 0 ? axis + dimensions : axis;
  TORCH_CHECK(normalized >= 0 && normalized < dimensions, "masked categorical axis is out of range");
  return normalized;
}

torch::Tensor expanded_boolean_mask(const torch::Tensor& logits, const torch::Tensor& mask) {
  return mask.to(torch::kBool).expand_as(logits);
}

struct MaskedReduction {
  torch::Tensor exponentials;
  torch::Tensor denominator;
  torch::Tensor maximum;
  torch::Tensor has_choice;
};

MaskedReduction masked_reduction(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  auto values = logits.to(torch::kFloat32);
  auto boolean_mask = expanded_boolean_mask(values, mask);
  auto masked_values = torch::where(boolean_mask, values, torch::full_like(values, -1.0e30));
  auto maximum = std::get<0>(masked_values.max(axis, true));
  auto shifted =
      torch::where(boolean_mask, values.sub(maximum), torch::full_like(values, -1.0e30));
  auto exponentials = shifted.exp();
  auto denominator = exponentials.sum(axis, true);
  auto has_choice = boolean_mask.any(axis, true);
  return {exponentials, denominator, maximum, has_choice};
}

torch::Tensor masked_softmax_reference(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  auto reduction = masked_reduction(logits, mask, axis);
  auto probabilities = reduction.exponentials.div(reduction.denominator.clamp_min(1.0e-30));
  return torch::where(mask, probabilities, torch::zeros_like(probabilities));
}

torch::Tensor masked_log_sum_exp_reference(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  auto reduction = masked_reduction(logits, mask, axis);
  auto normalizers = reduction.denominator.clamp_min(1.0e-30).log().add(reduction.maximum);
  return torch::where(reduction.has_choice, normalizers, torch::zeros_like(normalizers));
}

#if defined(DJL_USE_ROCM_KERNELS)

class MaskedSoftmaxFunction : public torch::autograd::Function<MaskedSoftmaxFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& logits, const torch::Tensor& mask) {
    auto probabilities = rocm::masked_softmax_forward(logits, mask);
    context->save_for_backward({probabilities, mask});
    context->saved_data["input_type"] = static_cast<int64_t>(logits.scalar_type());
    return probabilities;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto input_type =
        static_cast<torch::ScalarType>(context->saved_data["input_type"].toInt());
    auto gradient = rocm::masked_softmax_backward(
        gradient_outputs.at(0), saved.at(0), saved.at(1), input_type);
    return {gradient, torch::Tensor()};
  }
};

class MaskedLogSumExpFunction : public torch::autograd::Function<MaskedLogSumExpFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& logits, const torch::Tensor& mask) {
    auto normalizers = rocm::masked_log_sum_exp_forward(logits, mask);
    context->save_for_backward({logits, mask, normalizers});
    return normalizers;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    auto gradient = rocm::masked_log_sum_exp_backward(
        gradient_outputs.at(0), saved.at(0), saved.at(1), saved.at(2));
    return {gradient, torch::Tensor()};
  }
};

#endif

}  // namespace

torch::Tensor masked_softmax(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  const int64_t normalized_axis = normalize_axis(axis, logits.dim());
  auto boolean_mask = expanded_boolean_mask(logits, mask).contiguous();
#if defined(DJL_USE_ROCM_KERNELS)
  if (rocm::supports_masked_categorical(logits, boolean_mask, normalized_axis)) {
    if (at::GradMode::is_enabled() && logits.requires_grad()) {
      return MaskedSoftmaxFunction::apply(logits, boolean_mask);
    }
    return rocm::masked_softmax_forward(logits, boolean_mask);
  }
#endif
  return masked_softmax_reference(logits, boolean_mask, normalized_axis);
}

torch::Tensor masked_log_sum_exp(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis) {
  const int64_t normalized_axis = normalize_axis(axis, logits.dim());
  auto boolean_mask = expanded_boolean_mask(logits, mask).contiguous();
#if defined(DJL_USE_ROCM_KERNELS)
  if (rocm::supports_masked_categorical(logits, boolean_mask, normalized_axis)) {
    if (at::GradMode::is_enabled() && logits.requires_grad()) {
      return MaskedLogSumExpFunction::apply(logits, boolean_mask);
    }
    return rocm::masked_log_sum_exp_forward(logits, boolean_mask);
  }
#endif
  return masked_log_sum_exp_reference(logits, boolean_mask, normalized_axis);
}

}  // namespace djl::pytorch
