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

#include <vector>

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

void validate_grouped_pool_shapes(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values) {
  TORCH_CHECK(logits.dim() > 0, "grouped masked softmax logits must have at least one dimension");
  TORCH_CHECK(mask.dim() == logits.dim() + 1,
      "grouped masked softmax mask must add a trailing group dimension");
  TORCH_CHECK(values.dim() == logits.dim() + 1,
      "grouped masked softmax values must add a trailing feature dimension");
  for (int64_t dimension = 0; dimension < logits.dim() - 1; ++dimension) {
    TORCH_CHECK(mask.size(dimension) == logits.size(dimension) &&
            values.size(dimension) == logits.size(dimension),
        "grouped masked softmax leading dimensions must match");
  }
  const int64_t choices = logits.size(-1);
  TORCH_CHECK(choices > 0 && mask.size(-2) == choices && values.size(-2) == choices,
      "grouped masked softmax choice dimensions must match and be non-empty");
  TORCH_CHECK(mask.size(-1) > 0, "grouped masked softmax requires at least one group");
  TORCH_CHECK(values.size(-1) > 0, "grouped masked softmax requires at least one feature");
  TORCH_CHECK(mask.device() == logits.device() && values.device() == logits.device(),
      "grouped masked softmax tensors must use the same device");
}

void validate_indexed_pool_shapes(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices) {
  TORCH_CHECK(logits.dim() > 0,
      "indexed masked softmax logits must have at least one dimension");
  TORCH_CHECK(mask.sizes() == logits.sizes(),
      "indexed masked softmax mask must match logits");
  TORCH_CHECK(values.dim() == logits.dim() + 1,
      "indexed masked softmax values must add a trailing feature dimension");
  for (int64_t dimension = 0; dimension < logits.dim(); ++dimension) {
    TORCH_CHECK(values.size(dimension) == logits.size(dimension),
        "indexed masked softmax value dimensions must match logits");
  }
  TORCH_CHECK(values.size(-1) > 0,
      "indexed masked softmax requires a non-empty feature dimension");
  TORCH_CHECK(mask.device() == logits.device() && values.device() == logits.device(),
      "indexed masked softmax tensors must use the same device");
  TORCH_CHECK(!choice_indices.empty(),
      "indexed masked softmax requires at least one choice index");
  const int64_t choice_count = logits.size(-1);
  for (size_t index = 0; index < choice_indices.size(); ++index) {
    const int64_t choice = choice_indices[index];
    TORCH_CHECK(choice >= 0 && choice < choice_count,
        "indexed masked softmax choice index is out of range: ", choice);
    for (size_t previous = 0; previous < index; ++previous) {
      TORCH_CHECK(choice_indices[previous] != choice,
          "indexed masked softmax choice indices must be unique: ", choice);
    }
  }
}

torch::Tensor grouped_masked_softmax_pool_reference(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values) {
  const int64_t choice_axis = logits.dim() - 1;
  auto float_logits = logits.to(torch::kFloat32);
  auto boolean_mask = mask.to(torch::kBool);
  auto expanded_logits = float_logits.unsqueeze(-1).expand_as(boolean_mask);
  auto weights = masked_softmax_reference(expanded_logits, boolean_mask, choice_axis);
  auto pooled =
      weights.unsqueeze(-1).mul(values.to(torch::kFloat32).unsqueeze(-2)).sum(choice_axis);
  auto present = boolean_mask.any(choice_axis).unsqueeze(-1);
  return torch::where(present, pooled, torch::zeros_like(pooled))
      .movedim(choice_axis, 0)
      .contiguous();
}

torch::Tensor indexed_masked_softmax_pool_reference(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices) {
  auto index = torch::tensor(choice_indices.vec(),
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  auto selected_logits = logits.index_select(-1, index);
  auto selected_mask = mask.to(torch::kBool).index_select(-1, index);
  auto selected_values = values.index_select(-2, index);
  auto weights = masked_softmax_reference(selected_logits, selected_mask, -1);
  auto pooled =
      selected_values.to(torch::kFloat32).mul(weights.unsqueeze(-1)).sum(-2);
  auto present = selected_mask.any(-1).unsqueeze(-1);
  return torch::where(present, pooled, torch::zeros_like(pooled));
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

class IndexedMaskedSoftmaxPoolValueFunction
    : public torch::autograd::Function<IndexedMaskedSoftmaxPoolValueFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& logits, const torch::Tensor& mask,
      const torch::Tensor& values, std::vector<int64_t> choice_indices) {
    context->save_for_backward({logits, mask});
    context->saved_data["value_shape"] = values.sizes().vec();
    context->saved_data["value_type"] = static_cast<int64_t>(values.scalar_type());
    context->saved_data["choice_indices"] = choice_indices;
    return rocm::indexed_masked_softmax_pool_forward(
        logits, mask, values, choice_indices);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto value_shape = context->saved_data["value_shape"].toIntVector();
    const auto value_type = static_cast<torch::ScalarType>(
        context->saved_data["value_type"].toInt());
    const auto choice_indices =
        context->saved_data["choice_indices"].toIntVector();
    auto value_gradient = context->needs_input_grad(2)
        ? rocm::indexed_masked_softmax_pool_value_backward(
              gradient_outputs.at(0), saved.at(0), saved.at(1), value_shape,
              value_type, choice_indices)
        : torch::Tensor();
    return {torch::Tensor(), torch::Tensor(), value_gradient, torch::Tensor()};
  }
};

class GroupedMaskedSoftmaxPoolValueFunction
    : public torch::autograd::Function<GroupedMaskedSoftmaxPoolValueFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& logits, const torch::Tensor& mask,
      const torch::Tensor& values) {
    context->save_for_backward({logits, mask});
    context->saved_data["value_shape"] = values.sizes().vec();
    context->saved_data["value_type"] =
        static_cast<int64_t>(values.scalar_type());
    return rocm::grouped_masked_softmax_pool_forward(logits, mask, values);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto value_shape = context->saved_data["value_shape"].toIntVector();
    const auto value_type = static_cast<torch::ScalarType>(
        context->saved_data["value_type"].toInt());
    auto value_gradient = context->needs_input_grad(2)
        ? rocm::grouped_masked_softmax_pool_value_backward(
              gradient_outputs.at(0), saved.at(0), saved.at(1), value_shape,
              value_type)
        : torch::Tensor();
    return {torch::Tensor(), torch::Tensor(), value_gradient};
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

torch::Tensor grouped_masked_softmax_pool(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values) {
  validate_grouped_pool_shapes(logits, mask, values);
  auto boolean_mask = mask.to(torch::kBool).contiguous();
#if defined(DJL_USE_ROCM_KERNELS)
  auto contiguous_logits = logits.contiguous();
  auto contiguous_values = values.contiguous();
  if ((!at::GradMode::is_enabled() ||
          (!logits.requires_grad() && !values.requires_grad())) &&
      rocm::supports_grouped_masked_softmax_pool(
          contiguous_logits, boolean_mask, contiguous_values)) {
    return rocm::grouped_masked_softmax_pool_forward(
        contiguous_logits, boolean_mask, contiguous_values);
  }
  if (at::GradMode::is_enabled() && !logits.requires_grad() &&
      values.requires_grad() &&
      !at::globalContext().deterministicAlgorithms() &&
      rocm::supports_grouped_masked_softmax_pool_value_backward(
          contiguous_logits, boolean_mask, contiguous_values)) {
    return GroupedMaskedSoftmaxPoolValueFunction::apply(
        contiguous_logits, boolean_mask, contiguous_values);
  }
#endif
  return grouped_masked_softmax_pool_reference(logits, boolean_mask, values);
}

torch::Tensor indexed_masked_softmax_pool(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices) {
  validate_indexed_pool_shapes(logits, mask, values, choice_indices);
#if defined(DJL_USE_ROCM_KERNELS)
  auto contiguous_logits = logits.contiguous();
  auto contiguous_mask = mask.contiguous();
  auto contiguous_values = values.contiguous();
  if ((!at::GradMode::is_enabled() ||
          (!logits.requires_grad() && !values.requires_grad())) &&
      rocm::supports_indexed_masked_softmax_pool(
          contiguous_logits, contiguous_mask, contiguous_values, choice_indices)) {
    return rocm::indexed_masked_softmax_pool_forward(
        contiguous_logits, contiguous_mask, contiguous_values, choice_indices);
  }
  if (at::GradMode::is_enabled() && !logits.requires_grad() &&
      values.requires_grad() &&
      rocm::supports_indexed_masked_softmax_pool(
          contiguous_logits, contiguous_mask, contiguous_values, choice_indices)) {
    return IndexedMaskedSoftmaxPoolValueFunction::apply(contiguous_logits,
        contiguous_mask, contiguous_values, choice_indices.vec());
  }
#endif
  return indexed_masked_softmax_pool_reference(logits, mask, values, choice_indices);
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
