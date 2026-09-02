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

#include "djl_pytorch_projected_residual_mlp.h"

#include <ATen/autocast_mode.h>
#include <ATen/ops/addmm.h>
#include <torch/csrc/autograd/custom_function.h>

#if defined(DJL_USE_FUSION_KERNELS)
#include "djl_pytorch_fusion_kernels.h"
#endif

namespace djl::pytorch {
namespace {

bool is_supported_data_type(torch::ScalarType data_type) {
  return data_type == torch::kFloat16 || data_type == torch::kBFloat16 ||
      data_type == torch::kFloat32;
}

bool supports_projected_residual_mlp(const torch::Tensor& input,
    const torch::Tensor& combined_weight, const torch::Tensor& combined_bias,
    const torch::Tensor& output_weight) {
  if (!input.is_cuda() || input.dim() < 1 || combined_weight.dim() != 2 ||
      combined_bias.dim() != 1 || output_weight.dim() != 2 ||
      !is_supported_data_type(input.scalar_type()) ||
      combined_weight.scalar_type() != input.scalar_type() ||
      combined_bias.scalar_type() != input.scalar_type() ||
      output_weight.scalar_type() != input.scalar_type() ||
      combined_weight.device() != input.device() ||
      combined_bias.device() != input.device() ||
      output_weight.device() != input.device() ||
      !combined_weight.is_contiguous() || !combined_bias.is_contiguous() ||
      !output_weight.is_contiguous()) {
    return false;
  }
  const int64_t output_width = output_weight.size(0);
  const int64_t hidden_width = output_weight.size(1);
  return input.size(-1) > 0 && output_width > 0 && hidden_width > 0 &&
      combined_weight.size(0) == output_width + hidden_width &&
      combined_weight.size(1) == input.size(-1) &&
      combined_bias.size(0) == output_width + hidden_width;
}

torch::Tensor projected_residual_mlp_reference(const torch::Tensor& input,
    const torch::Tensor& combined_weight, const torch::Tensor& combined_bias,
    const torch::Tensor& output_weight) {
  auto combined = torch::nn::functional::linear(
      input, combined_weight, combined_bias);
  const int64_t output_width = output_weight.size(0);
  const int64_t hidden_width = output_weight.size(1);
  auto skip = combined.narrow(-1, 0, output_width);
  auto hidden = torch::silu(
      combined.narrow(-1, output_width, hidden_width));
  return torch::nn::functional::linear(
      hidden, output_weight, torch::Tensor()).add(skip);
}

class ProjectedResidualMlpFunction
    : public torch::autograd::Function<ProjectedResidualMlpFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& input, const torch::Tensor& combined_weight,
      const torch::Tensor& combined_bias, const torch::Tensor& output_weight) {
    auto result = projected_residual_mlp_forward(
        input, combined_weight, combined_bias, output_weight);
    const int64_t input_width = input.size(-1);
    const int64_t rows = input.numel() / input_width;
    context->save_for_backward({input.reshape({rows, input_width}),
        combined_weight, output_weight, result.combined});
    context->saved_data["input_shape"] = input.sizes().vec();
    context->set_materialize_grads(false);
    return result.output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto& gradient_output = gradient_outputs.at(0);
    if (!gradient_output.defined()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }

    const auto saved = context->get_saved_variables();
    const auto& flattened_input = saved.at(0);
    const auto& combined_weight = saved.at(1);
    const auto& output_weight = saved.at(2);
    const auto& combined = saved.at(3);
    const int64_t rows = flattened_input.size(0);
    const int64_t output_width = output_weight.size(0);
    const int64_t hidden_width = output_weight.size(1);
    auto gradient = gradient_output.reshape({rows, output_width}).contiguous();

    torch::Tensor input_gradient;
    torch::Tensor combined_weight_gradient;
    torch::Tensor combined_bias_gradient;
    torch::Tensor output_weight_gradient;

    auto preactivation = combined.narrow(1, output_width, hidden_width);
    auto hidden = torch::silu(preactivation);
    if (context->needs_input_grad(3)) {
      output_weight_gradient = torch::matmul(gradient.transpose(0, 1), hidden);
    }

    const bool needs_combined_gradient = context->needs_input_grad(0) ||
        context->needs_input_grad(1) || context->needs_input_grad(2);
    if (needs_combined_gradient) {
      auto hidden_gradient = torch::matmul(gradient, output_weight);
      auto activation_gradient =
          at::silu_backward(hidden_gradient, preactivation);
      auto combined_gradient = torch::cat({gradient, activation_gradient}, 1);
      if (context->needs_input_grad(0)) {
        auto input_shape = context->saved_data["input_shape"].toIntVector();
        input_gradient = torch::matmul(combined_gradient, combined_weight)
                             .reshape(input_shape);
      }
      if (context->needs_input_grad(1)) {
        combined_weight_gradient = torch::matmul(
            combined_gradient.transpose(0, 1), flattened_input);
      }
      if (context->needs_input_grad(2)) {
        combined_bias_gradient = combined_gradient.sum(0);
      }
    }
    return {input_gradient, combined_weight_gradient,
        combined_bias_gradient, output_weight_gradient};
  }
};

torch::Tensor cast_for_autocast(
    const torch::Tensor& tensor, torch::ScalarType data_type) {
  return tensor.scalar_type() == data_type
      ? tensor
      : at::autocast::cached_cast(
            data_type, tensor, at::DeviceType::CUDA);
}

}  // namespace

ProjectedResidualMlpForwardResult projected_residual_mlp_forward(
    const torch::Tensor& input, const torch::Tensor& combined_weight,
    const torch::Tensor& combined_bias, const torch::Tensor& output_weight,
    torch::Tensor combined, torch::Tensor activated, torch::Tensor output) {
  const int64_t input_width = input.size(-1);
  const int64_t rows = input.numel() / input_width;
  const int64_t output_width = output_weight.size(0);
  const int64_t hidden_width = output_weight.size(1);
  auto flattened_input = input.reshape({rows, input_width});
  if (!combined.defined()) {
    combined = torch::empty(
        {rows, output_width + hidden_width}, input.options());
  } else {
    combined = combined.reshape({rows, output_width + hidden_width});
  }
  if (!output.defined()) {
    output = torch::empty({rows, output_width}, input.options());
  } else {
    output = output.reshape({rows, output_width});
  }

  at::addmm_out(combined, combined_bias, flattened_input,
      combined_weight.transpose(0, 1));
#if defined(DJL_USE_FUSION_KERNELS)
  if (!activated.defined()) {
    activated = torch::empty({rows, hidden_width}, input.options());
  } else {
    activated = activated.reshape({rows, hidden_width});
  }
  fusion::LaunchProjectedResidualMlpPrepare(
      combined, activated, output, rows, output_width, hidden_width);
  output.addmm_(activated, output_weight.transpose(0, 1));
#else
  output.copy_(combined.narrow(1, 0, output_width));
  activated = torch::silu(combined.narrow(1, output_width, hidden_width));
  output.addmm_(activated, output_weight.transpose(0, 1));
#endif

  auto output_shape = input.sizes().vec();
  output_shape.back() = output_width;
  return {output.reshape(output_shape), std::move(combined),
      std::move(activated)};
}

torch::Tensor projected_residual_mlp(const torch::Tensor& input,
    const torch::Tensor& combined_weight, const torch::Tensor& combined_bias,
    const torch::Tensor& output_weight) {
#if defined(DJL_USE_FUSION_KERNELS)
  torch::Tensor operation_input = input;
  torch::Tensor operation_combined_weight = combined_weight;
  torch::Tensor operation_combined_bias = combined_bias;
  torch::Tensor operation_output_weight = output_weight;
  if (input.is_cuda() &&
      at::autocast::is_autocast_enabled(at::DeviceType::CUDA)) {
    const auto data_type =
        at::autocast::get_autocast_dtype(at::DeviceType::CUDA);
    if (is_supported_data_type(data_type)) {
      operation_input = cast_for_autocast(input, data_type);
      operation_combined_weight = cast_for_autocast(combined_weight, data_type);
      operation_combined_bias = cast_for_autocast(combined_bias, data_type);
      operation_output_weight = cast_for_autocast(output_weight, data_type);
    }
  }
  if (supports_projected_residual_mlp(operation_input,
          operation_combined_weight, operation_combined_bias,
          operation_output_weight)) {
    const bool needs_autograd = at::GradMode::is_enabled() &&
        (operation_input.requires_grad() ||
            operation_combined_weight.requires_grad() ||
            operation_combined_bias.requires_grad() ||
            operation_output_weight.requires_grad());
    if (needs_autograd) {
      return ProjectedResidualMlpFunction::apply(operation_input,
          operation_combined_weight, operation_combined_bias,
          operation_output_weight);
    }
    return projected_residual_mlp_forward(operation_input,
        operation_combined_weight, operation_combined_bias,
        operation_output_weight).output;
  }
#endif
  return projected_residual_mlp_reference(
      input, combined_weight, combined_bias, output_weight);
}

}  // namespace djl::pytorch
