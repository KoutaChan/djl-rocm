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

#include "djl_pytorch_layer_norm.h"

#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/nn/functional/normalization.h>

#include "djl_pytorch_kernel_backend.h"

namespace djl::pytorch {
namespace detail {

bool is_autocast_layer_norm_layout_supported(const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias, at::IntArrayRef normalized_shape) {
  if (!input.is_cuda() || !input.is_contiguous() || normalized_shape.empty() ||
      input.dim() < static_cast<int64_t>(normalized_shape.size()) ||
      input.sizes().slice(input.dim() - normalized_shape.size()) != normalized_shape ||
      !weight.defined() || !bias.defined() || !weight.is_contiguous() || !bias.is_contiguous() ||
      weight.sizes() != normalized_shape || weight.numel() <= 0 || bias.sizes() != weight.sizes() ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kBFloat16) ||
      (weight.scalar_type() != torch::kFloat32 && weight.scalar_type() != torch::kFloat16 &&
          weight.scalar_type() != torch::kBFloat16) ||
      bias.scalar_type() != weight.scalar_type() || weight.device() != input.device() ||
      bias.device() != input.device()) {
    return false;
  }
  return true;
}

}  // namespace detail
namespace {

std::vector<torch::Tensor> layer_norm_and_cast_reference(const torch::Tensor& input,
    at::IntArrayRef normalized_shape, const torch::Tensor& weight,
    const torch::Tensor& bias, double epsilon,
    torch::ScalarType converted_type) {
  auto normalized = torch::nn::functional::layer_norm(input,
      torch::nn::functional::LayerNormFuncOptions(normalized_shape.vec())
          .weight(weight)
          .bias(bias)
          .eps(epsilon));
  auto converted = normalized.scalar_type() == converted_type
      ? normalized.clone()
      : normalized.to(converted_type);
  return {std::move(normalized), std::move(converted)};
}

std::vector<torch::Tensor> residual_add_layer_norm_reference(const torch::Tensor& residual,
    const torch::Tensor& update, at::IntArrayRef normalized_shape,
    const torch::Tensor& weight, const torch::Tensor& bias, double epsilon) {
  auto summed_residual = residual.add(update);
  auto normalized = torch::nn::functional::layer_norm(summed_residual,
      torch::nn::functional::LayerNormFuncOptions(normalized_shape.vec())
          .weight(weight)
          .bias(bias)
          .eps(epsilon));
  return {std::move(normalized), std::move(summed_residual)};
}

#if defined(DJL_USE_ROCM_KERNELS)

class LayerNormAndCastFunction
    : public torch::autograd::Function<LayerNormAndCastFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& input, const torch::Tensor& weight,
      const torch::Tensor& bias, std::vector<int64_t> normalized_shape,
      double epsilon, int64_t converted_type) {
    auto result = rocm::autocast_layer_norm_and_cast(input, weight, bias,
        static_cast<float>(epsilon), static_cast<torch::ScalarType>(converted_type), true);
    context->save_for_backward(
        {input, weight, result.mean, result.reciprocal_standard_deviation});
    context->set_materialize_grads(false);
    context->saved_data["converted_type"] = converted_type;
    return {result.normalized, result.converted};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto converted_type = static_cast<torch::ScalarType>(
        context->saved_data["converted_type"].toInt());
    auto gradients = rocm::autocast_layer_norm_and_cast_backward(
        gradient_outputs.at(0), gradient_outputs.at(1), saved.at(0), saved.at(1),
        saved.at(2), saved.at(3), converted_type, context->needs_input_grad(0),
        context->needs_input_grad(1), context->needs_input_grad(2));
    return {gradients.input, gradients.weight, gradients.bias,
        torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

class ResidualAddLayerNormFunction
    : public torch::autograd::Function<ResidualAddLayerNormFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& residual, const torch::Tensor& update,
      const torch::Tensor& weight, const torch::Tensor& bias,
      std::vector<int64_t> normalized_shape, double epsilon,
      int64_t summed_type, int64_t normalized_type) {
    auto result = rocm::residual_add_layer_norm_forward(residual, update, weight, bias,
        normalized_shape, static_cast<float>(epsilon),
        static_cast<torch::ScalarType>(summed_type),
        static_cast<torch::ScalarType>(normalized_type), true);
    context->save_for_backward(
        {result.summed_residual, weight, result.mean, result.reciprocal_standard_deviation});
    context->set_materialize_grads(false);
    context->saved_data["residual_type"] = static_cast<int64_t>(residual.scalar_type());
    context->saved_data["update_type"] = static_cast<int64_t>(update.scalar_type());
    return {result.normalized, result.summed_residual};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto residual_type = static_cast<torch::ScalarType>(
        context->saved_data["residual_type"].toInt());
    const auto update_type = static_cast<torch::ScalarType>(
        context->saved_data["update_type"].toInt());
    auto gradients = rocm::residual_add_layer_norm_backward(
        gradient_outputs.at(0), gradient_outputs.at(1), saved.at(0), saved.at(1),
        saved.at(2), saved.at(3), residual_type, update_type,
        context->needs_input_grad(0), context->needs_input_grad(1),
        context->needs_input_grad(2), context->needs_input_grad(3));
    return {gradients.residual, gradients.update, gradients.weight, gradients.bias,
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

#endif

}  // namespace

std::vector<torch::Tensor> layer_norm_and_cast(const torch::Tensor& input,
    at::IntArrayRef normalized_shape, const torch::Tensor& weight,
    const torch::Tensor& bias, double epsilon,
    torch::ScalarType converted_type) {
#if defined(DJL_USE_ACCELERATOR_KERNELS)
  if (at::autocast::is_autocast_enabled(at::DeviceType::CUDA) &&
      kernel_backend::supports_autocast_layer_norm_and_cast(
          input, weight, bias, normalized_shape, converted_type)) {
    const bool needs_autograd = at::GradMode::is_enabled() &&
        (input.requires_grad() || weight.requires_grad() || bias.requires_grad());
#if defined(DJL_USE_ROCM_KERNELS)
    if (needs_autograd) {
      auto outputs = LayerNormAndCastFunction::apply(input, weight, bias,
          normalized_shape.vec(), epsilon, static_cast<int64_t>(converted_type));
      return {outputs.at(0), outputs.at(1)};
    }
    auto result = rocm::autocast_layer_norm_and_cast(input, weight, bias,
        static_cast<float>(epsilon), converted_type, false);
    return {std::move(result.normalized), std::move(result.converted)};
#else
    if (!needs_autograd) {
      return cuda::autocast_layer_norm_and_cast(
          input, weight, bias, static_cast<float>(epsilon), converted_type);
    }
#endif
  }
#endif
  return layer_norm_and_cast_reference(
      input, normalized_shape, weight, bias, epsilon, converted_type);
}

std::vector<torch::Tensor> residual_add_layer_norm(const torch::Tensor& residual,
    const torch::Tensor& update, at::IntArrayRef normalized_shape,
    const torch::Tensor& weight, const torch::Tensor& bias, double epsilon) {
#if defined(DJL_USE_ROCM_KERNELS)
  const auto summed_type = at::result_type(residual, update);
  const bool autocast_enabled = at::autocast::is_autocast_enabled(at::DeviceType::CUDA);
  const auto normalized_type = autocast_enabled ? torch::kFloat32 : summed_type;
  if (rocm::supports_residual_add_layer_norm(residual, update, weight, bias,
          normalized_shape, summed_type, normalized_type) &&
      (autocast_enabled || weight.scalar_type() == torch::kFloat32)) {
    const bool needs_autograd = at::GradMode::is_enabled() &&
        (residual.requires_grad() || update.requires_grad() || weight.requires_grad() ||
            bias.requires_grad());
    if (needs_autograd) {
      auto outputs = ResidualAddLayerNormFunction::apply(residual, update, weight, bias,
          normalized_shape.vec(), epsilon, static_cast<int64_t>(summed_type),
          static_cast<int64_t>(normalized_type));
      return {outputs.at(0), outputs.at(1)};
    }
    auto result = rocm::residual_add_layer_norm_forward(residual, update, weight, bias,
        normalized_shape, static_cast<float>(epsilon), summed_type, normalized_type, false);
    return {std::move(result.normalized), std::move(result.summed_residual)};
  }
#endif
  return residual_add_layer_norm_reference(
      residual, update, normalized_shape, weight, bias, epsilon);
}

}  // namespace djl::pytorch
