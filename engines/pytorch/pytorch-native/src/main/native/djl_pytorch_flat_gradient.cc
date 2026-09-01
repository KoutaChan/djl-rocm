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
#include "djl_pytorch_flat_gradient.h"

#include <ATen/ops/_foreach_add.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/csrc/autograd/autograd.h>

#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace djl::pytorch::gradient {

struct FlatGradientAccumulator {
  std::vector<torch::Tensor> parameters;
  std::vector<torch::Tensor> gradient_views;
  torch::Tensor gradient;
  std::optional<c10::Stream> bound_stream;
};

namespace {

c10::Stream GetCurrentStream(c10::Device device) {
  c10::DeviceGuard device_guard(device);
  c10::impl::VirtualGuardImpl guard_impl(device.type());
  return guard_impl.getStream(device);
}

void BindCurrentStream(FlatGradientAccumulator* accumulator) {
  if (accumulator->gradient.device().is_cpu()) {
    return;
  }
  c10::Stream current_stream = GetCurrentStream(accumulator->gradient.device());
  if (!accumulator->bound_stream.has_value()) {
    accumulator->bound_stream = current_stream;
    return;
  }
  TORCH_CHECK(accumulator->bound_stream.value() == current_stream,
      "Flat gradient accumulator operations must use the accelerator stream "
      "that executed the first backward or zero operation.");
}

void RecordTensorOnBoundStream(const torch::Tensor& tensor,
    const std::optional<c10::Stream>& bound_stream) {
  if (!bound_stream.has_value() || !tensor.defined() || tensor.numel() == 0 ||
      tensor.layout() != c10::kStrided) {
    return;
  }
  c10::DeviceGuard device_guard(tensor.device());
  c10::impl::VirtualGuardImpl guard_impl(tensor.device().type());
  guard_impl.recordDataPtrOnStream(tensor.storage().data_ptr(), bound_stream.value());
}

}  // namespace

FlatGradientAccumulator* NewFlatGradientAccumulator(
    std::vector<torch::Tensor> parameters, torch::Tensor gradient) {
  TORCH_CHECK(!parameters.empty(),
      "Flat gradient accumulation requires at least one parameter.");
  TORCH_CHECK(gradient.defined() && gradient.layout() == torch::kStrided &&
          gradient.dim() == 1 && gradient.is_contiguous(),
      "The flat gradient destination must be a contiguous rank-1 dense tensor.");

  std::unordered_set<const c10::TensorImpl*> seen;
  int64_t total_size = 0;
  for (const auto& parameter : parameters) {
    TORCH_CHECK(parameter.defined() && parameter.layout() == torch::kStrided,
        "Flat gradient parameters must be defined dense tensors.");
    TORCH_CHECK(parameter.is_leaf() && parameter.requires_grad(),
        "Flat gradient parameters must be leaf tensors that require gradients.");
    TORCH_CHECK(seen.insert(parameter.unsafeGetTensorImpl()).second,
        "Flat gradient parameters must not contain duplicates.");
    TORCH_CHECK(parameter.device() == gradient.device(),
        "Flat gradient parameters and destination must use the same device.");
    TORCH_CHECK(parameter.scalar_type() == gradient.scalar_type(),
        "Flat gradient parameters and destination must use the same data type.");
    TORCH_CHECK(parameter.numel() <= std::numeric_limits<int64_t>::max() - total_size,
        "Flat gradient parameter size overflow.");
    total_size += parameter.numel();
  }
  TORCH_CHECK(total_size == gradient.numel(),
      "The flat gradient destination size must equal the total parameter size.");

  std::vector<torch::Tensor> gradient_views;
  gradient_views.reserve(parameters.size());
  int64_t offset = 0;
  for (const auto& parameter : parameters) {
    gradient_views.push_back(
        gradient.narrow(0, offset, parameter.numel()).view(parameter.sizes()));
    offset += parameter.numel();
  }

  return new FlatGradientAccumulator{std::move(parameters),
      std::move(gradient_views), std::move(gradient), std::nullopt};
}

void BackwardFlatGradientAccumulator(FlatGradientAccumulator* accumulator,
    const torch::Tensor& target, const torch::Tensor& target_gradient) {
  TORCH_CHECK(accumulator != nullptr, "Flat gradient accumulator has been closed.");
  BindCurrentStream(accumulator);
  auto gradients = torch::autograd::grad({target}, accumulator->parameters,
      {target_gradient}, false, false, true);
  TORCH_CHECK(gradients.size() == accumulator->parameters.size(),
      "Autograd returned an unexpected flat gradient count.");

  std::vector<torch::Tensor> destination_views;
  std::vector<torch::Tensor> dense_gradients;
  destination_views.reserve(gradients.size());
  dense_gradients.reserve(gradients.size());
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    const auto& gradient = gradients[index];
    if (!gradient.defined()) {
      continue;
    }
    const auto& parameter = accumulator->parameters[index];
    TORCH_CHECK(gradient.layout() == torch::kStrided,
        "Flat gradient accumulation does not support sparse gradients.");
    TORCH_CHECK(gradient.sizes() == parameter.sizes(),
        "Autograd returned an unexpected parameter gradient shape.");
    TORCH_CHECK(gradient.device() == accumulator->gradient.device(),
        "Autograd returned a gradient on an unexpected device.");
    TORCH_CHECK(gradient.scalar_type() == accumulator->gradient.scalar_type(),
        "Autograd returned a gradient with an unexpected data type.");
    destination_views.push_back(accumulator->gradient_views[index]);
    dense_gradients.push_back(gradient.is_contiguous() ? gradient : gradient.contiguous());
  }

  if (!dense_gradients.empty()) {
    torch::NoGradGuard no_grad;
    at::_foreach_add_(destination_views, dense_gradients);
  }
}

void ZeroFlatGradientAccumulator(FlatGradientAccumulator* accumulator) {
  TORCH_CHECK(accumulator != nullptr, "Flat gradient accumulator has been closed.");
  BindCurrentStream(accumulator);
  torch::NoGradGuard no_grad;
  accumulator->gradient.zero_();
}

void DeleteFlatGradientAccumulator(FlatGradientAccumulator* accumulator) {
  if (accumulator == nullptr) {
    return;
  }
  RecordTensorOnBoundStream(accumulator->gradient, accumulator->bound_stream);
  for (const auto& parameter : accumulator->parameters) {
    RecordTensorOnBoundStream(parameter, accumulator->bound_stream);
  }
  delete accumulator;
}

}  // namespace djl::pytorch::gradient
