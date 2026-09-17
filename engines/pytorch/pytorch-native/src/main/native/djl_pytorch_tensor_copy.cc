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
#include "djl_pytorch_tensor_copy.h"

#include <ATen/ops/_foreach_copy.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <optional>
#include <utility>

namespace djl::pytorch {

struct TensorCopyPlan {
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  std::optional<c10::Stream> stream;
};

TensorCopyPlan* NewTensorCopyPlan(
    std::vector<torch::Tensor> sources, std::vector<torch::Tensor> destinations) {
  TORCH_CHECK(!sources.empty() && sources.size() == destinations.size(),
      "Tensor copy plans require equally sized, non-empty tensor lists.");
  auto device = sources.front().device();
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const auto& source = sources[index];
    const auto& destination = destinations[index];
    TORCH_CHECK(source.layout() == torch::kStrided &&
            destination.layout() == torch::kStrided,
        "Tensor copy plans require dense tensors.");
    TORCH_CHECK(source.device() == device && destination.device() == device,
        "Tensor copy plan tensors must be on one device.");
    TORCH_CHECK(source.sizes() == destination.sizes(),
        "Tensor copy plan pairs must have matching shapes.");
    TORCH_CHECK(source.scalar_type() == destination.scalar_type(),
        "Tensor copy plan pairs must have matching data types.");
  }
  return new TensorCopyPlan{
      std::move(sources), std::move(destinations), std::nullopt};
}

void CopyTensorCopyPlan(TensorCopyPlan* plan) {
  TORCH_CHECK(plan != nullptr, "Tensor copy plan has been closed.");
  auto device = plan->sources.front().device();
  c10::DeviceGuard device_guard(device);
  if (!device.is_cpu()) {
    c10::impl::VirtualGuardImpl guard(device.type());
    auto current = guard.getStream(device);
    if (!plan->stream.has_value()) {
      plan->stream = current;
    } else {
      TORCH_CHECK(plan->stream.value() == current,
          "Tensor copy plan operations must use the stream of the first copy.");
    }
  }
  torch::NoGradGuard no_grad;
  at::_foreach_copy_(plan->destinations, plan->sources);
}

void DeleteTensorCopyPlan(TensorCopyPlan* plan) {
  if (plan == nullptr) return;
  if (plan->stream.has_value()) {
    auto device = plan->sources.front().device();
    c10::DeviceGuard device_guard(device);
    c10::impl::VirtualGuardImpl guard(device.type());
    for (const auto& source : plan->sources) {
      if (source.numel() > 0) {
        guard.recordDataPtrOnStream(source.storage().data_ptr(), plan->stream.value());
      }
    }
    for (const auto& destination : plan->destinations) {
      if (destination.numel() > 0) {
        guard.recordDataPtrOnStream(destination.storage().data_ptr(), plan->stream.value());
      }
    }
  }
  delete plan;
}

}  // namespace djl::pytorch
