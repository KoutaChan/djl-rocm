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

#include "djl_pytorch_routing_masks.h"

#include "djl_pytorch_kernel_backend.h"

namespace djl::pytorch {
namespace {

void check_metadata_shape(const torch::Tensor& metadata, const torch::Tensor& mask) {
  TORCH_CHECK(metadata.dim() > 0, "routing metadata must have at least one dimension");
  TORCH_CHECK(metadata.sizes().slice(0, metadata.dim() - 1) == mask.sizes(),
      "mask shape must match the leading metadata dimensions");
  TORCH_CHECK(c10::isIntegralType(metadata.scalar_type(), false), "routing metadata must be integral");
  TORCH_CHECK(c10::isFloatingType(mask.scalar_type()), "routing mask must be floating point");
  TORCH_CHECK(metadata.device() == mask.device(), "routing metadata and mask must use the same device");
}

torch::Tensor categorical_masks_reference(const torch::Tensor& categories, const torch::Tensor& mask,
    const std::vector<int64_t>& field_indices, const std::vector<uint64_t>& category_sets) {
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> rules;
  rules.reserve(field_indices.size());
  for (size_t rule = 0; rule < field_indices.size(); ++rule) {
    auto field = categories.select(-1, field_indices[rule]);
    auto selected = torch::zeros_like(field, field.options().dtype(torch::kBool));
    uint64_t remaining = category_sets[rule];
    while (remaining != 0) {
      int64_t category = 0;
      while (((remaining >> category) & uint64_t{1}) == 0) {
        ++category;
      }
      selected.logical_or_(field.eq(category));
      remaining &= remaining - 1;
    }
    rules.push_back(selected.to(mask.scalar_type()).mul(mask).unsqueeze(-1));
  }
  return torch::cat(rules, -1).detach();
}

torch::Tensor binary_choice_masks_reference(const torch::Tensor& routes,
    const torch::Tensor& first_mask, const torch::Tensor& second_mask, int64_t representative_field,
    int64_t first_route_field, int64_t second_route_field, int64_t padding_value) {
  torch::NoGradGuard no_grad;
  auto combined = first_mask.add(second_mask);
  auto first_present = routes.select(-1, first_route_field).ne(padding_value)
      .to(combined.scalar_type()).mul(combined);
  auto second_present = routes.select(-1, second_route_field).ne(padding_value)
      .to(combined.scalar_type()).mul(combined);
  auto representative = routes.select(-1, representative_field).ne(padding_value)
      .to(combined.scalar_type()).mul(combined);
  return torch::stack({combined, first_present, second_present, representative}, -1).detach();
}

}  // namespace

torch::Tensor categorical_masks(const torch::Tensor& categories, const torch::Tensor& mask,
    const std::vector<int64_t>& field_indices, const std::vector<uint64_t>& category_sets) {
  torch::NoGradGuard no_grad;
  check_metadata_shape(categories, mask);
  TORCH_CHECK(!field_indices.empty() && field_indices.size() == category_sets.size(),
      "categorical rules must have matching non-empty field and category sets");
  for (const auto field : field_indices) {
    TORCH_CHECK(field >= 0 && field < categories.size(-1), "categorical field index is out of range");
  }
#if defined(DJL_USE_ACCELERATOR_KERNELS)
  if (kernel_backend::supports_categorical_masks(categories, mask, field_indices.size())) {
    return kernel_backend::categorical_masks(categories, mask, field_indices, category_sets);
  }
#endif
  return categorical_masks_reference(categories, mask, field_indices, category_sets);
}

torch::Tensor binary_choice_masks(const torch::Tensor& routes, const torch::Tensor& first_mask,
    const torch::Tensor& second_mask, int64_t representative_field, int64_t first_route_field,
    int64_t second_route_field, int64_t padding_value) {
  torch::NoGradGuard no_grad;
  check_metadata_shape(routes, first_mask);
  TORCH_CHECK(second_mask.sizes() == first_mask.sizes() && second_mask.scalar_type() == first_mask.scalar_type(),
      "binary choice masks must have identical shapes and data types");
  TORCH_CHECK(second_mask.device() == first_mask.device(), "binary choice masks must use the same device");
  for (const auto field : {representative_field, first_route_field, second_route_field}) {
    TORCH_CHECK(field >= 0 && field < routes.size(-1), "routing field index is out of range");
  }
#if defined(DJL_USE_ACCELERATOR_KERNELS)
  if (kernel_backend::supports_binary_choice_masks(routes, first_mask, second_mask)) {
    return kernel_backend::binary_choice_masks(routes, first_mask, second_mask, representative_field,
        first_route_field, second_route_field, padding_value);
  }
#endif
  return binary_choice_masks_reference(routes, first_mask, second_mask, representative_field,
      first_route_field, second_route_field, padding_value);
}

}  // namespace djl::pytorch
