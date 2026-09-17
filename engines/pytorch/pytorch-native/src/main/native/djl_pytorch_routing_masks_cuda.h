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

#ifndef DJL_PYTORCH_ROUTING_MASKS_CUDA_H
#define DJL_PYTORCH_ROUTING_MASKS_CUDA_H

#include <torch/types.h>

#include <cstdint>
#include <vector>

namespace djl::pytorch::cuda {

bool supports_categorical_masks(
    const torch::Tensor& categories, const torch::Tensor& mask, size_t rule_count);

bool supports_binary_choice_masks(const torch::Tensor& routes,
    const torch::Tensor& first_mask, const torch::Tensor& second_mask);

torch::Tensor categorical_masks(const torch::Tensor& categories, const torch::Tensor& mask,
    const std::vector<int64_t>& field_indices, const std::vector<uint64_t>& category_sets);

torch::Tensor binary_choice_masks(const torch::Tensor& routes, const torch::Tensor& first_mask,
    const torch::Tensor& second_mask, int64_t representative_field, int64_t first_route_field,
    int64_t second_route_field, int64_t padding_value);

}  // namespace djl::pytorch::cuda

#endif  // DJL_PYTORCH_ROUTING_MASKS_CUDA_H
