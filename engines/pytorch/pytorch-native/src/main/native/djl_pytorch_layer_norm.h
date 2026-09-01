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

#ifndef DJL_PYTORCH_LAYER_NORM_H
#define DJL_PYTORCH_LAYER_NORM_H

#include <torch/torch.h>

#include <vector>

namespace djl::pytorch {

std::vector<torch::Tensor> layer_norm_and_cast(const torch::Tensor& input,
    at::IntArrayRef normalized_shape, const torch::Tensor& weight,
    const torch::Tensor& bias, double epsilon,
    torch::ScalarType converted_type);

std::vector<torch::Tensor> residual_add_layer_norm(const torch::Tensor& residual,
    const torch::Tensor& update, at::IntArrayRef normalized_shape,
    const torch::Tensor& weight, const torch::Tensor& bias, double epsilon);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_LAYER_NORM_H
