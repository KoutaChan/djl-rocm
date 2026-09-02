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
#ifndef DJL_TORCH_DJL_PYTORCH_PROJECTED_RESIDUAL_MLP_H
#define DJL_TORCH_DJL_PYTORCH_PROJECTED_RESIDUAL_MLP_H

#include <torch/torch.h>

namespace djl::pytorch {

struct ProjectedResidualMlpForwardResult {
  torch::Tensor output;
  torch::Tensor combined;
  torch::Tensor activated;
};

ProjectedResidualMlpForwardResult projected_residual_mlp_forward(
    const torch::Tensor& input, const torch::Tensor& combined_weight,
    const torch::Tensor& combined_bias, const torch::Tensor& output_weight,
    torch::Tensor combined = {}, torch::Tensor activated = {},
    torch::Tensor output = {});

torch::Tensor projected_residual_mlp(const torch::Tensor& input,
    const torch::Tensor& combined_weight, const torch::Tensor& combined_bias,
    const torch::Tensor& output_weight);

}  // namespace djl::pytorch

#endif
