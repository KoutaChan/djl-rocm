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

#ifndef DJL_PYTORCH_NN_FUNCTIONAL_CUDA_H
#define DJL_PYTORCH_NN_FUNCTIONAL_CUDA_H

#include <torch/torch.h>

#include <vector>

namespace djl::pytorch::cuda {

bool supports_masked_embedding_residual_to_owned_tokens(const torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask);

torch::Tensor add_masked_embedding_residual_to_owned_tokens(torch::Tensor& tokens,
    const std::vector<torch::Tensor>& stored_indices, const torch::Tensor& embedding_table,
    const torch::Tensor& valid_mask, int64_t padding_index, bool mean_valid);

bool supports_broadcast_residual_to_owned_silu(const torch::Tensor& values,
    const torch::Tensor& residual, const torch::Tensor* mask);

void add_broadcast_residual_to_owned_and_silu(
    torch::Tensor& values, const torch::Tensor& residual, const torch::Tensor* mask);

bool supports_bias_and_broadcast_residual_to_owned_silu(const torch::Tensor& values,
    const torch::Tensor& bias, const torch::Tensor& residual, const torch::Tensor* mask);

void add_bias_and_broadcast_residual_to_owned_and_silu(torch::Tensor& values,
    const torch::Tensor& bias, const torch::Tensor& residual, const torch::Tensor* mask);

}  // namespace djl::pytorch::cuda

#endif  // DJL_PYTORCH_NN_FUNCTIONAL_CUDA_H
