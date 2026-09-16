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

#ifndef DJL_PYTORCH_MASKED_CATEGORICAL_CUDA_H
#define DJL_PYTORCH_MASKED_CATEGORICAL_CUDA_H

#include <torch/torch.h>

namespace djl::pytorch::cuda {

bool supports_masked_categorical(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

torch::Tensor masked_softmax_forward(const torch::Tensor& logits, const torch::Tensor& mask);

torch::Tensor masked_softmax_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& probabilities, const torch::Tensor& mask,
    torch::ScalarType input_type);

torch::Tensor masked_log_sum_exp_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, torch::Tensor* normalization = nullptr);

torch::Tensor masked_log_sum_exp_backward(const torch::Tensor& gradient_output,
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& normalization);

bool supports_indexed_masked_softmax_pool(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices);

torch::Tensor indexed_masked_softmax_pool_forward(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices);

torch::Tensor indexed_masked_softmax_pool_value_backward(
    const torch::Tensor& gradient_output, const torch::Tensor& logits,
    const torch::Tensor& mask, at::IntArrayRef value_shape,
    torch::ScalarType value_type, at::IntArrayRef choice_indices);

}  // namespace djl::pytorch::cuda

#endif  // DJL_PYTORCH_MASKED_CATEGORICAL_CUDA_H
