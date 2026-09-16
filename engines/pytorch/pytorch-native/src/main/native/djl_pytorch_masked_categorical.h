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

#ifndef DJL_PYTORCH_MASKED_CATEGORICAL_H
#define DJL_PYTORCH_MASKED_CATEGORICAL_H

#include <torch/torch.h>

namespace djl::pytorch {
namespace detail {

bool is_masked_categorical_layout_supported(
    const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

}  // namespace detail

torch::Tensor masked_softmax(const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

torch::Tensor grouped_masked_softmax_pool(
    const torch::Tensor& logits, const torch::Tensor& mask, const torch::Tensor& values);

torch::Tensor indexed_masked_softmax_pool(const torch::Tensor& logits,
    const torch::Tensor& mask, const torch::Tensor& values,
    at::IntArrayRef choice_indices);

torch::Tensor masked_log_sum_exp(const torch::Tensor& logits, const torch::Tensor& mask, int64_t axis);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_MASKED_CATEGORICAL_H
