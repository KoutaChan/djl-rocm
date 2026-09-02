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
#ifndef DJL_PYTORCH_ATTENTION_H
#define DJL_PYTORCH_ATTENTION_H

#include <torch/torch.h>

#include <optional>

namespace djl::pytorch {

/** Dispatches scaled dot-product attention while preserving PyTorch fallback semantics. */
torch::Tensor scaled_dot_product_attention(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const std::optional<torch::Tensor>& mask, double dropout, bool causal,
    const std::optional<double>& scale);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_ATTENTION_H
