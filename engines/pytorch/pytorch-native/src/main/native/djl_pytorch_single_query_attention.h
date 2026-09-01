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
#ifndef DJL_PYTORCH_SINGLE_QUERY_ATTENTION_H
#define DJL_PYTORCH_SINGLE_QUERY_ATTENTION_H

#include <torch/torch.h>

#include <optional>

namespace djl::pytorch::rocm {

bool supports_single_query_attention(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const std::optional<torch::Tensor>& mask);

bool supports_single_query_attention_backward(const torch::Tensor& query, const torch::Tensor& key);

struct SingleQueryAttentionForwardResult {
  torch::Tensor output;
  torch::Tensor probabilities;
};

SingleQueryAttentionForwardResult single_query_attention_forward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const std::optional<torch::Tensor>& mask, float scale, bool capture_probabilities);

struct SingleQueryAttentionGradients {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
};

SingleQueryAttentionGradients single_query_attention_backward(const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& probabilities, const torch::Tensor& gradient_output, float scale,
    bool needs_query_gradient, bool needs_key_gradient, bool needs_value_gradient);

}  // namespace djl::pytorch::rocm

#endif  // DJL_PYTORCH_SINGLE_QUERY_ATTENTION_H
