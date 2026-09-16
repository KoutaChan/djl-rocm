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

#ifndef DJL_PYTORCH_STRUCTURED_ATTENTION_CUDA_H
#define DJL_PYTORCH_STRUCTURED_ATTENTION_CUDA_H

#include "djl_pytorch_structured_attention.h"

namespace djl::pytorch::cuda {

bool supports_mapped_grouped_indexed_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids);

using detail::MappedGroupedIndexedAttentionForwardResult;

MappedGroupedIndexedAttentionForwardResult mapped_grouped_indexed_attention_forward(
    const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_group_indices, const torch::Tensor& shared_delta_table,
    const torch::Tensor& shared_delta_indices, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, float scale, bool capture_probabilities);

using detail::MappedGroupedIndexedAttentionGradients;

MappedGroupedIndexedAttentionGradients mapped_grouped_indexed_attention_backward(
    const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_group_indices, const torch::Tensor& shared_delta_table,
    const torch::Tensor& shared_delta_indices, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, const torch::Tensor& probabilities,
    const torch::Tensor& gradient_output, float scale, bool needs_query_gradient,
    bool needs_shared_key_value_gradient, bool needs_shared_delta_table_gradient,
    bool needs_indexed_delta_gradient);

}  // namespace djl::pytorch::cuda

#endif  // DJL_PYTORCH_STRUCTURED_ATTENTION_CUDA_H
