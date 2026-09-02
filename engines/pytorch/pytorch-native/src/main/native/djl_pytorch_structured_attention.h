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

#ifndef DJL_PYTORCH_STRUCTURED_ATTENTION_H
#define DJL_PYTORCH_STRUCTURED_ATTENTION_H

#include <torch/torch.h>

namespace djl::pytorch {

torch::Tensor indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, double scale);

torch::Tensor grouped_packed_attention(const torch::Tensor& query,
    const torch::Tensor& packed_key_value, const torch::Tensor& mask,
    int64_t heads, double scale);

torch::Tensor grouped_indexed_attention(const torch::Tensor& query, const torch::Tensor& shared_key_values,
    const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, double scale);

torch::Tensor mapped_grouped_indexed_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_group_indices,
    const torch::Tensor& shared_delta_table, const torch::Tensor& shared_delta_indices,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, double scale);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_STRUCTURED_ATTENTION_H
