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

#ifndef DJL_PYTORCH_ROW_OPS_CUDA_H
#define DJL_PYTORCH_ROW_OPS_CUDA_H

#include <torch/torch.h>

namespace djl::pytorch::cuda {

bool supports_embedding_with_offsets(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table);

torch::Tensor embedding_with_offsets_forward(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table);

bool supports_embedding_feature_pack(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features);

torch::Tensor embedding_feature_pack_forward(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features);

bool supports_segmented_lookup_sum(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices);

torch::Tensor segmented_lookup_sum_forward(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices);

bool supports_padded_batch_gather(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices);

torch::Tensor padded_batch_gather_forward(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices);

}  // namespace djl::pytorch::cuda

#endif  // DJL_PYTORCH_ROW_OPS_CUDA_H
