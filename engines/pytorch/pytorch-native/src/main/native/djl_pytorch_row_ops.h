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
#ifndef DJL_PYTORCH_ROW_OPS_H
#define DJL_PYTORCH_ROW_OPS_H

#include <torch/torch.h>

namespace djl::pytorch {

torch::Tensor embedding_with_offsets(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table);

torch::Tensor embedding_feature_pack(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features);

torch::Tensor scatter_rows(
    const torch::Tensor& rows, const torch::Tensor& row_indices, int64_t row_count);

torch::Tensor segmented_lookup_sum(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices);

torch::Tensor padded_batch_gather(
    const torch::Tensor& source, const torch::Tensor& stored_indices);

torch::Tensor padded_batch_gather_2d(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices);

torch::Tensor padded_batch_gather_by_batch_indices(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_ROW_OPS_H
