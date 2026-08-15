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

torch::Tensor scatter_rows(
    const torch::Tensor& rows, const torch::Tensor& row_indices, int64_t row_count);

}  // namespace djl::pytorch

#endif  // DJL_PYTORCH_ROW_OPS_H
