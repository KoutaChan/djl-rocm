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
#ifndef DJL_TORCH_DJL_PYTORCH_TENSOR_COPY_H
#define DJL_TORCH_DJL_PYTORCH_TENSOR_COPY_H

#include <torch/torch.h>

#include <vector>

namespace djl::pytorch {

struct TensorCopyPlan;

TensorCopyPlan* NewTensorCopyPlan(
    std::vector<torch::Tensor> sources, std::vector<torch::Tensor> destinations);
void CopyTensorCopyPlan(TensorCopyPlan* plan);
void DeleteTensorCopyPlan(TensorCopyPlan* plan);

}  // namespace djl::pytorch

#endif  // DJL_TORCH_DJL_PYTORCH_TENSOR_COPY_H
