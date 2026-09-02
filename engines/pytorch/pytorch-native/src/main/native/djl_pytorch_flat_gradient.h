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
#ifndef DJL_TORCH_DJL_PYTORCH_FLAT_GRADIENT_H
#define DJL_TORCH_DJL_PYTORCH_FLAT_GRADIENT_H

#include <torch/torch.h>

#include <vector>

namespace djl::pytorch::gradient {

struct FlatGradientAccumulator;
struct FlatGradientPacker;

FlatGradientAccumulator* NewFlatGradientAccumulator(
    std::vector<torch::Tensor> parameters, torch::Tensor gradient);
void BackwardFlatGradientAccumulator(FlatGradientAccumulator* accumulator,
    const torch::Tensor& target, const torch::Tensor& target_gradient);
void ZeroFlatGradientAccumulator(FlatGradientAccumulator* accumulator);
void DeleteFlatGradientAccumulator(FlatGradientAccumulator* accumulator);

FlatGradientPacker* NewFlatGradientPacker(
    std::vector<torch::Tensor> parameters, torch::Tensor destination);
void PackAndClearFlatGradients(
    FlatGradientPacker* packer, bool zero_missing_gradients);
void AccumulateAndClearFlatGradients(
    FlatGradientPacker* packer, bool zero_missing_gradients);
void ZeroFlatGradientPackerDestination(FlatGradientPacker* packer);
void ClearFlatGradientPackerParameterGradients(FlatGradientPacker* packer);
void DeleteFlatGradientPacker(FlatGradientPacker* packer);

}  // namespace djl::pytorch::gradient

#endif  // DJL_TORCH_DJL_PYTORCH_FLAT_GRADIENT_H
