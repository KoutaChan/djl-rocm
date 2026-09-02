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
#ifndef DJL_TORCH_DJL_PYTORCH_FUSION_H
#define DJL_TORCH_DJL_PYTORCH_FUSION_H

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>

namespace djl::pytorch::fusion {

struct FusionPlan;
struct FusionExecutable;
struct FusionSession;

/** Identifies the accelerator runtime backing fusion kernel launches. */
enum class FusionBackend : int32_t {
  kUnsupported = 0,
  kCuda = 1,
  kRocm = 2,
};

/** Returns the fusion backend compiled into this native library. */
FusionBackend GetFusionBackend();

FusionPlan* PrepareFusionPlan(
    c10::Device device, const int64_t* descriptor, std::size_t descriptor_size);
FusionExecutable* BindFusionPlan(const FusionPlan* plan,
    const int64_t* constant_handles, std::size_t constant_count);
FusionSession* NewFusionSession(const FusionExecutable* executable, int32_t buffer_count);
torch::Tensor GetFusionSessionOutput(
    const FusionSession* session, int32_t buffer_index, int32_t output_index);
void SubmitFusion(FusionSession* session, int32_t buffer_index,
    const int64_t* input_handles, std::size_t input_count,
    const int64_t* dimensions, std::size_t dimension_count);
void SynchronizeFusionOutput(FusionSession* session, int32_t buffer_index);

void DeleteFusionPlan(FusionPlan* plan);
void DeleteFusionExecutable(FusionExecutable* executable);
void DeleteFusionSession(FusionSession* session);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_H
