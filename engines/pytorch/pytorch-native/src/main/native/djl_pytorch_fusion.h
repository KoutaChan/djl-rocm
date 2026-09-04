/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not
 * use this file except in compliance with the License. A copy of the License is
 * located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
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

struct FusionPlanStats {
  int64_t executable_storage_bytes;
  int64_t persistent_storage_bytes;
  int64_t workspace_bytes;
  int64_t exported_output_bytes;
  int64_t arena_bytes;
  int64_t planner_version;
  int64_t logical_allocation_count;
  int64_t backing_allocation_count;
  int64_t alias_view_count;
  int64_t in_place_reuse_count;
  int64_t backend_workspace_upper_bound_bytes;
};

/** Identifies the accelerator runtime backing fusion kernel launches. */
enum class FusionBackend : int32_t {
  kUnsupported = 0,
  kCuda = 1,
  kRocm = 2,
};

/** Returns the fusion backend compiled into this native library. */
FusionBackend GetFusionBackend();

FusionPlan* PrepareFusionPlan(c10::Device device,
    const int64_t* descriptor, std::size_t descriptor_size,
    const int64_t* profile_descriptor, std::size_t profile_descriptor_size);
FusionPlanStats GetFusionPlanStats(const FusionPlan* plan, int32_t variant_index);
FusionExecutable* BindFusionPlan(const FusionPlan* plan, const int64_t* constant_handles, std::size_t constant_count);
FusionSession* NewFusionSession(
    const FusionExecutable* executable, int32_t variant_index,
    int32_t output_slot_count);
torch::Tensor GetFusionSessionOutput(
    const FusionSession* session, int32_t output_slot_index, int32_t output_index);
void SubmitFusion(FusionSession* session, int32_t output_slot_index,
    const int64_t* input_handles, std::size_t input_count,
    const int64_t* dimensions, std::size_t dimension_count);
void SynchronizeFusionOutput(FusionSession* session, int32_t output_slot_index);

void DeleteFusionPlan(FusionPlan* plan);
void DeleteFusionExecutable(FusionExecutable* executable);
void DeleteFusionSession(FusionSession* session);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_H
