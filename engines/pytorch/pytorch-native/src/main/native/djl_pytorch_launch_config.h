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
#ifndef DJL_PYTORCH_LAUNCH_CONFIG_H
#define DJL_PYTORCH_LAUNCH_CONFIG_H

namespace djl::pytorch::launch_environment {

inline constexpr char kFusionScratchPlanner[] = "DJL_FUSION_SCRATCH_PLANNER";
inline constexpr char kFusionIntermediatePlanner[] = "DJL_FUSION_INTERMEDIATE_PLANNER";
inline constexpr char kFusionInPlacePlanner[] = "DJL_FUSION_INPLACE_PLANNER";

/** Returns a default-on boolean setting loaded from the process environment. */
bool IsPlannerEnabled(const char* name);

}  // namespace djl::pytorch::launch_environment

#endif  // DJL_PYTORCH_LAUNCH_CONFIG_H
