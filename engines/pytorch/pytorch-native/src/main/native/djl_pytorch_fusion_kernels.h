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
#ifndef DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
#define DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H

#include <torch/torch.h>

#include <cstdint>

namespace djl::pytorch::fusion {

inline constexpr int32_t kMaximumOutputPackSources = 32;

struct OutputPackSource {
  const void* data;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
};

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width);

}  // namespace djl::pytorch::fusion

#endif  // DJL_TORCH_DJL_PYTORCH_FUSION_KERNELS_H
