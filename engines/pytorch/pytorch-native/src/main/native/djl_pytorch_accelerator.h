/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#ifndef DJL_TORCH_DJL_PYTORCH_ACCELERATOR_H
#define DJL_TORCH_DJL_PYTORCH_ACCELERATOR_H

#include <torch/torch.h>

#include <cstdint>

namespace djl_pytorch {
namespace accel {

struct CopyEvent;
struct HostBuffer;
struct StreamScope;

bool IsAvailable();

HostBuffer* AllocateHostBuffer(int64_t size, torch::ScalarType dtype);
void* GetHostBufferData(HostBuffer* buffer);
int64_t GetHostBufferSize(HostBuffer* buffer);
bool IsHostBufferPinned(HostBuffer* buffer);
void DeleteHostBuffer(HostBuffer* buffer);

void CopyFromHost(torch::Tensor& target, void* data, bool non_blocking = false);
void CopyFromHost(torch::Tensor& target, HostBuffer* buffer, bool non_blocking = false);
CopyEvent* CopyFromHostAsync(torch::Tensor& target, HostBuffer* buffer);
void SynchronizeCopyEvent(CopyEvent* event);
void DeleteCopyEvent(CopyEvent* event);

StreamScope* NewStreamScope();
void DeleteStreamScope(StreamScope* scope);

void EmptyCache();

}  // namespace accel
}  // namespace djl_pytorch

#endif  // DJL_TORCH_DJL_PYTORCH_ACCELERATOR_H
