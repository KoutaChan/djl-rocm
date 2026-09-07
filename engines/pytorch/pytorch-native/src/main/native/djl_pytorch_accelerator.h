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
#include <vector>

namespace djl_pytorch {
namespace accel {

struct CopyEvent;
struct DeviceEvent;
struct DeviceStream;
struct HostBuffer;
struct AcceleratorGraph;
struct StreamScope;

struct DeviceMemoryStats {
  int64_t allocated_bytes;
  int64_t peak_allocated_bytes;
  int64_t reserved_bytes;
  int64_t peak_reserved_bytes;
  int64_t active_bytes;
  int64_t peak_active_bytes;
  int64_t inactive_split_bytes;
  int64_t num_alloc_retries;
  int64_t num_ooms;
};

struct AllocatorStreamPool {
  uint64_t stream_id;
  uint64_t pool_id_high;
  uint64_t pool_id_low;
  bool is_large;
  int64_t reserved_bytes = 0;
  int64_t allocated_bytes = 0;
  int64_t active_bytes = 0;
  int64_t largest_inactive_block_bytes = 0;
  int64_t segment_count = 0;
};

bool IsAvailable();

HostBuffer* AllocateHostBuffer(int64_t size, torch::ScalarType dtype);
void* GetHostBufferData(HostBuffer* buffer);
int64_t GetHostBufferSize(HostBuffer* buffer);
bool IsHostBufferPinned(HostBuffer* buffer);
void DeleteHostBuffer(HostBuffer* buffer);

void CopyFromHost(torch::Tensor& target, void* data, bool non_blocking = false);
void CopyFromHost(torch::Tensor& target, HostBuffer* buffer, bool non_blocking = false);
void EnqueueCopyFrom(torch::Tensor& target, HostBuffer* buffer);
void EnqueueCopyTo(const torch::Tensor& source, HostBuffer* buffer);
CopyEvent* CopyFromHostAsync(torch::Tensor& target, HostBuffer* buffer);
CopyEvent* CopyToHostAsync(const torch::Tensor& source, HostBuffer* buffer);
void SynchronizeCopyEvent(CopyEvent* event);
void DeleteCopyEvent(CopyEvent* event);
void RecordStream(const torch::Tensor& tensor);

StreamScope* NewStreamScope();
StreamScope* NewStreamScope(c10::Device device);
DeviceStream* NewDeviceStream(c10::Device device);
StreamScope* OpenDeviceStream(DeviceStream* stream);
uint64_t GetStreamId(DeviceStream* stream);
void DeleteDeviceStream(DeviceStream* stream);
void DeleteStreamScope(StreamScope* scope);

DeviceEvent* NewDeviceEvent(c10::Device device);
void RecordDeviceEvent(DeviceEvent* event);
void WaitDeviceEvent(DeviceEvent* event);
bool QueryDeviceEvent(DeviceEvent* event);
void SynchronizeDeviceEvent(DeviceEvent* event);
void DeleteDeviceEvent(DeviceEvent* event);

AcceleratorGraph* NewAcceleratorGraph(c10::Device device);
void BeginAcceleratorGraphCapture(AcceleratorGraph* graph);
void EndAcceleratorGraphCapture(AcceleratorGraph* graph);
void ReplayAcceleratorGraph(AcceleratorGraph* graph);
void DeleteAcceleratorGraph(AcceleratorGraph* graph);

DeviceMemoryStats GetMemoryStats(c10::DeviceIndex device);
std::vector<AllocatorStreamPool> GetAllocatorSnapshot(c10::DeviceIndex device);
void ResetPeakMemoryStats(c10::DeviceIndex device);
void EmptyCache();

}  // namespace accel
}  // namespace djl_pytorch

#endif  // DJL_TORCH_DJL_PYTORCH_ACCELERATOR_H
