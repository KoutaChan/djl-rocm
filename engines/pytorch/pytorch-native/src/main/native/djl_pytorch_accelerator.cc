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
#include "djl_pytorch_accelerator.h"

#if __has_include(<ATen/DeviceAccelerator.h>)
#include <ATen/DeviceAccelerator.h>
#define DJL_HAS_DEVICE_ACCELERATOR 1
#else
#include <ATen/Context.h>
#define DJL_HAS_DEVICE_ACCELERATOR 0
#endif

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <optional>

namespace djl_pytorch {
namespace accel {

struct HostBuffer {
  torch::Tensor storage;
  bool pinned;
};

struct CopyEvent {
  c10::Event event;

  explicit CopyEvent(c10::DeviceType device_type) : event(device_type) {}
};

struct StreamScope {
  c10::Stream stream;
  c10::StreamGuard guard;

  explicit StreamScope(c10::Stream stream) : stream(stream), guard(this->stream) {}
};

std::optional<c10::DeviceType> GetAcceleratorType() {
#if DJL_HAS_DEVICE_ACCELERATOR
  return at::accelerator::getAccelerator();
#else
  if (torch::cuda::is_available()) {
    return c10::DeviceType::CUDA;
  }
  return std::nullopt;
#endif
}

bool IsAvailable() {
#if DJL_HAS_DEVICE_ACCELERATOR
  return GetAcceleratorType().has_value() && at::accelerator::deviceCount() > 0;
#else
  return torch::cuda::is_available();
#endif
}

bool IsAcceleratorDevice(c10::Device device) {
#if DJL_HAS_DEVICE_ACCELERATOR
  return at::accelerator::isAccelerator(device.type());
#else
  return device.is_cuda();
#endif
}

HostBuffer* AllocateHostBuffer(int64_t size, torch::ScalarType dtype) {
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(IsAvailable());
  torch::Tensor storage = torch::empty({size}, options);
  return new HostBuffer{storage, storage.is_pinned()};
}

void* GetHostBufferData(HostBuffer* buffer) {
  return buffer->storage.data_ptr();
}

int64_t GetHostBufferSize(HostBuffer* buffer) {
  return buffer->storage.numel() * buffer->storage.dtype().itemsize();
}

bool IsHostBufferPinned(HostBuffer* buffer) {
  return buffer->pinned;
}

void DeleteHostBuffer(HostBuffer* buffer) {
  delete buffer;
}

void CopyFromHost(torch::Tensor& target, void* data, bool non_blocking) {
  auto options = torch::TensorOptions().dtype(target.dtype()).requires_grad(false);
  torch::Tensor source = torch::from_blob(data, target.sizes(), options);
  target.copy_(source, non_blocking);
}

void CopyFromHost(torch::Tensor& target, HostBuffer* buffer, bool non_blocking) {
  torch::Tensor source = buffer->storage.narrow(0, 0, target.numel()).view(target.sizes());
  target.copy_(source, non_blocking);
}

CopyEvent* CopyFromHostAsync(torch::Tensor& target, HostBuffer* buffer) {
  if (!IsAcceleratorDevice(target.device()) || !buffer->pinned) {
    CopyFromHost(target, buffer);
    return nullptr;
  }
  c10::DeviceGuard device_guard(target.device());
  c10::impl::VirtualGuardImpl guard_impl(target.device().type());
  c10::Stream alloc_stream = guard_impl.getStream(target.device());
  c10::Stream stream = guard_impl.getStreamFromGlobalPool(target.device());
  // The caching allocator may hand `target` a block that was freed while kernels
  // previously enqueued on the allocation (compute) stream still read or write it.
  // That reuse is only implicitly safe for work enqueued on the same stream, so the
  // pool-stream copy must wait for everything already queued on the allocation
  // stream before writing into the block.
  if (stream != alloc_stream) {
    c10::Event dependency(target.device().type());
    dependency.record(alloc_stream);
    dependency.block(stream);
  }
  c10::StreamGuard stream_guard(stream);
  CopyFromHost(target, buffer, true);
  guard_impl.recordDataPtrOnStream(target.storage().data_ptr(), stream);
  auto* event = new CopyEvent(target.device().type());
  event->event.record(stream);
  if (stream != alloc_stream) {
    // The consumer continues on the allocation stream. Queue its dependency on
    // the asynchronous host copy without synchronizing the CPU.
    event->event.block(alloc_stream);
  }
  return event;
}

void SynchronizeCopyEvent(CopyEvent* event) {
  event->event.synchronize();
}

void DeleteCopyEvent(CopyEvent* event) {
  delete event;
}

void RecordTensorUseOnCurrentStream(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0 || tensor.layout() != c10::kStrided ||
      !IsAcceleratorDevice(tensor.device())) {
    return;
  }
  c10::DeviceGuard device_guard(tensor.device());
  c10::impl::VirtualGuardImpl guard_impl(tensor.device().type());
  c10::Stream stream = guard_impl.getStream(tensor.device());
  guard_impl.recordDataPtrOnStream(tensor.storage().data_ptr(), stream);
}

StreamScope* NewStreamScope() {
  if (!IsAvailable()) {
    return nullptr;
  }
  std::optional<c10::DeviceType> device_type = GetAcceleratorType();
  c10::impl::VirtualGuardImpl guard_impl(device_type.value());
  c10::Device device = guard_impl.getDevice();
  c10::Stream stream = guard_impl.getStreamFromGlobalPool(device);
  return new StreamScope(stream);
}

void DeleteStreamScope(StreamScope* scope) {
  delete scope;
}

void EmptyCache() {
  if (!IsAvailable()) {
    return;
  }
#if DJL_HAS_DEVICE_ACCELERATOR
  at::accelerator::emptyCache();
#else
  at::getDeviceAllocator(c10::DeviceType::CUDA)->emptyCache();
#endif
}

}  // namespace accel
}  // namespace djl_pytorch
