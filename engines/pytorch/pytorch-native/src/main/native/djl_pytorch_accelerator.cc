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

#if defined(DJL_USE_ACCELERATOR_GRAPH) && defined(USE_ROCM)
#include <ATen/hip/HIPGraph.h>
#elif defined(DJL_USE_ACCELERATOR_GRAPH)
#include <ATen/cuda/CUDAGraph.h>
#endif

#include <ATen/Context.h>
#if __has_include(<ATen/DeviceAccelerator.h>)
#include <ATen/DeviceAccelerator.h>
#define DJL_HAS_DEVICE_ACCELERATOR 1
#else
#define DJL_HAS_DEVICE_ACCELERATOR 0
#endif

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <exception>
#include <memory>
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

struct DeviceEvent {
  c10::Device device;
  c10::Event event;

  explicit DeviceEvent(c10::Device device) : device(device), event(device.type()) {}
};

struct DeviceStream {
  c10::Stream stream;

  explicit DeviceStream(c10::Stream stream) : stream(stream) {}
};

struct StreamScope {
  c10::Stream stream;
  c10::StreamGuard guard;

  explicit StreamScope(c10::Stream stream) : stream(stream), guard(this->stream) {}
};

namespace {

c10::Stream GetCurrentStream(c10::Device device) {
  c10::DeviceGuard device_guard(device);
  c10::impl::VirtualGuardImpl guard_impl(device.type());
  return guard_impl.getStream(device);
}

void CheckBuffer(const torch::Tensor& tensor, const HostBuffer* buffer) {
  TORCH_CHECK(buffer != nullptr, "host buffer must not be null");
  TORCH_CHECK(tensor.layout() == c10::kStrided, "pinned host transfer requires a strided tensor");
  TORCH_CHECK(tensor.scalar_type() == buffer->storage.scalar_type(),
      "tensor and host buffer data types must match");
  TORCH_CHECK(tensor.numel() <= buffer->storage.numel(), "host buffer is smaller than the tensor");
}

void CopyFromBuffer(torch::Tensor& target, HostBuffer* buffer, bool non_blocking) {
  torch::Tensor source = buffer->storage.narrow(0, 0, target.numel()).view(target.sizes());
  target.copy_(source, non_blocking);
}

[[noreturn]] void DrainStreamAndRethrow(
    const c10::Stream& stream, const std::exception_ptr& failure) {
  try {
    stream.synchronize();
  } catch (...) {
    // The original exception describes the operation that failed. If the
    // accelerator cannot drain its stream, callers must treat the device as
    // failed; replacing the original exception would not make the transfer
    // recoverable.
  }
  std::rethrow_exception(failure);
}

}  // namespace

#if defined(DJL_USE_ACCELERATOR_GRAPH)
struct AcceleratorGraph {
  c10::Device device;
  c10::Stream stream;
  at::cuda::CUDAGraph graph;
  std::unique_ptr<c10::StreamGuard> capture_guard;

  AcceleratorGraph(c10::Device device, c10::Stream stream)
      : device(device), stream(stream), graph(false) {}
};
#else
struct AcceleratorGraph {};
#endif

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

namespace {

void InitializeAccelerator() {
  TORCH_CHECK(IsAvailable(), "accelerator is unavailable");
  at::globalContext().lazyInitDevice(GetAcceleratorType().value());
}

}  // namespace

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
  CheckBuffer(target, buffer);
  CopyFromBuffer(target, buffer, non_blocking);
}

void EnqueueCopyFrom(torch::Tensor& target, HostBuffer* buffer) {
  CheckBuffer(target, buffer);
  if (!IsAcceleratorDevice(target.device()) || !buffer->pinned) {
    CopyFromBuffer(target, buffer, false);
    return;
  }

  c10::Stream stream = GetCurrentStream(target.device());
  CopyFromBuffer(target, buffer, true);
  c10::impl::VirtualGuardImpl guard_impl(target.device().type());
  guard_impl.recordDataPtrOnStream(target.storage().data_ptr(), stream);
}

void EnqueueCopyTo(const torch::Tensor& source, HostBuffer* buffer) {
  CheckBuffer(source, buffer);
  torch::Tensor target = buffer->storage.narrow(0, 0, source.numel()).view(source.sizes());
  if (!IsAcceleratorDevice(source.device()) || !buffer->pinned) {
    target.copy_(source);
    return;
  }

  c10::Stream stream = GetCurrentStream(source.device());
  target.copy_(source, true);
  c10::impl::VirtualGuardImpl guard_impl(source.device().type());
  guard_impl.recordDataPtrOnStream(source.storage().data_ptr(), stream);
}

CopyEvent* CopyFromHostAsync(torch::Tensor& target, HostBuffer* buffer) {
  CheckBuffer(target, buffer);
  if (!IsAcceleratorDevice(target.device()) || !buffer->pinned) {
    CopyFromBuffer(target, buffer, false);
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
  auto event = std::make_unique<CopyEvent>(target.device().type());
  try {
    CopyFromBuffer(target, buffer, true);
    guard_impl.recordDataPtrOnStream(target.storage().data_ptr(), stream);
    event->event.record(stream);
    if (stream != alloc_stream) {
      // The consumer continues on the allocation stream. Queue its dependency on
      // the asynchronous host copy without synchronizing the CPU.
      event->event.block(alloc_stream);
    }
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    DrainStreamAndRethrow(stream, failure);
  }
  return event.release();
}

CopyEvent* CopyToHostAsync(const torch::Tensor& source, HostBuffer* buffer) {
  CheckBuffer(source, buffer);
  torch::Tensor target = buffer->storage.narrow(0, 0, source.numel()).view(source.sizes());
  if (!IsAcceleratorDevice(source.device()) || !buffer->pinned) {
    target.copy_(source);
    return nullptr;
  }

  c10::DeviceGuard device_guard(source.device());
  c10::impl::VirtualGuardImpl guard_impl(source.device().type());
  c10::Stream producer_stream = guard_impl.getStream(source.device());
  c10::Stream copy_stream = guard_impl.getStreamFromGlobalPool(source.device());
  if (copy_stream != producer_stream) {
    c10::Event dependency(source.device().type());
    dependency.record(producer_stream);
    dependency.block(copy_stream);
  }
  c10::StreamGuard stream_guard(copy_stream);
  auto event = std::make_unique<CopyEvent>(source.device().type());
  try {
    target.copy_(source, true);
    guard_impl.recordDataPtrOnStream(source.storage().data_ptr(), copy_stream);
    event->event.record(copy_stream);
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    DrainStreamAndRethrow(copy_stream, failure);
  }
  return event.release();
}

void SynchronizeCopyEvent(CopyEvent* event) {
  event->event.synchronize();
}

void DeleteCopyEvent(CopyEvent* event) {
  delete event;
}

void RecordStream(const torch::Tensor& tensor) {
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
  return NewStreamScope(guard_impl.getDevice());
}

StreamScope* NewStreamScope(c10::Device device) {
  if (!IsAcceleratorDevice(device)) {
    return nullptr;
  }
  c10::impl::VirtualGuardImpl guard_impl(device.type());
  c10::Stream stream = guard_impl.getStreamFromGlobalPool(device);
  return new StreamScope(stream);
}

DeviceStream* NewDeviceStream(c10::Device device) {
  if (!IsAcceleratorDevice(device)) {
    return nullptr;
  }
  c10::impl::VirtualGuardImpl guard_impl(device.type());
  return new DeviceStream(guard_impl.getStreamFromGlobalPool(device));
}

StreamScope* OpenDeviceStream(DeviceStream* stream) {
  if (stream == nullptr) {
    return nullptr;
  }
  return new StreamScope(stream->stream);
}

void DeleteDeviceStream(DeviceStream* stream) {
  delete stream;
}

void DeleteStreamScope(StreamScope* scope) {
  delete scope;
}

DeviceEvent* NewDeviceEvent(c10::Device device) {
  if (!IsAcceleratorDevice(device)) {
    return nullptr;
  }
  return new DeviceEvent(device);
}

void RecordDeviceEvent(DeviceEvent* event) {
  if (event == nullptr) {
    return;
  }
  event->event.record(GetCurrentStream(event->device));
}

void WaitDeviceEvent(DeviceEvent* event) {
  if (event == nullptr) {
    return;
  }
  TORCH_CHECK(event->event.was_marked_for_recording(), "cannot wait for an event before it is recorded");
  event->event.block(GetCurrentStream(event->device));
}

bool QueryDeviceEvent(DeviceEvent* event) {
  return event == nullptr || event->event.query();
}

void SynchronizeDeviceEvent(DeviceEvent* event) {
  if (event != nullptr && event->event.was_marked_for_recording()) {
    event->event.synchronize();
  }
}

void DeleteDeviceEvent(DeviceEvent* event) {
  delete event;
}

AcceleratorGraph* NewAcceleratorGraph(c10::Device device) {
#if defined(DJL_USE_ACCELERATOR_GRAPH)
  TORCH_CHECK(IsAcceleratorDevice(device), "accelerator graph requires an accelerator device");
  c10::DeviceGuard device_guard(device);
  c10::impl::VirtualGuardImpl guard_impl(device.type());
  return new AcceleratorGraph(device, guard_impl.getStreamFromGlobalPool(device));
#else
  TORCH_CHECK(false, "accelerator graph is unavailable in this build");
#endif
}

void BeginAcceleratorGraphCapture(AcceleratorGraph* graph) {
#if defined(DJL_USE_ACCELERATOR_GRAPH)
  c10::DeviceGuard device_guard(graph->device);
  c10::impl::VirtualGuardImpl guard_impl(graph->device.type());
  c10::Stream caller_stream = guard_impl.getStream(graph->device);
  if (caller_stream != graph->stream) {
    c10::Event ready(graph->device.type());
    ready.record(caller_stream);
    ready.block(graph->stream);
  }
  graph->capture_guard = std::make_unique<c10::StreamGuard>(graph->stream);
#if defined(USE_ROCM)
  graph->graph.capture_begin({0, 0}, hipStreamCaptureModeThreadLocal);
#else
  graph->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
#endif
#else
  TORCH_CHECK(false, "accelerator graph is unavailable in this build");
#endif
}

void EndAcceleratorGraphCapture(AcceleratorGraph* graph) {
#if defined(DJL_USE_ACCELERATOR_GRAPH)
  c10::DeviceGuard device_guard(graph->device);
  graph->graph.capture_end();
  graph->capture_guard.reset();
#else
  TORCH_CHECK(false, "accelerator graph is unavailable in this build");
#endif
}

void ReplayAcceleratorGraph(AcceleratorGraph* graph) {
#if defined(DJL_USE_ACCELERATOR_GRAPH)
  c10::DeviceGuard device_guard(graph->device);
  c10::impl::VirtualGuardImpl guard_impl(graph->device.type());
  c10::Stream caller_stream = guard_impl.getStream(graph->device);
  if (caller_stream != graph->stream) {
    c10::Event ready(graph->device.type());
    ready.record(caller_stream);
    ready.block(graph->stream);
  }
  {
    c10::StreamGuard graph_guard(graph->stream);
    graph->graph.replay();
    if (caller_stream != graph->stream) {
      c10::Event complete(graph->device.type());
      complete.record(graph->stream);
      complete.block(caller_stream);
    }
  }
#else
  TORCH_CHECK(false, "accelerator graph is unavailable in this build");
#endif
}

void DeleteAcceleratorGraph(AcceleratorGraph* graph) {
#if defined(DJL_USE_ACCELERATOR_GRAPH)
  if (graph != nullptr) {
    c10::DeviceGuard device_guard(graph->device);
    delete graph;
  }
#else
  delete graph;
#endif
}

DeviceMemoryStats GetMemoryStats(c10::DeviceIndex device) {
  InitializeAccelerator();
  c10::CachingDeviceAllocator::DeviceStats stats;
#if DJL_HAS_DEVICE_ACCELERATOR
  stats = at::accelerator::getDeviceStats(device);
#else
  stats = at::getDeviceAllocator(c10::DeviceType::CUDA)->getDeviceStats(device);
#endif
  constexpr auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  const auto& allocated = stats.allocated_bytes[aggregate];
  const auto& reserved = stats.reserved_bytes[aggregate];
  const auto& active = stats.active_bytes[aggregate];
  return {allocated.current, allocated.peak, reserved.current, reserved.peak, active.current, active.peak};
}

void ResetPeakMemoryStats(c10::DeviceIndex device) {
  InitializeAccelerator();
#if DJL_HAS_DEVICE_ACCELERATOR
  at::accelerator::resetPeakStats(device);
#else
  at::getDeviceAllocator(c10::DeviceType::CUDA)->resetPeakStats(device);
#endif
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
