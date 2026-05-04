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

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

#if !defined(__ANDROID__) && defined(USE_CUDA)
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Bucket {
  std::vector<size_t> parameter_indices;
  std::vector<at::Tensor> views;
  std::vector<at::Tensor> tensors;
  std::vector<bool> received;
  at::Tensor buffer;
  c10::intrusive_ptr<c10d::Work> work;
  size_t pending{0};
  bool ready{false};
  bool launched{false};
};

struct ParameterLocation {
  size_t bucket_index{0};
  size_t view_index{0};
};

class NativeDistributedReducer : public std::enable_shared_from_this<NativeDistributedReducer> {
 public:
  NativeDistributedReducer(std::vector<at::Tensor> parameters, std::string master_host, int master_port, int rank,
      int world_size, int bucket_cap_mb, bool static_graph, bool average_gradients)
      : parameters_(std::move(parameters)),
        master_host_(std::move(master_host)),
        master_port_(master_port),
        rank_(rank),
        world_size_(world_size),
        bucket_cap_mb_(bucket_cap_mb),
        static_graph_(static_graph),
        average_gradients_(average_gradients) {}

  void Initialize() {
    (void) static_graph_;
    InitializeProcessGroup();
    BroadcastParameters();
    SelectTrainableParameters();
    parameter_locations_.resize(parameters_.size());
    BuildBuckets();
    RegisterHooks();
  }

  void PrepareForBackward() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (expecting_backward_ && !finalized_) {
      throw std::runtime_error("Previous distributed backward pass has not been finalized.");
    }
    for (auto& bucket : buckets_) {
      bucket.work.reset();
      bucket.pending = bucket.parameter_indices.size();
      bucket.ready = false;
      bucket.launched = false;
      std::fill(bucket.received.begin(), bucket.received.end(), false);
    }
    next_bucket_to_launch_ = 0;
    expecting_backward_ = true;
    finalized_ = false;
  }

  void MarkReady(size_t parameter_index, const at::Tensor& grad) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!expecting_backward_) {
      return;
    }
    const auto& location = parameter_locations_[parameter_index];
    auto& bucket = buckets_[location.bucket_index];
    if (bucket.received[location.view_index]) {
      return;
    }
    if (!grad.defined()) {
      throw std::runtime_error("Distributed reducer received an undefined gradient.");
    }
    {
      torch::NoGradGuard no_grad;
      bucket.views[location.view_index].copy_(grad.detach());
    }
    bucket.received[location.view_index] = true;
    --bucket.pending;
    if (bucket.pending == 0) {
      bucket.ready = true;
    }
    LaunchReadyBuckets();
  }

  void FinalizeBackward() {
    std::vector<c10::intrusive_ptr<c10d::Work>> works;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!expecting_backward_ || finalized_) {
        return;
      }
      LaunchReadyBuckets();
      works.reserve(buckets_.size());
      for (const auto& bucket : buckets_) {
        if (!bucket.launched) {
          throw std::runtime_error(
              "Some distributed gradients were not produced. "
              "Unused parameters are not supported by the native reducer yet.");
        }
        works.push_back(bucket.work);
      }
    }

    for (const auto& work : works) {
      work->wait();
    }

    std::lock_guard<std::mutex> guard(mutex_);
    torch::NoGradGuard no_grad;
    for (auto& bucket : buckets_) {
      if (average_gradients_) {
        bucket.buffer.div_(world_size_);
      }
      for (size_t i = 0; i < bucket.parameter_indices.size(); ++i) {
        auto& parameter = parameters_[bucket.parameter_indices[i]];
        auto grad = parameter.grad();
        if (!grad.defined()) {
          throw std::runtime_error("Distributed reducer cannot write an undefined parameter gradient.");
        }
        grad.copy_(bucket.views[i]);
      }
    }
    expecting_backward_ = false;
    finalized_ = true;
  }

 private:
  void InitializeProcessGroup() {
    c10d::TCPStoreOptions store_options;
    store_options.port = static_cast<std::uint16_t>(master_port_);
    store_options.isServer = rank_ == 0;
    store_options.numWorkers = static_cast<size_t>(world_size_);
    store_options.waitWorkers = true;
    store_options.timeout = std::chrono::milliseconds(300000);

    store_ = c10::make_intrusive<c10d::TCPStore>(master_host_, store_options);
    auto options = c10d::ProcessGroupNCCL::Options::create();
    process_group_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(store_, rank_, world_size_, options);
  }

  void BroadcastParameters() {
    c10d::BroadcastOptions options;
    options.rootRank = 0;
    options.rootTensor = 0;
    for (auto& parameter : parameters_) {
      if (!parameter.is_cuda()) {
        throw std::runtime_error("Native distributed training requires CUDA or ROCm tensors.");
      }
      if (parameter.is_sparse()) {
        throw std::runtime_error("Native distributed training does not support sparse parameters.");
      }
      std::vector<at::Tensor> tensors{parameter};
      process_group_->broadcast(tensors, options)->wait();
    }
  }

  void SelectTrainableParameters() {
    std::vector<at::Tensor> trainable;
    trainable.reserve(parameters_.size());
    for (auto& parameter : parameters_) {
      if (parameter.requires_grad()) {
        trainable.push_back(parameter);
      }
    }
    parameters_ = std::move(trainable);
    if (parameters_.empty()) {
      throw std::runtime_error("No parameters require gradients for native distributed training.");
    }
  }

  void BuildBuckets() {
    const int64_t max_bucket_bytes = static_cast<int64_t>(bucket_cap_mb_) * 1024 * 1024;
    std::vector<size_t> order(parameters_.size());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());

    std::vector<size_t> pending_indices;
    int64_t pending_bytes = 0;
    c10::ScalarType pending_dtype = parameters_[order.front()].scalar_type();
    c10::Device pending_device = parameters_[order.front()].device();

    auto flush_bucket = [&]() {
      if (pending_indices.empty()) {
        return;
      }
      AddBucket(pending_indices, pending_bytes);
      pending_indices.clear();
      pending_bytes = 0;
    };

    for (size_t parameter_index : order) {
      const auto& parameter = parameters_[parameter_index];
      if (!parameter.is_cuda()) {
        throw std::runtime_error("Native distributed training requires CUDA or ROCm tensors.");
      }
      if (parameter.is_sparse()) {
        throw std::runtime_error("Native distributed training does not support sparse gradients.");
      }
      const int64_t parameter_bytes = parameter.numel() * parameter.element_size();
      const bool same_type = parameter.scalar_type() == pending_dtype;
      const bool same_device = parameter.device() == pending_device;
      if (!pending_indices.empty()
          && (!same_type || !same_device || pending_bytes + parameter_bytes > max_bucket_bytes)) {
        flush_bucket();
      }
      if (pending_indices.empty()) {
        pending_dtype = parameter.scalar_type();
        pending_device = parameter.device();
      }
      pending_indices.push_back(parameter_index);
      pending_bytes += parameter_bytes;
    }
    flush_bucket();
  }

  void AddBucket(const std::vector<size_t>& parameter_indices, int64_t bucket_bytes) {
    if (bucket_bytes <= 0) {
      throw std::runtime_error("Invalid empty distributed gradient bucket.");
    }
    const auto& first = parameters_[parameter_indices.front()];
    const int64_t element_size = first.element_size();
    const int64_t num_elements = bucket_bytes / element_size;
    auto options = first.options().requires_grad(false);
    Bucket bucket;
    bucket.parameter_indices = parameter_indices;
    bucket.buffer = torch::empty({num_elements}, options);
    bucket.tensors = {bucket.buffer};
    bucket.received.resize(parameter_indices.size(), false);

    int64_t offset = 0;
    for (size_t parameter_index : parameter_indices) {
      const auto& parameter = parameters_[parameter_index];
      const int64_t numel = parameter.numel();
      bucket.views.push_back(bucket.buffer.narrow(0, offset, numel).view(parameter.sizes()));
      offset += numel;
    }

    const size_t bucket_index = buckets_.size();
    for (size_t i = 0; i < parameter_indices.size(); ++i) {
      parameter_locations_[parameter_indices[i]] = ParameterLocation{bucket_index, i};
    }
    buckets_.push_back(std::move(bucket));
  }

  void RegisterHooks() {
    auto weak_this = weak_from_this();
    for (size_t i = 0; i < parameters_.size(); ++i) {
      parameters_[i].register_hook([weak_this, i](const at::Tensor& grad) -> at::Tensor {
        if (auto reducer = weak_this.lock()) {
          reducer->MarkReady(i, grad);
        }
        return grad;
      });
    }
  }

  void LaunchReadyBuckets() {
    c10d::AllreduceOptions options;
    options.reduceOp = c10d::ReduceOp::SUM;
    while (next_bucket_to_launch_ < buckets_.size()) {
      auto& bucket = buckets_[next_bucket_to_launch_];
      if (!bucket.ready) {
        return;
      }
      bucket.work = process_group_->allreduce(bucket.tensors, options);
      bucket.launched = true;
      ++next_bucket_to_launch_;
    }
  }

  std::vector<at::Tensor> parameters_;
  std::string master_host_;
  int master_port_;
  int rank_;
  int world_size_;
  int bucket_cap_mb_;
  bool static_graph_;
  bool average_gradients_;
  c10::intrusive_ptr<c10d::Store> store_;
  c10::intrusive_ptr<c10d::ProcessGroupNCCL> process_group_;
  std::vector<Bucket> buckets_;
  std::vector<ParameterLocation> parameter_locations_;
  size_t next_bucket_to_launch_{0};
  bool expecting_backward_{false};
  bool finalized_{true};
  std::mutex mutex_;
};

using ReducerHandle = std::shared_ptr<NativeDistributedReducer>;

ReducerHandle& GetReducer(jlong jhandle) {
  if (jhandle == djl::utils::jni::NULL_PTR) {
    throw std::runtime_error("Distributed reducer has already been closed.");
  }
  return *reinterpret_cast<ReducerHandle*>(jhandle);
}

}  // namespace
#endif

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_distributedCreateReducer(JNIEnv* env, jobject jthis,
    jlongArray jparameter_handles, jstring jmaster_host, jint jmaster_port, jint jrank, jint jworld_size,
    jint jlocal_rank, jint jbucket_cap_mb, jboolean jstatic_graph, jboolean jfind_unused_parameters,
    jboolean javerage_gradients) {
  API_BEGIN()
#if !defined(__ANDROID__) && defined(USE_CUDA)
  if (jfind_unused_parameters == JNI_TRUE) {
    throw std::runtime_error("findUnusedParameters is not supported by the PyTorch native reducer yet.");
  }
  (void) jthis;
  (void) jlocal_rank;
  const auto parameter_handles = djl::utils::jni::GetVecFromJLongArray(env, jparameter_handles);
  std::vector<at::Tensor> parameters;
  parameters.reserve(parameter_handles.size());
  for (const auto handle : parameter_handles) {
    parameters.push_back(*reinterpret_cast<at::Tensor*>(handle));
  }
  const char* master_host_chars = env->GetStringUTFChars(jmaster_host, nullptr);
  std::string master_host(master_host_chars);
  env->ReleaseStringUTFChars(jmaster_host, master_host_chars);

  auto reducer = std::make_shared<NativeDistributedReducer>(std::move(parameters), std::move(master_host), jmaster_port,
      jrank, jworld_size, jbucket_cap_mb, jstatic_graph == JNI_TRUE, javerage_gradients == JNI_TRUE);
  reducer->Initialize();
  auto* handle = new ReducerHandle(std::move(reducer));
  return reinterpret_cast<uintptr_t>(handle);
#else
  (void) jthis;
  (void) jparameter_handles;
  (void) jmaster_host;
  (void) jmaster_port;
  (void) jrank;
  (void) jworld_size;
  (void) jlocal_rank;
  (void) jbucket_cap_mb;
  (void) jstatic_graph;
  (void) jfind_unused_parameters;
  (void) javerage_gradients;
  throw std::runtime_error("Native distributed training requires a CUDA or ROCm PyTorch build.");
#endif
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_distributedPrepareForBackward(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
#if !defined(__ANDROID__) && defined(USE_CUDA)
  (void) jthis;
  GetReducer(jhandle)->PrepareForBackward();
#else
  (void) jthis;
  (void) jhandle;
  throw std::runtime_error("Native distributed training requires a CUDA or ROCm PyTorch build.");
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_distributedFinalizeBackward(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
#if !defined(__ANDROID__) && defined(USE_CUDA)
  (void) jthis;
  GetReducer(jhandle)->FinalizeBackward();
#else
  (void) jthis;
  (void) jhandle;
  throw std::runtime_error("Native distributed training requires a CUDA or ROCm PyTorch build.");
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_distributedDeleteReducer(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
#if !defined(__ANDROID__) && defined(USE_CUDA)
  (void) jthis;
  auto* handle = reinterpret_cast<ReducerHandle*>(jhandle);
  delete handle;
#else
  (void) jthis;
  (void) jhandle;
  throw std::runtime_error("Native distributed training requires a CUDA or ROCm PyTorch build.");
#endif
  API_END()
}
