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
#include "djl_pytorch_java_buffer.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "ai_djl_pytorch_jni_cache.h"

namespace djl::pytorch {

namespace {

// References whose storage was freed on a thread without a JNIEnv, such as an autograd device thread. The next Java
// thread that shares or releases a buffer deletes them. The list is never destroyed, so a storage freed during static
// destruction can still use it.
struct PendingReferences {
  std::mutex mutex;
  std::vector<jobject> references;
  std::atomic<bool> any{false};
};

PendingReferences& GetPendingReferences() {
  static auto* pending = new PendingReferences();
  return *pending;
}

void DeletePendingReferences(JNIEnv* env) {
  auto& pending = GetPendingReferences();
  if (!pending.any.load(std::memory_order_acquire)) {
    return;
  }
  std::vector<jobject> references;
  {
    std::lock_guard<std::mutex> lock(pending.mutex);
    references.swap(pending.references);
    pending.any.store(false, std::memory_order_relaxed);
  }
  for (auto reference : references) {
    env->DeleteGlobalRef(reference);
  }
}

void ReleaseBuffer(void* reference) noexcept {
  JNIEnv* env = nullptr;
  if (JAVA_VM->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) == JNI_OK) {
    env->DeleteGlobalRef(static_cast<jobject>(reference));
    DeletePendingReferences(env);
    return;
  }
  // Attaching the thread could block during shutdown, so the reference waits for a Java thread.
  try {
    auto& pending = GetPendingReferences();
    std::lock_guard<std::mutex> lock(pending.mutex);
    pending.references.push_back(static_cast<jobject>(reference));
    pending.any.store(true, std::memory_order_release);
  } catch (...) {
    // The buffer stays reachable, which only keeps its memory longer.
  }
}

jobject NewBufferReference(JNIEnv* env, jobject buffer) {
  DeletePendingReferences(env);
  jobject reference = env->NewGlobalRef(buffer);
  TORCH_CHECK(reference != nullptr, "Unable to keep the direct buffer reachable");
  return reference;
}

}  // namespace

torch::Tensor FromDirectBuffer(
    JNIEnv* env, jobject buffer, c10::IntArrayRef sizes, const torch::TensorOptions& options) {
  void* data = env->GetDirectBufferAddress(buffer);
  // make_tensor releases the context if it fails
  return at::for_blob(data, sizes)
      .context(NewBufferReference(env, buffer), ReleaseBuffer)
      .options(options)
      .make_tensor();
}

c10::DataPtr DirectBufferDataPtr(JNIEnv* env, jobject buffer) {
  void* data = env->GetDirectBufferAddress(buffer);
  return c10::DataPtr(data, NewBufferReference(env, buffer), ReleaseBuffer, c10::Device(c10::kCPU));
}

torch::Tensor FromArray(JNIEnv* env, jobject array, jlong offset, jlong length, c10::IntArrayRef sizes,
    const torch::TensorOptions& options) {
  torch::Tensor tensor = torch::empty(sizes, options);
  const auto nbytes = static_cast<jlong>(tensor.nbytes());
  TORCH_CHECK(nbytes <= length, "The array has ", length, " bytes of data, but the tensor needs ", nbytes);
  if (nbytes == 0) {
    return tensor;
  }
  // a plain memory copy, unlike Get<Type>ArrayRegion, which copies element by element
  void* source = env->GetPrimitiveArrayCritical(static_cast<jarray>(array), nullptr);
  if (source == nullptr) {
    return torch::Tensor();
  }
  std::memcpy(tensor.data_ptr(), static_cast<const char*>(source) + offset, nbytes);
  env->ReleasePrimitiveArrayCritical(static_cast<jarray>(array), source, JNI_ABORT);
  return tensor;
}

}  // namespace djl::pytorch
