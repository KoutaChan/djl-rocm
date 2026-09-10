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
#include <djl/utils.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_jni_log.h"
#include "djl_pytorch_tensor_copy.h"

#include <utility>
#include <vector>

namespace {
std::vector<torch::Tensor> Tensors(JNIEnv* env, jlongArray handles) {
  const auto values = djl::utils::jni::GetVecFromJLongArray(env, handles);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(values.size());
  for (const auto value : values) {
    TORCH_CHECK(value != djl::utils::jni::NULL_PTR,
        "Tensor copy plan tensor handles must not be null.");
    tensors.push_back(*reinterpret_cast<torch::Tensor*>(value));
  }
  return tensors;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateTensorCopyPlan(
    JNIEnv* env, jobject jthis, jlongArray sources, jlongArray destinations) {
  API_BEGIN()
  (void) jthis;
  return reinterpret_cast<uintptr_t>(djl::pytorch::NewTensorCopyPlan(
      Tensors(env, sources), Tensors(env, destinations)));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTensorCopyPlanCopy(
    JNIEnv* env, jobject jthis, jlong handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::CopyTensorCopyPlan(
      reinterpret_cast<djl::pytorch::TensorCopyPlan*>(handle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteTensorCopyPlan(
    JNIEnv* env, jobject jthis, jlong handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::DeleteTensorCopyPlan(
      reinterpret_cast<djl::pytorch::TensorCopyPlan*>(handle));
  API_END()
}
