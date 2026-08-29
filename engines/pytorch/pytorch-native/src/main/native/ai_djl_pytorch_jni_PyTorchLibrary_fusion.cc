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
#include "djl_pytorch_fusion.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPrepareFusionPlan(
    JNIEnv* env, jobject jthis, jintArray jdevice, jobject jdescriptor) {
  API_BEGIN()
  const torch::Device device = utils::GetDeviceFromJDevice(env, jdevice);
  const auto* descriptor =
      static_cast<const int64_t*>(env->GetDirectBufferAddress(jdescriptor));
  const jlong capacity = env->GetDirectBufferCapacity(jdescriptor);
  TORCH_CHECK(capacity >= 0 && capacity % sizeof(int64_t) == 0 &&
          (capacity == 0 || descriptor != nullptr),
      "fusion descriptor must be a direct long buffer");
  return reinterpret_cast<uintptr_t>(djl::pytorch::fusion::PrepareFusionPlan(
      device, descriptor, static_cast<std::size_t>(capacity) / sizeof(int64_t)));
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchBindFusionPlan(
    JNIEnv* env, jobject jthis, jlong jplan, jobject jconstant_handles) {
  API_BEGIN()
  const auto* constant_handles =
      static_cast<const int64_t*>(env->GetDirectBufferAddress(jconstant_handles));
  const jlong capacity = env->GetDirectBufferCapacity(jconstant_handles);
  TORCH_CHECK(capacity >= 0 && capacity % sizeof(int64_t) == 0 &&
          (capacity == 0 || constant_handles != nullptr),
      "fusion constant handles must be a direct long buffer");
  return reinterpret_cast<uintptr_t>(djl::pytorch::fusion::BindFusionPlan(
      reinterpret_cast<const djl::pytorch::fusion::FusionPlan*>(jplan),
      constant_handles, static_cast<std::size_t>(capacity) / sizeof(int64_t)));
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateFusionSession(
    JNIEnv* env, jobject jthis, jlong jexecutable, jint jbuffer_count) {
  API_BEGIN()
  return reinterpret_cast<uintptr_t>(djl::pytorch::fusion::NewFusionSession(
      reinterpret_cast<const djl::pytorch::fusion::FusionExecutable*>(jexecutable),
      jbuffer_count));
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetFusionSessionOutput(
    JNIEnv* env, jobject jthis, jlong jsession, jint jbuffer_index, jint joutput_index) {
  API_BEGIN()
  torch::Tensor output = djl::pytorch::fusion::GetFusionSessionOutput(
      reinterpret_cast<const djl::pytorch::fusion::FusionSession*>(jsession),
      jbuffer_index, joutput_index);
  return reinterpret_cast<uintptr_t>(new torch::Tensor(std::move(output)));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSubmitFusion(
    JNIEnv* env, jobject jthis, jlong jsession, jint jbuffer_index,
    jobject jinput_handles, jobject jdimensions) {
  API_BEGIN()
  const auto* input_handles =
      static_cast<const int64_t*>(env->GetDirectBufferAddress(jinput_handles));
  const auto* dimensions =
      static_cast<const int64_t*>(env->GetDirectBufferAddress(jdimensions));
  const jlong input_capacity = env->GetDirectBufferCapacity(jinput_handles);
  const jlong dimension_capacity = env->GetDirectBufferCapacity(jdimensions);
  TORCH_CHECK(input_capacity >= 0 && input_capacity % sizeof(int64_t) == 0 &&
          (input_capacity == 0 || input_handles != nullptr),
      "fusion input handles must be a direct long buffer");
  TORCH_CHECK(dimension_capacity >= 0 && dimension_capacity % sizeof(int64_t) == 0 &&
          (dimension_capacity == 0 || dimensions != nullptr),
      "fusion dimensions must be a direct long buffer");
  djl::pytorch::fusion::SubmitFusion(
      reinterpret_cast<djl::pytorch::fusion::FusionSession*>(jsession),
      jbuffer_index, input_handles,
      static_cast<std::size_t>(input_capacity) / sizeof(int64_t), dimensions,
      static_cast<std::size_t>(dimension_capacity) / sizeof(int64_t));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSynchronizeFusionOutput(
    JNIEnv* env, jobject jthis, jlong jsession, jint jbuffer_index) {
  API_BEGIN()
  djl::pytorch::fusion::SynchronizeFusionOutput(
      reinterpret_cast<djl::pytorch::fusion::FusionSession*>(jsession),
      jbuffer_index);
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteFusionPlan(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl::pytorch::fusion::DeleteFusionPlan(
      reinterpret_cast<djl::pytorch::fusion::FusionPlan*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteFusionExecutable(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl::pytorch::fusion::DeleteFusionExecutable(
      reinterpret_cast<djl::pytorch::fusion::FusionExecutable*>(jhandle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteFusionSession(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  djl::pytorch::fusion::DeleteFusionSession(
      reinterpret_cast<djl::pytorch::fusion::FusionSession*>(jhandle));
  API_END()
}
