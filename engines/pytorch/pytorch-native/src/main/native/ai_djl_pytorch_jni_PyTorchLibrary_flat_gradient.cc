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
#include "djl_pytorch_flat_gradient.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_jni_log.h"

#include <utility>
#include <vector>

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateFlatGradientAccumulator(
    JNIEnv* env, jobject jthis, jlongArray jparameter_handles,
    jlong jgradient_handle) {
  API_BEGIN()
  (void) jthis;
  const auto parameter_handles =
      djl::utils::jni::GetVecFromJLongArray(env, jparameter_handles);
  std::vector<torch::Tensor> parameters;
  parameters.reserve(parameter_handles.size());
  for (const auto handle : parameter_handles) {
    TORCH_CHECK(handle != djl::utils::jni::NULL_PTR,
        "Flat gradient parameter handle must not be null.");
    parameters.push_back(*reinterpret_cast<torch::Tensor*>(handle));
  }
  TORCH_CHECK(jgradient_handle != djl::utils::jni::NULL_PTR,
      "Flat gradient destination handle must not be null.");
  auto gradient = *reinterpret_cast<torch::Tensor*>(jgradient_handle);
  return reinterpret_cast<uintptr_t>(
      djl::pytorch::gradient::NewFlatGradientAccumulator(
          std::move(parameters), std::move(gradient)));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlatGradientAccumulatorBackward(
    JNIEnv* env, jobject jthis, jlong jaccumulator_handle,
    jlong jtarget_handle, jlong jtarget_gradient_handle) {
  API_BEGIN()
  (void) jthis;
  auto* accumulator =
      reinterpret_cast<djl::pytorch::gradient::FlatGradientAccumulator*>(
          jaccumulator_handle);
  const auto& target = *reinterpret_cast<torch::Tensor*>(jtarget_handle);
  const auto& target_gradient =
      *reinterpret_cast<torch::Tensor*>(jtarget_gradient_handle);
  djl::pytorch::gradient::BackwardFlatGradientAccumulator(
      accumulator, target, target_gradient);
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchZeroFlatGradientAccumulator(
    JNIEnv* env, jobject jthis, jlong jaccumulator_handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::ZeroFlatGradientAccumulator(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientAccumulator*>(
          jaccumulator_handle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteFlatGradientAccumulator(
    JNIEnv* env, jobject jthis, jlong jaccumulator_handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::DeleteFlatGradientAccumulator(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientAccumulator*>(
          jaccumulator_handle));
  API_END()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCreateFlatGradientPacker(
    JNIEnv* env, jobject jthis, jlongArray jparameter_handles,
    jlong jdestination_handle) {
  API_BEGIN()
  (void) jthis;
  const auto parameter_handles =
      djl::utils::jni::GetVecFromJLongArray(env, jparameter_handles);
  std::vector<torch::Tensor> parameters;
  parameters.reserve(parameter_handles.size());
  for (const auto handle : parameter_handles) {
    TORCH_CHECK(handle != djl::utils::jni::NULL_PTR,
        "Flat gradient parameter handle must not be null.");
    parameters.push_back(*reinterpret_cast<torch::Tensor*>(handle));
  }
  TORCH_CHECK(jdestination_handle != djl::utils::jni::NULL_PTR,
      "Flat gradient destination handle must not be null.");
  auto destination = *reinterpret_cast<torch::Tensor*>(jdestination_handle);
  return reinterpret_cast<uintptr_t>(
      djl::pytorch::gradient::NewFlatGradientPacker(
          std::move(parameters), std::move(destination)));
  API_END_RETURN()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlatGradientPackerPackAndClear(
    JNIEnv* env, jobject jthis, jlong jpacker_handle,
    jboolean jzero_missing_gradients) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::PackAndClearFlatGradients(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientPacker*>(
          jpacker_handle),
      jzero_missing_gradients == JNI_TRUE);
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlatGradientPackerAccumulateAndClear(
    JNIEnv* env, jobject jthis, jlong jpacker_handle,
    jboolean jzero_missing_gradients) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::AccumulateAndClearFlatGradients(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientPacker*>(
          jpacker_handle),
      jzero_missing_gradients == JNI_TRUE);
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchZeroFlatGradientPackerDestination(
    JNIEnv* env, jobject jthis, jlong jpacker_handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::ZeroFlatGradientPackerDestination(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientPacker*>(
          jpacker_handle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchClearFlatGradientPackerParameterGradients(
    JNIEnv* env, jobject jthis, jlong jpacker_handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::ClearFlatGradientPackerParameterGradients(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientPacker*>(
          jpacker_handle));
  API_END()
}

extern "C" JNIEXPORT void JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteFlatGradientPacker(
    JNIEnv* env, jobject jthis, jlong jpacker_handle) {
  API_BEGIN()
  (void) jthis;
  djl::pytorch::gradient::DeleteFlatGradientPacker(
      reinterpret_cast<djl::pytorch::gradient::FlatGradientPacker*>(
          jpacker_handle));
  API_END()
}
