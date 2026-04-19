/*
 * Copyright 2025 KoutaChan. Licensed under the Apache License, Version 2.0.
 */
// Autocast ("torch.autocast") bindings. Thread-local PyTorch dispatcher flags
// that instruct matmul / conv / attention style ops to cast their inputs to
// a lower-precision dtype (typically BF16) and propagate that dtype through,
// while numerically sensitive ops (softmax, reductions, loss fns) stay in
// FP32. Mirrors the 9-function surface used by PyTorch's Python-level
// `torch.autocast` context manager: set/get enabled, set/get dtype, set/get
// cache enabled, increment/decrement nesting, clear cache.
//
// ROCm note: libtorch built with `PYTORCH_HIP_AS_CUDA=1` (our fork's setup)
// registers the autocast backend under `at::DeviceType::CUDA`, so consumers
// always pass device_type=1 (CUDA) regardless of NVIDIA vs AMD. CPU autocast
// uses device_type=0.

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastIsEnabled(
    JNIEnv* env, jobject jthis, jint jdevice_type) {
  API_BEGIN()
  const auto device_type = static_cast<at::DeviceType>(jdevice_type);
  return at::autocast::is_autocast_enabled(device_type) ? JNI_TRUE : JNI_FALSE;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastSetEnabled(
    JNIEnv* env, jobject jthis, jint jdevice_type, jboolean jenabled) {
  API_BEGIN()
  const auto device_type = static_cast<at::DeviceType>(jdevice_type);
  at::autocast::set_autocast_enabled(device_type, jenabled == JNI_TRUE);
  API_END()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastGetDtype(
    JNIEnv* env, jobject jthis, jint jdevice_type) {
  API_BEGIN()
  const auto device_type = static_cast<at::DeviceType>(jdevice_type);
  const auto dtype = at::autocast::get_autocast_dtype(device_type);
  return utils::GetDTypeFromScalarType(dtype);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastSetDtype(
    JNIEnv* env, jobject jthis, jint jdevice_type, jint jdtype) {
  API_BEGIN()
  const auto device_type = static_cast<at::DeviceType>(jdevice_type);
  const auto dtype = utils::GetScalarTypeFromDType(jdtype);
  at::autocast::set_autocast_dtype(device_type, dtype);
  API_END()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastIsCacheEnabled(
    JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return at::autocast::is_autocast_cache_enabled() ? JNI_TRUE : JNI_FALSE;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastSetCacheEnabled(
    JNIEnv* env, jobject jthis, jboolean jenabled) {
  API_BEGIN()
  at::autocast::set_autocast_cache_enabled(jenabled == JNI_TRUE);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastClearCache(
    JNIEnv* env, jobject jthis) {
  API_BEGIN()
  at::autocast::clear_cache();
  API_END()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastIncrementNesting(
    JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return static_cast<jint>(at::autocast::increment_nesting());
  API_END_RETURN()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAutocastDecrementNesting(
    JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return static_cast<jint>(at::autocast::decrement_nesting());
  API_END_RETURN()
}
