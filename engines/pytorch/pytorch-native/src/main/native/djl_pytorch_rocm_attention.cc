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
#include "djl_pytorch_rocm_attention.h"

#if defined(DJL_USE_AOTRITON)
#include <aotriton/util.h>
#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/Exception.h>

#include <mutex>
#endif

namespace djl::pytorch {

void prepare_rocm_attention_backend() {
#if defined(DJL_USE_AOTRITON)
  static std::once_flag initialize_flag;
  std::call_once(initialize_flag, []() {
    const c10::DeviceIndex device_count = c10::hip::device_count();
    for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
      const auto stream = c10::hip::getStreamFromPool(false, device_index);
      const hipStream_t native_stream = stream.stream();
      const auto gpu = aotriton::getGpuFromStream(native_stream);
      const int multiprocessor_count = aotriton::getMultiProcessorCount(native_stream);
      TORCH_CHECK(gpu != aotriton::GPU_ARCH_UNKNOWN,
          "AOTriton could not identify ROCm device ", device_index);
      TORCH_CHECK(multiprocessor_count > 0,
          "AOTriton reported no multiprocessors for ROCm device ", device_index);
    }
  });
#endif
}

}  // namespace djl::pytorch
