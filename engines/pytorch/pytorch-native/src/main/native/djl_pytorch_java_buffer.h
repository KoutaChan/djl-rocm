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
#ifndef DJL_TORCH_DJL_PYTORCH_JAVA_BUFFER_H
#define DJL_TORCH_DJL_PYTORCH_JAVA_BUFFER_H

#include <jni.h>
#include <torch/torch.h>

namespace djl::pytorch {

// Creates a CPU tensor over the memory of a direct buffer without a copy. Its storage holds a global reference to the
// buffer and deletes it when libtorch frees the storage, so views and saved tensors keep the memory valid.
torch::Tensor FromDirectBuffer(
    JNIEnv* env, jobject buffer, c10::IntArrayRef sizes, const torch::TensorOptions& options);

// Returns a CPU data pointer over the memory of a direct buffer that keeps the buffer reachable in the same way.
c10::DataPtr DirectBufferDataPtr(JNIEnv* env, jobject buffer);

// Creates a CPU tensor with a copy of the bytes of a Java primitive array that start at an offset. The length is the
// number of bytes from the offset that hold data. The tensor is undefined if a Java exception is pending afterwards.
torch::Tensor FromArray(JNIEnv* env, jobject array, jlong offset, jlong length, c10::IntArrayRef sizes,
    const torch::TensorOptions& options);

}  // namespace djl::pytorch

#endif  // DJL_TORCH_DJL_PYTORCH_JAVA_BUFFER_H
