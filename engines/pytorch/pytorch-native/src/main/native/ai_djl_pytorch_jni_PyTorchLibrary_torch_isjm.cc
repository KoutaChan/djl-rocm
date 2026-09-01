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
#include <ATen/ops/unique_dim.h>
#include <c10/core/DeviceGuard.h>
#include <djl/utils.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_fusion_kernels.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch tensor indexing, slicing, joining, mutating ops

#if defined(DJL_USE_ROCM_KERNELS)
namespace {

void record_concat_to_type_stream(const torch::Tensor& tensor) {
  c10::DeviceGuard device_guard(tensor.device());
  c10::impl::VirtualGuardImpl guard_impl(tensor.device().type());
  guard_impl.recordDataPtrOnStream(
      tensor.storage().data_ptr(), guard_impl.getStream(tensor.device()));
}

}  // namespace
#endif

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchReshape(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jshape) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->reshape(shape_vec));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqueeze__J(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->squeeze());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqueeze__JJ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->squeeze(jdim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUnsqueeze(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->unsqueeze(jdim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRot90(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jk, jlongArray jdims) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  auto vec = djl::utils::jni::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->rot90(jk, vec));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchExpand(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jshape) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->expand(shape_vec));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchStack(
    JNIEnv* env, jobject jthis, jlongArray jhandles, jlong jdim) {
  API_BEGIN()
  const std::vector<torch::Tensor> tensor_vec = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jhandles);
  const torch::Tensor* result_ptr = new torch::Tensor(torch::stack(tensor_vec, jdim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCat(
    JNIEnv* env, jobject jthis, jlongArray jhandles, jlong jdim) {
  API_BEGIN()
  const std::vector<torch::Tensor> tensor_vec = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jhandles);
  const torch::Tensor* result_ptr = new torch::Tensor(torch::cat(tensor_vec, jdim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchConcatToType(
    JNIEnv* env, jobject jthis, jlongArray jhandles, jlong jdim,
    jint jdata_type) {
  API_BEGIN()
  const std::vector<torch::Tensor> tensors =
      djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jhandles);
  TORCH_CHECK(!tensors.empty(), "concat-to-type requires at least one tensor");
  const torch::ScalarType output_type =
      utils::GetScalarTypeFromDType(jdata_type);
  const int64_t rank = tensors.front().dim();
  const int64_t dimension = jdim < 0 ? jdim + rank : jdim;
  TORCH_CHECK(dimension >= 0 && dimension < rank,
      "concat-to-type dimension is outside the input rank");

#if defined(DJL_USE_ROCM_KERNELS)
  bool native = dimension == rank - 1 && tensors.size() <=
      static_cast<size_t>(djl::pytorch::fusion::kMaximumOutputPackSources) &&
      (output_type == torch::kFloat16 || output_type == torch::kBFloat16 ||
       output_type == torch::kFloat32);
  int64_t output_width = 0;
  std::vector<int64_t> output_sizes(tensors.front().sizes().begin(),
      tensors.front().sizes().end());
  const auto device = tensors.front().device();
  for (const torch::Tensor& tensor : tensors) {
    native = native && tensor.dim() == rank && tensor.is_cuda() &&
        tensor.is_contiguous() && tensor.device() == device &&
        !tensor.requires_grad() &&
        (tensor.scalar_type() == torch::kFloat16 ||
         tensor.scalar_type() == torch::kBFloat16 ||
         tensor.scalar_type() == torch::kFloat32);
    if (tensor.dim() == rank) {
      for (int64_t axis = 0; axis < rank - 1; ++axis) {
        native = native && tensor.size(axis) == output_sizes[axis];
      }
      output_width += tensor.size(rank - 1);
    }
  }
  if (native) {
    c10::DeviceGuard device_guard(device);
    output_sizes[rank - 1] = output_width;
    torch::Tensor output = torch::empty(
        output_sizes, tensors.front().options().dtype(output_type));
    const int64_t row_count = output_width == 0
        ? 0
        : output.numel() / output_width;
    std::vector<djl::pytorch::fusion::OutputPackSource> sources;
    sources.reserve(tensors.size());
    int64_t destination_offset = 0;
    for (const torch::Tensor& tensor : tensors) {
      record_concat_to_type_stream(tensor);
      sources.push_back(djl::pytorch::fusion::OutputPackSource{
          tensor.data_ptr(), tensor.scalar_type(), tensor.size(rank - 1),
          destination_offset});
      destination_offset += tensor.size(rank - 1);
    }
    record_concat_to_type_stream(output);
    djl::pytorch::fusion::LaunchOutputPack(
        sources.data(), static_cast<int32_t>(sources.size()), output,
        row_count, output_width);
    const auto* result_ptr = new torch::Tensor(std::move(output));
    return reinterpret_cast<uintptr_t>(result_ptr);
  }
#endif

  std::vector<torch::Tensor> converted;
  converted.reserve(tensors.size());
  for (const torch::Tensor& tensor : tensors) {
    converted.push_back(tensor.to(output_type));
  }
  const auto* result_ptr =
      new torch::Tensor(torch::cat(converted, dimension));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__JJJ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jsize, jlong jdim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  std::vector<torch::Tensor> tensors = tensor_ptr->split(jsize, jdim);
  return djl::utils::jni::GetPtrArrayFromContainer<std::vector<torch::Tensor>, torch::Tensor>(env, tensors);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__J_3JJ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jindices, jlong jdim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> indices = djl::utils::jni::GetVecFromJLongArray(env, jindices);
  std::vector<torch::Tensor> tensors = tensor_ptr->split_with_sizes(indices, jdim);
  return djl::utils::jni::GetPtrArrayFromContainer<std::vector<torch::Tensor>, torch::Tensor>(env, tensors);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPermute(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jdims) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> dims = djl::utils::jni::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->permute(dims));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlip(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jdims) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> dims = djl::utils::jni::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->flip(dims));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTranspose(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim1, jlong jdim2) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->transpose(jdim1, jdim2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRepeat(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jrepeats) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> repeats = djl::utils::jni::GetVecFromJLongArray(env, jrepeats);
  const torch::Tensor* result_ptr = new torch::Tensor(tensor_ptr->repeat(repeats));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRepeatInterleave(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jrepeats, jlong jdim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const torch::Tensor* result_ptr = new torch::Tensor(tensor_ptr->repeat_interleave(jrepeats, jdim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNonZeros(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const torch::Tensor* result_ptr = new torch::Tensor(tensor_ptr->nonzero());

  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUnique(JNIEnv* env, jobject jthis,
    jlong jhandle, jlong jdim, jboolean jsorted, jboolean jreturn_inverse, jboolean jreturn_counts) {
  API_BEGIN()
  using namespace std;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> output_tuple;
  if (jdim < 0) {
    // negative jdim is a code for dim=None
    output_tuple = torch::_unique2(*tensor_ptr, jsorted, jreturn_inverse, jreturn_counts);
  } else {
    output_tuple = at::unique_dim(*tensor_ptr, jdim, jsorted, jreturn_inverse, jreturn_counts);
  }
  std::vector<jlong> jptrs;
  jptrs.push_back(reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<0>(output_tuple))));
  jptrs.push_back(reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<1>(output_tuple))));
  jptrs.push_back(reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<2>(output_tuple))));
  // Convert to jlongArray
  jlongArray jarray = env->NewLongArray(jptrs.size());
  env->SetLongArrayRegion(jarray, 0, jptrs.size(), jptrs.data());
  return jarray;
  API_END_RETURN()
}
