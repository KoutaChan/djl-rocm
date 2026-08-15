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
#include <djl/utils.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_rocm_kernels.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch neural network functional ops

namespace {

bool autograd_required(std::initializer_list<const torch::Tensor*> tensors) {
  if (!at::GradMode::is_enabled()) {
    return false;
  }
  return std::any_of(tensors.begin(), tensors.end(), [](const torch::Tensor* tensor) {
    return tensor != nullptr && tensor->requires_grad();
  });
}

torch::Tensor grouped_indexed_attention_reference(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas,
    const torch::Tensor& indexed_deltas, const torch::Tensor& indexed_shared_ids, int64_t queries_per_group,
    double scale) {
  const auto query_count = query.size(0);
  const auto heads = query.size(1);
  const auto key_size = query.size(2);
  const auto key_width = heads * key_size;
  const auto packed_width = shared_key_values.size(2);
  const auto value_size = (packed_width - key_width) / heads;
  const auto shared_tokens = shared_key_values.size(1);
  const auto indexed_tokens = indexed_deltas.size(1);

  auto group_indices = torch::arange(
      shared_key_values.size(0), torch::TensorOptions().device(query.device()).dtype(torch::kLong))
                           .repeat_interleave(queries_per_group);
  auto query_shared_key_values = shared_key_values.index_select(0, group_indices);
  auto stored_ids = indexed_shared_ids.to(torch::kLong);
  auto gather_indices = stored_ids.sub(1).clamp_min(0).unsqueeze(2).expand({query_count, indexed_tokens, packed_width});
  auto indexed_shared_key_values = query_shared_key_values.gather(1, gather_indices);

  auto shared_keys = query_shared_key_values.slice(2, 0, key_width)
                         .reshape({query_count, shared_tokens, heads, key_size})
                         .permute({0, 2, 1, 3});
  auto shared_delta_keys = shared_deltas.slice(2, 0, key_width)
                               .reshape({query_count, shared_tokens, heads, key_size})
                               .permute({0, 2, 1, 3});
  auto indexed_keys = indexed_shared_key_values.slice(2, 0, key_width)
                          .reshape({query_count, indexed_tokens, heads, key_size})
                          .permute({0, 2, 1, 3});
  auto indexed_delta_keys = indexed_deltas.slice(2, 0, key_width)
                                .reshape({query_count, indexed_tokens, heads, key_size})
                                .permute({0, 2, 1, 3});
  auto keys = torch::cat({shared_keys.add(shared_delta_keys), indexed_keys.add(indexed_delta_keys)}, 2);
  auto scores = query.unsqueeze(2).mul(keys).sum(3).mul(scale);
  auto indexed_present = stored_ids.ne(0).unsqueeze(1).expand({query_count, heads, indexed_tokens});
  auto indexed_scores = scores.slice(2, shared_tokens).masked_fill(indexed_present.logical_not(), -1.0e9);
  scores = torch::cat({scores.slice(2, 0, shared_tokens), indexed_scores}, 2);
  auto weights = scores.softmax(2);

  auto shared_values = query_shared_key_values.slice(2, key_width)
                           .reshape({query_count, shared_tokens, heads, value_size})
                           .permute({0, 2, 1, 3});
  auto shared_delta_values = shared_deltas.slice(2, key_width)
                                 .reshape({query_count, shared_tokens, heads, value_size})
                                 .permute({0, 2, 1, 3});
  auto indexed_values = indexed_shared_key_values.slice(2, key_width)
                            .reshape({query_count, indexed_tokens, heads, value_size})
                            .permute({0, 2, 1, 3});
  auto indexed_delta_values = indexed_deltas.slice(2, key_width)
                                  .reshape({query_count, indexed_tokens, heads, value_size})
                                  .permute({0, 2, 1, 3});
  auto values = torch::cat({shared_values.add(shared_delta_values), indexed_values.add(indexed_delta_values)}, 2);
  return weights.unsqueeze(3).mul(values).sum(2);
}

}  // namespace

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPad(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jshape, jdouble jvalue) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto* result_ptr = new torch::Tensor(pad(*tensor_ptr, c10::ArrayRef(shape_vec), "constant", jvalue));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSoftmax(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jint jdtype) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogSoftmax(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jint jdtype) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log_softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNOneHot(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jdepth) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::one_hot(*tensor_ptr, jdepth));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

// rms_norm: fused RMSNorm (PyTorch 2.4+). Normalises `input` along the
// trailing dims given by `normalized_shape`, then applies an affine `weight`
// when provided. `jeps` is always forwarded as the variance epsilon; the
// caller is expected to supply a sensible value (e.g. 1e-6f).
JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRmsNorm(
    JNIEnv* env, jobject jthis, jlong jinput, jlongArray jnormalized_shape, jlong jweight, jdouble jeps) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jnormalized_shape);
  std::optional<torch::Tensor> weight_opt;
  if (jweight != djl::utils::jni::NULL_PTR) {
    weight_opt = *reinterpret_cast<torch::Tensor*>(jweight);
  }
  auto result = at::rms_norm(
      *input_ptr, c10::ArrayRef<int64_t>(shape_vec), weight_opt, std::optional<double>(static_cast<double>(jeps)));
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

// scaled_dot_product_attention: fused attention that dispatches to
// FlashAttention / mem-efficient / math backends via PyTorch internals.
// query/key/value shape: [B, H, T, D]. attn_mask is an additive float bias
// broadcastable over [B, H, Q, K] (or 0 for no mask). Scale defaults to
// 1/sqrt(D) inside libtorch when scale is NaN.
JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchScaledDotProductAttention(JNIEnv* env,
    jobject jthis, jlong jquery, jlong jkey, jlong jvalue, jlong jmask, jdouble jdropout, jboolean jcausal,
    jdouble jscale) {
  API_BEGIN()
  const auto* q_ptr = reinterpret_cast<torch::Tensor*>(jquery);
  const auto* k_ptr = reinterpret_cast<torch::Tensor*>(jkey);
  const auto* v_ptr = reinterpret_cast<torch::Tensor*>(jvalue);
  std::optional<torch::Tensor> mask_opt;
  if (jmask != djl::utils::jni::NULL_PTR) {
    mask_opt = *reinterpret_cast<torch::Tensor*>(jmask);
  }
  const std::optional<double> scale_opt =
      std::isnan(jscale) ? std::nullopt : std::optional<double>(static_cast<double>(jscale));
  auto result = at::scaled_dot_product_attention(
      *q_ptr, *k_ptr, *v_ptr, mask_opt, static_cast<double>(jdropout), jcausal == JNI_TRUE, scale_opt);
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexedRelationBias(
    JNIEnv* env, jobject jthis, jlong jrelation_logits, jlong jrelation_bias, jlong jrelation_ids, jfloat jscale) {
  API_BEGIN()
  const auto* relation_logits_ptr = reinterpret_cast<torch::Tensor*>(jrelation_logits);
  const auto* relation_bias_ptr = reinterpret_cast<torch::Tensor*>(jrelation_bias);
  const auto* relation_ids_ptr = reinterpret_cast<torch::Tensor*>(jrelation_ids);
  torch::Tensor result;
#if defined(DJL_USE_ROCM_KERNELS)
  const bool fused = !autograd_required({relation_logits_ptr, relation_bias_ptr}) &&
                     djl::pytorch::rocm::can_use_indexed_relation_bias(
                         *relation_logits_ptr, *relation_bias_ptr, *relation_ids_ptr);
  if (fused) {
    result = djl::pytorch::rocm::indexed_relation_bias(
        *relation_logits_ptr, *relation_bias_ptr, *relation_ids_ptr, static_cast<float>(jscale));
  } else {
#endif
  const auto batch = relation_logits_ptr->size(0);
  const auto heads = relation_logits_ptr->size(1);
  const auto query_tokens = relation_logits_ptr->size(2);
  const auto key_tokens = relation_ids_ptr->size(-1);
  auto stored_ids = relation_ids_ptr->to(torch::kLong);
  auto relation_indices =
      relation_ids_ptr->dim() == 2
          ? stored_ids.reshape({1, 1, query_tokens, key_tokens}).expand({batch, heads, query_tokens, key_tokens})
          : stored_ids.reshape({batch, 1, query_tokens, key_tokens}).expand({batch, heads, query_tokens, key_tokens});
  result = relation_logits_ptr->gather(3, relation_indices).mul(jscale).add(*relation_bias_ptr);
#if defined(DJL_USE_ROCM_KERNELS)
  }
#endif
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGroupedIndexedScaledDotProductAttention(
    JNIEnv* env, jobject jthis, jlong jquery, jlong jshared_key_values, jlong jshared_deltas, jlong jindexed_deltas,
    jlong jindexed_shared_ids, jlong jqueries_per_group, jfloat jscale) {
  API_BEGIN()
  const auto* query_ptr = reinterpret_cast<torch::Tensor*>(jquery);
  const auto* shared_key_values_ptr = reinterpret_cast<torch::Tensor*>(jshared_key_values);
  const auto* shared_deltas_ptr = reinterpret_cast<torch::Tensor*>(jshared_deltas);
  const auto* indexed_deltas_ptr = reinterpret_cast<torch::Tensor*>(jindexed_deltas);
  const auto* indexed_shared_ids_ptr = reinterpret_cast<torch::Tensor*>(jindexed_shared_ids);
  const auto queries_per_group = static_cast<int64_t>(jqueries_per_group);

  TORCH_CHECK(query_ptr->dim() == 3, "query must have shape [queries, heads, key features]");
  TORCH_CHECK(shared_key_values_ptr->dim() == 3,
      "shared key/value storage must have shape [groups, shared tokens, packed features]");
  TORCH_CHECK(shared_deltas_ptr->dim() == 3 && indexed_deltas_ptr->dim() == 3,
      "shared and indexed deltas must have shape [queries, tokens, packed features]");
  TORCH_CHECK(indexed_shared_ids_ptr->dim() == 2,
      "indexed shared IDs must have shape [queries, indexed tokens]");
  TORCH_CHECK(query_ptr->size(1) > 0 && query_ptr->size(2) > 0 && shared_key_values_ptr->size(1) > 0,
      "attention heads, features, and shared-token count must be positive");
  TORCH_CHECK(queries_per_group > 0 && shared_key_values_ptr->size(0) * queries_per_group == query_ptr->size(0),
      "queries must be divided into equal consecutive groups");
  const auto key_width = query_ptr->size(1) * query_ptr->size(2);
  const auto packed_width = shared_key_values_ptr->size(2);
  TORCH_CHECK(packed_width > key_width && (packed_width - key_width) % query_ptr->size(1) == 0,
      "packed shared features must contain per-head keys followed by per-head values");
  TORCH_CHECK(shared_deltas_ptr->size(0) == query_ptr->size(0) &&
          shared_deltas_ptr->size(1) == shared_key_values_ptr->size(1) &&
          shared_deltas_ptr->size(2) == packed_width,
      "shared delta shape must match each query and shared token");
  TORCH_CHECK(indexed_deltas_ptr->size(0) == query_ptr->size(0) &&
          indexed_deltas_ptr->size(2) == packed_width &&
          indexed_shared_ids_ptr->size(0) == query_ptr->size(0) &&
          indexed_shared_ids_ptr->size(1) == indexed_deltas_ptr->size(1),
      "indexed deltas and IDs must describe the same query-token pairs");

  torch::Tensor result;
#if defined(DJL_USE_ROCM_KERNELS)
  const bool fused = !autograd_required(
                         {query_ptr, shared_key_values_ptr, shared_deltas_ptr, indexed_deltas_ptr}) &&
                     djl::pytorch::rocm::can_use_grouped_indexed_scaled_dot_product_attention(*query_ptr,
                         *shared_key_values_ptr, *shared_deltas_ptr, *indexed_deltas_ptr, *indexed_shared_ids_ptr,
                         queries_per_group);
  if (fused) {
    result = djl::pytorch::rocm::grouped_indexed_scaled_dot_product_attention(*query_ptr, *shared_key_values_ptr,
        *shared_deltas_ptr, *indexed_deltas_ptr, *indexed_shared_ids_ptr, queries_per_group,
        static_cast<float>(jscale));
  } else
#endif
  {
    result = grouped_indexed_attention_reference(*query_ptr, *shared_key_values_ptr, *shared_deltas_ptr,
        *indexed_deltas_ptr, *indexed_shared_ids_ptr, queries_per_group, static_cast<double>(jscale));
  }
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAddToOwnedResidualAndLayerNorm(
    JNIEnv* env, jobject jthis, jlong jresidual, jlong jupdate, jlong jweight, jlong jbias, jfloat jepsilon) {
  API_BEGIN()
  auto* residual_ptr = reinterpret_cast<torch::Tensor*>(jresidual);
  const auto* update_ptr = reinterpret_cast<torch::Tensor*>(jupdate);
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  const auto* bias_ptr = reinterpret_cast<torch::Tensor*>(jbias);
  TORCH_CHECK(!autograd_required({residual_ptr, update_ptr, weight_ptr, bias_ptr}),
      "owned residual update is an inference operation and does not support automatic differentiation");
  TORCH_CHECK(residual_ptr->dim() >= 1 && residual_ptr->size(-1) > 0,
      "residual must have a non-empty trailing dimension");
  TORCH_CHECK(residual_ptr->sizes() == update_ptr->sizes(), "residual and update shapes must match");
  TORCH_CHECK(weight_ptr->dim() == 1 && weight_ptr->size(0) == residual_ptr->size(-1) &&
          bias_ptr->sizes() == weight_ptr->sizes(),
      "LayerNorm parameters must match the residual trailing dimension");

  torch::Tensor result;
#if defined(DJL_USE_ROCM_KERNELS)
  if (djl::pytorch::rocm::can_use_owned_residual_layer_norm(
          *residual_ptr, *update_ptr, *weight_ptr, *bias_ptr)) {
    result = djl::pytorch::rocm::add_to_owned_residual_and_layer_norm(
        *residual_ptr, *update_ptr, *weight_ptr, *bias_ptr, static_cast<float>(jepsilon));
  } else
#endif
  {
    residual_ptr->add_(*update_ptr);
    result = torch::nn::functional::layer_norm(
        *residual_ptr, torch::nn::functional::LayerNormFuncOptions({residual_ptr->size(-1)})
                           .weight(*weight_ptr)
                           .bias(*bias_ptr)
                           .eps(jepsilon));
  }
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNInterpolate(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jsize, jint jmode, jboolean jalign_corners) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto size_vec = djl::utils::jni::GetVecFromJLongArray(env, jsize);

#if defined(__ANDROID__)
  torch::Tensor result;
  if (jmode == 0) {
    result = torch::upsample_nearest2d(*tensor_ptr, size_vec);
  } else if (jmode == 2) {
    result = torch::upsample_bilinear2d(*tensor_ptr, size_vec, jalign_corners);
  } else if (jmode == 3) {
    result = torch::upsample_bicubic2d(*tensor_ptr, size_vec, jalign_corners);
  } else {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "This kind of mode is not supported on Android");
    return reinterpret_cast<uintptr_t>(nullptr);
  }
  const auto* result_ptr = new torch::Tensor(result);
#else
  auto options =
      torch::nn::functional::InterpolateFuncOptions().size(size_vec).mode(utils::GetInterpolationMode(jmode));
  // kNearest, kArea interpolate can't set align_corners
  if (jmode != 0 && jmode != 5) {
    options = options.align_corners(jalign_corners).antialias(true);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::interpolate(*tensor_ptr, options));
#endif
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLinear(
    JNIEnv* env, jobject jthis, jlong jinput, jlong jweight, jlong jbias) {
  API_BEGIN()
  auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  torch::Tensor bias = {};
  if (jbias != djl::utils::jni::NULL_PTR) {
    bias = *reinterpret_cast<torch::Tensor*>(jbias);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::linear(*input_ptr, *weight_ptr, bias));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNConvNd(JNIEnv* env, jobject jthis, jlong jinput,
    jlong jweight, jlong jbias, jlongArray jstride, jlongArray jpadding, jlongArray jdilation, jint jgroups) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* weigtht_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  torch::Tensor bias = {};
  if (jbias != djl::utils::jni::NULL_PTR) {
    bias = *reinterpret_cast<torch::Tensor*>(jbias);
  }
  const std::vector<int64_t> strideVec = djl::utils::jni::GetVecFromJLongArray(env, jstride);
  const std::vector<int64_t> paddingVec = djl::utils::jni::GetVecFromJLongArray(env, jpadding);
  const std::vector<int64_t> dilationVec = djl::utils::jni::GetVecFromJLongArray(env, jdilation);

  torch::Tensor* result_ptr = nullptr;
  long dim = weigtht_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr =
        new torch::Tensor(torch::conv1d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
  } else if (dim == 2) {
    result_ptr =
        new torch::Tensor(torch::conv2d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
  } else if (dim == 3) {
    result_ptr =
        new torch::Tensor(torch::conv3d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNBatchNorm(JNIEnv* env, jobject jthis,
    jlong jinput, jlong jrunning_mean, jlong jrunning_var, jlong jweight, jlong jbias, jboolean jtraining,
    jdouble jmomentum, jdouble jeps) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* running_mean_ptr = reinterpret_cast<torch::Tensor*>(jrunning_mean);
  const auto* running_var_ptr = reinterpret_cast<torch::Tensor*>(jrunning_var);
  torch::Tensor weight = {};
  torch::Tensor bias = {};
  if (jweight != djl::utils::jni::NULL_PTR) {
    weight = *reinterpret_cast<torch::Tensor*>(jweight);
  }
  if (jbias != djl::utils::jni::NULL_PTR) {
    bias = *reinterpret_cast<torch::Tensor*>(jbias);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::batch_norm(*tensor_ptr, *running_mean_ptr,
      *running_var_ptr,
      torch::nn::functional::BatchNormFuncOptions().weight(weight).bias(bias).momentum(jmomentum).eps(jeps).training(
          jtraining)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLayerNorm(
    JNIEnv* env, jobject jthis, jlong jinput, jlongArray jnormalizedshape, jlong jweight, jlong jbias, jdouble jeps) {
  API_BEGIN()
#if defined(__ANDROID__)
  env->ThrowNew(ENGINE_EXCEPTION_CLASS, "layerNorm is not supported on Android.");
  return 0;
#else
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto normalized_shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jnormalizedshape);
  torch::Tensor weight = {};
  torch::Tensor bias = {};
  if (jweight != djl::utils::jni::NULL_PTR) {
    weight = *reinterpret_cast<torch::Tensor*>(jweight);
  }
  if (jbias != djl::utils::jni::NULL_PTR) {
    bias = *reinterpret_cast<torch::Tensor*>(jbias);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::layer_norm(*tensor_ptr,
      torch::nn::functional::LayerNormFuncOptions(normalized_shape_vec).weight(weight).bias(bias).eps(jeps)));
  return reinterpret_cast<uintptr_t>(result_ptr);
#endif
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNNormalize(
    JNIEnv* env, jobject jthis, jlong jinput, jdouble jp, jlong jdim, jdouble jeps) {
  API_BEGIN()
#if defined(__ANDROID__)
  env->ThrowNew(ENGINE_EXCEPTION_CLASS, "Normalize is not supported on Android.");
  return 0;
#else
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  auto options = torch::nn::functional::NormalizeFuncOptions();
  options.p(jp);
  options.dim(jdim);
  options.eps(jeps);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::normalize(*tensor_ptr, options));
  return reinterpret_cast<uintptr_t>(result_ptr);
#endif
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNDropout(
    JNIEnv* env, jobject jthis, jlong jinput, jdouble probability, jboolean jtraining) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::dropout(
      *tensor_ptr, torch::nn::functional::DropoutFuncOptions().p(probability).training(jtraining)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNRnn(JNIEnv* env, jobject jthis, jlong jinput,
    jlong jhx, jlongArray jparams, jboolean jhas_biases, jint jnum_layers, jint jactivation, jdouble jdrop_rate,
    jboolean jtraining, jboolean jbidirectional, jboolean jbatch_first) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* hx_ptr = reinterpret_cast<torch::Tensor*>(jhx);
  const std::vector<torch::Tensor> params = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jparams);

  std::tuple<torch::Tensor, torch::Tensor> outputs;
  if (jactivation == 0) {
    outputs = torch::rnn_relu(*input_ptr, *hx_ptr, torch::TensorList(params), jhas_biases, jnum_layers, jdrop_rate,
        jtraining, jbidirectional, jbatch_first);
  } else if (jactivation == 1) {
    outputs = torch::rnn_tanh(*input_ptr, *hx_ptr, torch::TensorList(params), jhas_biases, jnum_layers, jdrop_rate,
        jtraining, jbidirectional, jbatch_first);
  } else {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "can't find activation");
    return nullptr;
  }

  // process output
  jlongArray jarray = env->NewLongArray(2);
  std::vector<jlong> jptrs(2);
  jptrs[0] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<0>(outputs)));
  jptrs[1] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<1>(outputs)));
  env->SetLongArrayRegion(jarray, 0, 2, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNGru(JNIEnv* env, jobject jthis, jlong jinput,
    jlong jhx, jlongArray jparams, jboolean jhas_biases, jint jnum_layers, jdouble jdrop_rate, jboolean jtraining,
    jboolean jbidirectional, jboolean jbatch_first) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* hx_ptr = reinterpret_cast<torch::Tensor*>(jhx);
  const std::vector<torch::Tensor> params = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jparams);

  std::tuple<torch::Tensor, torch::Tensor> outputs = torch::gru(*input_ptr, *hx_ptr, torch::TensorList(params),
      jhas_biases, jnum_layers, jdrop_rate, jtraining, jbidirectional, jbatch_first);

  // process output
  jlongArray jarray = env->NewLongArray(2);
  std::vector<jlong> jptrs(2);
  jptrs[0] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<0>(outputs)));
  jptrs[1] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<1>(outputs)));
  env->SetLongArrayRegion(jarray, 0, 2, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLstm(JNIEnv* env, jobject jthis,
    jlong jinput, jlongArray jhx, jlongArray jparams, jboolean jhas_biases, jint jnum_layers, jdouble jdrop_rate,
    jboolean jtraining, jboolean jbidirectional, jboolean jbatch_first) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const std::vector<torch::Tensor> hx = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jhx);
  const std::vector<torch::Tensor> params = djl::utils::jni::GetObjectVecFromJHandles<torch::Tensor>(env, jparams);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> outputs = torch::lstm(*input_ptr, torch::TensorList(hx),
      torch::TensorList(params), jhas_biases, jnum_layers, jdrop_rate, jtraining, jbidirectional, jbatch_first);

  // process output
  jlongArray jarray = env->NewLongArray(3);
  std::vector<jlong> jptrs(3);
  jptrs[0] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<0>(outputs)));
  jptrs[1] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<1>(outputs)));
  jptrs[2] = reinterpret_cast<uintptr_t>(new torch::Tensor(std::get<2>(outputs)));
  env->SetLongArrayRegion(jarray, 0, 3, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNRelu(JNIEnv* env, jobject jthis, jlong jinput) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  // FIIXME the compiled libtorch have reference error
  // use torch::relu() for now until the fix
  const auto* result_ptr = new torch::Tensor(torch::relu(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSoftPlus(
    JNIEnv* env, jobject jthis, jlong jinput) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::softplus(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSoftSign(
    JNIEnv* env, jobject jthis, jlong jinput) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::softsign(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLeakyRelu(
    JNIEnv* env, jobject jthis, jlong jinput, jdouble jnegative_slope) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::leaky_relu(
      *tensor_ptr, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(jnegative_slope)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNElu(
    JNIEnv* env, jobject jthis, jlong jinput, jdouble jalpha) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr =
      new torch::Tensor(torch::nn::functional::elu(*tensor_ptr, torch::nn::functional::ELUFuncOptions().alpha(jalpha)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSelu(JNIEnv* env, jobject jthis, jlong jinput) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  // FIIXME the compiled libtorch have reference error
  // use torch::selu() for now until the fix
  const auto* result_ptr = new torch::Tensor(torch::selu(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNGelu(JNIEnv* env, jobject jthis, jlong jinput) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::gelu(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNMaxPool(JNIEnv* env, jobject jthis, jlong jhandle,
    jlongArray jkernel, jlongArray jstride, jlongArray jpadding, jboolean jceil_mode) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> kernel_vec = djl::utils::jni::GetVecFromJLongArray(env, jkernel);
  const std::vector<int64_t> stride_vec = djl::utils::jni::GetVecFromJLongArray(env, jstride);
  const std::vector<int64_t> padding_vec = djl::utils::jni::GetVecFromJLongArray(env, jpadding);
  torch::Tensor* result_ptr = nullptr;
  long dim = tensor_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::max_pool1d(*tensor_ptr, torch::nn::functional::MaxPool1dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  } else if (dim == 2) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::max_pool2d(*tensor_ptr, torch::nn::functional::MaxPool2dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  } else if (dim == 3) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::max_pool3d(*tensor_ptr, torch::nn::functional::MaxPool3dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAvgPool(JNIEnv* env, jobject jthis, jlong jinput,
    jlongArray jkernel_size, jlongArray jstride, jlongArray jpaddiing, jboolean jceil_mode,
    jboolean jcount_include_pad) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const std::vector<int64_t> kernel_vec = djl::utils::jni::GetVecFromJLongArray(env, jkernel_size);
  const std::vector<int64_t> stride_vec = djl::utils::jni::GetVecFromJLongArray(env, jstride);
  const std::vector<int64_t> padding_vec = djl::utils::jni::GetVecFromJLongArray(env, jpaddiing);

  torch::Tensor* result_ptr = nullptr;
  long dim = tensor_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::avg_pool1d(*tensor_ptr, torch::nn::functional::AvgPool1dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  } else if (dim == 2) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::avg_pool2d(*tensor_ptr, torch::nn::functional::AvgPool2dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  } else if (dim == 3) {
    result_ptr = new torch::Tensor(
        torch::nn::functional::avg_pool3d(*tensor_ptr, torch::nn::functional::AvgPool3dFuncOptions(kernel_vec)
                                                           .stride(stride_vec)
                                                           .padding(padding_vec)
                                                           .ceil_mode(jceil_mode)));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAdaptiveAvgPool(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray joutput_size) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> output_vec = djl::utils::jni::GetVecFromJLongArray(env, joutput_size);

  torch::Tensor* result_ptr = nullptr;
  long dim = tensor_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool1d(
        *tensor_ptr, torch::nn::functional::AdaptiveAvgPool1dFuncOptions(output_vec)));
  } else if (dim == 2) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool2d(
        *tensor_ptr, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(output_vec)));
  } else if (dim == 3) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool3d(
        *tensor_ptr, torch::nn::functional::AdaptiveAvgPool3dFuncOptions(output_vec)));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAdaptiveMaxPool(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray joutput_size) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> output_vec = djl::utils::jni::GetVecFromJLongArray(env, joutput_size);

  torch::Tensor* result_ptr = nullptr;
  long dim = tensor_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool1d(
        *tensor_ptr, torch::nn::functional::AdaptiveMaxPool1dFuncOptions(output_vec)));
  } else if (dim == 2) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool2d(
        *tensor_ptr, torch::nn::functional::AdaptiveMaxPool2dFuncOptions(output_vec)));
  } else if (dim == 3) {
    result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool3d(
        *tensor_ptr, torch::nn::functional::AdaptiveMaxPool3dFuncOptions(output_vec)));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLpPool(JNIEnv* env, jobject jthis, jlong jinput,
    jdouble jnorm_type, jlongArray jkernel_size, jlongArray jstride, jboolean jceil_mode) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const std::vector<int64_t> kernel_vec = djl::utils::jni::GetVecFromJLongArray(env, jkernel_size);
  const std::vector<int64_t> stride_vec = djl::utils::jni::GetVecFromJLongArray(env, jstride);

  torch::Tensor* result_ptr = nullptr;
  long dim = tensor_ptr->dim() - 2;
  if (dim == 1) {
    result_ptr = new torch::Tensor(torch::nn::functional::lp_pool1d(*tensor_ptr,
        torch::nn::functional::LPPool1dFuncOptions(jnorm_type, kernel_vec).stride(stride_vec).ceil_mode(jceil_mode)));
  } else if (dim == 2) {
    result_ptr = new torch::Tensor(torch::nn::functional::lp_pool2d(*tensor_ptr,
        torch::nn::functional::LPPool2dFuncOptions(jnorm_type, kernel_vec).stride(stride_vec).ceil_mode(jceil_mode)));
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNEmbedding(
    JNIEnv* env, jobject jthis, jlong jinput, jlong jweight, jboolean jsparse) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  auto* result_ptr = new torch::Tensor(torch::nn::functional::embedding(
      *tensor_ptr, *weight_ptr, torch::nn::functional::EmbeddingFuncOptions().sparse(jsparse)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}
