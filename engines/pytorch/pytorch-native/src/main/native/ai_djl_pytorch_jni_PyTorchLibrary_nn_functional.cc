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

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_rocm_kernels.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch neural network functional ops

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
JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRmsNorm(JNIEnv* env, jobject jthis, jlong jinput,
    jlongArray jnormalized_shape, jlong jweight, jdouble jeps) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jinput);
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jnormalized_shape);
  std::optional<torch::Tensor> weight_opt;
  if (jweight != djl::utils::jni::NULL_PTR) {
    weight_opt = *reinterpret_cast<torch::Tensor*>(jweight);
  }
  auto result = at::rms_norm(*input_ptr, c10::ArrayRef<int64_t>(shape_vec), weight_opt,
      std::optional<double>(static_cast<double>(jeps)));
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

// scaled_dot_product_attention: fused attention that dispatches to
// FlashAttention / mem-efficient / math backends via PyTorch internals.
// query/key/value shape: [B, H, T, D]. attn_mask is an additive float bias
// broadcastable over [B, H, Q, K] (or 0 for no mask). Scale defaults to
// 1/sqrt(D) inside libtorch.
JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchScaledDotProductAttention(
    JNIEnv* env, jobject jthis, jlong jquery, jlong jkey, jlong jvalue, jlong jmask, jdouble jdropout,
    jboolean jcausal) {
  API_BEGIN()
  const auto* q_ptr = reinterpret_cast<torch::Tensor*>(jquery);
  const auto* k_ptr = reinterpret_cast<torch::Tensor*>(jkey);
  const auto* v_ptr = reinterpret_cast<torch::Tensor*>(jvalue);
  std::optional<torch::Tensor> mask_opt;
  if (jmask != djl::utils::jni::NULL_PTR) {
    mask_opt = *reinterpret_cast<torch::Tensor*>(jmask);
  }
  auto result = at::scaled_dot_product_attention(*q_ptr, *k_ptr, *v_ptr, mask_opt,
      static_cast<double>(jdropout), jcausal == JNI_TRUE, std::nullopt);
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTileRelationAttention(
    JNIEnv* env, jobject jthis, jlong jquery, jlong jkey, jlong jvalue, jlong jrelation_key,
    jlong jrelation_bias, jlong jrelation_ids, jfloat jscale) {
  API_BEGIN()
  const auto* query_ptr = reinterpret_cast<torch::Tensor*>(jquery);
  const auto* key_ptr = reinterpret_cast<torch::Tensor*>(jkey);
  const auto* value_ptr = reinterpret_cast<torch::Tensor*>(jvalue);
  const auto* relation_key_ptr = reinterpret_cast<torch::Tensor*>(jrelation_key);
  const auto* relation_bias_ptr = reinterpret_cast<torch::Tensor*>(jrelation_bias);
  const auto* relation_ids_ptr = reinterpret_cast<torch::Tensor*>(jrelation_ids);
#if defined(DJL_USE_ROCM_KERNELS)
  auto result = djl::pytorch::rocm::tile_relation_attention(
      *query_ptr, *key_ptr, *value_ptr, *relation_key_ptr, *relation_bias_ptr, *relation_ids_ptr,
      static_cast<float>(jscale));
#else
  const auto batch = query_ptr->size(0);
  const auto heads = query_ptr->size(1);
  const auto tiles = query_ptr->size(2);
  auto relation_logits = torch::matmul(*query_ptr, *relation_key_ptr);
  auto relation_indices = relation_ids_ptr->reshape({1, 1, tiles, tiles}).expand({batch, heads, tiles, tiles});
  auto attention_mask = relation_logits.gather(3, relation_indices).mul(jscale).add(*relation_bias_ptr);
  auto result = at::scaled_dot_product_attention(
      *query_ptr, *key_ptr, *value_ptr, attention_mask, 0.0, false, std::nullopt);
#endif
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTileRelationMask(
    JNIEnv* env, jobject jthis, jlong jrelation_logits, jlong jrelation_bias,
    jlong jrelation_ids, jfloat jscale) {
  API_BEGIN()
  const auto* relation_logits_ptr = reinterpret_cast<torch::Tensor*>(jrelation_logits);
  const auto* relation_bias_ptr = reinterpret_cast<torch::Tensor*>(jrelation_bias);
  const auto* relation_ids_ptr = reinterpret_cast<torch::Tensor*>(jrelation_ids);
#if defined(DJL_USE_ROCM_KERNELS)
  auto result = djl::pytorch::rocm::tile_relation_mask(
      *relation_logits_ptr, *relation_bias_ptr, *relation_ids_ptr,
      static_cast<float>(jscale));
#else
  const auto batch = relation_logits_ptr->size(0);
  const auto heads = relation_logits_ptr->size(1);
  const auto tiles = relation_ids_ptr->size(0);
  auto relation_indices =
      relation_ids_ptr->reshape({1, 1, tiles, tiles}).expand({batch, heads, tiles, tiles});
  auto result =
      relation_logits_ptr->gather(3, relation_indices).mul(jscale).add(*relation_bias_ptr);
#endif
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTransitionTileAttention(
    JNIEnv* env, jobject jthis, jlong jquery, jlong jtile_key_value,
    jlong jrelation_key_value, jlong jwait_key_value, jlong jwait_tile_ids,
    jlong jcandidates_per_state, jfloat jscale) {
  API_BEGIN()
  const auto* query_ptr = reinterpret_cast<torch::Tensor*>(jquery);
  const auto* tile_key_value_ptr = reinterpret_cast<torch::Tensor*>(jtile_key_value);
  const auto* relation_key_value_ptr = reinterpret_cast<torch::Tensor*>(jrelation_key_value);
  const auto* wait_key_value_ptr = reinterpret_cast<torch::Tensor*>(jwait_key_value);
  const auto* wait_tile_ids_ptr = reinterpret_cast<torch::Tensor*>(jwait_tile_ids);
#if defined(DJL_USE_ROCM_KERNELS)
  auto result = djl::pytorch::rocm::transition_tile_attention(
      *query_ptr, *tile_key_value_ptr, *relation_key_value_ptr, *wait_key_value_ptr,
      *wait_tile_ids_ptr, static_cast<int64_t>(jcandidates_per_state),
      static_cast<float>(jscale));
#else
  const auto candidate_count = query_ptr->size(0);
  const auto heads = query_ptr->size(1);
  const auto key_size = query_ptr->size(2);
  const auto key_width = heads * key_size;
  const auto value_size = (tile_key_value_ptr->size(2) - key_width) / heads;
  auto state_indices = torch::arange(
                           tile_key_value_ptr->size(0),
                           torch::TensorOptions().device(query_ptr->device()).dtype(torch::kLong))
                           .repeat_interleave(jcandidates_per_state);
  auto candidate_tile_key_value = tile_key_value_ptr->index_select(0, state_indices);
  auto stored_wait_ids = wait_tile_ids_ptr->to(torch::kLong);
  auto wait_indices = stored_wait_ids.sub(1).clamp_min(0).unsqueeze(2).expand(
      {candidate_count, stored_wait_ids.size(1), tile_key_value_ptr->size(2)});
  auto wait_tile_key_value = candidate_tile_key_value.gather(1, wait_indices);
  auto tile_keys = candidate_tile_key_value.slice(2, 0, key_width)
                       .reshape({candidate_count, 34, heads, key_size})
                       .permute({0, 2, 1, 3});
  auto relation_keys = relation_key_value_ptr->slice(2, 0, key_width)
                           .reshape({candidate_count, 34, heads, key_size})
                           .permute({0, 2, 1, 3});
  auto wait_keys = wait_key_value_ptr->slice(2, 0, key_width)
                       .reshape({candidate_count, 13, heads, key_size})
                       .permute({0, 2, 1, 3});
  auto wait_tile_keys = wait_tile_key_value.slice(2, 0, key_width)
                            .reshape({candidate_count, 13, heads, key_size})
                            .permute({0, 2, 1, 3});
  auto keys = torch::cat({tile_keys.add(relation_keys), wait_keys.add(wait_tile_keys)}, 2);
  auto scores = query_ptr->unsqueeze(2).mul(keys).sum(3).mul(jscale);
  auto wait_present = stored_wait_ids.ne(0).unsqueeze(1).expand({candidate_count, heads, 13});
  scores.slice(2, 34, 47).masked_fill_(wait_present.logical_not(), -1.0e9);
  auto weights = scores.softmax(2);
  auto tile_values = candidate_tile_key_value.slice(2, key_width)
                         .reshape({candidate_count, 34, heads, value_size})
                         .permute({0, 2, 1, 3});
  auto relation_values = relation_key_value_ptr->slice(2, key_width)
                             .reshape({candidate_count, 34, heads, value_size})
                             .permute({0, 2, 1, 3});
  auto wait_values = wait_key_value_ptr->slice(2, key_width)
                         .reshape({candidate_count, 13, heads, value_size})
                         .permute({0, 2, 1, 3});
  auto wait_tile_values = wait_tile_key_value.slice(2, key_width)
                              .reshape({candidate_count, 13, heads, value_size})
                              .permute({0, 2, 1, 3});
  auto values = torch::cat(
      {tile_values.add(relation_values), wait_values.add(wait_tile_values)}, 2);
  auto result = weights.unsqueeze(3).mul(values).sum(2);
#endif
  const auto* result_ptr = new torch::Tensor(std::move(result));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchResidualLayerNormInPlace(
    JNIEnv* env, jobject jthis, jlong jresidual, jlong jupdate, jlong jweight,
    jlong jbias, jfloat jepsilon) {
  API_BEGIN()
  auto* residual_ptr = reinterpret_cast<torch::Tensor*>(jresidual);
  const auto* update_ptr = reinterpret_cast<torch::Tensor*>(jupdate);
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  const auto* bias_ptr = reinterpret_cast<torch::Tensor*>(jbias);
#if defined(DJL_USE_ROCM_KERNELS)
  auto result = djl::pytorch::rocm::residual_layer_norm_in_place(
      *residual_ptr, *update_ptr, *weight_ptr, *bias_ptr, static_cast<float>(jepsilon));
#else
  residual_ptr->add_(*update_ptr);
  auto result = torch::nn::functional::layer_norm(
      *residual_ptr,
      torch::nn::functional::LayerNormFuncOptions({residual_ptr->size(-1)})
          .weight(*weight_ptr)
          .bias(*bias_ptr)
          .eps(jepsilon));
#endif
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
