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

#include <ATen/Functions.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_rocm_kernels.h"
#endif

// The file is the implementation for PyTorch training operations

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_adamUpdate(JNIEnv* env, jobject jthis, jlong jweight,
    jlong jgrad, jlong jmean, jlong jvariance, jfloat learning_rate, jfloat learning_rate_bias_correction,
    jfloat weight_decay, jfloat rescale_grad, jfloat clip_grad, jfloat beta1, jfloat beta2, jfloat eps,
    jboolean adamw) {
  API_BEGIN()
  torch::autograd::AutoGradMode no_autograd_guard{false};
  auto& weight = *reinterpret_cast<torch::Tensor*>(jweight);
  const auto& gradient = *reinterpret_cast<torch::Tensor*>(jgrad);
  auto& mean = *reinterpret_cast<torch::Tensor*>(jmean);
  auto& variance = *reinterpret_cast<torch::Tensor*>(jvariance);
#if defined(DJL_USE_ROCM_KERNELS)
  if (djl::pytorch::rocm::supports_fused_adam_update(weight, gradient, mean, variance)) {
    djl::pytorch::rocm::fused_adam_update(weight, gradient, mean, variance, learning_rate,
        learning_rate_bias_correction, weight_decay, rescale_grad, clip_grad, beta1, beta2, eps, adamw);
    return;
  }
#endif
  const auto grad = gradient.clone();
  // following this formula: rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
  if (rescale_grad != 1.0) {
    grad.mul_(rescale_grad);
  }
  if (clip_grad >= 0.0) {
    // Add clip grad option
    grad.clamp_max_(clip_grad);
  }
  if (!adamw) {
    // rescaled_grad is obtained here
    grad.add_(weight, weight_decay);
  } else {
    weight.sub_(weight.mul(learning_rate).mul(weight_decay));
  }
  mean.mul_(beta1).add_(grad, 1 - beta1);
  variance.mul_(beta2).addcmul_(grad, grad, 1 - beta2);
  weight.sub_(mean.mul(learning_rate_bias_correction).div(variance.sqrt().add(eps)));
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_sgdUpdate(JNIEnv* env, jobject jthis, jlong jweight,
    jlong jgrad, jlong jstate, jfloat learning_rate, jfloat weight_decay, jfloat rescale_grad, jfloat clip_grad,
    jfloat momentum) {
  API_BEGIN()
  // disable gradient calculation
  torch::autograd::AutoGradMode no_autograd_guard{false};
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  // use clone to avoid input grad change
  auto grad = reinterpret_cast<torch::Tensor*>(jgrad)->clone();
  // following this formula: rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
  if (rescale_grad != 1.0) {
    grad.mul_(rescale_grad);
  }
  // TODO: MXNet convension, if < 0, it won't clip
  if (clip_grad >= 0.0) {
    // Add clip grad option
    grad.clamp_max_(clip_grad);
  }
  grad.add_(*weight_ptr, weight_decay).mul_(learning_rate);
  // TODO: implementation in DJL is different than PyTorch with missing dampening and nesterov
  if (momentum == 0.0) {
    weight_ptr->sub_(grad);
  } else {
    const auto* state_ptr = reinterpret_cast<torch::Tensor*>(jstate);
    state_ptr->mul_(momentum).add_(grad);
    weight_ptr->sub_(*state_ptr);
  }
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_zeroGrad(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  // the check is only for batch_size < # of gpus
  // where some required_grad weights never call backward
  // TODO we should avoid the create parameter but not applying backward
  if (weight_ptr->grad().defined()) {
    weight_ptr->grad().zero_();
  }
  API_END()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUnscaleGradientsAndCheckFinite(
    JNIEnv* env, jobject jthis, jlongArray jgradient_handles, jfloat jinverse_scale) {
  API_BEGIN()
  (void) jthis;
  torch::NoGradGuard no_grad;
  const auto gradient_handles = djl::utils::jni::GetVecFromJLongArray(env, jgradient_handles);
  if (gradient_handles.empty()) {
    throw std::invalid_argument("GradScaler requires at least one gradient tensor.");
  }

  std::vector<at::Tensor> gradients;
  gradients.reserve(gradient_handles.size());
  for (const auto handle : gradient_handles) {
    gradients.push_back(*reinterpret_cast<at::Tensor*>(handle));
  }

  const auto& first = gradients.front();
  std::vector<at::Tensor> tensors_to_unscale;
  tensors_to_unscale.reserve(gradients.size());
  for (const auto& gradient : gradients) {
    if (gradient.device() != first.device() || gradient.scalar_type() != first.scalar_type()) {
      throw std::invalid_argument("Fused GradScaler inputs must share a device and data type.");
    }
    tensors_to_unscale.push_back(gradient.is_sparse() ? gradient._values() : gradient);
  }

  const auto scalar_options = first.options().dtype(at::kFloat).layout(at::kStrided).requires_grad(false);
  auto found_non_finite = at::zeros({}, scalar_options);
  const auto inverse_scale = at::full({}, static_cast<double>(jinverse_scale), scalar_options);
  for (const auto& gradient : gradients) {
    if (gradient.is_sparse() && gradient.scalar_type() == at::kHalf) {
      // PyTorch coalesces scaled FP16 sparse gradients before unscale because summing duplicate
      // indices can overflow even when each stored value is finite. The optimizer can retain the
      // original sparse layout, but the finite check must still observe that coalesced result.
      const auto coalesced = gradient.coalesce();
      found_non_finite.add_(at::logical_not(at::isfinite(coalesced._values())).any().to(at::kFloat));
    }
  }
#if defined(USE_ROCM)
  if (first.is_cuda() && first.scalar_type() == at::kBFloat16) {
    // ROCm does not currently register the fused AMP kernel for BF16. Keep the
    // finite flag on the device across all gradients so only the final result synchronizes.
    for (auto& tensor : tensors_to_unscale) {
      tensor.mul_(static_cast<double>(jinverse_scale));
      found_non_finite.add_(at::logical_not(at::isfinite(tensor)).any().to(at::kFloat));
    }
  } else {
    at::_amp_foreach_non_finite_check_and_unscale_(tensors_to_unscale, found_non_finite, inverse_scale);
  }
#else
  at::_amp_foreach_non_finite_check_and_unscale_(tensors_to_unscale, found_non_finite, inverse_scale);
#endif
  return found_non_finite.item<float>() == 0.0f ? JNI_TRUE : JNI_FALSE;
  API_END_RETURN()
}
