/* Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0 */
#include "djl_pytorch_packed_relation_attention.h"

#include <torch/csrc/autograd/custom_function.h>

#include <cmath>
#include <limits>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_jni_log.h"

namespace djl::pytorch {
namespace {
void validate(const torch::Tensor& q, const torch::Tensor& kv,
              const torch::Tensor& mask, const torch::Tensor& codes,
              const torch::Tensor& table, int64_t heads, int64_t entries,
              double scale) {
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && mask.dim() == 2 &&
                  codes.dim() == 4 && table.dim() == 2,
              "packed relation attention requires Q[B,Q,A], KV[B,K,2A], "
              "mask[B,K], codes[B,Q,K,W], table[R,H+A]");
  TORCH_CHECK(
      heads > 0 && entries > 0 && entries <= 16 && std::isfinite(scale),
      "packed relation attention invalid heads, segment size, or scale");
  const int64_t segments = (table.size(0) + entries - 1) / entries;
  TORCH_CHECK(q.size(1) > 0 && q.size(2) > 0 && q.size(2) % heads == 0 &&
                  kv.size(1) > 0 && kv.size(0) == q.size(0) &&
                  kv.size(2) == 2 * q.size(2) && mask.size(0) == q.size(0) &&
                  mask.size(1) == kv.size(1) && codes.size(0) == q.size(0) &&
                  codes.size(1) == q.size(1) && codes.size(2) == kv.size(1) &&
                  codes.size(3) == (segments + 3) / 4 && table.size(0) > 0 &&
                  table.size(1) == heads + q.size(2),
              "packed relation attention incompatible shapes");
  TORCH_CHECK(q.is_floating_point() && q.scalar_type() == kv.scalar_type() &&
                  codes.scalar_type() == torch::kInt16 &&
                  table.is_floating_point(),
              "packed relation attention expects matching floating Q/KV, INT16 "
              "codes and a floating table");
  TORCH_CHECK(q.device() == kv.device() && q.device() == mask.device() &&
                  q.device() == codes.device() && q.device() == table.device(),
              "packed relation attention tensors must share a device");
}

torch::Tensor reference(const torch::Tensor& query, const torch::Tensor& kv,
                        const torch::Tensor& mask, const torch::Tensor& codes,
                        const torch::Tensor& table, int64_t heads,
                        int64_t entries, double scale) {
  const auto b = query.size(0), nq = query.size(1), a = query.size(2),
             nk = kv.size(1), d = a / heads;
  const auto segments = (table.size(0) + entries - 1) / entries;
  auto valid_memory = mask.ne(0).unsqueeze(-1);
  auto memory =
      torch::where(valid_memory, kv, torch::zeros_like(kv)).to(torch::kFloat32);
  auto q = query.to(torch::kFloat32).view({b, nq, heads, d}).transpose(1, 2);
  auto k = memory.slice(2, 0, a).reshape({b, nk, heads, d}).transpose(1, 2);
  auto v = memory.slice(2, a).reshape({b, nk, heads, d}).transpose(1, 2);
  torch::Tensor relation;
  auto unsigned_codes = codes.to(torch::kInt32).bitwise_and(65535);
  for (int64_t segment = 0; segment < segments; ++segment) {
    auto code = unsigned_codes.select(3, segment / 4)
                    .bitwise_right_shift((segment % 4) * 4)
                    .bitwise_and(15);
    auto ids = code.add(segment * entries).to(torch::kLong);
    TORCH_CHECK(code.max().item<int64_t>() < entries &&
                    ids.max().item<int64_t>() < table.size(0),
                "packed relation code exceeds its segment");
    auto part = table.to(torch::kFloat32)
                    .index_select(0, ids.reshape({-1}))
                    .view({b, nq, nk, heads + a});
    relation = relation.defined() ? relation.add(part) : part;
  }
  auto valid = mask.ne(0).view({b, 1, 1, nk});
  auto scores = q.matmul(k.transpose(2, 3))
                    .mul(scale)
                    .add(relation.slice(3, 0, heads).permute({0, 3, 1, 2}));
  auto safe_scores = torch::where(valid, scores, torch::zeros_like(scores));
  auto attention_valid = valid.logical_or(valid.any(3, true).logical_not());
  auto probabilities = torch::where(
      valid,
      torch::where(
          attention_valid, safe_scores,
          torch::full_like(scores, -std::numeric_limits<double>::infinity()))
          .softmax(3),
      torch::zeros_like(scores));
  auto values = v.unsqueeze(2).add(relation.slice(3, heads)
                                       .view({b, nq, nk, heads, d})
                                       .permute({0, 3, 1, 2, 4}));
  values = torch::where(valid.unsqueeze(-1), values, torch::zeros_like(values));
  return probabilities.unsqueeze(-1)
      .mul(values)
      .sum(3)
      .transpose(1, 2)
      .reshape({b, nq, a})
      .to(query.scalar_type());
}

#if defined(DJL_USE_ROCM_KERNELS)
class PackedRelationFunction
    : public torch::autograd::Function<PackedRelationFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const torch::Tensor& q, const torch::Tensor& kv,
                               const torch::Tensor& mask,
                               const torch::Tensor& codes,
                               const torch::Tensor& table, int64_t heads,
                               int64_t entries, double scale) {
    auto result = rocm::packed_relation_forward(q, kv, mask, codes, table,
                                                heads, entries, scale, true);
    ctx->save_for_backward(
        {q, kv, mask, codes, table, result.probabilities, result.float_output});
    ctx->saved_data["heads"] = heads;
    ctx->saved_data["entries"] = entries;
    ctx->saved_data["scale"] = scale;
    return result.output;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list outputs) {
    auto x = ctx->get_saved_variables();
    auto gradients = rocm::packed_relation_backward(
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], outputs[0].contiguous(),
        ctx->saved_data["heads"].toInt(), ctx->saved_data["entries"].toInt(),
        ctx->saved_data["scale"].toDouble());
    return {gradients[0], gradients[1],    torch::Tensor(), torch::Tensor(),
            gradients[2], torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};
#endif
}  // namespace

torch::Tensor packed_relation_attention(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& mask,
    const torch::Tensor& codes, const torch::Tensor& table, int64_t heads,
    int64_t entries, double scale) {
  validate(q, kv, mask, codes, table, heads, entries, scale);
  if (q.size(0) == 0) {
    return q.mul(0).add(kv.sum().mul(0)).add(table.sum().mul(0));
  }
  if (!q.is_cuda()) {
    return reference(q, kv, mask, codes, table, heads, entries, scale);
  }
#if defined(DJL_USE_ROCM_KERNELS)
  auto query = q.contiguous();
  auto memory = kv.contiguous();
  auto valid = mask.ne(0).contiguous();
  auto packed = codes.contiguous();
  // Cast outside the custom Function so autograd converts its FP32 table
  // gradient back to a mixed-precision parameter without a long-lived
  // parameter-side cache.
  auto relations = table.to(torch::kFloat32).contiguous();
  if (torch::GradMode::is_enabled() &&
      (q.requires_grad() || kv.requires_grad() || table.requires_grad())) {
    return PackedRelationFunction::apply(query, memory, valid, packed,
                                         relations, heads, entries, scale);
  }
  return rocm::packed_relation_forward(query, memory, valid, packed, relations,
                                       heads, entries, scale, false)
      .output;
#else
  TORCH_CHECK(
      false,
      "packed relation GPU attention requires the ROCm native kernel build");
#endif
}
}  // namespace djl::pytorch

extern "C" JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPackedRelationScaledDotProductAttention(
    JNIEnv* env, jobject, jlong q, jlong kv, jlong mask, jlong codes,
    jlong table, jlong heads, jlong entries, jfloat scale) {
  API_BEGIN()
  auto result = djl::pytorch::packed_relation_attention(
      *reinterpret_cast<torch::Tensor*>(q),
      *reinterpret_cast<torch::Tensor*>(kv),
      *reinterpret_cast<torch::Tensor*>(mask),
      *reinterpret_cast<torch::Tensor*>(codes),
      *reinterpret_cast<torch::Tensor*>(table), heads, entries, scale);
  return reinterpret_cast<uintptr_t>(new torch::Tensor(std::move(result)));
  API_END_RETURN()
}
