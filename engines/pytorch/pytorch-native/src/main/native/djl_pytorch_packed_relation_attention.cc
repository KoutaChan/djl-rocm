/* Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0 */
#include "djl_pytorch_packed_relation_attention.h"

#include <ATen/autocast_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/ScopeExit.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <vector>

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
  TORCH_CHECK(q.is_floating_point() && kv.is_floating_point() && codes.scalar_type() == torch::kInt16 &&
          table.is_floating_point() && (!q.is_cuda() || q.scalar_type() == kv.scalar_type()),
      "packed relation attention expects floating Q/KV (matching on the GPU), INT16 codes and a floating table");
  TORCH_CHECK(q.device() == kv.device() && q.device() == mask.device() &&
                  q.device() == codes.device() && q.device() == table.device(),
              "packed relation attention tensors must share a device");
}

// The table row that each segment reads for every (row, query, key) triple: the first row of the segment plus the
// unsigned four-bit code of the segment, lowest nibble first.
std::vector<torch::Tensor> segment_rows(const torch::Tensor& codes, int64_t rows, int64_t entries) {
  const auto segments = (rows + entries - 1) / entries;
  std::vector<torch::Tensor> ids;
  ids.reserve(segments);
  for (int64_t segment = 0; segment < segments; ++segment) {
    auto code = codes.select(3, segment / 4)
                    .to(torch::kInt32)
                    .bitwise_and(65535)
                    .bitwise_right_shift((segment % 4) * 4)
                    .bitwise_and(15);
    const auto largest = code.max().item<int64_t>();
    TORCH_CHECK(largest < entries && segment * entries + largest < rows, "packed relation code exceeds its segment");
    ids.push_back(code.add(segment * entries).to(torch::kLong));
  }
  return ids;
}

// The relation of every triple: the table rows of its segments, added from segment 0.
torch::Tensor relation_rows(const torch::Tensor& table, const std::vector<torch::Tensor>& ids) {
  torch::Tensor relation;
  for (const auto& segment : ids) {
    auto part = torch::embedding(table, segment);
    relation = relation.defined() ? relation.add(part) : part;
  }
  return relation;
}

// Attends FP32 rows with the operations of the portable decomposition (NDArrayEx) in its order, so the output and
// the gradients keep its bits. No operation mixes rows, so a block of rows gives the bits of the same rows of the
// whole batch.
torch::Tensor attend(const torch::Tensor& query, const torch::Tensor& key_value, const torch::Tensor& mask,
    const torch::Tensor& relation, int64_t heads, double scale) {
  const auto batch = query.size(0);
  const auto queries = query.size(1);
  const auto width = query.size(2);
  const auto keys = key_value.size(1);
  const auto features = width / heads;
  auto valid = mask.ne(0).reshape({batch, 1, 1, keys}).detach();
  auto valid_memory = mask.ne(0).unsqueeze(2).expand(key_value.sizes());
  auto safe_memory = torch::where(valid_memory, key_value, torch::zeros_like(key_value));
  auto q = query.reshape({batch, queries, heads, features}).transpose(1, 2);
  auto k = safe_memory.slice(2, 0, width).reshape({batch, keys, heads, features}).transpose(1, 2);
  auto v = safe_memory.slice(2, width).reshape({batch, keys, heads, features}).transpose(1, 2);
  auto bias = relation.slice(3, 0, heads).permute({0, 3, 1, 2});
  auto scores = q.matmul(k.transpose(2, 3)).mul(scale).add(bias);
  auto full_valid = valid.expand(scores.sizes());
  auto present = valid.sum(3, true).gt(0).expand(scores.sizes());
  auto safe_scores = torch::where(full_valid, scores, torch::zeros_like(scores));
  auto attention_valid = torch::where(present, full_valid, torch::ones_like(full_valid));
  auto excluded = torch::zeros_like(scores).add(-std::numeric_limits<double>::infinity());
  auto probabilities = torch::where(full_valid,
      torch::where(attention_valid, safe_scores, excluded).softmax(3, torch::kFloat32), torch::zeros_like(scores));
  auto relative_values =
      relation.slice(3, heads).reshape({batch, queries, keys, heads, features}).permute({0, 3, 1, 2, 4});
  auto values = v.unsqueeze(2).add(relative_values);
  values = torch::where(full_valid.unsqueeze(4).expand(values.sizes()), values, torch::zeros_like(values));
  return probabilities.unsqueeze(4).mul(values).sum(3).transpose(1, 2).reshape({batch, queries, width});
}

// Whole rows with at most block_triples (row, query, key) triples, and at least one row.
int64_t rows_per_block(const torch::Tensor& query, const torch::Tensor& key_value, int64_t block_triples) {
  return std::max<int64_t>(1, block_triples / (query.size(1) * key_value.size(1)));
}

// Attends FP32 inputs one block of rows at a time.
torch::Tensor attend_blocks(const torch::Tensor& query, const torch::Tensor& key_value, const torch::Tensor& mask,
    const torch::Tensor& codes, const torch::Tensor& table, int64_t heads, int64_t entries, double scale,
    int64_t block_triples) {
  const auto batch = query.size(0);
  const auto step = rows_per_block(query, key_value, block_triples);
  auto output = torch::empty(query.sizes(), query.options());
  for (int64_t start = 0; start < batch; start += step) {
    const auto length = std::min(step, batch - start);
    auto relation = relation_rows(table, segment_rows(codes.narrow(0, start, length), table.size(0), entries));
    output.narrow(0, start, length)
        .copy_(attend(query.narrow(0, start, length), key_value.narrow(0, start, length),
            mask.narrow(0, start, length), relation, heads, scale));
  }
  return output;
}

// Packed relation attention on the CPU for a batch of several blocks that holds the tensors of one block of rows at a
// time. The forward pass saves only its inputs and the backward pass recomputes each block with the CPU autocast state
// of the forward pass.
// Every segment accumulates its table gradient from zero over the triples in order, as one embedding backward over
// the whole batch does, and the segments are added last segment first, the order in which the autograd graph of the
// decomposition delivers them.
class PackedRelationCpuFunction : public torch::autograd::Function<PackedRelationCpuFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& query,
      const torch::Tensor& key_value, const torch::Tensor& mask, const torch::Tensor& codes,
      const torch::Tensor& table, int64_t heads, int64_t entries, double scale, int64_t block_triples) {
    context->save_for_backward({query, key_value, mask, codes, table});
    context->saved_data["heads"] = heads;
    context->saved_data["entries"] = entries;
    context->saved_data["scale"] = scale;
    context->saved_data["block_triples"] = block_triples;
    context->saved_data["autocast_enabled"] = at::autocast::is_autocast_enabled(at::DeviceType::CPU);
    context->saved_data["autocast_dtype"] =
        static_cast<int64_t>(at::autocast::get_autocast_dtype(at::DeviceType::CPU));
    return attend_blocks(query, key_value, mask, codes, table, heads, entries, scale, block_triples);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    const auto saved = context->get_saved_variables();
    const auto& query = saved.at(0);
    const auto& key_value = saved.at(1);
    const auto& mask = saved.at(2);
    const auto& codes = saved.at(3);
    const auto& table = saved.at(4);
    const auto heads = context->saved_data["heads"].toInt();
    const auto entries = context->saved_data["entries"].toInt();
    const auto scale = context->saved_data["scale"].toDouble();
    const auto& gradient = gradient_outputs.at(0);
    const bool create_graph = at::GradMode::is_enabled();
    c10::InferenceMode inference_mode{false};
    // Leaving inference mode enables grad mode; keep the grad mode of this backward pass.
    torch::autograd::AutoGradMode grad_mode{create_graph};
    torch::autograd::variable_list result(9);
    if (create_graph) {
      // Higher-order derivatives differentiate the graph of the whole batch.
      auto output = recompute(context, [&]() {
        return attend(query, key_value, mask, relation_rows(table, segment_rows(codes, table.size(0), entries)),
            heads, scale);
      });
      torch::autograd::variable_list inputs;
      std::vector<size_t> input_indices;
      for (size_t index : {size_t{0}, size_t{1}, size_t{4}}) {
        if (context->needs_input_grad(index)) {
          inputs.push_back(saved.at(index));
          input_indices.push_back(index);
        }
      }
      auto gradients =
          torch::autograd::grad({output}, inputs, {gradient.to(output.scalar_type())}, std::nullopt, true, true);
      for (size_t index = 0; index < gradients.size(); ++index) {
        result[input_indices[index]] = gradients[index];
      }
      return result;
    }

    const bool needs_query = context->needs_input_grad(0);
    const bool needs_memory = context->needs_input_grad(1);
    const bool needs_table = context->needs_input_grad(4);
    const auto batch = query.size(0);
    const auto step = rows_per_block(query, key_value, context->saved_data["block_triples"].toInt());
    auto query_gradient = needs_query ? torch::empty(query.sizes(), query.options()) : torch::Tensor();
    auto memory_gradient = needs_memory ? torch::empty(key_value.sizes(), key_value.options()) : torch::Tensor();
    std::vector<torch::Tensor> segment_gradients;
    if (needs_table) {
      const auto segments = (table.size(0) + entries - 1) / entries;
      for (int64_t segment = 0; segment < segments; ++segment) {
        segment_gradients.push_back(torch::zeros({table.size(0), table.size(1)}, table.options()));
      }
    }
    for (int64_t start = 0; start < batch; start += step) {
      const auto length = std::min(step, batch - start);
      auto ids = segment_rows(codes.narrow(0, start, length), table.size(0), entries);
      torch::Tensor block_query = query.narrow(0, start, length).detach();
      torch::Tensor block_memory = key_value.narrow(0, start, length).detach();
      torch::Tensor relation = relation_rows(table, ids);
      block_query.requires_grad_(needs_query);
      block_memory.requires_grad_(needs_memory);
      relation.requires_grad_(needs_table);
      auto output = recompute(context,
          [&]() { return attend(block_query, block_memory, mask.narrow(0, start, length), relation, heads, scale); });
      torch::autograd::variable_list inputs;
      if (needs_query) {
        inputs.push_back(block_query);
      }
      if (needs_memory) {
        inputs.push_back(block_memory);
      }
      if (needs_table) {
        inputs.push_back(relation);
      }
      auto gradients =
          torch::autograd::grad({output}, inputs, {gradient.narrow(0, start, length).to(output.scalar_type())});
      size_t next = 0;
      if (needs_query) {
        query_gradient.narrow(0, start, length).copy_(gradients[next++]);
      }
      if (needs_memory) {
        memory_gradient.narrow(0, start, length).copy_(gradients[next++]);
      }
      if (needs_table) {
        auto relation_gradient = gradients[next].reshape({-1, table.size(1)});
        for (size_t segment = 0; segment < ids.size(); ++segment) {
          segment_gradients[segment].index_add_(0, ids[segment].reshape(-1), relation_gradient);
        }
      }
    }
    result[0] = query_gradient;
    result[1] = memory_gradient;
    if (needs_table) {
      auto table_gradient = segment_gradients.back();
      for (auto segment = segment_gradients.size() - 1; segment-- > 0;) {
        table_gradient.add_(segment_gradients[segment]);
      }
      result[4] = table_gradient;
    }
    return result;
  }

 private:
  // Records a recomputation of the forward pass with its CPU autocast state.
  template <typename Forward>
  static torch::Tensor recompute(torch::autograd::AutogradContext* context, Forward forward) {
    c10::impl::ForceDispatchKeyGuard restore_dispatch;
    const auto previous_dtype = at::autocast::get_autocast_dtype(at::DeviceType::CPU);
    auto restore_dtype = c10::make_scope_exit(
        [previous_dtype]() { at::autocast::set_autocast_dtype(at::DeviceType::CPU, previous_dtype); });
    at::autocast::set_autocast_enabled(at::DeviceType::CPU, context->saved_data["autocast_enabled"].toBool());
    at::autocast::set_autocast_dtype(
        at::DeviceType::CPU, static_cast<at::ScalarType>(context->saved_data["autocast_dtype"].toInt()));
    torch::autograd::AutoGradMode enable_grad{true};
    return forward();
  }
};

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

torch::Tensor packed_relation_attention(const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& mask,
    const torch::Tensor& codes, const torch::Tensor& table, int64_t heads, int64_t entries, double scale,
    int64_t block_triples) {
  validate(q, kv, mask, codes, table, heads, entries, scale);
  TORCH_CHECK(block_triples > 0, "packed relation attention block size must be positive");
  if (q.size(0) == 0) {
    return q.mul(0).add(kv.sum().mul(0)).add(table.sum().mul(0));
  }
  if (!q.is_cuda()) {
    // Cast outside the custom Function, as the decomposition does, so autograd converts its FP32 gradients back.
    auto query = q.to(torch::kFloat32);
    auto memory = kv.to(torch::kFloat32);
    auto relations = table.to(torch::kFloat32);
    const bool gradients =
        torch::GradMode::is_enabled() && (q.requires_grad() || kv.requires_grad() || table.requires_grad());
    torch::Tensor output;
    if (!gradients) {
      output = attend_blocks(query, memory, mask, codes, relations, heads, entries, scale, block_triples);
    } else if (rows_per_block(query, memory, block_triples) >= q.size(0)) {
      // A batch within one block records the decomposition itself, so its backward pass recomputes nothing.
      output = attend(query, memory, mask, relation_rows(relations, segment_rows(codes, table.size(0), entries)),
          heads, scale);
    } else {
      output =
          PackedRelationCpuFunction::apply(query, memory, mask, codes, relations, heads, entries, scale, block_triples);
    }
    return output.to(q.scalar_type());
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
    jlong table, jlong heads, jlong entries, jfloat scale, jlong block_triples) {
  API_BEGIN()
  auto result = djl::pytorch::packed_relation_attention(
      *reinterpret_cast<torch::Tensor*>(q),
      *reinterpret_cast<torch::Tensor*>(kv),
      *reinterpret_cast<torch::Tensor*>(mask),
      *reinterpret_cast<torch::Tensor*>(codes),
      *reinterpret_cast<torch::Tensor*>(table), heads, entries, scale, block_triples);
  return reinterpret_cast<uintptr_t>(new torch::Tensor(std::move(result)));
  API_END_RETURN()
}
