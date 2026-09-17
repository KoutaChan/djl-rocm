/* Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0 */
#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <limits>

#include "djl_pytorch_packed_relation_attention.h"
#include "djl_pytorch_packed_relation_attention_kernels.h"

namespace djl::pytorch::rocm {
namespace {
using namespace packed_relation_detail;

Layout layout(const torch::Tensor& q, const torch::Tensor& kv,
              const torch::Tensor& table, int64_t heads, int64_t entries) {
  const int64_t segments = (table.size(0) + entries - 1) / entries;
  TORCH_CHECK(kv.size(1) <= 4096 && q.size(2) / heads <= 256,
              "packed relation ROCm kernel supports at most4096 keys and256 "
              "features per head");
  TORCH_CHECK(
      q.size(0) * q.size(1) * heads <= std::numeric_limits<uint32_t>::max(),
      "packed relation attention grid exceeds its native limit");
  return {q.size(1), kv.size(1),        q.size(2),
          heads,     q.size(2) / heads, table.size(0),
          entries,   segments,          (segments + 3) / 4};
}

void check_launch() {
  auto status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess,
              "packed relation attention kernel launch failed: ",
              hipGetErrorString(status));
}
}  // namespace

PackedRelationForward packed_relation_forward(
    const torch::Tensor& query, const torch::Tensor& key_value,
    const torch::Tensor& mask, const torch::Tensor& codes,
    const torch::Tensor& table, int64_t heads, int64_t entries, double scale,
    bool save_backward) {
  const c10::DeviceGuard device_guard(query.device());
  const auto l = layout(query, key_value, table, heads, entries);
  auto output = torch::empty_like(query);
  auto probabilities =
      save_backward ? torch::empty({query.size(0), l.queries, heads, l.keys},
                                   query.options().dtype(torch::kFloat32))
                    : torch::Tensor();
  auto float_output =
      save_backward
          ? torch::empty_like(query, query.options().dtype(torch::kFloat32))
          : torch::Tensor();
  const auto stream =
      c10::hip::getCurrentHIPStream(query.get_device()).stream();
  const auto blocks = static_cast<uint32_t>(query.size(0) * l.queries * heads);
  const auto shared_bytes = (l.keys + kThreads) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(),
      "packed_relation_forward", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(forward_kernel<scalar_t>), dim3(blocks),
            dim3(kThreads), shared_bytes, stream, query.data_ptr<scalar_t>(),
            key_value.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
            codes.data_ptr<int16_t>(), table.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            save_backward ? float_output.data_ptr<float>() : nullptr,
            save_backward ? probabilities.data_ptr<float>() : nullptr, l,
            static_cast<float>(scale));
      });
  check_launch();
  return {output, probabilities, float_output};
}

std::vector<torch::Tensor> packed_relation_backward(
    const torch::Tensor& query, const torch::Tensor& key_value,
    const torch::Tensor& mask, const torch::Tensor& codes,
    const torch::Tensor& table, const torch::Tensor& probabilities,
    const torch::Tensor& float_output, const torch::Tensor& grad_output,
    int64_t heads, int64_t entries, double scale) {
  const c10::DeviceGuard device_guard(query.device());
  const auto l = layout(query, key_value, table, heads, entries);
  auto grad_q =
      torch::empty_like(query, query.options().dtype(torch::kFloat32));
  auto grad_kv =
      torch::zeros_like(key_value, key_value.options().dtype(torch::kFloat32));
  auto grad_table = torch::zeros_like(table);
  const auto stream =
      c10::hip::getCurrentHIPStream(query.get_device()).stream();
  const auto blocks = static_cast<uint32_t>(query.size(0) * l.queries * heads);
  const auto shared_bytes = (l.keys + kThreads) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(),
      "packed_relation_backward", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(backward_kernel<scalar_t>), dim3(blocks),
            dim3(kThreads), shared_bytes, stream, query.data_ptr<scalar_t>(),
            key_value.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
            codes.data_ptr<int16_t>(), table.data_ptr<float>(),
            probabilities.data_ptr<float>(), float_output.data_ptr<float>(),
            grad_output.data_ptr<scalar_t>(), grad_q.data_ptr<float>(),
            grad_kv.data_ptr<float>(), grad_table.data_ptr<float>(), l,
            static_cast<float>(scale));
      });
  check_launch();
  return {grad_q.to(query.scalar_type()), grad_kv.to(key_value.scalar_type()),
          grad_table};
}
}  // namespace djl::pytorch::rocm
