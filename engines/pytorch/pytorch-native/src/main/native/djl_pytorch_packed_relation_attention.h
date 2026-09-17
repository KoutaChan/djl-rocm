/* Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <torch/torch.h>

namespace djl::pytorch {
torch::Tensor packed_relation_attention(
    const torch::Tensor& query, const torch::Tensor& key_value,
    const torch::Tensor& mask, const torch::Tensor& packed_codes,
    const torch::Tensor& table, int64_t heads, int64_t entries_per_segment,
    double scale);

#if defined(DJL_USE_ROCM_KERNELS)
namespace rocm {
struct PackedRelationForward {
  torch::Tensor output;
  torch::Tensor probabilities;
  torch::Tensor float_output;
};
PackedRelationForward packed_relation_forward(const torch::Tensor& query,
                                              const torch::Tensor& key_value,
                                              const torch::Tensor& mask,
                                              const torch::Tensor& codes,
                                              const torch::Tensor& table,
                                              int64_t heads, int64_t entries,
                                              double scale, bool save_backward);
std::vector<torch::Tensor> packed_relation_backward(
    const torch::Tensor& query, const torch::Tensor& key_value,
    const torch::Tensor& mask, const torch::Tensor& codes,
    const torch::Tensor& table, const torch::Tensor& probabilities,
    const torch::Tensor& float_output, const torch::Tensor& grad_output,
    int64_t heads, int64_t entries, double scale);
}  // namespace rocm
#endif
}  // namespace djl::pytorch
