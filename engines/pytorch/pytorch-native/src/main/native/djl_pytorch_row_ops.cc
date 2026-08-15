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

#include "djl_pytorch_row_ops.h"

#include <torch/csrc/autograd/custom_function.h>

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_rocm_kernels.h"
#endif

namespace djl::pytorch {
namespace {

torch::Tensor scatter_rows_reference(
    const torch::Tensor& rows, const torch::Tensor& row_indices, int64_t row_count) {
  auto output_shape = rows.sizes().vec();
  output_shape[0] = row_count;
  return torch::zeros(output_shape, rows.options()).index_copy(0, row_indices, rows);
}

#if defined(DJL_USE_ROCM_KERNELS)

class ScatterRowsFunction : public torch::autograd::Function<ScatterRowsFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& rows, const torch::Tensor& row_indices,
      const torch::Tensor& output) {
    context->save_for_backward({row_indices});
    return rocm::scatter_rows_forward(rows, row_indices, output);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context, torch::autograd::variable_list gradient_outputs) {
    auto row_indices = context->get_saved_variables().at(0);
    auto gradient_rows = rocm::scatter_rows_backward(gradient_outputs.at(0), row_indices);
    return {gradient_rows, torch::Tensor(), torch::Tensor()};
  }
};

#endif

}  // namespace

torch::Tensor scatter_rows(
    const torch::Tensor& rows, const torch::Tensor& row_indices, int64_t row_count) {
  TORCH_CHECK(rows.dim() > 0, "scatter rows requires at least one row dimension");
  TORCH_CHECK(row_count >= 0, "scatter row count must be nonnegative");
  auto indices = row_indices.reshape({-1}).contiguous();
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "scatter row indices must be int64");
  TORCH_CHECK(indices.numel() == rows.size(0),
      "scatter row index count must equal the number of compact rows");
#if defined(DJL_USE_ROCM_KERNELS)
  if (rocm::supports_scatter_rows(rows, indices)) {
    auto output_shape = rows.sizes().vec();
    output_shape[0] = row_count;
    auto output = torch::empty(output_shape, rows.options());
    if (at::GradMode::is_enabled() && rows.requires_grad()) {
      return ScatterRowsFunction::apply(rows, indices, output);
    }
    return rocm::scatter_rows_forward(rows, indices, output);
  }
#endif
  return scatter_rows_reference(rows, indices, row_count);
}

}  // namespace djl::pytorch
