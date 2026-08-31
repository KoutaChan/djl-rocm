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

#include <ATen/ops/embedding.h>
#include <ATen/ops/embedding_dense_backward.h>
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

torch::Tensor segmented_lookup_sum_reference(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices) {
  const int64_t segment_count = stored_indices.size(-1);
  const int64_t entries_per_segment = lookup_table.size(0) / segment_count;
  auto offsets = torch::arange(segment_count, stored_indices.options().dtype(torch::kInt64))
                     .mul(entries_per_segment);
  auto row_indices = stored_indices.to(torch::kInt64)
                         .clamp(1, entries_per_segment)
                         .sub(1)
                         .add(offsets)
                         .reshape({-1});
  auto gathered_shape = stored_indices.sizes().vec();
  gathered_shape.insert(
      gathered_shape.end(), lookup_table.sizes().begin() + 1, lookup_table.sizes().end());
  return lookup_table.index_select(0, row_indices)
      .reshape(gathered_shape)
      .sum(stored_indices.dim() - 1);
}

bool is_index_type(const torch::Tensor& indices) {
  return indices.scalar_type() == torch::kInt16 || indices.scalar_type() == torch::kInt32 ||
      indices.scalar_type() == torch::kInt64;
}

bool broadcasts_to(const torch::Tensor& source, const torch::Tensor& destination) {
  if (source.dim() > destination.dim()) {
    return false;
  }
  for (int64_t axis = 1; axis <= source.dim(); ++axis) {
    const int64_t source_size = source.size(source.dim() - axis);
    const int64_t destination_size = destination.size(destination.dim() - axis);
    if (source_size != 1 && source_size != destination_size) {
      return false;
    }
  }
  return true;
}

torch::Tensor embedding_with_offsets_reference(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table) {
  return at::embedding(table, raw_ids.add(offsets));
}

torch::Tensor embedding_feature_pack_reference(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features) {
  auto embeddings = embedding_with_offsets_reference(raw_ids, offsets, table);
  return torch::cat(
      {embeddings.reshape({raw_ids.size(0), -1}), features}, 1);
}

std::vector<int64_t> padded_gather_output_shape(
    const torch::Tensor& source, const torch::Tensor& stored_indices, int64_t indexed_dimensions) {
  auto output_shape = stored_indices.sizes().vec();
  output_shape.insert(
      output_shape.end(), source.sizes().begin() + indexed_dimensions, source.sizes().end());
  return output_shape;
}

torch::Tensor padded_gather_reference(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& outer_stored_indices,
    const torch::Tensor& inner_stored_indices) {
  const bool has_inner_indices = inner_stored_indices.defined();
  const int64_t indexed_dimensions = has_inner_indices ? 3 : 2;
  const int64_t outer_entries = source.size(1);
  const int64_t inner_entries = has_inner_indices ? source.size(2) : 1;
  auto outer = outer_stored_indices.to(torch::kInt64);
  auto present = outer.gt(0).logical_and(outer.le(outer_entries));
  auto row_indices = outer.clamp(1, outer_entries).sub(1);
  if (has_inner_indices) {
    auto inner = inner_stored_indices.to(torch::kInt64);
    present = present.logical_and(inner.gt(0)).logical_and(inner.le(inner_entries));
    row_indices = row_indices.mul(inner_entries).add(inner.clamp(1, inner_entries).sub(1));
  }
  auto batches = batch_indices.to(torch::kInt64);
  present = present.logical_and(batches.ge(0)).logical_and(batches.lt(source.size(0)));
  row_indices = batches.clamp(0, source.size(0) - 1)
                    .mul(outer_entries * inner_entries)
                    .add(row_indices)
                    .reshape({-1});

  const int64_t table_rows = source.size(0) * outer_entries * inner_entries;
  const int64_t row_width = source.numel() / table_rows;
  auto output_shape =
      padded_gather_output_shape(source, outer_stored_indices, indexed_dimensions);
  auto gathered = source.reshape({table_rows, row_width}).index_select(0, row_indices);
  auto padding_mask = present.to(source.scalar_type());
  for (int64_t axis = indexed_dimensions; axis < source.dim(); ++axis) {
    padding_mask = padding_mask.unsqueeze(-1);
  }
  return gathered.reshape(output_shape).mul(padding_mask);
}

torch::Tensor implicit_batch_indices(const torch::Tensor& stored_indices, int64_t batch_count) {
  auto shape = std::vector<int64_t>(stored_indices.dim(), 1);
  shape[0] = batch_count;
  return torch::arange(batch_count, stored_indices.options().dtype(torch::kInt64))
      .reshape(shape)
      .expand(stored_indices.sizes());
}

#if defined(DJL_USE_ROCM_KERNELS)

class EmbeddingWithOffsetsFunction
    : public torch::autograd::Function<EmbeddingWithOffsetsFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* context,
      const torch::Tensor& raw_ids, const torch::Tensor& offsets,
      const torch::Tensor& table) {
    context->save_for_backward({raw_ids, offsets});
    context->saved_data["table_rows"] = table.size(0);
    return rocm::embedding_with_offsets_forward(raw_ids, offsets, table);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradient_outputs) {
    auto saved = context->get_saved_variables();
    auto indices = saved.at(0).add(saved.at(1));
    const int64_t table_rows = context->saved_data["table_rows"].toInt();
    auto table_gradient = at::embedding_dense_backward(
        gradient_outputs.at(0), indices, table_rows, -1, false);
    return {torch::Tensor(), torch::Tensor(), table_gradient};
  }
};

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

torch::Tensor embedding_with_offsets(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table) {
  TORCH_CHECK(is_index_type(raw_ids), "raw IDs must be int16, int32, or int64");
  TORCH_CHECK(is_index_type(offsets), "offsets must be int16, int32, or int64");
  TORCH_CHECK(!(raw_ids.scalar_type() == torch::kInt16 &&
                  offsets.scalar_type() == torch::kInt16),
      "raw IDs and offsets cannot both be int16");
  TORCH_CHECK(broadcasts_to(offsets, raw_ids),
      "offsets must be broadcastable to raw IDs");
  TORCH_CHECK(table.dim() == 2 && (table.is_floating_point() || table.is_complex()),
      "embedding table must be a rank-two floating-point tensor");
  TORCH_CHECK(raw_ids.device() == offsets.device() && raw_ids.device() == table.device(),
      "raw IDs, offsets, and embedding table must use the same device");
#if defined(DJL_USE_ROCM_KERNELS)
  if (rocm::supports_embedding_with_offsets(raw_ids, offsets, table)) {
    if (at::GradMode::is_enabled() && table.requires_grad()) {
      return EmbeddingWithOffsetsFunction::apply(raw_ids, offsets, table);
    }
    return rocm::embedding_with_offsets_forward(raw_ids, offsets, table);
  }
#endif
  return embedding_with_offsets_reference(raw_ids, offsets, table);
}

torch::Tensor embedding_feature_pack(const torch::Tensor& raw_ids,
    const torch::Tensor& offsets, const torch::Tensor& table,
    const torch::Tensor& features) {
  TORCH_CHECK(raw_ids.dim() == 2,
      "embedding feature pack IDs must be shaped [rows, fields]");
  TORCH_CHECK(features.dim() == 2 && features.size(0) == raw_ids.size(0) &&
          features.size(1) > 0,
      "embedding feature pack features must be shaped [rows, width]");
  TORCH_CHECK(features.scalar_type() == table.scalar_type() && features.is_floating_point(),
      "embedding feature pack table and features must use one floating-point type");
  TORCH_CHECK(features.device() == raw_ids.device(),
      "embedding feature pack inputs must use the same device");
  TORCH_CHECK(is_index_type(raw_ids), "raw IDs must be int16, int32, or int64");
  TORCH_CHECK(is_index_type(offsets), "offsets must be int16, int32, or int64");
  TORCH_CHECK(!(raw_ids.scalar_type() == torch::kInt16 &&
                  offsets.scalar_type() == torch::kInt16),
      "raw IDs and offsets cannot both be int16");
  TORCH_CHECK(broadcasts_to(offsets, raw_ids),
      "offsets must be broadcastable to raw IDs");
  TORCH_CHECK(table.dim() == 2 && table.is_floating_point(),
      "embedding table must be a rank-two floating-point tensor");
  TORCH_CHECK(raw_ids.device() == offsets.device() && raw_ids.device() == table.device(),
      "raw IDs, offsets, table, and features must use the same device");
#if defined(DJL_USE_ROCM_KERNELS)
  if (rocm::supports_embedding_feature_pack(raw_ids, offsets, table, features) &&
      !(at::GradMode::is_enabled() && (table.requires_grad() || features.requires_grad()))) {
    return rocm::embedding_feature_pack_forward(raw_ids, offsets, table, features);
  }
#endif
  return embedding_feature_pack_reference(raw_ids, offsets, table, features);
}

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

torch::Tensor segmented_lookup_sum(
    const torch::Tensor& lookup_table, const torch::Tensor& stored_indices) {
  TORCH_CHECK(lookup_table.dim() >= 2,
      "segmented lookup sum requires a table with at least two dimensions");
  TORCH_CHECK(stored_indices.dim() >= 1 && stored_indices.size(-1) > 0,
      "segmented lookup sum requires a nonempty trailing segment dimension");
  TORCH_CHECK(lookup_table.size(0) > 0 &&
          lookup_table.size(0) % stored_indices.size(-1) == 0,
      "lookup table rows must divide evenly across the stored-index segments");
  TORCH_CHECK(is_index_type(stored_indices),
      "segmented lookup indices must be int16, int32, or int64");
  TORCH_CHECK(lookup_table.device() == stored_indices.device(),
      "segmented lookup table and stored indices must share a device");
#if defined(DJL_USE_ROCM_KERNELS)
  if (!at::GradMode::is_enabled() &&
      rocm::supports_segmented_lookup_sum(lookup_table, stored_indices)) {
    return rocm::segmented_lookup_sum_forward(lookup_table, stored_indices);
  }
#endif
  return segmented_lookup_sum_reference(lookup_table, stored_indices);
}

torch::Tensor padded_batch_gather(
    const torch::Tensor& source, const torch::Tensor& stored_indices) {
  TORCH_CHECK(source.dim() >= 2,
      "padded batch gather requires source with at least two dimensions");
  TORCH_CHECK(stored_indices.dim() >= 1 && stored_indices.size(0) == source.size(0),
      "stored indices must begin with the source batch dimension");
  TORCH_CHECK(source.size(0) > 0 && source.size(1) > 0,
      "padded batch gather requires nonempty batch and table dimensions");
  TORCH_CHECK(is_index_type(stored_indices), "stored indices must be int16, int32, or int64");
#if defined(DJL_USE_ROCM_KERNELS)
  if (!at::GradMode::is_enabled() &&
      rocm::supports_padded_batch_gather(source, stored_indices)) {
    return rocm::padded_batch_gather_forward(source, stored_indices);
  }
#endif
  return padded_gather_reference(
      source, implicit_batch_indices(stored_indices, source.size(0)), stored_indices, {});
}

torch::Tensor padded_batch_gather_2d(const torch::Tensor& source,
    const torch::Tensor& outer_stored_indices, const torch::Tensor& inner_stored_indices) {
  TORCH_CHECK(source.dim() >= 3,
      "two-dimensional padded batch gather requires source with at least three dimensions");
  TORCH_CHECK(outer_stored_indices.sizes() == inner_stored_indices.sizes(),
      "outer and inner stored indices must have identical shapes");
  TORCH_CHECK(outer_stored_indices.dim() >= 1 &&
          outer_stored_indices.size(0) == source.size(0),
      "stored indices must begin with the source batch dimension");
  TORCH_CHECK(source.size(0) > 0 && source.size(1) > 0 && source.size(2) > 0,
      "padded batch gather requires nonempty batch and table dimensions");
  TORCH_CHECK(is_index_type(outer_stored_indices) && is_index_type(inner_stored_indices),
      "stored indices must be int16, int32, or int64");
#if defined(DJL_USE_ROCM_KERNELS)
  if (!at::GradMode::is_enabled() && rocm::supports_padded_batch_gather_2d(
          source, outer_stored_indices, inner_stored_indices)) {
    return rocm::padded_batch_gather_2d_forward(
        source, outer_stored_indices, inner_stored_indices);
  }
#endif
  return padded_gather_reference(source,
      implicit_batch_indices(outer_stored_indices, source.size(0)),
      outer_stored_indices, inner_stored_indices);
}

torch::Tensor padded_batch_gather_by_batch_indices(const torch::Tensor& source,
    const torch::Tensor& batch_indices, const torch::Tensor& stored_indices) {
  TORCH_CHECK(source.dim() >= 2,
      "padded batch gather requires source with at least two dimensions");
  TORCH_CHECK(batch_indices.sizes() == stored_indices.sizes(),
      "batch and stored indices must have identical shapes");
  TORCH_CHECK(source.size(0) > 0 && source.size(1) > 0,
      "padded batch gather requires nonempty batch and table dimensions");
  TORCH_CHECK(is_index_type(batch_indices) && is_index_type(stored_indices),
      "batch and stored indices must be int16, int32, or int64");
#if defined(DJL_USE_ROCM_KERNELS)
  if (!at::GradMode::is_enabled() && rocm::supports_padded_batch_gather_by_batch_indices(
          source, batch_indices, stored_indices)) {
    return rocm::padded_batch_gather_by_batch_indices_forward(
        source, batch_indices, stored_indices);
  }
#endif
  return padded_gather_reference(source, batch_indices, stored_indices, {});
}

}  // namespace djl::pytorch
