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

#include "djl_pytorch_routing_masks_cuda.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <optional>

namespace djl::pytorch::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaximumCategoricalMaskRules = 32;

struct CategoricalMaskRules {
  int32_t count;
  int64_t fields[kMaximumCategoricalMaskRules];
  uint64_t category_sets[kMaximumCategoricalMaskRules];
};

bool is_fused_floating(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kFloat32 ||
      tensor.scalar_type() == torch::kFloat16 ||
      tensor.scalar_type() == torch::kBFloat16;
}

bool is_supported_index_type(const torch::Tensor& tensor) {
  const auto type = tensor.scalar_type();
  return type == torch::kUInt8 || type == torch::kInt8 || type == torch::kInt16 ||
      type == torch::kInt32 || type == torch::kInt64;
}

std::optional<int64_t> uniform_element_stride(const torch::Tensor& tensor) {
  if (tensor.numel() == 0 || tensor.dim() == 0) {
    return 1;
  }
  const int64_t stride = tensor.stride(-1);
  if (stride <= 0) {
    return std::nullopt;
  }
  int64_t expected_stride = stride;
  for (int64_t dimension = tensor.dim() - 1; dimension >= 0; --dimension) {
    if (tensor.size(dimension) > 1 && tensor.stride(dimension) != expected_stride) {
      return std::nullopt;
    }
    expected_stride *= tensor.size(dimension);
  }
  return stride;
}

int launch_blocks(int64_t elements) {
  return static_cast<int>(std::min<int64_t>(
      (elements + kThreadsPerBlock - 1) / kThreadsPerBlock, 65535));
}

template <typename index_t, typename scalar_t>
__global__ void categorical_masks_kernel(const index_t* categories, const scalar_t* mask,
    scalar_t* output, int64_t row_count, int64_t field_count, CategoricalMaskRules rules) {
  const int64_t element_count = row_count * rules.count;
  for (int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_index < element_count;
       output_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = output_index / rules.count;
    const int rule = static_cast<int>(output_index - row * rules.count);
    const int64_t category =
        static_cast<int64_t>(categories[row * field_count + rules.fields[rule]]);
    const bool selected = category >= 0 && category < 64 &&
        ((rules.category_sets[rule] >> category) & uint64_t{1}) != 0;
    // Preserve the reference multiply, including non-finite mask values.
    output[output_index] = static_cast<scalar_t>(
        static_cast<float>(selected) * static_cast<float>(mask[row]));
  }
}

template <typename index_t, typename scalar_t>
__global__ void binary_choice_masks_kernel(const index_t* routes, const scalar_t* first_mask,
    const scalar_t* second_mask, scalar_t* output, int64_t row_count, int64_t field_count,
    int64_t first_mask_stride, int64_t second_mask_stride, int64_t representative_field,
    int64_t first_route_field, int64_t second_route_field, int64_t padding_value) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < row_count;
       row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    // The reference rounds the sum to the mask dtype before applying each mask.
    const scalar_t combined = static_cast<scalar_t>(
        static_cast<float>(first_mask[row * first_mask_stride]) +
        static_cast<float>(second_mask[row * second_mask_stride]));
    const int64_t route_offset = row * field_count;
    const int64_t output_offset = row * 4;
    output[output_offset] = combined;
    output[output_offset + 1] = static_cast<scalar_t>(
        static_cast<float>(routes[route_offset + first_route_field] != padding_value) *
        static_cast<float>(combined));
    output[output_offset + 2] = static_cast<scalar_t>(
        static_cast<float>(routes[route_offset + second_route_field] != padding_value) *
        static_cast<float>(combined));
    output[output_offset + 3] = static_cast<scalar_t>(
        static_cast<float>(routes[route_offset + representative_field] != padding_value) *
        static_cast<float>(combined));
  }
}

}  // namespace

bool supports_categorical_masks(
    const torch::Tensor& categories, const torch::Tensor& mask, size_t rule_count) {
  return categories.is_cuda() && categories.is_contiguous() &&
      is_supported_index_type(categories) &&
      mask.is_cuda() && mask.is_contiguous() && is_fused_floating(mask) &&
      categories.device() == mask.device() &&
      rule_count > 0 && rule_count <= kMaximumCategoricalMaskRules;
}

bool supports_binary_choice_masks(const torch::Tensor& routes,
    const torch::Tensor& first_mask, const torch::Tensor& second_mask) {
  return routes.is_cuda() && routes.is_contiguous() &&
      is_supported_index_type(routes) &&
      first_mask.is_cuda() && uniform_element_stride(first_mask).has_value() &&
      second_mask.is_cuda() && uniform_element_stride(second_mask).has_value() &&
      is_fused_floating(first_mask) &&
      second_mask.scalar_type() == first_mask.scalar_type() &&
      routes.device() == first_mask.device() && second_mask.device() == first_mask.device();
}

torch::Tensor categorical_masks(const torch::Tensor& categories, const torch::Tensor& mask,
    const std::vector<int64_t>& field_indices, const std::vector<uint64_t>& category_sets) {
  TORCH_CHECK(supports_categorical_masks(categories, mask, field_indices.size()),
      "categorical masks received an unsupported CUDA tensor layout");
  CategoricalMaskRules rules{};
  rules.count = static_cast<int32_t>(field_indices.size());
  for (size_t index = 0; index < field_indices.size(); ++index) {
    rules.fields[index] = field_indices[index];
    rules.category_sets[index] = category_sets[index];
  }
  auto output_shape = categories.sizes().vec();
  output_shape.back() = rules.count;
  auto output = torch::empty(output_shape, mask.options().requires_grad(false));
  if (output.numel() == 0) {
    return output;
  }
  c10::cuda::CUDAGuard device_guard(categories.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(categories.get_device()).stream();
  AT_DISPATCH_INTEGRAL_TYPES(categories.scalar_type(), "categorical_masks_index", [&] {
    using index_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf, torch::kBFloat16, mask.scalar_type(), "categorical_masks", [&] {
          categorical_masks_kernel<index_t, scalar_t>
              <<<launch_blocks(output.numel()), kThreadsPerBlock, 0, stream>>>(
                  categories.data_ptr<index_t>(), mask.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(), mask.numel(), categories.size(-1), rules);
        });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor binary_choice_masks(const torch::Tensor& routes, const torch::Tensor& first_mask,
    const torch::Tensor& second_mask, int64_t representative_field, int64_t first_route_field,
    int64_t second_route_field, int64_t padding_value) {
  TORCH_CHECK(supports_binary_choice_masks(routes, first_mask, second_mask),
      "binary choice masks received an unsupported CUDA tensor layout");
  auto output_shape = routes.sizes().vec();
  output_shape.back() = 4;
  auto output = torch::empty(output_shape, first_mask.options().requires_grad(false));
  if (output.numel() == 0) {
    return output;
  }
  c10::cuda::CUDAGuard device_guard(routes.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(routes.get_device()).stream();
  const int64_t first_mask_stride = *uniform_element_stride(first_mask);
  const int64_t second_mask_stride = *uniform_element_stride(second_mask);
  AT_DISPATCH_INTEGRAL_TYPES(routes.scalar_type(), "binary_choice_masks_index", [&] {
    using index_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf, torch::kBFloat16, first_mask.scalar_type(), "binary_choice_masks", [&] {
          binary_choice_masks_kernel<index_t, scalar_t>
              <<<launch_blocks(first_mask.numel()), kThreadsPerBlock, 0, stream>>>(
                  routes.data_ptr<index_t>(), first_mask.data_ptr<scalar_t>(),
                  second_mask.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                  first_mask.numel(), routes.size(-1), first_mask_stride, second_mask_stride,
                  representative_field, first_route_field, second_route_field, padding_value);
        });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace djl::pytorch::cuda
