#include <ATen/Dispatch.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "djl_pytorch_rocm_kernels.h"

namespace djl::pytorch::rocm {
namespace {

constexpr int kElementwiseThreads = 256;
constexpr int kResidualNormThreads = 256;

void check_hip_kernel_launch(const char* kernel_name) {
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, kernel_name, " launch failed: ", hipGetErrorString(error));
}

bool is_fused_floating(const torch::Tensor& tensor) {
  return tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16 ||
         tensor.scalar_type() == torch::kBFloat16;
}

template <typename scalar_t, typename index_t>
__global__ void indexed_relation_bias_kernel(const scalar_t* relation_logits, const scalar_t* relation_bias,
    const index_t* relation_ids, scalar_t* output, int64_t element_count, int64_t heads, int64_t query_tokens,
    int64_t key_tokens, int64_t relation_count, int64_t logits_stride_batch, int64_t logits_stride_head,
    int64_t logits_stride_query, int64_t logits_stride_relation, int64_t bias_stride_batch, int64_t bias_stride_head,
    int64_t bias_stride_query, int64_t bias_stride_key, int64_t relation_id_stride_batch,
    int64_t relation_id_stride_query, int64_t relation_id_stride_key, float scale) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < element_count;
      index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t coordinate = index;
    const int64_t key_token = coordinate % key_tokens;
    coordinate /= key_tokens;
    const int64_t query_token = coordinate % query_tokens;
    coordinate /= query_tokens;
    const int64_t head = coordinate % heads;
    const int64_t batch = coordinate / heads;
    const int64_t relation =
        static_cast<int64_t>(relation_ids[batch * relation_id_stride_batch + query_token * relation_id_stride_query +
                                          key_token * relation_id_stride_key]);
    if (relation < 0 || relation >= relation_count) {
      output[index] = static_cast<scalar_t>(NAN);
      continue;
    }
    const float logit =
        static_cast<float>(relation_logits[batch * logits_stride_batch + head * logits_stride_head +
                                           query_token * logits_stride_query + relation * logits_stride_relation]);
    const float bias = static_cast<float>(relation_bias[batch * bias_stride_batch + head * bias_stride_head +
                                                        query_token * bias_stride_query + key_token * bias_stride_key]);
    output[index] = static_cast<scalar_t>(logit * scale + bias);
  }
}

template <typename scalar_t, typename index_t>
__global__ void grouped_indexed_attention_kernel(const scalar_t* query, const scalar_t* shared_key_values,
    const scalar_t* shared_deltas, const scalar_t* indexed_deltas, const index_t* indexed_shared_ids, scalar_t* output,
    int64_t queries_per_group, int64_t heads, int64_t key_features, int64_t value_features, int64_t shared_tokens,
    int64_t indexed_tokens, int64_t query_stride_query, int64_t query_stride_head, int64_t query_stride_feature,
    int64_t shared_stride_group, int64_t shared_stride_token, int64_t shared_stride_feature,
    int64_t shared_delta_stride_query, int64_t shared_delta_stride_token, int64_t shared_delta_stride_feature,
    int64_t indexed_delta_stride_query, int64_t indexed_delta_stride_token, int64_t indexed_delta_stride_feature,
    int64_t indexed_id_stride_query, int64_t indexed_id_stride_token, float scale) {
  extern __shared__ float token_weights[];
  const int64_t query_head = blockIdx.x;
  const int64_t query_index = query_head / heads;
  const int64_t head = query_head % heads;
  const int64_t group = query_index / queries_per_group;
  const int lane = threadIdx.x;
  const int64_t total_tokens = shared_tokens + indexed_tokens;
  const int64_t key_offset = head * key_features;
  const int64_t value_offset = heads * key_features + head * value_features;

  float local_maximum = -std::numeric_limits<float>::infinity();
  for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
    const bool shared_token = attended_token < shared_tokens;
    const int64_t token = shared_token ? attended_token : attended_token - shared_tokens;
    const int64_t stored_id =
        shared_token ? attended_token + 1
                     : static_cast<int64_t>(
                           indexed_shared_ids[query_index * indexed_id_stride_query + token * indexed_id_stride_token]);
    const bool present = stored_id > 0 && stored_id <= shared_tokens;
    float score = -std::numeric_limits<float>::infinity();
    if (present) {
      score = 0.0f;
      const int64_t shared_index = stored_id - 1;
      for (int64_t feature = 0; feature < key_features; feature++) {
        const int64_t packed_feature = key_offset + feature;
        const float base =
            static_cast<float>(shared_key_values[group * shared_stride_group + shared_index * shared_stride_token +
                                                 packed_feature * shared_stride_feature]);
        const float delta =
            shared_token
                ? static_cast<float>(
                      shared_deltas[query_index * shared_delta_stride_query + token * shared_delta_stride_token +
                                    packed_feature * shared_delta_stride_feature])
                : static_cast<float>(
                      indexed_deltas[query_index * indexed_delta_stride_query + token * indexed_delta_stride_token +
                                     packed_feature * indexed_delta_stride_feature]);
        const float query_value = static_cast<float>(
            query[query_index * query_stride_query + head * query_stride_head + feature * query_stride_feature]);
        score += query_value * (base + delta);
      }
      score *= scale;
    }
    token_weights[attended_token] = score;
    local_maximum = fmaxf(local_maximum, score);
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, offset));
  }
  const float maximum = __shfl(local_maximum, 0);
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t attended_token = lane; attended_token < total_tokens; attended_token += warpSize) {
    const float score = token_weights[attended_token];
    const float weight = score == -std::numeric_limits<float>::infinity() ? 0.0f : expf(score - maximum);
    token_weights[attended_token] = weight;
    local_sum += weight;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down(local_sum, offset);
  }
  const float inverse_sum = 1.0f / __shfl(local_sum, 0);
  __syncthreads();

  for (int64_t feature = lane; feature < value_features; feature += warpSize) {
    const int64_t packed_feature = value_offset + feature;
    float context = 0.0f;
    for (int64_t attended_token = 0; attended_token < total_tokens; attended_token++) {
      const bool shared_token = attended_token < shared_tokens;
      const int64_t token = shared_token ? attended_token : attended_token - shared_tokens;
      const int64_t stored_id =
          shared_token
              ? attended_token + 1
              : static_cast<int64_t>(
                    indexed_shared_ids[query_index * indexed_id_stride_query + token * indexed_id_stride_token]);
      if (stored_id > 0 && stored_id <= shared_tokens) {
        const int64_t shared_index = stored_id - 1;
        const float base =
            static_cast<float>(shared_key_values[group * shared_stride_group + shared_index * shared_stride_token +
                                                 packed_feature * shared_stride_feature]);
        const float delta =
            shared_token
                ? static_cast<float>(
                      shared_deltas[query_index * shared_delta_stride_query + token * shared_delta_stride_token +
                                    packed_feature * shared_delta_stride_feature])
                : static_cast<float>(
                      indexed_deltas[query_index * indexed_delta_stride_query + token * indexed_delta_stride_token +
                                     packed_feature * indexed_delta_stride_feature]);
        context += token_weights[attended_token] * (base + delta);
      }
    }
    output[(query_index * heads + head) * value_features + feature] = static_cast<scalar_t>(context * inverse_sum);
  }
}

template <typename residual_t, typename update_t, typename parameter_t>
__global__ void residual_add_layer_norm_in_place_kernel(residual_t* residual, const update_t* update,
    const parameter_t* weight, const parameter_t* bias, residual_t* output, int64_t row_count, int64_t width,
    float epsilon) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (row >= row_count) {
    return;
  }

  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * width;
  for (int64_t feature = lane; feature < width; feature += warpSize) {
    const int64_t index = row_offset + feature;
    const float value = static_cast<float>(residual[index]) + static_cast<float>(update[index]);
    residual[index] = static_cast<residual_t>(value);
    count++;
    const float delta = value - mean;
    mean += delta / count;
    moment += delta * (value - mean);
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int other_count = __shfl_down(count, offset);
    if (lane + offset < warpSize && other_count > 0) {
      const int combined_count = count + other_count;
      const float delta = other_mean - mean;
      moment += other_moment + delta * delta * static_cast<float>(count) * static_cast<float>(other_count) /
                                   static_cast<float>(combined_count);
      mean += delta * static_cast<float>(other_count) / static_cast<float>(combined_count);
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / static_cast<float>(width) + epsilon);

  for (int64_t feature = lane; feature < width; feature += warpSize) {
    const int64_t index = row_offset + feature;
    const float normalized = (static_cast<float>(residual[index]) - mean) * inverse_standard_deviation;
    output[index] =
        static_cast<residual_t>(normalized * static_cast<float>(weight[feature]) + static_cast<float>(bias[feature]));
  }
}

hipDeviceProp_t device_properties(int device) {
  hipDeviceProp_t properties;
  C10_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  TORCH_CHECK(
      properties.warpSize == 32 || properties.warpSize == 64, "ROCm kernels require a 32- or 64-lane wavefront");
  return properties;
}

int residual_norm_waves_per_block(int device) {
  const auto properties = device_properties(device);
  return kResidualNormThreads / properties.warpSize;
}

}  // namespace

torch::Tensor indexed_relation_bias(const torch::Tensor& relation_logits, const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids, float scale) {
  TORCH_CHECK(relation_logits.is_cuda(), "indexed relation bias requires a ROCm tensor");
  TORCH_CHECK(is_fused_floating(relation_logits), "relation logits have unsupported dtype");
  TORCH_CHECK(relation_logits.scalar_type() == relation_bias.scalar_type(), "relation logits/bias dtype mismatch");
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt32 || relation_ids.scalar_type() == torch::kInt64,
      "relation IDs must be int32 or int64");
  TORCH_CHECK(relation_logits.dim() == 4, "relation logits must be rank 4");
  const int64_t batch = relation_logits.size(0);
  const int64_t heads = relation_logits.size(1);
  const int64_t query_tokens = relation_logits.size(2);
  const int64_t relation_count = relation_logits.size(3);
  TORCH_CHECK(query_tokens > 0 && relation_count > 0, "relation logits cannot be empty");
  TORCH_CHECK(relation_ids.dim() == 2 || relation_ids.dim() == 3, "relation IDs must be rank 2 or 3");
  TORCH_CHECK(relation_ids.size(-2) == query_tokens, "relation ID query dimension mismatch");
  TORCH_CHECK(relation_ids.dim() == 2 || relation_ids.size(0) == batch, "relation ID batch dimension mismatch");
  const int64_t key_tokens = relation_ids.size(-1);
  TORCH_CHECK(key_tokens > 0, "relation IDs cannot have an empty key dimension");
  TORCH_CHECK(relation_bias.device() == relation_logits.device() && relation_ids.device() == relation_logits.device(),
      "indexed relation bias inputs must be on the same device");

  c10::DeviceGuard device_guard(relation_logits.device());
  torch::Tensor expanded_bias = relation_bias.expand({batch, heads, query_tokens, key_tokens});
  torch::Tensor output = torch::empty({batch, heads, query_tokens, key_tokens},
      relation_logits.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t element_count = output.numel();
  if (element_count == 0) {
    return output;
  }
  const int blocks =
      static_cast<int>(std::min<int64_t>((element_count + kElementwiseThreads - 1) / kElementwiseThreads, 4096));
  const int64_t id_batch_stride = relation_ids.dim() == 2 ? 0 : relation_ids.stride(0);
  const int64_t id_query_stride = relation_ids.dim() == 2 ? relation_ids.stride(0) : relation_ids.stride(1);
  const int64_t id_key_stride = relation_ids.dim() == 2 ? relation_ids.stride(1) : relation_ids.stride(2);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(relation_logits.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, relation_logits.scalar_type(), "indexed_relation_bias", [&] {
        if (relation_ids.scalar_type() == torch::kInt32) {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(indexed_relation_bias_kernel<scalar_t, int32_t>), dim3(blocks),
              dim3(kElementwiseThreads), 0, stream, relation_logits.data_ptr<scalar_t>(),
              expanded_bias.data_ptr<scalar_t>(), relation_ids.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
              element_count, heads, query_tokens, key_tokens, relation_count, relation_logits.stride(0),
              relation_logits.stride(1), relation_logits.stride(2), relation_logits.stride(3), expanded_bias.stride(0),
              expanded_bias.stride(1), expanded_bias.stride(2), expanded_bias.stride(3), id_batch_stride,
              id_query_stride, id_key_stride, scale);
        } else {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(indexed_relation_bias_kernel<scalar_t, int64_t>), dim3(blocks),
              dim3(kElementwiseThreads), 0, stream, relation_logits.data_ptr<scalar_t>(),
              expanded_bias.data_ptr<scalar_t>(), relation_ids.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
              element_count, heads, query_tokens, key_tokens, relation_count, relation_logits.stride(0),
              relation_logits.stride(1), relation_logits.stride(2), relation_logits.stride(3), expanded_bias.stride(0),
              expanded_bias.stride(1), expanded_bias.stride(2), expanded_bias.stride(3), id_batch_stride,
              id_query_stride, id_key_stride, scale);
        }
      });
  check_hip_kernel_launch("indexed_relation_bias");
  return output;
}

torch::Tensor grouped_indexed_scaled_dot_product_attention(const torch::Tensor& query,
    const torch::Tensor& shared_key_values, const torch::Tensor& shared_deltas, const torch::Tensor& indexed_deltas,
    const torch::Tensor& indexed_shared_ids, int64_t queries_per_group, float scale) {
  TORCH_CHECK(query.is_cuda(), "grouped indexed attention requires a ROCm tensor");
  TORCH_CHECK(query.dim() == 3, "query must be rank 3");
  TORCH_CHECK(shared_key_values.dim() == 3, "shared key/value data must be rank 3");
  TORCH_CHECK(shared_deltas.dim() == 3, "shared deltas must be rank 3");
  TORCH_CHECK(indexed_deltas.dim() == 3, "indexed deltas must be rank 3");
  TORCH_CHECK(indexed_shared_ids.dim() == 2, "indexed shared IDs must be rank 2");
  TORCH_CHECK(queries_per_group > 0, "queries per group must be positive");
  const int64_t query_count = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t key_features = query.size(2);
  const int64_t group_count = shared_key_values.size(0);
  const int64_t shared_tokens = shared_key_values.size(1);
  const int64_t packed_width = shared_key_values.size(2);
  const int64_t indexed_tokens = indexed_deltas.size(1);
  const int64_t key_width = heads * key_features;
  TORCH_CHECK(query_count == group_count * queries_per_group, "query/group shape mismatch");
  TORCH_CHECK(heads > 0 && key_features > 0 && shared_tokens > 0, "attention dimensions must be positive");
  TORCH_CHECK(packed_width > key_width && (packed_width - key_width) % heads == 0,
      "packed key/value width is incompatible with the query");
  const int64_t value_features = (packed_width - key_width) / heads;
  TORCH_CHECK(shared_deltas.size(0) == query_count && shared_deltas.size(1) == shared_tokens &&
                  shared_deltas.size(2) == packed_width,
      "shared delta shape mismatch");
  TORCH_CHECK(indexed_deltas.size(0) == query_count && indexed_deltas.size(1) == indexed_tokens &&
                  indexed_deltas.size(2) == packed_width,
      "indexed delta shape mismatch");
  TORCH_CHECK(indexed_shared_ids.size(0) == query_count && indexed_shared_ids.size(1) == indexed_tokens,
      "indexed shared ID shape mismatch");
  TORCH_CHECK(query.scalar_type() == shared_key_values.scalar_type() &&
                  query.scalar_type() == shared_deltas.scalar_type() &&
                  query.scalar_type() == indexed_deltas.scalar_type(),
      "grouped indexed attention dtype mismatch");
  TORCH_CHECK(is_fused_floating(query), "grouped indexed attention has unsupported dtype");
  TORCH_CHECK(indexed_shared_ids.scalar_type() == torch::kInt32 || indexed_shared_ids.scalar_type() == torch::kInt64,
      "indexed shared IDs must be int32 or int64");
  TORCH_CHECK(shared_key_values.device() == query.device() && shared_deltas.device() == query.device() &&
                  indexed_deltas.device() == query.device() && indexed_shared_ids.device() == query.device(),
      "grouped indexed attention inputs must be on the same device");

  c10::DeviceGuard device_guard(query.device());
  torch::Tensor output = torch::empty(
      {query_count, heads, value_features}, query.options().memory_format(torch::MemoryFormat::Contiguous));
  if (query_count == 0) {
    return output;
  }
  const auto properties = device_properties(query.get_device());
  const int64_t total_tokens = shared_tokens + indexed_tokens;
  const size_t shared_memory_bytes = static_cast<size_t>(total_tokens) * sizeof(float);
  TORCH_CHECK(shared_memory_bytes <= static_cast<size_t>(properties.sharedMemPerBlock),
      "grouped indexed attention token count exceeds per-block shared memory");
  const int64_t grid_size = query_count * heads;
  TORCH_CHECK(
      grid_size <= static_cast<int64_t>(properties.maxGridSize[0]), "grouped indexed attention grid is too large");
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(), "grouped_indexed_attention", [&] {
        if (indexed_shared_ids.scalar_type() == torch::kInt32) {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_indexed_attention_kernel<scalar_t, int32_t>), dim3(grid_size),
              dim3(properties.warpSize), shared_memory_bytes, stream, query.data_ptr<scalar_t>(),
              shared_key_values.data_ptr<scalar_t>(), shared_deltas.data_ptr<scalar_t>(),
              indexed_deltas.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
              queries_per_group, heads, key_features, value_features, shared_tokens, indexed_tokens, query.stride(0),
              query.stride(1), query.stride(2), shared_key_values.stride(0), shared_key_values.stride(1),
              shared_key_values.stride(2), shared_deltas.stride(0), shared_deltas.stride(1), shared_deltas.stride(2),
              indexed_deltas.stride(0), indexed_deltas.stride(1), indexed_deltas.stride(2),
              indexed_shared_ids.stride(0), indexed_shared_ids.stride(1), scale);
        } else {
          hipLaunchKernelGGL(HIP_KERNEL_NAME(grouped_indexed_attention_kernel<scalar_t, int64_t>), dim3(grid_size),
              dim3(properties.warpSize), shared_memory_bytes, stream, query.data_ptr<scalar_t>(),
              shared_key_values.data_ptr<scalar_t>(), shared_deltas.data_ptr<scalar_t>(),
              indexed_deltas.data_ptr<scalar_t>(), indexed_shared_ids.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
              queries_per_group, heads, key_features, value_features, shared_tokens, indexed_tokens, query.stride(0),
              query.stride(1), query.stride(2), shared_key_values.stride(0), shared_key_values.stride(1),
              shared_key_values.stride(2), shared_deltas.stride(0), shared_deltas.stride(1), shared_deltas.stride(2),
              indexed_deltas.stride(0), indexed_deltas.stride(1), indexed_deltas.stride(2),
              indexed_shared_ids.stride(0), indexed_shared_ids.stride(1), scale);
        }
      });
  check_hip_kernel_launch("grouped_indexed_attention");
  return output;
}

torch::Tensor residual_add_layer_norm_in_place(torch::Tensor& residual, const torch::Tensor& update,
    const torch::Tensor& weight, const torch::Tensor& bias, float epsilon) {
  TORCH_CHECK(residual.is_cuda(), "residual add LayerNorm requires a ROCm tensor");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(update.is_contiguous(), "residual update must be contiguous");
  TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "LayerNorm parameters must be contiguous");
  TORCH_CHECK(residual.sizes() == update.sizes(), "residual/update shape mismatch");
  TORCH_CHECK(is_fused_floating(residual) && is_fused_floating(update), "residual/update dtype is unsupported");
  TORCH_CHECK(residual.dim() >= 1 && residual.size(-1) > 0, "residual must have a non-empty trailing dimension");
  const int64_t width = residual.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == width, "LayerNorm weight shape mismatch");
  TORCH_CHECK(bias.sizes() == weight.sizes(), "LayerNorm bias shape mismatch");
  TORCH_CHECK(weight.scalar_type() == bias.scalar_type(), "LayerNorm parameter dtype mismatch");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == residual.scalar_type(),
      "LayerNorm parameters must be float32 or match the residual dtype");
  TORCH_CHECK(update.device() == residual.device() && weight.device() == residual.device() &&
                  bias.device() == residual.device(),
      "residual add LayerNorm inputs must be on the same device");

  c10::DeviceGuard device_guard(residual.device());
  torch::Tensor output = torch::empty_like(residual, residual.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t row_count = residual.numel() / width;
  if (row_count == 0) {
    return output;
  }
  const int waves_per_block = residual_norm_waves_per_block(residual.get_device());
  const dim3 grid((row_count + waves_per_block - 1) / waves_per_block);
  const dim3 block(kResidualNormThreads);
  const hipStream_t stream = c10::hip::getCurrentHIPStream(residual.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, residual.scalar_type(), "residual_add_layer_norm", [&] {
        using residual_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, update.scalar_type(), "residual_add_layer_norm_update", [&] {
              using update_t = scalar_t;
              if (weight.scalar_type() == torch::kFloat32) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(residual_add_layer_norm_in_place_kernel<residual_t, update_t, float>), grid, block,
                    0, stream, residual.data_ptr<residual_t>(), update.data_ptr<update_t>(), weight.data_ptr<float>(),
                    bias.data_ptr<float>(), output.data_ptr<residual_t>(), row_count, width, epsilon);
              } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(residual_add_layer_norm_in_place_kernel<residual_t, update_t, residual_t>), grid,
                    block, 0, stream, residual.data_ptr<residual_t>(), update.data_ptr<update_t>(),
                    weight.data_ptr<residual_t>(), bias.data_ptr<residual_t>(), output.data_ptr<residual_t>(),
                    row_count, width, epsilon);
              }
            });
      });
  check_hip_kernel_launch("residual_add_layer_norm");
  return output;
}

}  // namespace djl::pytorch::rocm
