#include "djl_pytorch_rocm_kernels.h"

#include <ATen/Dispatch.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>

namespace djl::pytorch::rocm {
namespace {

constexpr int kTileCount = 34;
constexpr int kHeadSize = 16;
constexpr int kRelationCount = 23;
constexpr int kThreads = 256;
constexpr int kMaximumWaves = 8;
constexpr int kMaskThreads = 256;
constexpr int kTransitionTileCount = 34;
constexpr int kTransitionWaitCount = 13;
constexpr int kTransitionTokenCount = kTransitionTileCount + kTransitionWaitCount;
constexpr int kTransitionHeads = 4;
constexpr int kTransitionKeySize = 8;
constexpr int kTransitionValueSize = 16;
constexpr int kTransitionKeyWidth = kTransitionHeads * kTransitionKeySize;
constexpr int kTransitionThreads = 128;
constexpr int kTransitionMaximumWaves = 4;
constexpr int kResidualNormWidth = 256;
constexpr int kResidualNormThreads = 256;
template <typename scalar_t>
__global__ void tile_relation_mask_kernel(
    const scalar_t* relation_logits,
    const scalar_t* relation_bias,
    const int64_t* relation_ids,
    scalar_t* output,
    int64_t element_count,
    int64_t heads,
    int64_t logits_stride_batch,
    int64_t logits_stride_head,
    int64_t logits_stride_query,
    int64_t logits_stride_relation,
    int64_t bias_stride_head,
    int64_t bias_stride_query,
    int64_t bias_stride_key,
    int64_t relation_id_stride_query,
    int64_t relation_id_stride_key,
    float scale) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t coordinate = index;
    const int64_t key_tile = coordinate % kTileCount;
    coordinate /= kTileCount;
    const int64_t query_tile = coordinate % kTileCount;
    coordinate /= kTileCount;
    const int64_t head = coordinate % heads;
    const int64_t batch = coordinate / heads;
    const int64_t relation =
        relation_ids[query_tile * relation_id_stride_query
                     + key_tile * relation_id_stride_key];
    const float logit = static_cast<float>(
        relation_logits[batch * logits_stride_batch + head * logits_stride_head
                        + query_tile * logits_stride_query
                        + relation * logits_stride_relation]);
    const scalar_t scaled = static_cast<scalar_t>(logit * scale);
    output[index] = static_cast<scalar_t>(
        static_cast<float>(scaled)
        + static_cast<float>(
            relation_bias[head * bias_stride_head + query_tile * bias_stride_query
                          + key_tile * bias_stride_key]));
  }
}

template <typename scalar_t>
__global__ void tile_relation_attention_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const float* relation_key,
    const float* relation_bias,
    const int64_t* relation_ids,
    scalar_t* output,
    int64_t heads,
    int64_t query_stride_batch,
    int64_t query_stride_head,
    int64_t query_stride_tile,
    int64_t query_stride_feature,
    int64_t key_stride_batch,
    int64_t key_stride_head,
    int64_t key_stride_tile,
    int64_t key_stride_feature,
    int64_t value_stride_batch,
    int64_t value_stride_head,
    int64_t value_stride_tile,
    int64_t value_stride_feature,
    int64_t relation_key_stride_head,
    int64_t relation_key_stride_feature,
    int64_t relation_key_stride_relation,
    int64_t relation_bias_stride_head,
    int64_t relation_bias_stride_query,
    int64_t relation_bias_stride_key,
    int64_t relation_id_stride_query,
    int64_t relation_id_stride_key,
    float scale) {
  __shared__ float shared_keys[kTileCount * kHeadSize];
  __shared__ float shared_values[kTileCount * kHeadSize];
  __shared__ float shared_relation_keys[kRelationCount * kHeadSize];
  __shared__ float shared_queries[kMaximumWaves * kHeadSize];
  __shared__ float shared_scores[kMaximumWaves * kTileCount];

  const int64_t batch_head = blockIdx.x;
  const int64_t batch = batch_head / heads;
  const int64_t head = batch_head % heads;
  const int thread = threadIdx.x;

  for (int index = thread; index < kTileCount * kHeadSize; index += blockDim.x) {
    const int tile = index / kHeadSize;
    const int feature = index % kHeadSize;
    shared_keys[index] = static_cast<float>(
        key[batch * key_stride_batch + head * key_stride_head + tile * key_stride_tile
            + feature * key_stride_feature]);
    shared_values[index] = static_cast<float>(
        value[batch * value_stride_batch + head * value_stride_head + tile * value_stride_tile
              + feature * value_stride_feature]);
  }
  for (int index = thread; index < kRelationCount * kHeadSize; index += blockDim.x) {
    const int relation = index / kHeadSize;
    const int feature = index % kHeadSize;
    shared_relation_keys[index] =
        relation_key[head * relation_key_stride_head + feature * relation_key_stride_feature
                     + relation * relation_key_stride_relation];
  }
  __syncthreads();

  const int lane = thread % warpSize;
  const int wave = thread / warpSize;
  const int wave_count = blockDim.x / warpSize;
  for (int query_tile = wave; query_tile < kTileCount; query_tile += wave_count) {
    if (lane < kHeadSize) {
      shared_queries[wave * kHeadSize + lane] = static_cast<float>(
          query[batch * query_stride_batch + head * query_stride_head
                + query_tile * query_stride_tile + lane * query_stride_feature]);
    }
    __syncwarp();

    float local_max = -std::numeric_limits<float>::infinity();
    for (int key_tile = lane; key_tile < kTileCount; key_tile += warpSize) {
      const int64_t relation =
          relation_ids[query_tile * relation_id_stride_query
                       + key_tile * relation_id_stride_key];
      float content_score = 0.0f;
      float relation_score = 0.0f;
      for (int feature = 0; feature < kHeadSize; feature++) {
        const float query_value = shared_queries[wave * kHeadSize + feature];
        content_score += query_value * shared_keys[key_tile * kHeadSize + feature];
        relation_score +=
            query_value * shared_relation_keys[relation * kHeadSize + feature];
      }
      const float score =
          (content_score + relation_score) * scale
          + relation_bias[head * relation_bias_stride_head
                          + query_tile * relation_bias_stride_query
                          + key_tile * relation_bias_stride_key];
      shared_scores[wave * kTileCount + key_tile] = score;
      local_max = fmaxf(local_max, score);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    const float maximum = __shfl(local_max, 0);

    float local_sum = 0.0f;
    for (int key_tile = lane; key_tile < kTileCount; key_tile += warpSize) {
      const float probability = expf(shared_scores[wave * kTileCount + key_tile] - maximum);
      shared_scores[wave * kTileCount + key_tile] = probability;
      local_sum += probability;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down(local_sum, offset);
    }
    const float inverse_sum = 1.0f / __shfl(local_sum, 0);

    for (int feature = lane; feature < kHeadSize; feature += warpSize) {
      float context = 0.0f;
      for (int key_tile = 0; key_tile < kTileCount; key_tile++) {
        context += shared_scores[wave * kTileCount + key_tile]
                   * shared_values[key_tile * kHeadSize + feature];
      }
      const int64_t output_index =
          ((batch * heads + head) * kTileCount + query_tile) * kHeadSize + feature;
      output[output_index] = static_cast<scalar_t>(context * inverse_sum);
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void transition_tile_attention_kernel(
    const scalar_t* query,
    const scalar_t* tile_key_value,
    const scalar_t* relation_key_value,
    const scalar_t* wait_key_value,
    const index_t* wait_tile_ids,
    scalar_t* output,
    int64_t candidates_per_state,
    int64_t query_stride_candidate,
    int64_t query_stride_head,
    int64_t query_stride_feature,
    int64_t tile_stride_state,
    int64_t tile_stride_tile,
    int64_t tile_stride_feature,
    int64_t relation_stride_candidate,
    int64_t relation_stride_tile,
    int64_t relation_stride_feature,
    int64_t wait_stride_candidate,
    int64_t wait_stride_tile,
    int64_t wait_stride_feature,
    int64_t wait_id_stride_candidate,
    int64_t wait_id_stride_tile,
    float scale) {
  __shared__ float token_weights[kTransitionMaximumWaves * kTransitionTokenCount];
  __shared__ float shared_queries[kTransitionMaximumWaves * kTransitionKeySize];

  const int thread = threadIdx.x;
  const int lane = thread % warpSize;
  const int wave = thread / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int head_groups = kTransitionHeads / waves_per_block;
  const int64_t candidate = blockIdx.x / head_groups;
  const int head = (blockIdx.x % head_groups) * waves_per_block + wave;
  const int64_t state = candidate / candidates_per_state;
  if (lane < kTransitionKeySize) {
    shared_queries[wave * kTransitionKeySize + lane] = static_cast<float>(
        query[candidate * query_stride_candidate + head * query_stride_head
              + lane * query_stride_feature]);
  }
  __syncwarp();

  float local_maximum = -std::numeric_limits<float>::infinity();
  for (int attended_token = lane; attended_token < kTransitionTokenCount;
       attended_token += warpSize) {
    const bool tile_token = attended_token < kTransitionTileCount;
    const int token = tile_token ? attended_token : attended_token - kTransitionTileCount;
    const int64_t called_tile =
        tile_token
            ? -1
            : static_cast<int64_t>(
                  wait_tile_ids[candidate * wait_id_stride_candidate
                                + token * wait_id_stride_tile]);
    float score = -std::numeric_limits<float>::infinity();
    if (tile_token || called_tile > 0) {
      score = 0.0f;
      for (int feature = 0; feature < kTransitionKeySize; feature++) {
        const int key_feature = head * kTransitionKeySize + feature;
        float key_value;
        if (tile_token) {
          key_value = static_cast<float>(
                          tile_key_value[state * tile_stride_state + token * tile_stride_tile
                                         + key_feature * tile_stride_feature])
                      + static_cast<float>(
                          relation_key_value[candidate * relation_stride_candidate
                                             + token * relation_stride_tile
                                             + key_feature * relation_stride_feature]);
        } else {
          key_value = static_cast<float>(
                          wait_key_value[candidate * wait_stride_candidate
                                         + token * wait_stride_tile
                                         + key_feature * wait_stride_feature])
                      + static_cast<float>(
                          tile_key_value[state * tile_stride_state
                                         + (called_tile - 1) * tile_stride_tile
                                         + key_feature * tile_stride_feature]);
        }
        score += shared_queries[wave * kTransitionKeySize + feature] * key_value;
      }
      score *= scale;
    }
    token_weights[wave * kTransitionTokenCount + attended_token] = score;
    local_maximum = fmaxf(local_maximum, score);
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_maximum = fmaxf(local_maximum, __shfl_down(local_maximum, offset));
  }
  const float maximum = __shfl(local_maximum, 0);

  float local_sum = 0.0f;
  for (int attended_token = lane; attended_token < kTransitionTokenCount;
       attended_token += warpSize) {
    const int weight_index = wave * kTransitionTokenCount + attended_token;
    const float score = token_weights[weight_index];
    const float weight =
        score == -std::numeric_limits<float>::infinity() ? 0.0f : expf(score - maximum);
    token_weights[weight_index] = weight;
    local_sum += weight;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down(local_sum, offset);
  }
  const float inverse_sum = 1.0f / __shfl(local_sum, 0);
  __syncthreads();

  if (lane < kTransitionValueSize) {
    const int value_feature = kTransitionKeyWidth + head * kTransitionValueSize + lane;
    float context = 0.0f;
    for (int tile = 0; tile < kTransitionTileCount; tile++) {
      const float value = static_cast<float>(
                              tile_key_value[state * tile_stride_state + tile * tile_stride_tile
                                             + value_feature * tile_stride_feature])
                          + static_cast<float>(
                              relation_key_value[candidate * relation_stride_candidate
                                                 + tile * relation_stride_tile
                                                 + value_feature * relation_stride_feature]);
      context += token_weights[wave * kTransitionTokenCount + tile] * value;
    }
    for (int wait = 0; wait < kTransitionWaitCount; wait++) {
      const int64_t called_tile = static_cast<int64_t>(
          wait_tile_ids[candidate * wait_id_stride_candidate + wait * wait_id_stride_tile]);
      if (called_tile > 0) {
        const float value = static_cast<float>(
                                wait_key_value[candidate * wait_stride_candidate
                                               + wait * wait_stride_tile
                                               + value_feature * wait_stride_feature])
                            + static_cast<float>(
                                tile_key_value[state * tile_stride_state
                                               + (called_tile - 1) * tile_stride_tile
                                               + value_feature * tile_stride_feature]);
        context += token_weights[wave * kTransitionTokenCount + kTransitionTileCount + wait]
                   * value;
      }
    }
    output[(candidate * kTransitionHeads + head) * kTransitionValueSize + lane] =
        static_cast<scalar_t>(context * inverse_sum);
  }
}

template <typename residual_t, typename update_t, typename parameter_t>
__global__ void residual_layer_norm_in_place_kernel(
    residual_t* residual,
    const update_t* update,
    const parameter_t* weight,
    const parameter_t* bias,
    residual_t* output,
    int64_t row_count,
    float epsilon) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  if (row >= row_count) {
    return;
  }

  // gfx11 executes wave32 while older ROCm targets commonly execute wave64.
  // Reserve the larger per-lane slice so the same kernel is valid for both.
  float values[kResidualNormWidth / 32];
  float mean = 0.0f;
  float moment = 0.0f;
  int count = 0;
  const int64_t row_offset = row * kResidualNormWidth;
  for (int feature = lane, local = 0; feature < kResidualNormWidth;
       feature += warpSize, local++) {
    const int64_t index = row_offset + feature;
    const float value =
        static_cast<float>(residual[index]) + static_cast<float>(update[index]);
    values[local] = value;
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
      moment += other_moment
                + delta * delta * static_cast<float>(count * other_count)
                      / static_cast<float>(combined_count);
      mean += delta * static_cast<float>(other_count) / static_cast<float>(combined_count);
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  const float inverse_standard_deviation =
      rsqrtf(__shfl(moment, 0) / static_cast<float>(kResidualNormWidth) + epsilon);

  for (int feature = lane, local = 0; feature < kResidualNormWidth;
       feature += warpSize, local++) {
    const float normalized = (values[local] - mean) * inverse_standard_deviation;
    output[row_offset + feature] = static_cast<residual_t>(
        normalized * static_cast<float>(weight[feature])
        + static_cast<float>(bias[feature]));
  }
}

void require_shape(const torch::Tensor& tensor, std::initializer_list<int64_t> suffix, const char* name) {
  TORCH_CHECK(tensor.dim() == static_cast<int64_t>(suffix.size()) + 2, name, " has invalid rank");
  int64_t dimension = tensor.dim() - static_cast<int64_t>(suffix.size());
  for (int64_t expected : suffix) {
    TORCH_CHECK(tensor.size(dimension++) == expected, name, " has invalid shape");
  }
}

int transition_waves_per_block(int device) {
  static thread_local int cached_device = -1;
  static thread_local int cached_waves = 0;
  if (device != cached_device) {
    hipDeviceProp_t properties;
    C10_HIP_CHECK(hipGetDeviceProperties(&properties, device));
    cached_device = device;
    cached_waves = kTransitionThreads / properties.warpSize;
  }
  return cached_waves;
}

int residual_norm_waves_per_block(int device) {
  static thread_local int cached_device = -1;
  static thread_local int cached_waves = 0;
  if (device != cached_device) {
    hipDeviceProp_t properties;
    C10_HIP_CHECK(hipGetDeviceProperties(&properties, device));
    TORCH_CHECK(
        properties.warpSize == 32 || properties.warpSize == 64,
        "residual LayerNorm requires a 32- or 64-lane ROCm wavefront");
    cached_device = device;
    cached_waves = kResidualNormThreads / properties.warpSize;
  }
  return cached_waves;
}

}  // namespace

torch::Tensor tile_relation_mask(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids,
    float scale) {
  TORCH_CHECK(relation_logits.is_cuda(), "tile relation mask requires a ROCm tensor");
  TORCH_CHECK(
      relation_logits.scalar_type() == relation_bias.scalar_type(),
      "relation logits/bias dtype mismatch");
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt64, "relation IDs must be int64");
  TORCH_CHECK(
      relation_logits.dim() == 4 && relation_logits.size(2) == kTileCount
          && relation_logits.size(3) == kRelationCount,
      "relation logits must be [batch,heads,34,23]");
  TORCH_CHECK(
      relation_bias.dim() == 4 && relation_bias.size(0) == 1
          && relation_bias.size(1) == relation_logits.size(1)
          && relation_bias.size(2) == kTileCount && relation_bias.size(3) == kTileCount,
      "relation bias must be [1,heads,34,34]");
  TORCH_CHECK(
      relation_ids.dim() == 2 && relation_ids.size(0) == kTileCount
          && relation_ids.size(1) == kTileCount,
      "relation IDs must be [34,34]");

  c10::cuda::CUDAGuard device_guard(relation_logits.device());
  torch::Tensor output = torch::empty(
      {relation_logits.size(0), relation_logits.size(1), kTileCount, kTileCount},
      relation_logits.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t element_count = output.numel();
  const int blocks = static_cast<int>((element_count + kMaskThreads - 1) / kMaskThreads);
  const hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, relation_logits.scalar_type(), "tile_relation_mask", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(tile_relation_mask_kernel<scalar_t>),
            dim3(blocks),
            dim3(kMaskThreads),
            0,
            stream,
            relation_logits.data_ptr<scalar_t>(),
            relation_bias.data_ptr<scalar_t>(),
            relation_ids.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            element_count,
            relation_logits.size(1),
            relation_logits.stride(0),
            relation_logits.stride(1),
            relation_logits.stride(2),
            relation_logits.stride(3),
            relation_bias.stride(1),
            relation_bias.stride(2),
            relation_bias.stride(3),
            relation_ids.stride(0),
            relation_ids.stride(1),
            scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor tile_relation_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& relation_key,
    const torch::Tensor& relation_bias,
    const torch::Tensor& relation_ids,
    float scale) {
  TORCH_CHECK(query.is_cuda(), "tile relation attention requires a ROCm tensor");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(relation_key.scalar_type() == torch::kFloat32, "relation key must be float32");
  TORCH_CHECK(relation_bias.scalar_type() == torch::kFloat32, "relation bias must be float32");
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt64, "relation IDs must be int64");
  require_shape(query, {kTileCount, kHeadSize}, "query");
  TORCH_CHECK(key.sizes() == query.sizes(), "key shape mismatch");
  TORCH_CHECK(value.sizes() == query.sizes(), "value shape mismatch");
  TORCH_CHECK(
      relation_key.dim() == 4 && relation_key.size(0) == 1
          && relation_key.size(1) == query.size(1) && relation_key.size(2) == kHeadSize
          && relation_key.size(3) == kRelationCount,
      "relation key has invalid shape");
  TORCH_CHECK(
      relation_bias.dim() == 4 && relation_bias.size(0) == 1
          && relation_bias.size(1) == query.size(1) && relation_bias.size(2) == kTileCount
          && relation_bias.size(3) == kTileCount,
      "relation bias has invalid shape");
  TORCH_CHECK(
      relation_ids.dim() == 2 && relation_ids.size(0) == kTileCount
          && relation_ids.size(1) == kTileCount,
      "relation IDs have invalid shape");

  c10::cuda::CUDAGuard device_guard(query.device());
  torch::Tensor output = torch::empty_like(query, query.options().memory_format(torch::MemoryFormat::Contiguous));
  const dim3 grid(query.size(0) * query.size(1));
  const dim3 block(kThreads);
  const hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(), "tile_relation_attention", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(tile_relation_attention_kernel<scalar_t>),
            grid,
            block,
            0,
            stream,
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            relation_key.data_ptr<float>(),
            relation_bias.data_ptr<float>(),
            relation_ids.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            query.size(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            relation_key.stride(1),
            relation_key.stride(2),
            relation_key.stride(3),
            relation_bias.stride(1),
            relation_bias.stride(2),
            relation_bias.stride(3),
            relation_ids.stride(0),
            relation_ids.stride(1),
            scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor transition_tile_attention(
    const torch::Tensor& query,
    const torch::Tensor& tile_key_value,
    const torch::Tensor& relation_key_value,
    const torch::Tensor& wait_key_value,
    const torch::Tensor& wait_tile_ids,
    int64_t candidates_per_state,
    float scale) {
  TORCH_CHECK(query.is_cuda(), "transition tile attention requires a ROCm tensor");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == kTransitionHeads
                  && query.size(2) == kTransitionKeySize,
              "query must be [candidates,4,8]");
  TORCH_CHECK(tile_key_value.dim() == 3 && tile_key_value.size(1) == kTransitionTileCount
                  && tile_key_value.size(2) == kTransitionKeyWidth
                                                + kTransitionHeads * kTransitionValueSize,
              "tile key/value must be [states,34,96]");
  TORCH_CHECK(relation_key_value.dim() == 3
                  && relation_key_value.size(0) == query.size(0)
                  && relation_key_value.size(1) == kTransitionTileCount
                  && relation_key_value.size(2) == tile_key_value.size(2),
              "relation key/value must be [candidates,34,96]");
  TORCH_CHECK(wait_key_value.dim() == 3 && wait_key_value.size(0) == query.size(0)
                  && wait_key_value.size(1) == kTransitionWaitCount
                  && wait_key_value.size(2) == tile_key_value.size(2),
              "wait key/value must be [candidates,13,96]");
  TORCH_CHECK(wait_tile_ids.dim() == 2 && wait_tile_ids.size(0) == query.size(0)
                  && wait_tile_ids.size(1) == kTransitionWaitCount,
              "wait tile IDs must be [candidates,13]");
  TORCH_CHECK(candidates_per_state > 0
                  && query.size(0) == tile_key_value.size(0) * candidates_per_state,
              "candidate/state shape mismatch");
  TORCH_CHECK(query.scalar_type() == tile_key_value.scalar_type()
                  && query.scalar_type() == relation_key_value.scalar_type()
                  && query.scalar_type() == wait_key_value.scalar_type(),
              "transition attention dtype mismatch: query=", query.scalar_type(),
              " tile=", tile_key_value.scalar_type(),
              " relation=", relation_key_value.scalar_type(),
              " wait=", wait_key_value.scalar_type());
  TORCH_CHECK(wait_tile_ids.scalar_type() == torch::kInt32
                  || wait_tile_ids.scalar_type() == torch::kInt64,
              "wait tile IDs must be int32 or int64");

  c10::cuda::CUDAGuard device_guard(query.device());
  torch::Tensor output = torch::empty(
      {query.size(0), kTransitionHeads, kTransitionValueSize}, query.options());
  const int waves_per_block = transition_waves_per_block(query.get_device());
  TORCH_CHECK(kTransitionHeads % waves_per_block == 0, "unsupported ROCm wavefront size");
  const dim3 grid(query.size(0) * (kTransitionHeads / waves_per_block));
  const dim3 block(kTransitionThreads);
  const hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, query.scalar_type(), "transition_tile_attention", [&] {
        if (wait_tile_ids.scalar_type() == torch::kInt32) {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(transition_tile_attention_kernel<scalar_t, int32_t>),
              grid,
              block,
              0,
              stream,
              query.data_ptr<scalar_t>(),
              tile_key_value.data_ptr<scalar_t>(),
              relation_key_value.data_ptr<scalar_t>(),
              wait_key_value.data_ptr<scalar_t>(),
              wait_tile_ids.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              candidates_per_state,
              query.stride(0),
              query.stride(1),
              query.stride(2),
              tile_key_value.stride(0),
              tile_key_value.stride(1),
              tile_key_value.stride(2),
              relation_key_value.stride(0),
              relation_key_value.stride(1),
              relation_key_value.stride(2),
              wait_key_value.stride(0),
              wait_key_value.stride(1),
              wait_key_value.stride(2),
              wait_tile_ids.stride(0),
              wait_tile_ids.stride(1),
              scale);
        } else {
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(transition_tile_attention_kernel<scalar_t, int64_t>),
              grid,
              block,
              0,
              stream,
              query.data_ptr<scalar_t>(),
              tile_key_value.data_ptr<scalar_t>(),
              relation_key_value.data_ptr<scalar_t>(),
              wait_key_value.data_ptr<scalar_t>(),
              wait_tile_ids.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              candidates_per_state,
              query.stride(0),
              query.stride(1),
              query.stride(2),
              tile_key_value.stride(0),
              tile_key_value.stride(1),
              tile_key_value.stride(2),
              relation_key_value.stride(0),
              relation_key_value.stride(1),
              relation_key_value.stride(2),
              wait_key_value.stride(0),
              wait_key_value.stride(1),
              wait_key_value.stride(2),
              wait_tile_ids.stride(0),
              wait_tile_ids.stride(1),
              scale);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor residual_layer_norm_in_place(
    torch::Tensor& residual,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float epsilon) {
  TORCH_CHECK(residual.is_cuda(), "residual LayerNorm requires a ROCm tensor");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(update.is_contiguous(), "residual update must be contiguous");
  TORCH_CHECK(residual.sizes() == update.sizes(), "residual/update shape mismatch");
  TORCH_CHECK(
      update.scalar_type() == torch::kFloat32 || update.scalar_type() == torch::kFloat16
          || update.scalar_type() == torch::kBFloat16,
      "residual update must have a floating dtype");
  TORCH_CHECK(
      residual.dim() >= 1 && residual.size(-1) == kResidualNormWidth,
      "residual LayerNorm requires a trailing width of 256");
  TORCH_CHECK(
      weight.dim() == 1 && weight.size(0) == kResidualNormWidth,
      "LayerNorm weight must be [256]");
  TORCH_CHECK(bias.sizes() == weight.sizes(), "LayerNorm bias shape mismatch");
  TORCH_CHECK(weight.scalar_type() == bias.scalar_type(), "LayerNorm parameter dtype mismatch");
  TORCH_CHECK(
      weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == residual.scalar_type(),
      "LayerNorm parameters must be float32 or match the residual dtype");

  c10::cuda::CUDAGuard device_guard(residual.device());
  torch::Tensor output = torch::empty_like(
      residual, residual.options().memory_format(torch::MemoryFormat::Contiguous));
  const int64_t row_count = residual.numel() / kResidualNormWidth;
  const int waves_per_block = residual_norm_waves_per_block(residual.get_device());
  const dim3 grid((row_count + waves_per_block - 1) / waves_per_block);
  const dim3 block(kResidualNormThreads);
  const hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, residual.scalar_type(), "residual_layer_norm", [&] {
        using residual_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf,
            torch::kBFloat16,
            update.scalar_type(),
            "residual_layer_norm_update",
            [&] {
              using update_t = scalar_t;
              if (weight.scalar_type() == torch::kFloat32) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        residual_layer_norm_in_place_kernel<residual_t, update_t, float>),
                    grid,
                    block,
                    0,
                    stream,
                    residual.data_ptr<residual_t>(),
                    update.data_ptr<update_t>(),
                    weight.data_ptr<float>(),
                    bias.data_ptr<float>(),
                    output.data_ptr<residual_t>(),
                    row_count,
                    epsilon);
              } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        residual_layer_norm_in_place_kernel<residual_t, update_t, residual_t>),
                    grid,
                    block,
                    0,
                    stream,
                    residual.data_ptr<residual_t>(),
                    update.data_ptr<update_t>(),
                    weight.data_ptr<residual_t>(),
                    bias.data_ptr<residual_t>(),
                    output.data_ptr<residual_t>(),
                    row_count,
                    epsilon);
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace djl::pytorch::rocm
