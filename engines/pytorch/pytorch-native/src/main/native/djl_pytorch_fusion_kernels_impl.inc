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
#include "djl_pytorch_fusion_kernels.h"
#include "djl_pytorch_fusion_backend.h"
#include "djl_pytorch_launch_config.h"

#include <ATen/Context.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/version.h>

#if defined(DJL_USE_ROCM_KERNELS)
#include <ATen/hip/HIPContextLight.h>
#include <ATen/hip/tunable/Tunable.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt.h>
#else
#include <ATen/cuda/CUDAContextLight.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace djl::pytorch::fusion {
namespace {

constexpr int kThreadsPerBlock = 256;

// Bound register storage per lane; larger rows retain the streaming implementation.
constexpr int kMaximumNormalizationCachedValuesPerLane = 32;

template <typename Launch>
void DispatchNormalizationCache(int64_t hidden_width, int device, Launch&& launch) {
  const int wave_size = at::cuda::getDeviceProperties(device)->warpSize;
  const int64_t values_per_lane = (hidden_width - 1) / wave_size + 1;
  if (values_per_lane <= 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (values_per_lane <= 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (values_per_lane <= 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (values_per_lane <= 8) {
    launch(std::integral_constant<int, 8>{});
  } else if (values_per_lane <= 16) {
    launch(std::integral_constant<int, 16>{});
  } else if (values_per_lane <= kMaximumNormalizationCachedValuesPerLane) {
    launch(std::integral_constant<int, kMaximumNormalizationCachedValuesPerLane>{});
  } else {
    launch(std::integral_constant<int, 0>{});
  }
}

#if defined(DJL_USE_ROCM_KERNELS)

void CheckHipBlasLt(hipblasStatus_t status, const char* operation) {
  TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, operation,
      " failed with hipBLASLt status ", static_cast<int>(status));
}

hipDataType HipBlasLtDataType(torch::ScalarType data_type) {
  switch (data_type) {
    case torch::kFloat16:
      return HIP_R_16F;
    case torch::kBFloat16:
      return HIP_R_16BF;
    case torch::kFloat32:
      return HIP_R_32F;
    default:
      TORCH_CHECK(false, "unsupported hipBLASLt data type");
  }
}

enum class FusionMatmulEpilogue : uint8_t {
  kNone,
  kBias,
  kSwishBias,
};

struct FusionMatmulKey {
  int32_t device;
  torch::ScalarType data_type;
  int64_t batch_count;
  int64_t rows;
  int64_t columns;
  int64_t reduction;
  int64_t left_row_stride;
  int64_t right_leading_dimension;
  int64_t output_row_stride;
  int64_t left_batch_stride;
  int64_t right_batch_stride;
  int64_t output_batch_stride;
  bool right_transposed;
  FusionMatmulEpilogue epilogue;

  bool operator==(const FusionMatmulKey& other) const {
    return device == other.device && data_type == other.data_type &&
        batch_count == other.batch_count && rows == other.rows &&
        columns == other.columns && reduction == other.reduction &&
        left_row_stride == other.left_row_stride &&
        right_leading_dimension == other.right_leading_dimension &&
        output_row_stride == other.output_row_stride &&
        left_batch_stride == other.left_batch_stride &&
        right_batch_stride == other.right_batch_stride &&
        output_batch_stride == other.output_batch_stride &&
        right_transposed == other.right_transposed &&
        epilogue == other.epilogue;
  }
};

void HashCombine(std::size_t& hash, std::size_t value) {
  hash ^= value + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
}

struct FusionMatmulKeyHash {
  std::size_t operator()(const FusionMatmulKey& key) const {
    std::size_t hash = std::hash<int32_t>{}(key.device);
    HashCombine(hash, std::hash<int>{}(static_cast<int>(key.data_type)));
    HashCombine(hash, std::hash<int64_t>{}(key.batch_count));
    HashCombine(hash, std::hash<int64_t>{}(key.rows));
    HashCombine(hash, std::hash<int64_t>{}(key.columns));
    HashCombine(hash, std::hash<int64_t>{}(key.reduction));
    HashCombine(hash, std::hash<int64_t>{}(key.left_row_stride));
    HashCombine(hash,
        std::hash<int64_t>{}(key.right_leading_dimension));
    HashCombine(hash, std::hash<int64_t>{}(key.output_row_stride));
    HashCombine(hash, std::hash<int64_t>{}(key.left_batch_stride));
    HashCombine(hash, std::hash<int64_t>{}(key.right_batch_stride));
    HashCombine(hash, std::hash<int64_t>{}(key.output_batch_stride));
    HashCombine(hash, std::hash<bool>{}(key.right_transposed));
    HashCombine(hash, std::hash<int>{}(static_cast<int>(key.epilogue)));
    return hash;
  }
};

void LogFusionMatmulGeometry(const FusionMatmulKey& key,
    const char* result, std::size_t workspace_size) {
  if (!launch_environment::GetMatmulDiagnostics().geometry) {
    return;
  }
  std::fprintf(stderr,
      "[djl-rocm-matmul] geometry result=%s device=%d dtype=%d "
      "batch=%lld rows=%lld columns=%lld reduction=%lld "
      "left_row_stride=%lld right_ld=%lld output_row_stride=%lld "
      "left_batch_stride=%lld right_batch_stride=%lld "
      "output_batch_stride=%lld right_transposed=%d epilogue=%d "
      "workspace=%zu\n",
      result, key.device, static_cast<int>(key.data_type),
      static_cast<long long>(key.batch_count),
      static_cast<long long>(key.rows),
      static_cast<long long>(key.columns),
      static_cast<long long>(key.reduction),
      static_cast<long long>(key.left_row_stride),
      static_cast<long long>(key.right_leading_dimension),
      static_cast<long long>(key.output_row_stride),
      static_cast<long long>(key.left_batch_stride),
      static_cast<long long>(key.right_batch_stride),
      static_cast<long long>(key.output_batch_stride),
      key.right_transposed ? 1 : 0, static_cast<int>(key.epilogue),
      workspace_size);
}

void ConfigureBatchLayout(hipblasLtMatrixLayout_t layout,
    int64_t batch_count, int64_t batch_stride, const char* name) {
  TORCH_CHECK(batch_count >= 1 &&
          batch_count <= std::numeric_limits<int32_t>::max(),
      name, " batch count exceeds the hipBLASLt range");
  const int32_t count = static_cast<int32_t>(batch_count);
  CheckHipBlasLt(hipblasLtMatrixLayoutSetAttribute(layout,
      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)), name);
  CheckHipBlasLt(hipblasLtMatrixLayoutSetAttribute(layout,
      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
      sizeof(batch_stride)), name);
}

struct FusionMatmulAlgorithm {
  hipblasLtMatmulAlgo_t value{};
  std::size_t workspace_size = 0;
  bool supported = false;
};

struct FusionMatmulDescriptorKey {
  torch::ScalarType data_type;
  int64_t columns;
  int64_t reduction;
  int64_t left_row_stride;
  int64_t right_leading_dimension;
  int64_t output_row_stride;
  bool right_transposed;
  FusionMatmulEpilogue epilogue;

  explicit FusionMatmulDescriptorKey(const FusionMatmulKey& key)
      : data_type(key.data_type), columns(key.columns),
        reduction(key.reduction), left_row_stride(key.left_row_stride),
        right_leading_dimension(key.right_leading_dimension),
        output_row_stride(key.output_row_stride),
        right_transposed(key.right_transposed), epilogue(key.epilogue) {}

  bool operator==(const FusionMatmulDescriptorKey& other) const {
    return data_type == other.data_type && columns == other.columns &&
        reduction == other.reduction &&
        left_row_stride == other.left_row_stride &&
        right_leading_dimension == other.right_leading_dimension &&
        output_row_stride == other.output_row_stride &&
        right_transposed == other.right_transposed &&
        epilogue == other.epilogue;
  }
};

struct FusionMatmulDescriptorKeyHash {
  std::size_t operator()(const FusionMatmulDescriptorKey& key) const {
    std::size_t hash =
        std::hash<int>{}(static_cast<int>(key.data_type));
    HashCombine(hash, std::hash<int64_t>{}(key.columns));
    HashCombine(hash, std::hash<int64_t>{}(key.reduction));
    HashCombine(hash, std::hash<int64_t>{}(key.left_row_stride));
    HashCombine(hash,
        std::hash<int64_t>{}(key.right_leading_dimension));
    HashCombine(hash, std::hash<int64_t>{}(key.output_row_stride));
    HashCombine(hash, std::hash<bool>{}(key.right_transposed));
    HashCombine(hash, std::hash<int>{}(static_cast<int>(key.epilogue)));
    return hash;
  }
};

// List nodes keep indexed keys stable across LRU moves and map rehashes.
// Callers hold the stream context's mutex during lookup and use.
template <typename Key, typename Value, typename Hash>
class FusionMatmulCache {
 public:
  explicit FusionMatmulCache(std::size_t capacity) : capacity_(capacity) {}

  FusionMatmulCache(const FusionMatmulCache&) = delete;
  FusionMatmulCache& operator=(const FusionMatmulCache&) = delete;

  template <typename Factory>
  Value& GetOrCreate(const Key& key, Factory&& factory) {
    ++lookups_;
    const auto found = index_.find(&key);
    if (found != index_.end()) {
      entries_.splice(entries_.begin(), entries_, found->second);
      return found->second->value;
    }
    entries_.emplace_front(key, factory());
    try {
      index_.emplace(&entries_.front().key, entries_.begin());
    } catch (...) {
      entries_.pop_front();
      throw;
    }
    ++misses_;
    if (entries_.size() > capacity_) {
      index_.erase(&entries_.back().key);
      entries_.pop_back();
      ++evictions_;
    }
    return entries_.front().value;
  }

  std::size_t Size() const {
    return entries_.size();
  }

  std::size_t Lookups() const {
    return lookups_;
  }

  std::size_t Misses() const {
    return misses_;
  }

  std::size_t Evictions() const {
    return evictions_;
  }

 private:
  struct Entry {
    Key key;
    Value value;

    Entry(const Key& entry_key, Value&& entry_value)
        : key(entry_key), value(std::move(entry_value)) {}
  };

  struct PointerHash {
    std::size_t operator()(const Key* key) const {
      return Hash{}(*key);
    }
  };

  struct PointerEqual {
    bool operator()(const Key* left, const Key* right) const {
      return *left == *right;
    }
  };

  std::size_t capacity_;
  std::list<Entry> entries_;
  std::unordered_map<const Key*, typename std::list<Entry>::iterator,
      PointerHash, PointerEqual> index_;
  std::size_t lookups_ = 0;
  std::size_t misses_ = 0;
  std::size_t evictions_ = 0;
};

// Opaque hipBLASLt descriptors own Tensile problem state. Retain that state
// per matrix family, not per observed row count or batch geometry.
class FusionMatmulDescriptors {
 public:
  FusionMatmulDescriptors(const FusionMatmulKey& key,
      const void* heuristic_bias_pointer)
      : rows_(key.rows), batch_count_(key.batch_count),
        left_batch_stride_(key.left_batch_stride),
        right_batch_stride_(key.right_batch_stride),
        output_batch_stride_(key.output_batch_stride),
        epilogue_(key.epilogue) {
    try {
      const hipDataType data_type = HipBlasLtDataType(key.data_type);
      CheckHipBlasLt(hipblasLtMatmulDescCreate(
          &operation_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
          "hipblasLtMatmulDescCreate");
      if (key.right_transposed) {
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        CheckHipBlasLt(hipblasLtMatmulDescSetAttribute(operation_,
            HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
            "hipblasLtMatmulDescSetAttribute(right transpose)");
      }
      if (key.epilogue != FusionMatmulEpilogue::kNone) {
        hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        if (key.epilogue == FusionMatmulEpilogue::kSwishBias) {
#if defined(DJL_USE_HIPBLASLT_SWISH_BIAS)
          epilogue = HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
#else
          TORCH_CHECK(false,
              "hipBLASLt Swish-bias epilogue is unavailable in this build");
#endif
        }
        CheckHipBlasLt(hipblasLtMatmulDescSetAttribute(operation_,
            HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
            "hipblasLtMatmulDescSetAttribute(epilogue)");
        // hipBLASLt requires a non-null device pointer while selecting a bias
        // epilogue algorithm. Run() replaces this context-owned placeholder
        // with the invocation's bias pointer before enqueueing the matmul.
        TORCH_CHECK(heuristic_bias_pointer != nullptr,
            "hipBLASLt bias heuristic requires a device placeholder");
        CheckHipBlasLt(hipblasLtMatmulDescSetAttribute(operation_,
            HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &heuristic_bias_pointer,
            sizeof(heuristic_bias_pointer)),
            "hipblasLtMatmulDescSetAttribute(heuristic bias)");
        CheckHipBlasLt(hipblasLtMatmulDescSetAttribute(operation_,
            HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &data_type,
            sizeof(data_type)),
            "hipblasLtMatmulDescSetAttribute(bias type)");
      }

      // A row-major [M,K] * [K,N] product is the same memory as the
      // column-major [N,K] * [K,M] -> [N,M] product with operands reversed.
      const int64_t right_layout_rows =
          key.right_transposed ? key.reduction : key.columns;
      const int64_t right_layout_columns =
          key.right_transposed ? key.columns : key.reduction;
      CheckHipBlasLt(hipblasLtMatrixLayoutCreate(&right_, data_type,
          right_layout_rows, right_layout_columns,
          key.right_leading_dimension),
          "hipblasLtMatrixLayoutCreate(right)");
      CheckHipBlasLt(hipblasLtMatrixLayoutCreate(&left_, data_type,
          key.reduction, key.rows, key.left_row_stride),
          "hipblasLtMatrixLayoutCreate(left)");
      CheckHipBlasLt(hipblasLtMatrixLayoutCreate(&output_, data_type,
          key.columns, key.rows, key.output_row_stride),
          "hipblasLtMatrixLayoutCreate(output)");
      ConfigureBatchLayout(right_, key.batch_count, key.right_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(right batch)");
      ConfigureBatchLayout(left_, key.batch_count, key.left_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(left batch)");
      ConfigureBatchLayout(output_, key.batch_count, key.output_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(output batch)");
    } catch (...) {
      Reset();
      throw;
    }
  }

  ~FusionMatmulDescriptors() {
    Reset();
  }

  FusionMatmulDescriptors(const FusionMatmulDescriptors&) = delete;
  FusionMatmulDescriptors& operator=(const FusionMatmulDescriptors&) = delete;

  void Configure(const FusionMatmulKey& key) {
    if (rows_ != key.rows) {
      const uint64_t rows = static_cast<uint64_t>(key.rows);
      CheckHipBlasLt(hipblasLtMatrixLayoutSetAttribute(left_,
          HIPBLASLT_MATRIX_LAYOUT_COLS, &rows, sizeof(rows)),
          "hipblasLtMatrixLayoutSetAttribute(left columns)");
      CheckHipBlasLt(hipblasLtMatrixLayoutSetAttribute(output_,
          HIPBLASLT_MATRIX_LAYOUT_COLS, &rows, sizeof(rows)),
          "hipblasLtMatrixLayoutSetAttribute(output columns)");
    }
    if (batch_count_ != key.batch_count ||
        left_batch_stride_ != key.left_batch_stride) {
      ConfigureBatchLayout(left_, key.batch_count, key.left_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(left batch)");
    }
    if (batch_count_ != key.batch_count ||
        right_batch_stride_ != key.right_batch_stride) {
      ConfigureBatchLayout(right_, key.batch_count, key.right_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(right batch)");
    }
    if (batch_count_ != key.batch_count ||
        output_batch_stride_ != key.output_batch_stride) {
      ConfigureBatchLayout(output_, key.batch_count, key.output_batch_stride,
          "hipblasLtMatrixLayoutSetAttribute(output batch)");
    }
    rows_ = key.rows;
    batch_count_ = key.batch_count;
    left_batch_stride_ = key.left_batch_stride;
    right_batch_stride_ = key.right_batch_stride;
    output_batch_stride_ = key.output_batch_stride;
  }

  FusionMatmulAlgorithm SelectAlgorithm(hipblasLtHandle_t handle,
      const FusionMatmulKey& key, const void* heuristic_bias_pointer) {
    hipblasLtMatmulPreference_t preference = nullptr;
    try {
      // The previous invocation's bias may already have been released.
      // Heuristics use the context's stable placeholder instead.
      if (epilogue_ != FusionMatmulEpilogue::kNone) {
        SetBiasPointer(heuristic_bias_pointer);
      }

      CheckHipBlasLt(hipblasLtMatmulPreferenceCreate(&preference),
          "hipblasLtMatmulPreferenceCreate");
      const size_t workspace_bound =
          at::cuda::getCUDABlasLtWorkspaceSize();
      CheckHipBlasLt(hipblasLtMatmulPreferenceSetAttribute(preference,
          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bound,
          sizeof(workspace_bound)),
          "hipblasLtMatmulPreferenceSetAttribute");
      hipblasLtMatmulHeuristicResult_t result{};
      int result_count = 0;
      const hipblasStatus_t heuristic_status =
          hipblasLtMatmulAlgoGetHeuristic(
          handle, operation_, right_, left_,
          output_, output_, preference, 1, &result, &result_count);
      const hipblasStatus_t destroy_status =
          hipblasLtMatmulPreferenceDestroy(preference);
      preference = nullptr;
      CheckHipBlasLt(destroy_status, "hipblasLtMatmulPreferenceDestroy");
      if (heuristic_status == HIPBLAS_STATUS_NOT_SUPPORTED) {
        LogFusionMatmulGeometry(
            key, "heuristic-not-supported", 0);
        return {};
      }
      CheckHipBlasLt(
          heuristic_status, "hipblasLtMatmulAlgoGetHeuristic");
      TORCH_CHECK(result_count == 0 || result_count == 1,
          "hipblasLtMatmulAlgoGetHeuristic returned an invalid result count: ",
          result_count);
      if (result_count == 0 ||
          result.state == HIPBLAS_STATUS_NOT_SUPPORTED) {
        LogFusionMatmulGeometry(key,
            result_count == 0 ? "no-algorithm" : "algorithm-not-supported",
            0);
        return {};
      }
      CheckHipBlasLt(
          result.state, "hipblasLtMatmulAlgoGetHeuristic result");
      TORCH_CHECK(result.workspaceSize <= workspace_bound,
          "hipblasLt heuristic exceeded the requested workspace bound");
      LogFusionMatmulGeometry(key, "supported", result.workspaceSize);
      return {result.algo, result.workspaceSize, true};
    } catch (...) {
      if (preference != nullptr) {
        hipblasLtMatmulPreferenceDestroy(preference);
      }
      throw;
    }
  }

  void Run(hipblasLtHandle_t handle, backend::Stream stream,
      const FusionMatmulAlgorithm& algorithm,
      torch::Tensor& output, const torch::Tensor& left,
      const torch::Tensor& right, const torch::Tensor* bias,
      void* workspace, std::size_t workspace_capacity, float beta) {
    constexpr float alpha = 1.0f;
    TORCH_CHECK(algorithm.supported,
        "cannot run an unsupported direct ROCm matmul geometry");
    TORCH_CHECK(algorithm.workspace_size <= workspace_capacity,
        "Fusion matmul workspace is smaller than the selected algorithm requires");
    TORCH_CHECK(algorithm.workspace_size == 0 || workspace != nullptr,
        "Fusion matmul workspace must not be null");
    if (epilogue_ != FusionMatmulEpilogue::kNone) {
      TORCH_CHECK(bias != nullptr,
          "Fusion matmul bias epilogue requires a bias tensor");
      SetBiasPointer(bias->const_data_ptr());
    }
    CheckHipBlasLt(hipblasLtMatmul(
        handle, operation_, &alpha,
        right.const_data_ptr(), right_, left.const_data_ptr(), left_,
        &beta, output.const_data_ptr(), output_, output.data_ptr(), output_,
        &algorithm.value, workspace, algorithm.workspace_size, stream),
        "hipblasLtMatmul(Fusion)");
  }

 private:
  void SetBiasPointer(const void* bias_pointer) {
    CheckHipBlasLt(hipblasLtMatmulDescSetAttribute(operation_,
        HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
        sizeof(bias_pointer)),
        "hipblasLtMatmulDescSetAttribute(bias)");
  }

  void Reset() noexcept {
    if (output_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(output_);
      output_ = nullptr;
    }
    if (left_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(left_);
      left_ = nullptr;
    }
    if (right_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(right_);
      right_ = nullptr;
    }
    if (operation_ != nullptr) {
      hipblasLtMatmulDescDestroy(operation_);
      operation_ = nullptr;
    }
  }

  hipblasLtMatmulDesc_t operation_ = nullptr;
  hipblasLtMatrixLayout_t right_ = nullptr;
  hipblasLtMatrixLayout_t left_ = nullptr;
  hipblasLtMatrixLayout_t output_ = nullptr;
  int64_t rows_;
  int64_t batch_count_;
  int64_t left_batch_stride_;
  int64_t right_batch_stride_;
  int64_t output_batch_stride_;
  FusionMatmulEpilogue epilogue_ = FusionMatmulEpilogue::kNone;
};

#endif

int64_t CheckedMultiply(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " factors must not be negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
      name, " exceeds the supported range");
  return left * right;
}

int64_t RequiredBlockCount(int64_t element_count) {
  return element_count / kThreadsPerBlock +
      (element_count % kThreadsPerBlock == 0 ? 0 : 1);
}

struct OutputPackArguments {
  const void* sources[kMaximumOutputPackSources];
  int64_t widths[kMaximumOutputPackSources];
  int64_t destination_offsets[kMaximumOutputPackSources];
  int32_t data_types[kMaximumOutputPackSources];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct SegmentedOutputPackArguments {
  const void* sources[kMaximumOutputPackSources];
  int32_t data_types[kMaximumOutputPackSources];
  int64_t source_prefix_counts[kMaximumOutputPackSources];
  int64_t source_token_counts[kMaximumOutputPackSources];
  int64_t token_offsets[kMaximumOutputPackSources];
  int64_t token_counts[kMaximumOutputPackSources];
  int64_t hidden_widths[kMaximumOutputPackSources];
  int64_t destination_offsets[kMaximumOutputPackSources];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct BinaryBranchBlendArguments {
  const void* baseline_context;
  const void* selected_context;
  const void* selected_logit;
  const void* baseline_presence;
  const void* selected_presence;
  float* output;
  int64_t row_count;
  int64_t width;
  int32_t selected_logit_data_type;
  int32_t baseline_presence_data_type;
  int32_t selected_presence_data_type;
};

struct AffineInputPackArguments {
  const void* sources[kMaximumAffineTerms];
  int64_t widths[kMaximumAffineTerms];
  int64_t destination_offsets[kMaximumAffineTerms];
  int32_t data_types[kMaximumAffineTerms];
  void* output;
  int64_t row_count;
  int64_t output_width;
};

struct AffineFinalizeArguments {
  const void* sources[kMaximumAffineGroups];
  int64_t source_rows_per_batch[kMaximumAffineGroups];
  int64_t source_strides[kMaximumAffineGroups][kMaximumAffinePrefixRank];
  const void* bias;
  void* output;
  int64_t batch_count;
  int64_t output_prefix[kMaximumAffinePrefixRank];
  int64_t output_prefix_rows;
  int64_t output_width;
  int32_t source_count;
  int32_t output_prefix_rank;
  int32_t activation;
};

struct IndexedAffineInputPackArguments {
  const void* sources[kMaximumIndexedAffineSources];
  int64_t widths[kMaximumIndexedAffineSources];
  int64_t destination_offsets[kMaximumIndexedAffineSources];
  int64_t index_divisors[kMaximumIndexedAffineSources];
  int64_t source_rows[kMaximumIndexedAffineSources];
  int32_t data_types[kMaximumIndexedAffineSources];
  const void* indices;
  void* output;
  int64_t active_rows;
  int64_t destination_rows;
  int64_t output_width;
};

struct IndexedAffineFinalizeArguments {
  const void* indices;
  const void* hidden;
  const void* hidden_bias;
  const void* output_weight;
  const void* output_bias;
  void* output;
  int64_t active_rows;
  int64_t destination_rows;
  int64_t hidden_width;
  int64_t output_width;
  int32_t activation;
};

template <typename output_t, typename scalar_t>
__device__ void pack_source(const OutputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const scalar_t*>(arguments.sources[source_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    output[row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(static_cast<float>(source[index]));
  }
}

template <typename output_t>
__global__ void output_pack_kernel(OutputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_source<output_t, c10::Half>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_source<output_t, c10::BFloat16>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_source<output_t, float>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename output_t, typename source_t>
__device__ void pack_segmented_source(
    const SegmentedOutputPackArguments& arguments, int32_t segment_index) {
  const auto* source =
      static_cast<const source_t*>(arguments.sources[segment_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t prefix_count = arguments.source_prefix_counts[segment_index];
  const int64_t source_token_count =
      arguments.source_token_counts[segment_index];
  const int64_t token_offset = arguments.token_offsets[segment_index];
  const int64_t token_count = arguments.token_counts[segment_index];
  const int64_t hidden_width = arguments.hidden_widths[segment_index];
  const int64_t width = prefix_count * token_count * hidden_width;
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    const int64_t destination_token = column / hidden_width;
    const int64_t feature = column - destination_token * hidden_width;
    const int64_t prefix = destination_token / token_count;
    const int64_t token = destination_token - prefix * token_count + token_offset;
    const int64_t source_element =
        ((row * prefix_count + prefix) * source_token_count + token) *
            hidden_width +
        feature;
    output[row * arguments.output_width +
        arguments.destination_offsets[segment_index] + column] =
        static_cast<output_t>(static_cast<float>(source[source_element]));
  }
}

template <typename output_t>
__global__ void segmented_output_pack_kernel(
    SegmentedOutputPackArguments arguments) {
  const int32_t segment_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[segment_index])) {
    case torch::kFloat16:
      pack_segmented_source<output_t, c10::Half>(arguments, segment_index);
      break;
    case torch::kBFloat16:
      pack_segmented_source<output_t, c10::BFloat16>(arguments, segment_index);
      break;
    case torch::kFloat32:
      pack_segmented_source<output_t, float>(arguments, segment_index);
      break;
    default:
      break;
  }
}

__device__ float load_fusion_float(
    const void* data, int64_t index, int32_t data_type) {
  switch (static_cast<torch::ScalarType>(data_type)) {
    case torch::kFloat16:
      return static_cast<float>(static_cast<const c10::Half*>(data)[index]);
    case torch::kBFloat16:
      return static_cast<float>(static_cast<const c10::BFloat16*>(data)[index]);
    case torch::kFloat32:
      return static_cast<const float*>(data)[index];
    default:
      return 0.0f;
  }
}

template <typename score_t, typename mask_t, typename value_t>
__global__ void mapped_grouped_masked_softmax_pool_kernel(
    const score_t* scores, const mask_t* masks, const value_t* values,
    const int32_t* destination_metadata, float* contexts,
    mask_t* presence, int64_t batch_count, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count,
    int64_t output_batch_capacity) {
  __shared__ float probabilities[256];
  __shared__ int32_t has_values;
  const int64_t row_count = batch_count * destination_count;
  for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const int64_t batch = row / destination_count;
    const int64_t destination = row - batch * destination_count;
    const int32_t* metadata = destination_metadata + destination * 4;
    const int32_t source_group = metadata[0];
    const int64_t output_row =
        static_cast<int64_t>(metadata[2]) * output_batch_capacity +
        batch * static_cast<int64_t>(metadata[1]) + metadata[3];
    if (source_group < 0) {
      if (threadIdx.x == 0) {
        presence[output_row] = static_cast<mask_t>(0.0f);
      }
      for (int64_t column = threadIdx.x; column < width;
           column += blockDim.x) {
        contexts[output_row * width + column] = 0.0f;
      }
      continue;
    }

    if (threadIdx.x < warpSize) {
      const int lane = threadIdx.x;
      float maximum = -INFINITY;
      int selected_count = 0;
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        const int64_t mask_index =
            (batch * candidate_count + candidate) * group_count +
            source_group;
        if (static_cast<float>(masks[mask_index]) != 0.0f) {
          const float score = static_cast<float>(
              scores[batch * candidate_count + candidate]);
          maximum = isnan(maximum) || isnan(score)
              ? NAN
              : fmaxf(maximum, score);
          ++selected_count;
        }
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        const float other_maximum = __shfl_down(maximum, offset);
        maximum = isnan(maximum) || isnan(other_maximum)
            ? NAN
            : fmaxf(maximum, other_maximum);
        selected_count += __shfl_down(selected_count, offset);
      }
      maximum = __shfl(maximum, 0);
      selected_count = __shfl(selected_count, 0);

      float local_denominator = 0.0f;
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        const int64_t mask_index =
            (batch * candidate_count + candidate) * group_count +
            source_group;
        const float probability =
            selected_count > 0 &&
                    static_cast<float>(masks[mask_index]) != 0.0f
                ? expf(static_cast<float>(
                           scores[batch * candidate_count + candidate]) -
                      maximum)
                : 0.0f;
        probabilities[candidate] = probability;
        local_denominator += probability;
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_denominator += __shfl_down(local_denominator, offset);
      }
      const float denominator = __shfl(local_denominator, 0);
      for (int64_t candidate = lane; candidate < candidate_count;
           candidate += warpSize) {
        if (probabilities[candidate] != 0.0f) {
          probabilities[candidate] /= denominator;
        }
      }
      if (lane == 0) {
        has_values = selected_count > 0;
        presence[output_row] = static_cast<mask_t>(
            selected_count > 0 ? 1.0f : 0.0f);
      }
    }
    __syncthreads();
    if (!has_values) {
      for (int64_t column = threadIdx.x; column < width;
           column += blockDim.x) {
        contexts[output_row * width + column] = 0.0f;
      }
      // All threads take this branch. Finish the row before the shared
      // probabilities and presence flag are reused by the next grid-stride row.
      __syncthreads();
      continue;
    }
    for (int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      float sum = 0.0f;
      for (int64_t candidate = 0; candidate < candidate_count;
           ++candidate) {
        const float probability = probabilities[candidate];
        const int64_t mask_index =
            (batch * candidate_count + candidate) * group_count +
            source_group;
        if (static_cast<float>(masks[mask_index]) != 0.0f) {
          sum += probability *
              static_cast<float>(
                  values[(batch * candidate_count + candidate) * width +
                      column]);
        }
      }
      contexts[output_row * width + column] = sum;
    }
    __syncthreads();
  }
}

template <typename baseline_t, typename selected_t>
__global__ void binary_branch_blend_kernel(
    BinaryBranchBlendArguments arguments) {
  const auto* baseline_context =
      static_cast<const baseline_t*>(arguments.baseline_context);
  const auto* selected_context =
      static_cast<const selected_t*>(arguments.selected_context);
  __shared__ float baseline_presence;
  __shared__ float selected_presence;
  __shared__ float baseline_weight;
  __shared__ float selected_weight;
  for (int64_t row = blockIdx.x; row < arguments.row_count;
       row += gridDim.x) {
    if (threadIdx.x == 0) {
      baseline_presence = load_fusion_float(
          arguments.baseline_presence, row,
          arguments.baseline_presence_data_type);
      selected_presence = load_fusion_float(
          arguments.selected_presence, row,
          arguments.selected_presence_data_type);
      const float joint_presence = baseline_presence * selected_presence;
      float selected_probability = 0.0f;
      if (joint_presence != 0.0f) {
        const float selected_logit = load_fusion_float(arguments.selected_logit,
            row, arguments.selected_logit_data_type);
        selected_probability = 1.0f / (1.0f + expf(-selected_logit));
      }
      baseline_weight =
          baseline_presence - joint_presence * selected_probability;
      selected_weight = selected_presence - joint_presence +
          joint_presence * selected_probability;
    }
    __syncthreads();
    const int64_t row_offset = row * arguments.width;
    for (int64_t column = threadIdx.x; column < arguments.width;
         column += blockDim.x) {
      const int64_t index = row_offset + column;
      float value = 0.0f;
      if (baseline_presence != 0.0f) {
        value += baseline_weight * static_cast<float>(baseline_context[index]);
      }
      if (selected_presence != 0.0f) {
        value += selected_weight * static_cast<float>(selected_context[index]);
      }
      arguments.output[index] = value;
    }
    __syncthreads();
  }
}

template <typename output_t, typename input_t>
__device__ void pack_affine_source(
    const AffineInputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const input_t*>(arguments.sources[source_index]);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.row_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const int64_t column = index - row * width;
    output[row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(static_cast<float>(source[index]));
  }
}

template <typename output_t>
__global__ void affine_input_pack_kernel(AffineInputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_affine_source<output_t, c10::Half>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_affine_source<output_t, c10::BFloat16>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_affine_source<output_t, float>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename scalar_t>
__global__ void affine_finalize_kernel(AffineFinalizeArguments arguments) {
  auto* output = static_cast<scalar_t*>(arguments.output);
  const auto* bias = static_cast<const scalar_t*>(arguments.bias);
  const int64_t output_row_count = arguments.batch_count * arguments.output_prefix_rows;
  const int64_t element_count = output_row_count * arguments.output_width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t output_row = index / arguments.output_width;
    const int64_t column = index - output_row * arguments.output_width;
    const int64_t batch = output_row / arguments.output_prefix_rows;
    int64_t prefix_index = output_row - batch * arguments.output_prefix_rows;
    int64_t coordinates[kMaximumAffinePrefixRank];
    for (int32_t axis = arguments.output_prefix_rank - 1; axis >= 0; --axis) {
      coordinates[axis] = prefix_index % arguments.output_prefix[axis];
      prefix_index /= arguments.output_prefix[axis];
    }

    float sum = bias == nullptr ? 0.0f : static_cast<float>(bias[column]);
    for (int32_t source_index = 0; source_index < arguments.source_count; ++source_index) {
      int64_t source_row = batch * arguments.source_rows_per_batch[source_index];
      for (int32_t axis = 0; axis < arguments.output_prefix_rank; ++axis) {
        source_row += coordinates[axis] * arguments.source_strides[source_index][axis];
      }
      const auto* source = static_cast<const scalar_t*>(arguments.sources[source_index]);
      sum += static_cast<float>(
          source[source_row * arguments.output_width + column]);
    }
    if (arguments.activation == static_cast<int32_t>(AffineActivation::kSilu)) {
      sum = sum / (1.0f + expf(-sum));
    }
    output[index] = static_cast<scalar_t>(sum);
  }
}

template <typename scalar_t>
__global__ void projected_residual_mlp_prepare_kernel(
    const scalar_t* combined, scalar_t* activated, scalar_t* residual,
    int64_t row_count, int64_t output_width, int64_t hidden_width) {
  const int64_t combined_width = output_width + hidden_width;
  const int64_t element_count = row_count * combined_width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / combined_width;
    const int64_t column = index - row * combined_width;
    float value = static_cast<float>(combined[index]);
    if (column < output_width) {
      residual[row * output_width + column] = static_cast<scalar_t>(value);
    } else {
      activated[row * hidden_width + column - output_width] =
          static_cast<scalar_t>(value / (1.0f + expf(-value)));
    }
  }
}

template <typename output_t, typename input_t, typename index_t>
__device__ void pack_indexed_affine_source(
    const IndexedAffineInputPackArguments& arguments, int32_t source_index) {
  const auto* source = static_cast<const input_t*>(arguments.sources[source_index]);
  const auto* indices = static_cast<const index_t*>(arguments.indices);
  auto* output = static_cast<output_t*>(arguments.output);
  const int64_t width = arguments.widths[source_index];
  const int64_t element_count = arguments.active_rows * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_row = index / width;
    const int64_t column = index - active_row * width;
    const int64_t destination_row = static_cast<int64_t>(indices[active_row]);
    if (destination_row < 0 || destination_row >= arguments.destination_rows) {
      continue;
    }
    const int64_t source_row = destination_row / arguments.index_divisors[source_index];
    if (source_row >= arguments.source_rows[source_index]) {
      continue;
    }
    output[active_row * arguments.output_width +
        arguments.destination_offsets[source_index] + column] =
        static_cast<output_t>(static_cast<float>(source[source_row * width + column]));
  }
}

template <typename output_t, typename index_t>
__global__ void indexed_affine_input_pack_kernel(
    IndexedAffineInputPackArguments arguments) {
  const int32_t source_index = static_cast<int32_t>(blockIdx.y);
  switch (static_cast<torch::ScalarType>(arguments.data_types[source_index])) {
    case torch::kFloat16:
      pack_indexed_affine_source<output_t, c10::Half, index_t>(arguments, source_index);
      break;
    case torch::kBFloat16:
      pack_indexed_affine_source<output_t, c10::BFloat16, index_t>(arguments, source_index);
      break;
    case torch::kFloat32:
      pack_indexed_affine_source<output_t, float, index_t>(arguments, source_index);
      break;
    default:
      break;
  }
}

template <typename scalar_t, typename index_t>
__global__ void indexed_affine_finalize_kernel(
    IndexedAffineFinalizeArguments arguments) {
  const auto* indices = static_cast<const index_t*>(arguments.indices);
  const auto* hidden = static_cast<const scalar_t*>(arguments.hidden);
  const auto* hidden_bias = static_cast<const scalar_t*>(arguments.hidden_bias);
  const auto* output_weight = static_cast<const scalar_t*>(arguments.output_weight);
  const auto* output_bias = static_cast<const scalar_t*>(arguments.output_bias);
  auto* output = static_cast<scalar_t*>(arguments.output);
  const int64_t element_count = arguments.active_rows * arguments.output_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t active_row = index / arguments.output_width;
    const int64_t output_column = index - active_row * arguments.output_width;
    float sum = 0.0f;
    for (int64_t hidden_column = lane;
         hidden_column < arguments.hidden_width; hidden_column += warpSize) {
      float value = static_cast<float>(
          hidden[active_row * arguments.hidden_width + hidden_column]);
      if (hidden_bias != nullptr) {
        value += static_cast<float>(hidden_bias[hidden_column]);
      }
      if (arguments.activation == static_cast<int32_t>(AffineActivation::kSilu)) {
        value = value / (1.0f + expf(-value));
        value = static_cast<float>(static_cast<scalar_t>(value));
      }
      sum += value * static_cast<float>(
          output_weight[output_column * arguments.hidden_width + hidden_column]);
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
      if (output_bias != nullptr) {
        sum += static_cast<float>(output_bias[output_column]);
      }
      const int64_t destination_row = static_cast<int64_t>(indices[active_row]);
      if (destination_row >= 0 && destination_row < arguments.destination_rows) {
        output[destination_row * arguments.output_width + output_column] =
            static_cast<scalar_t>(sum);
      }
    }
  }
}

constexpr int kTransformerMaximumTokens = 8;
constexpr int kTransformerAttentionHeads = 4;
constexpr int kTransformerAttentionWidth = 128;
constexpr int kTransformerHeadWidth = 32;

template <typename scalar_t>
__global__ void indexed_relation_attention_bias_kernel(
    const scalar_t* relation_logits, const int16_t* relation_ids,
    const scalar_t* pair_bias, scalar_t* attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale) {
  const int64_t element_count = batch_count * attention_heads * token_count *
      padded_token_count;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t coordinate = index;
    const int64_t key = coordinate % padded_token_count;
    coordinate /= padded_token_count;
    const int64_t query = coordinate % token_count;
    coordinate /= token_count;
    const int64_t head = coordinate % attention_heads;
    const int64_t batch = coordinate / attention_heads;
    if (key >= token_count) {
      attention_bias[index] = static_cast<scalar_t>(0.0f);
      continue;
    }
    const int64_t relation = static_cast<int64_t>(
        relation_ids[query * token_count + key]);
    const int64_t query_row = batch * token_count + query;
    const int64_t relation_logit_index =
        (head * batch_count * token_count + query_row) * relation_count + relation;
    const int64_t pair_bias_index =
        (head * token_count + query) * token_count + key;
    attention_bias[index] = static_cast<scalar_t>(
        static_cast<float>(relation_logits[relation_logit_index]) * scale +
        static_cast<float>(pair_bias[pair_bias_index]));
  }
}

__device__ float TransformerWaveSum(float value) {
  for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return value;
}

template <typename scalar_t, typename norm_t, int fixed_hidden_width, int cached_values_per_lane>
__global__ void transformer_copy_layer_norm_kernel(const scalar_t* input, scalar_t* state, scalar_t* normalized,
    const norm_t* weight, const norm_t* bias, int64_t active_rows, float epsilon, bool copy_input,
    int64_t runtime_hidden_width) {
  const int64_t hidden_width = fixed_hidden_width > 0 ? fixed_hidden_width : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    float values[cached_values_per_lane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * hidden_width;
#pragma unroll
    for (int32_t index = 0; index < cached_values_per_lane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        const int64_t offset = row_offset + feature;
        const float value = static_cast<float>(input[offset]);
        values[index] = value;
        if (copy_input) {
          state[offset] = static_cast<scalar_t>(value);
        }
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll
    for (int32_t index = 0; index < cached_values_per_lane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        normalized[row_offset + feature] = static_cast<scalar_t>(
            (values[index] - mean) * inverse_standard_deviation * static_cast<float>(weight[feature]) +
            static_cast<float>(bias[feature]));
      }
    }
  }
}

template <typename scalar_t>
__global__ void transformer_attention_kernel(const scalar_t* query_key_value,
    scalar_t* context, int64_t batch_count, int64_t token_count) {
  const int64_t batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }
  const int32_t feature = threadIdx.x;
  const int32_t lane = feature % warpSize;
  const int32_t wave = feature / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  __shared__ float probabilities
      [kTransformerAttentionHeads * kTransformerMaximumTokens *
       kTransformerMaximumTokens];

  const int64_t qkv_batch_offset =
      batch * token_count * (3 * kTransformerAttentionWidth);
  constexpr float scale = 0.1767766952966369f;
  const int32_t attention_row_count =
      kTransformerAttentionHeads * static_cast<int32_t>(token_count);
  for (int32_t attention_row = wave; attention_row < attention_row_count;
       attention_row += waves_per_block) {
    const int32_t head = attention_row / token_count;
    const int32_t query = attention_row - head * token_count;
    float logits[kTransformerMaximumTokens];
    for (int32_t key = 0; key < token_count; ++key) {
      float product = 0.0f;
      if (lane < kTransformerHeadWidth) {
        const int64_t query_offset =
            qkv_batch_offset + query * (3 * kTransformerAttentionWidth) +
            head * kTransformerHeadWidth + lane;
        const int64_t key_offset =
            qkv_batch_offset + key * (3 * kTransformerAttentionWidth) +
            kTransformerAttentionWidth + head * kTransformerHeadWidth + lane;
        product = static_cast<float>(query_key_value[query_offset]) *
            static_cast<float>(query_key_value[key_offset]);
      }
      product = TransformerWaveSum(product);
      if (lane == 0) {
        logits[key] = product * scale;
      }
    }
    if (lane == 0) {
      float maximum = logits[0];
      for (int32_t key = 1; key < token_count; ++key) {
        maximum = fmaxf(maximum, logits[key]);
      }
      float denominator = 0.0f;
      for (int32_t key = 0; key < token_count; ++key) {
        logits[key] = expf(logits[key] - maximum);
        denominator += logits[key];
      }
      const int32_t probability_offset =
          (head * kTransformerMaximumTokens + query) *
          kTransformerMaximumTokens;
      for (int32_t key = 0; key < token_count; ++key) {
        probabilities[probability_offset + key] = logits[key] / denominator;
      }
    }
  }
  __syncthreads();

  const int32_t context_elements =
      static_cast<int32_t>(token_count) * kTransformerAttentionWidth;
  for (int32_t index = feature; index < context_elements;
       index += kThreadsPerBlock) {
    const int32_t query = index / kTransformerAttentionWidth;
    const int32_t attention_feature =
        index - query * kTransformerAttentionWidth;
    const int32_t head = attention_feature / kTransformerHeadWidth;
    float value = 0.0f;
    const int32_t probability_offset =
        (head * kTransformerMaximumTokens + query) *
        kTransformerMaximumTokens;
    for (int32_t key = 0; key < token_count; ++key) {
      const int64_t value_offset =
          qkv_batch_offset + key * (3 * kTransformerAttentionWidth) +
          2 * kTransformerAttentionWidth + attention_feature;
      value += probabilities[probability_offset + key] *
          static_cast<float>(query_key_value[value_offset]);
    }
    context[(batch * token_count + query) * kTransformerAttentionWidth +
        attention_feature] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, typename norm_t, bool residual>
__global__ void transformer_dynamic_layer_norm_kernel(const scalar_t* input, scalar_t* state, scalar_t* normalized,
    const scalar_t* update_bias, const norm_t* weight, const norm_t* bias, int64_t active_rows, int64_t hidden_width,
    float epsilon, bool copy_input, bool round_update_before_residual) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    float mean = 0.0f;
    float moment = 0.0f;
    int64_t count = 0;
    const int64_t row_offset = row * hidden_width;
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      const int64_t offset = row_offset + feature;
      float value = static_cast<float>(input[offset]);
      if constexpr (residual) {
        value += static_cast<float>(update_bias[feature]);
        if (round_update_before_residual) {
          value = static_cast<float>(static_cast<scalar_t>(value));
        }
        value += static_cast<float>(state[offset]);
        state[offset] = static_cast<scalar_t>(value);
        if (!round_update_before_residual) {
          value = static_cast<float>(state[offset]);
        }
      } else if (copy_input) {
        state[offset] = static_cast<scalar_t>(value);
      }
      ++count;
      const float delta = value - mean;
      mean += delta / count;
      moment += delta * (value - mean);
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int64_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int64_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      const int64_t offset = row_offset + feature;
      const float value = static_cast<float>(residual ? state[offset] : input[offset]);
      normalized[offset] =
          static_cast<scalar_t>((value - mean) * inverse_standard_deviation * static_cast<float>(weight[feature]) +
                                static_cast<float>(bias[feature]));
    }
  }
}

template <typename scalar_t>
__global__ void transformer_dynamic_attention_kernel(const scalar_t* query_key_value, scalar_t* context,
    int64_t attention_rows, int64_t token_count, int64_t attention_heads, int64_t attention_width, int64_t head_width) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  extern __shared__ float probabilities[];
  const float scale = rsqrtf(static_cast<float>(head_width));
  for (int64_t row = blockIdx.x; row < attention_rows; row += gridDim.x) {
    const int64_t query = row % token_count;
    const int64_t head = (row / token_count) % attention_heads;
    const int64_t batch = row / (token_count * attention_heads);
    const int64_t qkv_batch_offset = batch * token_count * (3 * attention_width);
    const int64_t query_offset = qkv_batch_offset + query * (3 * attention_width) + head * head_width;
    for (int64_t key = wave; key < token_count; key += waves_per_block) {
      const int64_t key_offset = qkv_batch_offset + key * (3 * attention_width) + attention_width + head * head_width;
      float product = 0.0f;
      for (int64_t feature = lane; feature < head_width; feature += warpSize) {
        product += static_cast<float>(query_key_value[query_offset + feature]) *
                   static_cast<float>(query_key_value[key_offset + feature]);
      }
      product = TransformerWaveSum(product);
      if (lane == 0) {
        probabilities[key] = product * scale;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float maximum = probabilities[0];
      for (int64_t key = 1; key < token_count; ++key) {
        maximum = fmaxf(maximum, probabilities[key]);
      }
      float denominator = 0.0f;
      for (int64_t key = 0; key < token_count; ++key) {
        probabilities[key] = expf(probabilities[key] - maximum);
        denominator += probabilities[key];
      }
      for (int64_t key = 0; key < token_count; ++key) {
        probabilities[key] /= denominator;
      }
    }
    __syncthreads();
    for (int64_t feature = threadIdx.x; feature < head_width; feature += blockDim.x) {
      float value = 0.0f;
      for (int64_t key = 0; key < token_count; ++key) {
        const int64_t value_offset =
            qkv_batch_offset + key * (3 * attention_width) + 2 * attention_width + head * head_width + feature;
        value += probabilities[key] * static_cast<float>(query_key_value[value_offset]);
      }
      context[(batch * token_count + query) * attention_width + head * head_width + feature] =
          static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void transformer_bias_silu_kernel(
    scalar_t* values, const scalar_t* bias, int64_t element_count,
    int64_t width, bool round_bias_before_activation) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float value = static_cast<float>(values[index]) +
        static_cast<float>(bias[index % width]);
    if (round_bias_before_activation) {
      value = static_cast<float>(static_cast<scalar_t>(value));
    }
    values[index] = static_cast<scalar_t>(value / (1.0f + expf(-value)));
  }
}

template <typename scalar_t, typename norm_t, int fixed_hidden_width, int cached_values_per_lane>
__global__ void transformer_residual_layer_norm_kernel(scalar_t* state, const scalar_t* update,
    const scalar_t* update_bias, const norm_t* norm_weight, const norm_t* norm_bias, scalar_t* normalized,
    int64_t active_rows, float epsilon, bool round_update_before_residual, int64_t runtime_hidden_width) {
  const int64_t hidden_width = fixed_hidden_width > 0 ? fixed_hidden_width : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    float values[cached_values_per_lane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * hidden_width;
#pragma unroll
    for (int32_t index = 0; index < cached_values_per_lane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        const int64_t offset = row_offset + feature;
        float update_value = static_cast<float>(update[offset]) + static_cast<float>(update_bias[feature]);
        if (round_update_before_residual) {
          update_value = static_cast<float>(static_cast<scalar_t>(update_value));
        }
        const float value = static_cast<float>(state[offset]) + update_value;
        const scalar_t rounded = static_cast<scalar_t>(value);
        state[offset] = rounded;
        values[index] = static_cast<float>(rounded);
        ++count;
        const float statistics_value = round_update_before_residual ? value : values[index];
        const float delta = statistics_value - mean;
        mean += delta / count;
        moment += delta * (statistics_value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll
    for (int32_t index = 0; index < cached_values_per_lane; ++index) {
      const int32_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        normalized[row_offset + feature] = static_cast<scalar_t>(
            (values[index] - mean) * inverse_standard_deviation * static_cast<float>(norm_weight[feature]) +
            static_cast<float>(norm_bias[feature]));
      }
    }
  }
}

constexpr int kSingleQuerySpecializedHiddenWidth = 256;
constexpr int kSingleQuerySpecializedTokens = 256;
constexpr int kSingleQuerySpecializedHeadWidth = 32;
constexpr int kSingleQueryCachedValuesPerLane = kSingleQuerySpecializedHiddenWidth / 32;
// Reserve space for the static softmax scalars in the portable 48 KiB budget.
constexpr int64_t kSingleQuerySharedMemoryBudgetBytes =
    kFusionAttentionSharedMemoryBytes - kSingleQueryReadoutSharedMemoryReserve;

template <typename mask_t>
__device__ bool single_query_valid(const mask_t* mask, int64_t index) {
  return static_cast<float>(mask[index]) != 0.0f;
}

template <typename memory_t, typename scalar_t, typename mask_t, bool specialized>
__global__ void single_query_seed_input_kernel(const memory_t* memory, const memory_t* query_source,
    const mask_t* valid_mask, scalar_t* seed_input, int64_t batch_count, int64_t token_count,
    int64_t runtime_hidden_width, int64_t query_token_count, int64_t query_index) {
  const int64_t hidden_width = specialized ? kSingleQuerySpecializedHiddenWidth : runtime_hidden_width;
  const int64_t batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }
  __shared__ int32_t valid_count;
  if (threadIdx.x == 0) {
    int32_t count = 0;
    for (int64_t token = 0; token < token_count; ++token) {
      count += single_query_valid(
          valid_mask, batch * token_count + token) ? 1 : 0;
    }
    valid_count = count;
  }
  __syncthreads();
  for (int32_t feature = threadIdx.x; feature < hidden_width; feature += blockDim.x) {
    seed_input[batch * (2 * hidden_width) + feature] =
        static_cast<scalar_t>(query_source[(batch * query_token_count + query_index) * hidden_width + feature]);
    float sum = 0.0f;
    if (valid_count != 0) {
      for (int64_t token = 0; token < token_count; ++token) {
        if (single_query_valid(
                valid_mask, batch * token_count + token)) {
          sum += static_cast<float>(memory[(batch * token_count + token) * hidden_width + feature]);
        }
      }
      sum /= valid_count;
    }
    seed_input[batch * (2 * hidden_width) + hidden_width + feature] = static_cast<scalar_t>(sum);
  }
}

template <typename memory_t, typename scalar_t, typename mask_t, bool specialized>
__global__ void single_query_attention_kernel(const scalar_t* seed_products, const scalar_t* seed_bias,
    const memory_t* memory, const mask_t* valid_mask, const scalar_t* query_weight, const scalar_t* query_bias,
    const scalar_t* key_value_weight, scalar_t* state, scalar_t* context, int64_t batch_count, int64_t readout_count,
    int64_t token_count, int64_t runtime_hidden_width, int64_t attention_heads, int64_t attention_width) {
  const int64_t hidden_width = specialized ? kSingleQuerySpecializedHiddenWidth : runtime_hidden_width;
  const int64_t batch = blockIdx.y;
  const int64_t readout_head = blockIdx.x;
  const int64_t readout = readout_head / attention_heads;
  const int64_t head = readout_head - readout * attention_heads;
  if (batch >= batch_count || readout >= readout_count) {
    return;
  }
  const int64_t head_width = attention_width / attention_heads;
  using valid_index_t = std::conditional_t<specialized, uint8_t, uint32_t>;
  const int64_t query_capacity = specialized ? kSingleQuerySpecializedHeadWidth : head_width;
  const int64_t token_capacity = specialized ? kSingleQuerySpecializedTokens : token_count;
  extern __shared__ float shared_values[];
  float* query = shared_values;
  float* direction = query + query_capacity;
  float* probabilities = direction + hidden_width;
  float* pooled = probabilities + token_capacity;
  valid_index_t* valid_tokens = reinterpret_cast<valid_index_t*>(pooled + hidden_width);
  __shared__ float softmax_maximum;
  __shared__ float softmax_denominator;
  __shared__ int64_t softmax_valid_count;
  __shared__ float reduction_values[32];
  const int64_t seed_offset = (batch * readout_count + readout) * hidden_width;
  const int64_t state_offset = (readout * batch_count + batch) * hidden_width;
  const int64_t readout_hidden_offset = readout * hidden_width;
  if (head == 0) {
    for (int32_t feature = threadIdx.x; feature < hidden_width; feature += blockDim.x) {
      state[state_offset + feature] = seed_products[seed_offset + feature];
    }
  }
  const int32_t projection_lane = threadIdx.x % warpSize;
  const int32_t projection_wave = threadIdx.x / warpSize;
  const int32_t projection_waves = blockDim.x / warpSize;
  for (int64_t head_feature = projection_wave; head_feature < head_width;
       head_feature += projection_waves) {
    const int64_t attention_feature = head * head_width + head_feature;
    float value = 0.0f;
    const int64_t weight_offset = (readout * attention_width + attention_feature) * hidden_width;
    for (int32_t feature = projection_lane; feature < hidden_width; feature += warpSize) {
      const float seed = static_cast<float>(seed_products[seed_offset + feature]) +
          static_cast<float>(seed_bias[readout_hidden_offset + feature]);
      value += seed * static_cast<float>(query_weight[weight_offset + feature]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset);
    }
    if (projection_lane == 0) {
      query[head_feature] = value + static_cast<float>(
          query_bias[readout * attention_width + attention_feature]);
    }
  }
  __syncthreads();

  for (int32_t feature = threadIdx.x; feature < hidden_width; feature += blockDim.x) {
    float value = 0.0f;
    for (int64_t head_feature = 0; head_feature < head_width;
         ++head_feature) {
      const int64_t attention_feature = head * head_width + head_feature;
      const int64_t key_weight_offset = (readout * (2 * attention_width) + attention_feature) * hidden_width + feature;
      value += query[head_feature] *
          static_cast<float>(key_value_weight[key_weight_offset]);
    }
    direction[feature] = value * rsqrtf(static_cast<float>(head_width));
  }
  __syncthreads();

  // Cooperate on each token's dot product so adjacent lanes read adjacent
  // memory features. Accumulate in FP32; only the reduction order changes.
  const int32_t score_lane = threadIdx.x % warpSize;
  const int32_t score_wave = threadIdx.x / warpSize;
  const int32_t score_waves = blockDim.x / warpSize;
  for (int64_t token = score_wave; token < token_count; token += score_waves) {
    const bool valid = single_query_valid(
        valid_mask, batch * token_count + token);
    float score = -INFINITY;
    if (valid) {
      score = 0.0f;
      const int64_t memory_offset = (batch * token_count + token) * hidden_width;
      for (int32_t feature = score_lane; feature < hidden_width; feature += warpSize) {
        score += static_cast<float>(static_cast<scalar_t>(memory[memory_offset + feature])) *
            direction[feature];
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        score += __shfl_down(score, offset);
      }
    }
    if (score_lane == 0) {
      probabilities[token] = score;
    }
  }
  __syncthreads();
  float maximum = -INFINITY;
  int valid_count = 0;
  for (int64_t token = threadIdx.x; token < token_count; token += blockDim.x) {
    if (single_query_valid(valid_mask, batch * token_count + token)) {
      const float score = probabilities[token];
      maximum = isnan(maximum) || isnan(score) ? NAN : fmaxf(maximum, score);
      ++valid_count;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other = __shfl_down(maximum, offset);
    maximum = isnan(maximum) || isnan(other) ? NAN : fmaxf(maximum, other);
    valid_count += __shfl_down(valid_count, offset);
  }
  if (score_lane == 0) {
    reduction_values[score_wave] = maximum;
    reduction_values[16 + score_wave] = static_cast<float>(valid_count);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float value = -INFINITY;
    int count = 0;
    for (int wave = 0; wave < score_waves; ++wave) {
      const float other = reduction_values[wave];
      value = isnan(value) || isnan(other) ? NAN : fmaxf(value, other);
      count += static_cast<int>(reduction_values[16 + wave]);
    }
    softmax_maximum = value;
    softmax_valid_count = count;
  }
  __syncthreads();
  for (int64_t token = threadIdx.x; token < token_count; token += blockDim.x) {
    probabilities[token] = softmax_valid_count != 0 && single_query_valid(
            valid_mask, batch * token_count + token)
        ? expf(probabilities[token] - softmax_maximum)
        : 0.0f;
  }
  __syncthreads();
  float denominator = 0.0f;
  for (int64_t token = threadIdx.x; token < token_count; token += blockDim.x) {
    denominator += probabilities[token];
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    denominator += __shfl_down(denominator, offset);
  }
  if (score_lane == 0) reduction_values[score_wave] = denominator;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int wave = 0; wave < score_waves; ++wave) sum += reduction_values[wave];
    softmax_denominator = sum;
  }
  __syncthreads();
  if (softmax_valid_count != 0) {
    for (int64_t token = threadIdx.x; token < token_count; token += blockDim.x) {
      probabilities[token] /= softmax_denominator;
    }
  }
  __syncthreads();

  for (int32_t feature = threadIdx.x; feature < hidden_width; feature += blockDim.x) {
    float value = 0.0f;
    for (int64_t token = 0; token < token_count; ++token) {
      if (!single_query_valid(valid_mask, batch * token_count + token)) continue;
      const float probability = probabilities[token];
      value += probability * static_cast<float>(
          static_cast<scalar_t>(memory[(batch * token_count + token) * hidden_width + feature]));
    }
    pooled[feature] = value;
  }
  __syncthreads();
  for (int64_t head_feature = projection_wave; head_feature < head_width;
       head_feature += projection_waves) {
    const int64_t attention_feature = head * head_width + head_feature;
    float value = 0.0f;
    const int64_t value_weight_offset =
        (readout * (2 * attention_width) + attention_width + attention_feature) * hidden_width;
    for (int32_t feature = projection_lane; feature < hidden_width; feature += warpSize) {
      value += pooled[feature] *
          static_cast<float>(key_value_weight[value_weight_offset + feature]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset);
    }
    if (projection_lane == 0) {
      context[(readout * batch_count + batch) * attention_width +
          attention_feature] = static_cast<scalar_t>(value);
    }
  }
}

template <typename scalar_t, typename norm_t, bool specialized, int cached_values_per_lane>
__global__ void single_query_residual_layer_norm_kernel(const scalar_t* state, const scalar_t* update,
    const scalar_t* state_bias, const scalar_t* update_bias, const norm_t* norm_weight, const norm_t* norm_bias,
    scalar_t* output, int64_t batch_count, int64_t readout_count, int64_t runtime_hidden_width,
    int64_t output_batch_stride, float epsilon, bool batch_major_output) {
  constexpr bool cached = cached_values_per_lane > 0;
  constexpr int unroll_factor = cached ? cached_values_per_lane : 1;
  const int64_t hidden_width = specialized ? kSingleQuerySpecializedHiddenWidth : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int64_t active_rows = batch_count * readout_count;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t readout = row / batch_count;
    const int64_t batch = row - readout * batch_count;
    float values[cached ? cached_values_per_lane : 1];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * hidden_width;
    const int64_t iterations = cached ? cached_values_per_lane : (hidden_width + warpSize - 1) / warpSize;
#pragma unroll unroll_factor
    for (int64_t index = 0; index < iterations; ++index) {
      const int64_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        float value = static_cast<float>(state[row_offset + feature]) +
                      static_cast<float>(update[row_offset + feature]) +
                      static_cast<float>(update_bias[readout * hidden_width + feature]);
        if (state_bias != nullptr) {
          value += static_cast<float>(state_bias[readout * hidden_width + feature]);
        }
        value = static_cast<float>(static_cast<scalar_t>(value));
        if constexpr (cached) {
          values[index] = value;
        }
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll unroll_factor
    for (int64_t index = 0; index < iterations; ++index) {
      const int64_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        float value;
        if constexpr (cached) {
          value = values[index];
        } else {
          value = static_cast<float>(state[row_offset + feature]) + static_cast<float>(update[row_offset + feature]) +
                  static_cast<float>(update_bias[readout * hidden_width + feature]);
          if (state_bias != nullptr) {
            value += static_cast<float>(state_bias[readout * hidden_width + feature]);
          }
          value = static_cast<float>(static_cast<scalar_t>(value));
        }
        const float normalized = (value - mean) * inverse_standard_deviation *
                                     static_cast<float>(norm_weight[readout * hidden_width + feature]) +
                                 static_cast<float>(norm_bias[readout * hidden_width + feature]);
        const int64_t output_offset = batch_major_output
                                          ? (readout * output_batch_stride + batch) * hidden_width + feature
                                          : row_offset + feature;
        output[output_offset] = static_cast<scalar_t>(normalized);
      }
    }
  }
}

template <typename scalar_t, typename norm_t, bool specialized, int cached_values_per_lane>
__global__ void single_query_layer_norm_kernel(const scalar_t* input, const norm_t* norm_weight,
    const norm_t* norm_bias, scalar_t* output, int64_t batch_count, int64_t readout_count, int64_t runtime_hidden_width,
    float epsilon) {
  constexpr bool cached = cached_values_per_lane > 0;
  constexpr int unroll_factor = cached ? cached_values_per_lane : 1;
  const int64_t hidden_width = specialized ? kSingleQuerySpecializedHiddenWidth : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int64_t active_rows = batch_count * readout_count;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
       row < active_rows;
       row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t readout = row / batch_count;
    float values[cached ? cached_values_per_lane : 1];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t row_offset = row * hidden_width;
    const int64_t iterations = cached ? cached_values_per_lane : (hidden_width + warpSize - 1) / warpSize;
#pragma unroll unroll_factor
    for (int64_t index = 0; index < iterations; ++index) {
      const int64_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        const float value = static_cast<float>(input[row_offset + feature]);
        if constexpr (cached) {
          values[index] = value;
        }
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count /
            combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll unroll_factor
    for (int64_t index = 0; index < iterations; ++index) {
      const int64_t feature = lane + index * warpSize;
      if (feature < hidden_width) {
        float value;
        if constexpr (cached) {
          value = values[index];
        } else {
          value = static_cast<float>(input[row_offset + feature]);
        }
        output[row_offset + feature] =
            static_cast<scalar_t>((value - mean) * inverse_standard_deviation *
                                      static_cast<float>(norm_weight[readout * hidden_width + feature]) +
                                  static_cast<float>(norm_bias[readout * hidden_width + feature]));
      }
    }
  }
}

template <typename scalar_t>
__global__ void single_query_bias_silu_kernel(scalar_t* values,
    const scalar_t* bias, int64_t batch_count, int64_t readout_count,
    int64_t width) {
  const int64_t element_count = batch_count * readout_count * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < element_count;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t readout = index / (batch_count * width);
    const int64_t feature = index % width;
    const float value = static_cast<float>(values[index]) +
        static_cast<float>(bias[readout * width + feature]);
    values[index] = static_cast<scalar_t>(
        value / (1.0f + expf(-value)));
  }
}

constexpr int kIndexedLocalMaximumTokens = 32;
constexpr int kIndexedLocalAttentionHeads = 4;
constexpr int kIndexedLocalAttentionWidth = 64;
constexpr int kIndexedLocalHeadWidth = 16;

template <typename scalar_t>
struct IndexedLocalDenseInput {
  const scalar_t* data;
  int64_t hidden_width;

  __device__ const scalar_t* row(int64_t dense_row) const { return data + dense_row * hidden_width; }
};

template <typename scalar_t>
struct IndexedLocalSegmentedInput {
  const void* data[kMaximumIndexedLocalTransformerSegments];
  int64_t token_offsets[kMaximumIndexedLocalTransformerSegments];
  int64_t token_counts[kMaximumIndexedLocalTransformerSegments];
  int64_t token_count;
  int64_t hidden_width;
  int32_t segment_count;

  __device__ const scalar_t* row(int64_t dense_row) const {
    const int64_t token = dense_row % token_count;
    const int64_t group_row = dense_row / token_count;
    for (int32_t segment = 0; segment < segment_count; ++segment) {
      const int64_t local_token = token - token_offsets[segment];
      if (local_token >= 0 && local_token < token_counts[segment]) {
        const int64_t segment_row = group_row * token_counts[segment] + local_token;
        return static_cast<const scalar_t*>(data[segment]) + segment_row * hidden_width;
      }
    }
    return nullptr;
  }
};

template <typename scalar_t, typename index_t, typename input_t, int fixed_hidden_width, int cached_values_per_lane>
__global__ void indexed_local_gather_normalize_kernel(input_t input, const index_t* indices,
    const void* input_norm_weight, const void* input_norm_bias, int32_t input_norm_data_type,
    const void* attention_norm_weight, const void* attention_norm_bias, int32_t attention_norm_data_type,
    scalar_t* output, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    float epsilon, int64_t runtime_hidden_width) {
  const int64_t hidden_width = fixed_hidden_width > 0 ? fixed_hidden_width : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[cached_values_per_lane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const scalar_t* input_row = input.row(dense_row);
    if (input_row == nullptr) {
      continue;
    }
    const int64_t output_offset = dense_row * hidden_width;
    const int64_t normalized_offset = row * hidden_width;
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        const float value = static_cast<float>(input_row[feature]);
        values[item] = value;
        ++count;
        const float delta = value - mean;
        mean += delta / count;
        moment += delta * (value - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);

    float second_mean = 0.0f;
    float second_moment = 0.0f;
    count = 0;
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        const float value = (values[item] - mean) * inverse_standard_deviation *
                                load_fusion_float(input_norm_weight, feature, input_norm_data_type) +
                            load_fusion_float(input_norm_bias, feature, input_norm_data_type);
        const scalar_t rounded = static_cast<scalar_t>(value);
        output[output_offset + feature] = rounded;
        values[item] = static_cast<float>(rounded);
        ++count;
        const float delta = values[item] - second_mean;
        second_mean += delta / count;
        second_moment += delta * (values[item] - second_mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(second_mean, offset);
      const float other_moment = __shfl_down(second_moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - second_mean;
        second_moment += other_moment + delta * delta * count * other_count / combined_count;
        second_mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    second_mean = __shfl(second_mean, 0);
    const float second_inverse_standard_deviation = rsqrtf(__shfl(second_moment, 0) / hidden_width + epsilon);
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        normalized[normalized_offset + feature] =
            static_cast<scalar_t>((values[item] - second_mean) * second_inverse_standard_deviation *
                                      load_fusion_float(attention_norm_weight, feature, attention_norm_data_type) +
                                  load_fusion_float(attention_norm_bias, feature, attention_norm_data_type));
      }
    }
  }
}

template <typename scalar_t>
__device__ void indexed_local_row_moments(
    const scalar_t* values, int64_t width, float epsilon, float& mean, float& inverse_standard_deviation) {
  const int32_t lane = threadIdx.x % warpSize;
  float moment = 0.0f;
  int64_t count = 0;
  mean = 0.0f;
  for (int64_t feature = lane; feature < width; feature += warpSize) {
    const float value = static_cast<float>(values[feature]);
    ++count;
    const float delta = value - mean;
    mean += delta / count;
    moment += delta * (value - mean);
  }
  for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
    const float other_mean = __shfl_down(mean, offset);
    const float other_moment = __shfl_down(moment, offset);
    const int64_t other_count = __shfl_down(count, offset);
    if (lane + offset < warpSize && other_count > 0) {
      const int64_t combined_count = count + other_count;
      const float delta = other_mean - mean;
      moment += other_moment + delta * delta * count * other_count / combined_count;
      mean += delta * other_count / combined_count;
      count = combined_count;
    }
  }
  mean = __shfl(mean, 0);
  inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / width + epsilon);
}

template <typename scalar_t, typename index_t, typename input_t>
__global__ void indexed_local_gather_normalize_dynamic_kernel(input_t input, const index_t* indices,
    const void* input_norm_weight, const void* input_norm_bias, int32_t input_norm_data_type,
    const void* attention_norm_weight, const void* attention_norm_bias, int32_t attention_norm_data_type,
    scalar_t* output, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    int64_t hidden_width, float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    const scalar_t* input_row = input.row(dense_row);
    if (input_row == nullptr) {
      continue;
    }
    scalar_t* output_row = output + dense_row * hidden_width;
    float mean;
    float inverse_standard_deviation;
    indexed_local_row_moments(input_row, hidden_width, epsilon, mean, inverse_standard_deviation);
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      output_row[feature] =
          static_cast<scalar_t>((static_cast<float>(input_row[feature]) - mean) * inverse_standard_deviation *
                                    load_fusion_float(input_norm_weight, feature, input_norm_data_type) +
                                load_fusion_float(input_norm_bias, feature, input_norm_data_type));
    }
    indexed_local_row_moments(output_row, hidden_width, epsilon, mean, inverse_standard_deviation);
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      normalized[row * hidden_width + feature] =
          static_cast<scalar_t>((static_cast<float>(output_row[feature]) - mean) * inverse_standard_deviation *
                                    load_fusion_float(attention_norm_weight, feature, attention_norm_data_type) +
                                load_fusion_float(attention_norm_bias, feature, attention_norm_data_type));
    }
  }
}

template <typename scalar_t, typename index_t, bool finalize>
__global__ void indexed_local_residual_normalize_dynamic_kernel(scalar_t* output, const scalar_t* update,
    const index_t* indices, const scalar_t* update_bias, const void* norm_weight, const void* norm_bias,
    int32_t norm_data_type, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    int64_t hidden_width, float epsilon) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    scalar_t* output_row = output + dense_row * hidden_width;
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      output_row[feature] = static_cast<scalar_t>(static_cast<float>(output_row[feature]) +
                                                  static_cast<float>(update[row * hidden_width + feature]) +
                                                  static_cast<float>(update_bias[feature]));
    }
    float mean;
    float inverse_standard_deviation;
    indexed_local_row_moments(output_row, hidden_width, epsilon, mean, inverse_standard_deviation);
    scalar_t* normalized_row = finalize ? output_row : normalized + row * hidden_width;
    for (int64_t feature = lane; feature < hidden_width; feature += warpSize) {
      normalized_row[feature] =
          static_cast<scalar_t>((static_cast<float>(output_row[feature]) - mean) * inverse_standard_deviation *
                                    load_fusion_float(norm_weight, feature, norm_data_type) +
                                load_fusion_float(norm_bias, feature, norm_data_type));
    }
  }
}

template <typename index_t>
__device__ int64_t indexed_local_lower_bound(const index_t* indices, int64_t active_rows, int64_t target) {
  int64_t low = 0;
  int64_t high = active_rows;
  while (low < high) {
    const int64_t middle = low + (high - low) / 2;
    if (static_cast<int64_t>(indices[middle]) < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

template <typename scalar_t, typename index_t>
__global__ void indexed_local_attention_kernel(
    scalar_t* query_key_value, const index_t* indices, int64_t active_rows, int64_t dense_rows, int64_t token_count) {
  const int64_t group = blockIdx.x;
  const int64_t group_count = dense_rows / token_count;
  if (group >= group_count) {
    return;
  }
  __shared__ int64_t group_begin;
  __shared__ int64_t group_end;
  __shared__ float probabilities[(kThreadsPerBlock / kIndexedLocalHeadWidth) * kIndexedLocalMaximumTokens];
  if (threadIdx.x == 0) {
    group_begin = indexed_local_lower_bound(indices, active_rows, group * token_count);
    group_end = indexed_local_lower_bound(indices, active_rows, (group + 1) * token_count);
  }
  __syncthreads();
  const int64_t selected_tokens = group_end - group_begin;
  if (selected_tokens == 0) {
    return;
  }
  const int32_t lane = threadIdx.x % kIndexedLocalHeadWidth;
  const int32_t subgroup = threadIdx.x / kIndexedLocalHeadWidth;
  const int32_t head = subgroup % kIndexedLocalAttentionHeads;
  const int32_t query_slot = subgroup / kIndexedLocalAttentionHeads;
  const int32_t queries_per_block = blockDim.x / (kIndexedLocalHeadWidth * kIndexedLocalAttentionHeads);
  constexpr float scale = 0.25f;
  for (int64_t query_base = 0; query_base < selected_tokens; query_base += queries_per_block) {
    const int64_t query = query_base + query_slot;
    const bool active_head = query < selected_tokens;
    if (active_head) {
      for (int64_t key = 0; key < selected_tokens; ++key) {
        const int64_t query_offset =
            (group_begin + query) * (3 * kIndexedLocalAttentionWidth) + head * kIndexedLocalHeadWidth + lane;
        const int64_t key_offset = (group_begin + key) * (3 * kIndexedLocalAttentionWidth) +
                                   kIndexedLocalAttentionWidth + head * kIndexedLocalHeadWidth + lane;
        float product =
            static_cast<float>(query_key_value[query_offset]) * static_cast<float>(query_key_value[key_offset]);
        for (int32_t offset = kIndexedLocalHeadWidth / 2; offset > 0; offset /= 2) {
          product += __shfl_down(product, offset, kIndexedLocalHeadWidth);
        }
        if (lane == 0) {
          probabilities[subgroup * kIndexedLocalMaximumTokens + key] = product * scale;
        }
      }
    }
    __syncthreads();
    if (active_head && lane == 0) {
      float maximum = probabilities[subgroup * kIndexedLocalMaximumTokens];
      for (int64_t key = 1; key < selected_tokens; ++key) {
        maximum = fmaxf(maximum, probabilities[subgroup * kIndexedLocalMaximumTokens + key]);
      }
      float denominator = 0.0f;
      for (int64_t key = 0; key < selected_tokens; ++key) {
        const int64_t offset = subgroup * kIndexedLocalMaximumTokens + key;
        const float value = expf(probabilities[offset] - maximum);
        probabilities[offset] = value;
        denominator += value;
      }
      for (int64_t key = 0; key < selected_tokens; ++key) {
        probabilities[subgroup * kIndexedLocalMaximumTokens + key] /= denominator;
      }
    }
    __syncthreads();
    if (active_head) {
      float value = 0.0f;
      for (int64_t key = 0; key < selected_tokens; ++key) {
        const int64_t value_offset = (group_begin + key) * (3 * kIndexedLocalAttentionWidth) +
                                     2 * kIndexedLocalAttentionWidth + head * kIndexedLocalHeadWidth + lane;
        value +=
            probabilities[subgroup * kIndexedLocalMaximumTokens + key] * static_cast<float>(query_key_value[value_offset]);
      }
      query_key_value[(group_begin + query) * (3 * kIndexedLocalAttentionWidth) + head * kIndexedLocalHeadWidth +
                      lane] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename index_t>
__global__ void indexed_local_attention_dynamic_kernel(const scalar_t* query_key_value, const index_t* indices,
    scalar_t* context, int64_t active_rows, int64_t token_count, int64_t attention_heads, int64_t attention_width) {
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int64_t head_width = attention_width / attention_heads;
  const float scale = rsqrtf(static_cast<float>(head_width));
  for (int64_t item = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; item < active_rows * attention_heads;
      item += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t row = item / attention_heads;
    const int64_t head = item % attention_heads;
    const int64_t group = static_cast<int64_t>(indices[row]) / token_count;
    int64_t group_begin = 0;
    int64_t group_end = 0;
    if (lane == 0) {
      group_begin = indexed_local_lower_bound(indices, active_rows, group * token_count);
      group_end = indexed_local_lower_bound(indices, active_rows, (group + 1) * token_count);
    }
    group_begin = __shfl(group_begin, 0);
    group_end = __shfl(group_end, 0);
    const int64_t query_offset = row * (3 * attention_width) + head * head_width;
    // Keep Q immutable until every output feature has consumed it. Online softmax
    // bounds register and shared-memory usage independently of the token count.
    for (int64_t feature_base = 0; feature_base < head_width; feature_base += warpSize) {
      const int64_t feature = feature_base + lane;
      float maximum = -INFINITY;
      float denominator = 0.0f;
      float accumulator = 0.0f;
      for (int64_t key = group_begin; key < group_end; ++key) {
        const int64_t key_offset = key * (3 * attention_width) + attention_width + head * head_width;
        float product = 0.0f;
        for (int64_t component = lane; component < head_width; component += warpSize) {
          product += static_cast<float>(query_key_value[query_offset + component]) *
                     static_cast<float>(query_key_value[key_offset + component]);
        }
        for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
          product += __shfl_down(product, offset);
        }
        const float score = __shfl(product, 0) * scale;
        const float next_maximum = fmaxf(maximum, score);
        const float previous_scale = expf(maximum - next_maximum);
        const float probability = expf(score - next_maximum);
        denominator = denominator * previous_scale + probability;
        if (feature < head_width) {
          const int64_t value_offset = key_offset + attention_width + feature;
          accumulator = accumulator * previous_scale + probability * static_cast<float>(query_key_value[value_offset]);
        }
        maximum = next_maximum;
      }
      if (feature < head_width) {
        context[row * attention_width + head * head_width + feature] = static_cast<scalar_t>(accumulator / denominator);
      }
    }
  }
}

template <typename scalar_t, typename index_t, int fixed_hidden_width, int cached_values_per_lane>
__global__ void indexed_local_residual_normalize_kernel(scalar_t* output, const scalar_t* update,
    const index_t* indices, const scalar_t* update_bias, const void* norm_weight, const void* norm_bias,
    int32_t norm_data_type, scalar_t* normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows,
    float epsilon, int64_t runtime_hidden_width) {
  const int64_t hidden_width = fixed_hidden_width > 0 ? fixed_hidden_width : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[cached_values_per_lane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t compact_offset = row * hidden_width;
    const int64_t output_offset = dense_row * hidden_width;
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        const float value = static_cast<float>(output[output_offset + feature]) +
                            static_cast<float>(update[compact_offset + feature]) +
                            static_cast<float>(update_bias[feature]);
        const scalar_t rounded = static_cast<scalar_t>(value);
        output[output_offset + feature] = rounded;
        values[item] = static_cast<float>(rounded);
        ++count;
        const float delta = values[item] - mean;
        mean += delta / count;
        moment += delta * (values[item] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        normalized[compact_offset + feature] =
            static_cast<scalar_t>((values[item] - mean) * inverse_standard_deviation *
                                      load_fusion_float(norm_weight, feature, norm_data_type) +
                                  load_fusion_float(norm_bias, feature, norm_data_type));
      }
    }
  }
}

template <typename scalar_t, typename index_t, int fixed_hidden_width, int cached_values_per_lane>
__global__ void indexed_local_finalize_kernel(scalar_t* output, const scalar_t* update, const index_t* indices,
    const scalar_t* update_bias, const void* norm_weight, const void* norm_bias, int32_t norm_data_type,
    int64_t active_offset, int64_t active_rows, int64_t dense_rows, float epsilon, int64_t runtime_hidden_width) {
  const int64_t hidden_width = fixed_hidden_width > 0 ? fixed_hidden_width : runtime_hidden_width;
  const int32_t lane = threadIdx.x % warpSize;
  const int32_t wave = threadIdx.x / warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * waves_per_block + wave; row < active_rows;
      row += static_cast<int64_t>(gridDim.x) * waves_per_block) {
    const int64_t dense_row = static_cast<int64_t>(indices[active_offset + row]);
    if (dense_row < 0 || dense_row >= dense_rows) {
      continue;
    }
    float values[cached_values_per_lane];
    float mean = 0.0f;
    float moment = 0.0f;
    int32_t count = 0;
    const int64_t compact_offset = row * hidden_width;
    const int64_t output_offset = dense_row * hidden_width;
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        const float value = static_cast<float>(output[output_offset + feature]) +
                            static_cast<float>(update[compact_offset + feature]) +
                            static_cast<float>(update_bias[feature]);
        values[item] = static_cast<float>(static_cast<scalar_t>(value));
        ++count;
        const float delta = values[item] - mean;
        mean += delta / count;
        moment += delta * (values[item] - mean);
      }
    }
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_mean = __shfl_down(mean, offset);
      const float other_moment = __shfl_down(moment, offset);
      const int32_t other_count = __shfl_down(count, offset);
      if (lane + offset < warpSize && other_count > 0) {
        const int32_t combined_count = count + other_count;
        const float delta = other_mean - mean;
        moment += other_moment + delta * delta * count * other_count / combined_count;
        mean += delta * other_count / combined_count;
        count = combined_count;
      }
    }
    mean = __shfl(mean, 0);
    const float inverse_standard_deviation = rsqrtf(__shfl(moment, 0) / hidden_width + epsilon);
#pragma unroll
    for (int32_t item = 0; item < cached_values_per_lane; ++item) {
      const int32_t feature = lane + item * warpSize;
      if (feature < hidden_width) {
        output[output_offset + feature] =
            static_cast<scalar_t>((values[item] - mean) * inverse_standard_deviation *
                                      load_fusion_float(norm_weight, feature, norm_data_type) +
                                  load_fusion_float(norm_bias, feature, norm_data_type));
      }
    }
  }
}

void CheckKernelLaunch(const char* kernel_name) {
  backend::CheckLastLaunch(kernel_name);
}

}  // namespace

#if defined(DJL_USE_ROCM_KERNELS)

namespace {

struct FusionMatmulStreamKey {
  int32_t device;
  std::uintptr_t stream;

  bool operator==(const FusionMatmulStreamKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct FusionMatmulStreamKeyHash {
  std::size_t operator()(const FusionMatmulStreamKey& key) const {
    std::size_t hash = std::hash<int32_t>{}(key.device);
    HashCombine(hash, std::hash<std::uintptr_t>{}(key.stream));
    return hash;
  }
};

std::uintptr_t FusionMatmulStreamIdentity(backend::Stream stream) {
  return reinterpret_cast<std::uintptr_t>(stream);
}

}  // namespace

enum class FusionMatmulExecutionResult : uint8_t {
  kExecuted,
  kUnsupported,
  kRetry,
};

class FusionMatmulStreamContext {
 public:
  FusionMatmulStreamContext(int32_t device, backend::Stream stream)
      : device_(device), stream_(FusionMatmulStreamIdentity(stream)) {
    const auto& diagnostics = launch_environment::GetMatmulDiagnostics();
    c10::DeviceGuard device_guard(
        c10::Device(c10::DeviceType::CUDA, device_));
    bias_placeholder_ = torch::empty({1},
        torch::TensorOptions()
            .device(c10::Device(c10::DeviceType::CUDA, device_))
            .dtype(torch::kUInt8));
    CheckHipBlasLt(hipblasLtCreate(&handle_), "hipblasLtCreate");
    if (diagnostics.context) {
      std::fprintf(stderr,
          "[djl-rocm-matmul] create device=%d stream=%p\n",
          device_, reinterpret_cast<void*>(stream));
    }
  }

  FusionMatmulStreamContext(const FusionMatmulStreamContext&) = delete;
  FusionMatmulStreamContext& operator=(
      const FusionMatmulStreamContext&) = delete;

  bool Matches(int32_t device, backend::Stream stream) const {
    return device_ == device &&
        stream_.load(std::memory_order_acquire) ==
        FusionMatmulStreamIdentity(stream);
  }

  std::unique_lock<std::mutex> Lock() {
    return std::unique_lock<std::mutex>(mutex_);
  }

  std::unique_lock<std::mutex> TryLock() {
    return std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
  }

  void PinLocked() {
    TORCH_CHECK(pin_count_ < std::numeric_limits<std::size_t>::max(),
        "direct ROCm matmul context pin count overflow");
    ++pin_count_;
  }

  void PrepareAndPinForGraphLocked(c10::Device device) {
    TORCH_CHECK(device.index() == device_,
        "direct ROCm matmul graph pin changed devices");
    EnsureWorkspaceCapacityLocked(
        at::cuda::getCUDABlasLtWorkspaceSize(), device);
    PinLocked();
  }

  void CheckPinnedLocked() const {
    TORCH_CHECK(pin_count_ > 0,
        "direct ROCm matmul context is not pinned");
  }

  void UnpinLockedAfterValidation() noexcept {
    --pin_count_;
  }

  bool CanRebindIfIdleLocked(
      backend::Stream stream, backend::Stream* previous_stream) {
    TORCH_CHECK(previous_stream != nullptr,
        "direct ROCm matmul rebind requires the previous stream");
    if (pin_count_ != 0) {
      return false;
    }
    const backend::Stream bound_stream = BoundStreamLocked();
    if (bound_stream == stream) {
      *previous_stream = bound_stream;
      return true;
    }
    c10::DeviceGuard device_guard(
        c10::Device(c10::DeviceType::CUDA, device_));
    const hipError_t query_status = hipStreamQuery(bound_stream);
    if (query_status == hipErrorNotReady) {
      return false;
    }
    TORCH_CHECK(query_status == hipSuccess,
        "hipStreamQuery failed while recycling a direct ROCm matmul "
        "context: ", hipGetErrorString(query_status));
    *previous_stream = bound_stream;
    return true;
  }

  void RebindLocked(backend::Stream stream) {
    const backend::Stream previous_stream = BoundStreamLocked();
    stream_.store(
        FusionMatmulStreamIdentity(stream), std::memory_order_release);
    if (launch_environment::GetMatmulDiagnostics().context) {
      std::fprintf(stderr,
          "[djl-rocm-matmul] rebind device=%d old_stream=%p "
          "new_stream=%p\n",
          device_, reinterpret_cast<void*>(previous_stream),
          reinterpret_cast<void*>(stream));
      LogCacheStatsLocked();
    }
  }

  FusionMatmulExecutionResult ResolvePlanLocked(
      backend::Stream stream, const FusionMatmulKey& key,
      FusionMatmulAlgorithm** algorithm,
      FusionMatmulDescriptors** descriptors) {
    *algorithm = nullptr;
    *descriptors = nullptr;
    if (!MatchesLocked(key.device, stream)) {
      return FusionMatmulExecutionResult::kRetry;
    }
    *descriptors = descriptors_.GetOrCreate(
        FusionMatmulDescriptorKey(key), [&] {
          return std::make_unique<FusionMatmulDescriptors>(
              key, bias_placeholder_.const_data_ptr());
        }).get();
    (*descriptors)->Configure(key);
    bool algorithm_created = false;
    *algorithm = &algorithms_.GetOrCreate(key, [&] {
      algorithm_created = true;
      return (*descriptors)->SelectAlgorithm(
          handle_, key, bias_placeholder_.const_data_ptr());
    });
    if (algorithm_created && launch_environment::GetMatmulDiagnostics().context &&
        (algorithms_.Misses() <= 4 || algorithms_.Misses() % 4096 == 0)) {
      LogCacheStatsLocked();
    }
    if (!(*algorithm)->supported) {
      return FusionMatmulExecutionResult::kUnsupported;
    }
    return FusionMatmulExecutionResult::kExecuted;
  }

  FusionMatmulExecutionResult ExecutePlanLocked(
      backend::Stream stream, const FusionMatmulKey& key,
      const FusionMatmulAlgorithm& algorithm,
      FusionMatmulDescriptors& descriptors, torch::Tensor& output,
      const torch::Tensor& left, const torch::Tensor& right,
      const torch::Tensor* bias, float beta) {
    if (!MatchesLocked(key.device, stream)) {
      return FusionMatmulExecutionResult::kRetry;
    }
    TORCH_CHECK(algorithm.supported,
        "direct ROCm matmul algorithm became unsupported");
    if (!output.defined()) {
      TORCH_CHECK(beta == 0.0f,
          "deferred direct ROCm matmul output requires an overwrite");
      if (key.output_batch_stride == 0) {
        output = torch::empty({key.rows, key.columns},
            left.options().requires_grad(false));
      } else {
        output = torch::empty(
            {key.batch_count, key.rows, key.columns},
            left.options().requires_grad(false));
      }
    }
    const std::size_t required_workspace = algorithm.workspace_size;
    void* workspace = AcquireWorkspace(required_workspace, left.device());
    descriptors.Run(handle_, stream, algorithm, output, left, right,
        bias, workspace, WorkspaceCapacity(), beta);
    return FusionMatmulExecutionResult::kExecuted;
  }

 private:
  void LogCacheStatsLocked() const {
    std::fprintf(stderr,
        "[djl-rocm-matmul] cache device=%d stream=%p "
        "algorithms=%zu algorithm_lookups=%zu algorithm_misses=%zu "
        "algorithm_evictions=%zu algorithm_bytes=%zu "
        "descriptor_families=%zu descriptor_misses=%zu "
        "descriptor_evictions=%zu\n",
        device_, reinterpret_cast<void*>(BoundStreamLocked()),
        algorithms_.Size(), algorithms_.Lookups(), algorithms_.Misses(),
        algorithms_.Evictions(), sizeof(FusionMatmulAlgorithm),
        descriptors_.Size(), descriptors_.Misses(),
        descriptors_.Evictions());
  }

  backend::Stream BoundStreamLocked() const {
    return reinterpret_cast<backend::Stream>(
        stream_.load(std::memory_order_relaxed));
  }

  bool MatchesLocked(int32_t device, backend::Stream stream) const {
    return device_ == device &&
        stream_.load(std::memory_order_relaxed) ==
        FusionMatmulStreamIdentity(stream);
  }

  void EnsureWorkspaceCapacityLocked(
      std::size_t required_bytes, c10::Device device) {
    TORCH_CHECK(required_bytes <=
            static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
        "direct ROCm matmul workspace exceeds the supported range");
    c10::impl::VirtualGuardImpl guard_impl(device.type());
    const c10::Stream stream = guard_impl.getStream(device);
    if (workspace_.defined()) {
      TORCH_CHECK(workspace_.device() == device,
          "direct ROCm matmul workspace changed devices");
    }
    if (required_bytes > 0 &&
        (!workspace_.defined() ||
            workspace_.numel() < static_cast<int64_t>(required_bytes))) {
      TORCH_CHECK(pin_count_ == 0,
          "direct ROCm matmul workspace cannot grow while an accelerator "
          "graph retains its address");
      if (launch_environment::GetMatmulDiagnostics().context) {
        std::fprintf(stderr,
            "[djl-rocm-matmul] workspace-grow device=%d stream=%p "
            "previous=%zu required=%zu\n",
            device_, reinterpret_cast<void*>(BoundStreamLocked()),
            WorkspaceCapacity(), required_bytes);
      }
      if (workspace_.defined()) {
        guard_impl.recordDataPtrOnStream(
            workspace_.storage().data_ptr(), stream);
      }
      workspace_ = torch::empty({static_cast<int64_t>(required_bytes)},
          torch::TensorOptions().device(device).dtype(torch::kUInt8));
    }
  }

  void* AcquireWorkspace(
      std::size_t required_bytes, c10::Device device) {
    EnsureWorkspaceCapacityLocked(required_bytes, device);
    return workspace_.defined() ? workspace_.data_ptr() : nullptr;
  }

  std::size_t WorkspaceCapacity() const {
    return workspace_.defined()
        ? static_cast<std::size_t>(workspace_.numel())
        : 0;
  }

  int32_t device_;
  std::atomic<std::uintptr_t> stream_;
  hipblasLtHandle_t handle_ = nullptr;
  std::mutex mutex_;
  // Descriptors retain opaque library allocations, so cache fewer families
  // than compact algorithms. Bound both caches across model changes.
  FusionMatmulCache<FusionMatmulKey, FusionMatmulAlgorithm,
      FusionMatmulKeyHash> algorithms_{262144};
  FusionMatmulCache<FusionMatmulDescriptorKey,
      std::unique_ptr<FusionMatmulDescriptors>,
      FusionMatmulDescriptorKeyHash> descriptors_{256};
  torch::Tensor bias_placeholder_;
  torch::Tensor workspace_;
  std::size_t pin_count_ = 0;
};

namespace {

thread_local FusionMatmulStreamContext* current_matmul_stream_context =
    nullptr;

struct FusionMatmulStreamRegistry {
  std::mutex mutex;
  std::unordered_map<FusionMatmulStreamKey,
      FusionMatmulStreamContext*, FusionMatmulStreamKeyHash>
      by_stream;
  std::unordered_map<int32_t, std::vector<FusionMatmulStreamContext*>>
      by_device;
};

FusionMatmulStreamRegistry& MatmulStreamRegistry() {
  // The c10 stream pool is process-global and bounded. Retaining contexts
  // avoids hipblasLtDestroy's device synchronization; idle contexts are
  // rebound instead of retaining one large workspace per historical stream.
  static auto* registry = new FusionMatmulStreamRegistry();
  return *registry;
}

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 11

struct PyTorchLtWorkspaceLifetimeRegistry {
  std::mutex mutex;
  std::unordered_map<FusionMatmulStreamKey, std::size_t,
      FusionMatmulStreamKeyHash> graph_pins_by_stream;
};

PyTorchLtWorkspaceLifetimeRegistry& PyTorchLtWorkspaceLifetime() {
  // Thread-local destructors may run during process shutdown. Keep the
  // coordinator alive because cross-library destruction order is unspecified.
  static auto* registry = new PyTorchLtWorkspaceLifetimeRegistry();
  return *registry;
}

bool IsPyTorchLtWorkspaceStreamGraphPinnedLocked(
    const PyTorchLtWorkspaceLifetimeRegistry& registry,
    int32_t device, void* stream) {
  const auto found = registry.graph_pins_by_stream.find({device,
      reinterpret_cast<std::uintptr_t>(stream)});
  return found != registry.graph_pins_by_stream.end() &&
      found->second != 0;
}

struct RetiredPyTorchLtWorkspace {
  at::DataPtr workspace;
  hipStream_t stream;
  c10::DeviceIndex device;
};

static_assert(
    std::is_nothrow_move_constructible_v<RetiredPyTorchLtWorkspace>,
    "retiring a PyTorch Lt workspace must not throw after map removal");

void ReleaseRetiredPyTorchLtWorkspaces(
    std::vector<RetiredPyTorchLtWorkspace>& retired) noexcept {
  for (auto& entry : retired) {
    try {
      c10::cuda::CUDACachingAllocator::recordStream(
          entry.workspace,
          c10::hip::getStreamFromExternal(entry.stream, entry.device));
    } catch (...) {
      // The normal path never synchronizes. If allocator bookkeeping is
      // unexpectedly unavailable during teardown, wait before releasing;
      // on a failed wait, intentionally retain the allocation.
      bool synchronized = false;
      try {
        c10::DeviceGuard device_guard(c10::Device(
            c10::DeviceType::CUDA, entry.device));
        synchronized = hipStreamSynchronize(entry.stream) == hipSuccess;
      } catch (...) {
        // Fall through to the safe leak below.
      }
      if (!synchronized) {
        static_cast<void>(entry.workspace.release_context());
      }
    }
  }
}

struct RegisteredPyTorchLtHandle {
  int32_t device;
  void* handle;
};

class ThreadPyTorchLtWorkspaceRetirement {
 public:
  ThreadPyTorchLtWorkspaceRetirement() = default;
  ThreadPyTorchLtWorkspaceRetirement(
      const ThreadPyTorchLtWorkspaceRetirement&) = delete;
  ThreadPyTorchLtWorkspaceRetirement& operator=(
      const ThreadPyTorchLtWorkspaceRetirement&) = delete;

  ~ThreadPyTorchLtWorkspaceRetirement() noexcept {
    try {
      std::vector<RetiredPyTorchLtWorkspace> retired;
      PyTorchLtWorkspaceLifetimeRegistry& lifetime =
          PyTorchLtWorkspaceLifetime();
      {
        std::lock_guard<std::mutex> lifetime_lock(lifetime.mutex);
        at::cuda::WorkspaceMapWithMutex& workspaces =
            at::cuda::cublaslt_handle_stream_to_workspace();
        using WorkspaceMap = decltype(workspaces.map);
        static_assert(std::is_same_v<typename WorkspaceMap::mapped_type,
            at::DataPtr>,
            "PyTorch 2.11 Lt workspace map ABI changed");
        std::unique_lock<std::shared_mutex> workspace_lock(
            workspaces.mutex);
        std::vector<WorkspaceMap::iterator> candidates;
        candidates.reserve(workspaces.map.size());
        for (auto entry = workspaces.map.begin();
             entry != workspaces.map.end(); ++entry) {
          const int32_t device = entry->second.device().index();
          const void* handle = std::get<0>(entry->first);
          const bool registered = std::find_if(handles_.begin(),
              handles_.end(), [device, handle](const auto& candidate) {
                return candidate.device == device &&
                    candidate.handle == handle;
              }) != handles_.end();
          if (registered &&
              !IsPyTorchLtWorkspaceStreamGraphPinnedLocked(lifetime,
                  device, std::get<1>(entry->first))) {
            candidates.push_back(entry);
          }
        }
        retired.reserve(candidates.size());
        for (const auto candidate : candidates) {
          const c10::DeviceIndex device =
              candidate->second.device().index();
          retired.push_back({std::move(candidate->second),
              reinterpret_cast<hipStream_t>(
                  std::get<1>(candidate->first)), device});
          workspaces.map.erase(candidate);
        }
      }
        ReleaseRetiredPyTorchLtWorkspaces(retired);
    } catch (...) {
      // Thread teardown cannot report failures. Retaining a workspace is safe.
    }
  }

  bool IsDeviceRegistered(int32_t device) const {
    return std::find_if(handles_.begin(), handles_.end(),
        [device](const auto& candidate) {
          return candidate.device == device;
        }) != handles_.end();
  }

  void Register(int32_t device, void* handle) {
    if (!IsDeviceRegistered(device)) {
      handles_.push_back({device, handle});
    }
  }

 private:
  std::vector<RegisteredPyTorchLtHandle> handles_;
};

void RegisterCurrentThreadPyTorchLtWorkspaceRetirement(c10::Device device) {
  thread_local ThreadPyTorchLtWorkspaceRetirement* retirement = nullptr;
  if (retirement != nullptr &&
      retirement->IsDeviceRegistered(device.index())) {
    return;
  }
  c10::DeviceGuard device_guard(device);
  // Construct PyTorch's PoolWindow first. Reverse TLS destruction then erases
  // workspace entries before this thread returns its Lt handle to that pool.
  void* handle = reinterpret_cast<void*>(
      at::cuda::getCurrentCUDABlasLtHandle());
  if (retirement == nullptr) {
    thread_local ThreadPyTorchLtWorkspaceRetirement owned_retirement;
    retirement = &owned_retirement;
  }
  retirement->Register(device.index(), handle);
}

void PinCurrentPyTorchLtWorkspaceStream(
    c10::Device device, backend::Stream stream) {
  PyTorchLtWorkspaceLifetimeRegistry& lifetime =
      PyTorchLtWorkspaceLifetime();
  std::lock_guard<std::mutex> lock(lifetime.mutex);
  std::size_t& pins = lifetime.graph_pins_by_stream[{device.index(),
      FusionMatmulStreamIdentity(stream)}];
  TORCH_CHECK(pins < std::numeric_limits<std::size_t>::max(),
      "PyTorch ROCm Lt workspace graph pin count overflow");
  ++pins;
}

void UnpinCurrentPyTorchLtWorkspaceStream(
    c10::Device device, backend::Stream stream) {
  PyTorchLtWorkspaceLifetimeRegistry& lifetime =
      PyTorchLtWorkspaceLifetime();
  std::lock_guard<std::mutex> lock(lifetime.mutex);
  const auto pin = lifetime.graph_pins_by_stream.find({device.index(),
      FusionMatmulStreamIdentity(stream)});
  TORCH_CHECK(pin != lifetime.graph_pins_by_stream.end() &&
          pin->second != 0,
      "PyTorch ROCm Lt workspace stream is not graph-pinned");
  if (--pin->second == 0) {
    lifetime.graph_pins_by_stream.erase(pin);
  }
}

#else

void RegisterCurrentThreadPyTorchLtWorkspaceRetirement(c10::Device) {}
void PinCurrentPyTorchLtWorkspaceStream(c10::Device, backend::Stream) {}
void UnpinCurrentPyTorchLtWorkspaceStream(c10::Device, backend::Stream) {}

#endif

struct FusionMatmulStreamContextLease {
  FusionMatmulStreamContext* context;
  std::unique_lock<std::mutex> lock;
};

FusionMatmulStreamContextLease ResolveMatmulStreamContextLease(
    c10::Device device, backend::Stream stream) {
  TORCH_CHECK(device.is_cuda(),
      "direct ROCm matmul requires a ROCm device");
  c10::DeviceGuard device_guard(device);
  RegisterCurrentThreadPyTorchLtWorkspaceRetirement(device);
  FusionMatmulStreamRegistry& registry = MatmulStreamRegistry();
  const FusionMatmulStreamKey key{device.index(),
      FusionMatmulStreamIdentity(stream)};
  for (;;) {
    FusionMatmulStreamContext* mapped_context = nullptr;
    std::vector<FusionMatmulStreamContext*> device_contexts;
    {
      std::lock_guard<std::mutex> registry_lock(registry.mutex);
      auto found = registry.by_stream.find(key);
      if (found != registry.by_stream.end()) {
        mapped_context = found->second;
      } else {
        auto device_found = registry.by_device.find(device.index());
        if (device_found != registry.by_device.end()) {
          device_contexts = device_found->second;
        }
      }
    }

    if (mapped_context != nullptr) {
      std::unique_lock<std::mutex> context_lock = mapped_context->Lock();
      if (!mapped_context->Matches(device.index(), stream)) {
        continue;
      }
      current_matmul_stream_context = mapped_context;
      return FusionMatmulStreamContextLease{
          mapped_context, std::move(context_lock)};
    }

    bool retry_mapped_context = false;
    for (FusionMatmulStreamContext* context : device_contexts) {
      std::unique_lock<std::mutex> context_lock = context->TryLock();
      if (!context_lock.owns_lock()) {
        continue;
      }
      backend::Stream previous_stream = nullptr;
      if (!context->CanRebindIfIdleLocked(stream, &previous_stream)) {
        continue;
      }
      {
        std::lock_guard<std::mutex> registry_lock(registry.mutex);
        if (registry.by_stream.find(key) != registry.by_stream.end()) {
          retry_mapped_context = true;
        } else {
          const FusionMatmulStreamKey previous_key{device.index(),
              FusionMatmulStreamIdentity(previous_stream)};
          auto previous = registry.by_stream.find(previous_key);
          if (previous != registry.by_stream.end() &&
              previous->second == context) {
            registry.by_stream.erase(previous);
          }
          context->RebindLocked(stream);
          registry.by_stream.emplace(key, context);
        }
      }
      if (retry_mapped_context) {
        break;
      }
      current_matmul_stream_context = context;
      return FusionMatmulStreamContextLease{
          context, std::move(context_lock)};
    }
    if (retry_mapped_context) {
      continue;
    }

    {
      std::lock_guard<std::mutex> registry_lock(registry.mutex);
      if (registry.by_stream.find(key) != registry.by_stream.end()) {
        continue;
      }
      auto* context =
          new FusionMatmulStreamContext(device.index(), stream);
      std::unique_lock<std::mutex> context_lock = context->Lock();
      registry.by_device[device.index()].push_back(context);
      registry.by_stream.emplace(key, context);
      current_matmul_stream_context = context;
      return FusionMatmulStreamContextLease{
          context, std::move(context_lock)};
    }
  }
}

struct FusionMatmulPlanLease {
  FusionMatmulStreamContext* context;
  backend::Stream stream;
  FusionMatmulAlgorithm* algorithm;
  FusionMatmulDescriptors* descriptors;
  FusionMatmulExecutionResult result;
  std::unique_lock<std::mutex> lock;
};

FusionMatmulPlanLease ResolveMatmulPlan(
    const FusionMatmulKey& key) {
  const c10::Device device(c10::DeviceType::CUDA, key.device);
  c10::DeviceGuard device_guard(device);
  for (;;) {
    const backend::Stream stream =
        backend::GetCurrentStream(device.index());
    FusionMatmulStreamContext* context = nullptr;
    std::unique_lock<std::mutex> context_lock;
    if (current_matmul_stream_context != nullptr &&
        current_matmul_stream_context->Matches(device.index(), stream)) {
      context = current_matmul_stream_context;
      context_lock = context->Lock();
    } else {
      FusionMatmulStreamContextLease context_lease =
          ResolveMatmulStreamContextLease(device, stream);
      context = context_lease.context;
      context_lock = std::move(context_lease.lock);
    }
    FusionMatmulAlgorithm* algorithm = nullptr;
    FusionMatmulDescriptors* descriptors = nullptr;
    const FusionMatmulExecutionResult result =
        context->ResolvePlanLocked(
            stream, key, &algorithm, &descriptors);
    if (result == FusionMatmulExecutionResult::kRetry) {
      current_matmul_stream_context = nullptr;
      continue;
    }
    return FusionMatmulPlanLease{context, stream, algorithm, descriptors,
        result, std::move(context_lock)};
  }
}

FusionMatmulExecutionResult TryExecuteOnCurrentMatmulStream(
    const FusionMatmulKey& key, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor* bias, float beta) {
  FusionMatmulPlanLease lease = ResolveMatmulPlan(key);
  if (lease.result != FusionMatmulExecutionResult::kExecuted) {
    return lease.result;
  }
  return lease.context->ExecutePlanLocked(lease.stream, key,
      *lease.algorithm, *lease.descriptors, output, left, right, bias, beta);
}

}  // namespace

void PinCurrentRocmMatmulStreamContext(c10::Device device) {
  TORCH_CHECK(device.is_cuda(),
      "direct ROCm matmul context pin requires a ROCm device");
  c10::DeviceGuard device_guard(device);
  const backend::Stream stream =
      backend::GetCurrentStream(device.index());
  RegisterCurrentThreadPyTorchLtWorkspaceRetirement(device);
  PinCurrentPyTorchLtWorkspaceStream(device, stream);
  try {
    if (current_matmul_stream_context != nullptr &&
        current_matmul_stream_context->Matches(device.index(), stream)) {
      std::unique_lock<std::mutex> lock =
          current_matmul_stream_context->Lock();
      if (current_matmul_stream_context->Matches(device.index(), stream)) {
        current_matmul_stream_context->PrepareAndPinForGraphLocked(device);
        return;
      }
      current_matmul_stream_context = nullptr;
    }
    FusionMatmulStreamContextLease lease =
        ResolveMatmulStreamContextLease(device, stream);
    lease.context->PrepareAndPinForGraphLocked(device);
  } catch (...) {
    UnpinCurrentPyTorchLtWorkspaceStream(device, stream);
    throw;
  }
}

void UnpinCurrentRocmMatmulStreamContext(c10::Device device) {
  TORCH_CHECK(device.is_cuda(),
      "direct ROCm matmul context unpin requires a ROCm device");
  c10::DeviceGuard device_guard(device);
  const backend::Stream stream =
      backend::GetCurrentStream(device.index());
  FusionMatmulStreamRegistry& registry = MatmulStreamRegistry();
  const FusionMatmulStreamKey key{device.index(),
      FusionMatmulStreamIdentity(stream)};
  FusionMatmulStreamContext* context = nullptr;
  {
    std::lock_guard<std::mutex> registry_lock(registry.mutex);
    auto found = registry.by_stream.find(key);
    TORCH_CHECK(found != registry.by_stream.end(),
        "pinned direct ROCm matmul stream context is missing");
    context = found->second;
  }
  std::unique_lock<std::mutex> context_lock = context->Lock();
  TORCH_CHECK(context->Matches(device.index(), stream),
      "pinned direct ROCm matmul stream context changed streams");
  context->CheckPinnedLocked();

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 11
  // No path acquires a matmul context lock while holding the lifetime lock.
  // Keep this order and validate both counters before changing either one so
  // a failed unpin remains retryable.
  PyTorchLtWorkspaceLifetimeRegistry& lifetime =
      PyTorchLtWorkspaceLifetime();
  std::unique_lock<std::mutex> lifetime_lock(lifetime.mutex);
  const auto pin = lifetime.graph_pins_by_stream.find({device.index(),
      FusionMatmulStreamIdentity(stream)});
  TORCH_CHECK(pin != lifetime.graph_pins_by_stream.end() &&
          pin->second != 0,
      "PyTorch ROCm Lt workspace stream is not graph-pinned");

  std::vector<RetiredPyTorchLtWorkspace> retired;
  if (pin->second == 1) {
    // Graph streams are exclusive DJL-owned leases. Once the last graph pin is
    // released, no unrelated operation can still use this raw stream, so also
    // retire workspaces skipped by a capture thread's TLS destructor.
    at::cuda::WorkspaceMapWithMutex& workspaces =
        at::cuda::cublaslt_handle_stream_to_workspace();
    using WorkspaceMap = decltype(workspaces.map);
    static_assert(std::is_same_v<typename WorkspaceMap::mapped_type,
        at::DataPtr>,
        "PyTorch 2.11 Lt workspace map ABI changed");
    std::unique_lock<std::shared_mutex> workspace_lock(workspaces.mutex);
    const std::uintptr_t stream_identity =
        FusionMatmulStreamIdentity(stream);
    std::size_t candidate_count = 0;
    for (const auto& entry : workspaces.map) {
      if (entry.second.device().index() == device.index() &&
          reinterpret_cast<std::uintptr_t>(
              std::get<1>(entry.first)) == stream_identity) {
        ++candidate_count;
      }
    }
    retired.reserve(candidate_count);
    for (auto entry = workspaces.map.begin();
         entry != workspaces.map.end();) {
      if (entry->second.device().index() != device.index() ||
          reinterpret_cast<std::uintptr_t>(
              std::get<1>(entry->first)) != stream_identity) {
        ++entry;
        continue;
      }
      const auto candidate = entry++;
      const c10::DeviceIndex workspace_device =
          candidate->second.device().index();
      retired.push_back({std::move(candidate->second),
          reinterpret_cast<hipStream_t>(
              std::get<1>(candidate->first)),
          workspace_device});
      workspaces.map.erase(candidate);
    }
  }
  if (--pin->second == 0) {
    lifetime.graph_pins_by_stream.erase(pin);
  }
  context->UnpinLockedAfterValidation();
  lifetime_lock.unlock();
  context_lock.unlock();
  ReleaseRetiredPyTorchLtWorkspaces(retired);
#else
  context->UnpinLockedAfterValidation();
#endif
}

class FusionMatmulWorkspace {
 public:
  FusionMatmulExecutionResult TryExecute(const FusionMatmulKey& key,
      torch::Tensor& output, const torch::Tensor& left,
      const torch::Tensor& right, const torch::Tensor* bias, float beta) {
    return TryExecuteOnCurrentMatmulStream(
        key, output, left, right, bias, beta);
  }
};

bool IsFusionMatmulSupported() {
  return true;
}

bool IsLinearBiasSiluSupported() {
#if defined(DJL_USE_HIPBLASLT_SWISH_BIAS)
  return true;
#else
  return false;
#endif
}

std::size_t GetFusionMatmulWorkspaceUpperBoundBytes() {
  return at::cuda::getCUDABlasLtWorkspaceSize();
}

class FusionMatmulPlan {};

std::shared_ptr<FusionMatmulPlan> CreateFusionMatmulPlan() {
  return std::make_shared<FusionMatmulPlan>();
}

std::shared_ptr<FusionMatmulWorkspace> CreateFusionMatmulWorkspace() {
  return std::make_shared<FusionMatmulWorkspace>();
}

FusionMatmulKey BuildFusionMatmulKey(torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    FusionMatmulEpilogue epilogue, const torch::Tensor* bias) {
  TORCH_CHECK(output.is_cuda() && left.is_cuda() && right.is_cuda(),
      "Fusion matmul tensors must be on a ROCm device");
  TORCH_CHECK(output.device() == left.device() &&
          output.device() == right.device(),
      "Fusion matmul tensors must be on one device");
  TORCH_CHECK(output.scalar_type() == left.scalar_type() &&
          output.scalar_type() == right.scalar_type(),
      "Fusion matmul tensors must use one data type");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 ||
          output.scalar_type() == torch::kBFloat16 ||
          output.scalar_type() == torch::kFloat32,
      "Fusion matmul requires FLOAT16, BFLOAT16, or FLOAT32 tensors");
  TORCH_CHECK(output.dim() == left.dim() && output.dim() == right.dim() &&
          (output.dim() == 2 || output.dim() == 3),
      "Fusion matmul requires matching rank-2 or rank-3 tensors");

  const int64_t matrix_axis = output.dim() - 2;
  const int64_t batch_count = output.dim() == 3 ? output.size(0) : 1;
  const int64_t rows = left.size(matrix_axis);
  const int64_t reduction = left.size(matrix_axis + 1);
  const int64_t columns = right.size(matrix_axis + 1);
  TORCH_CHECK(rows > 0 && columns > 0 && reduction > 0 &&
          right.size(matrix_axis) == reduction &&
          output.size(matrix_axis) == rows &&
          output.size(matrix_axis + 1) == columns,
      "Fusion matmul tensor shapes are inconsistent");
  if (output.dim() == 3) {
    TORCH_CHECK(left.size(0) == batch_count && right.size(0) == batch_count,
        "Fusion batched matmul batch counts are inconsistent");
  }
  TORCH_CHECK(left.stride(matrix_axis + 1) == 1 &&
          output.stride(matrix_axis + 1) == 1,
      "Fusion matmul requires contiguous left and output final dimensions");
  const int64_t left_row_stride = left.stride(matrix_axis);
  const int64_t output_row_stride = output.stride(matrix_axis);
  const bool right_row_major = right.stride(matrix_axis + 1) == 1 &&
      right.stride(matrix_axis) >= columns;
  const bool right_transpose_view = right.stride(matrix_axis) == 1 &&
      right.stride(matrix_axis + 1) >= reduction;
  TORCH_CHECK(right_row_major || right_transpose_view,
      "Fusion matmul right operand must be row-major or a transpose view");
  const bool right_transposed = !right_row_major;
  int64_t right_leading_dimension;
  if (right_transposed) {
    right_leading_dimension = right.stride(matrix_axis + 1);
  } else {
    right_leading_dimension = right.stride(matrix_axis);
  }
  TORCH_CHECK(left_row_stride >= reduction && output_row_stride >= columns,
      "Fusion matmul left or output leading dimension is too small");
  const int64_t left_batch_stride =
      output.dim() == 3 ? left.stride(0) : 0;
  const int64_t right_batch_stride =
      output.dim() == 3 ? right.stride(0) : 0;
  const int64_t output_batch_stride =
      output.dim() == 3 ? output.stride(0) : 0;
  TORCH_CHECK(left_batch_stride >= 0 && right_batch_stride >= 0 &&
          output_batch_stride >= 0,
      "Fusion matmul does not support negative batch strides");
  TORCH_CHECK(output.dim() == 2 || output_batch_stride >= CheckedMultiply(
          rows, output_row_stride, "Fusion matmul output batch span"),
      "Fusion matmul output batches must not overlap");
  if (epilogue != FusionMatmulEpilogue::kNone) {
    TORCH_CHECK(bias != nullptr && bias->is_cuda() &&
            bias->device() == output.device() &&
            bias->scalar_type() == output.scalar_type() &&
            bias->dim() == 1 && bias->size(0) == columns &&
            bias->is_contiguous(),
        "Fusion matmul bias tensor is inconsistent");
  }
  return FusionMatmulKey{output.get_device(), output.scalar_type(),
      batch_count, rows, columns, reduction, left_row_stride,
      right_leading_dimension, output_row_stride, left_batch_stride,
      right_batch_stride, output_batch_stride, right_transposed, epilogue};
}

FusionMatmulKey BuildRocmInferenceMatmulKey(
    const torch::Tensor& left, const torch::Tensor& right,
    FusionMatmulEpilogue epilogue) {
  const int64_t matrix_axis = left.dim() - 2;
  const int64_t batch_count = left.dim() == 3 ? left.size(0) : 1;
  const int64_t rows = left.size(matrix_axis);
  const int64_t columns = right.size(matrix_axis + 1);
  const bool right_row_major =
      right.stride(matrix_axis + 1) == 1 &&
      right.stride(matrix_axis) >= columns;
  const bool right_transposed = !right_row_major;
  const int64_t right_leading_dimension =
      right_transposed
      ? right.stride(matrix_axis + 1)
      : right.stride(matrix_axis);
  const int64_t output_batch_stride = left.dim() == 3
      ? rows * columns
      : 0;
  return FusionMatmulKey{left.get_device(), left.scalar_type(),
      batch_count, rows, columns, left.size(matrix_axis + 1),
      left.stride(matrix_axis), right_leading_dimension, columns,
      left.dim() == 3 ? left.stride(0) : 0,
      right.dim() == 3 ? right.stride(0) : 0,
      output_batch_stride,
      right_transposed, epilogue};
}

bool CanUseDirectRocmInferenceMatmul() {
  const at::BlasBackend preferred_backend =
      at::globalContext().blasPreferredBackend();
  return c10::InferenceMode::is_enabled() &&
      !at::globalContext().deterministicAlgorithms() &&
      !at::globalContext()._SMCarveout_EXPERIMENTAL().has_value() &&
      !at::cuda::tunable::getTuningContext()->IsTunableOpEnabled() &&
      preferred_backend == at::BlasBackend::Cublaslt;
}

bool HasOnlyPositiveSizesAndNonNegativeStrides(
    const torch::Tensor& tensor) {
  for (const int64_t size : tensor.sizes()) {
    if (size <= 0) {
      return false;
    }
  }
  for (const int64_t stride : tensor.strides()) {
    if (stride < 0) {
      return false;
    }
  }
  return true;
}

bool SupportsRocmInferenceTensor(
    const torch::Tensor& tensor, int64_t minimum_dimensions) {
  return tensor.defined() && tensor.is_cuda() &&
      tensor.layout() == c10::kStrided &&
      tensor.dim() >= minimum_dimensions && !tensor.has_names() &&
      !at::isTensorSubclassLike(tensor) && !tensor.is_conj() &&
      !tensor.is_neg() &&
      HasOnlyPositiveSizesAndNonNegativeStrides(tensor);
}

bool TryMultiplyPositive(
    int64_t left, int64_t right, int64_t* product) {
  if (product == nullptr || left <= 0 || right <= 0 ||
      left > std::numeric_limits<int64_t>::max() / right) {
    return false;
  }
  *product = left * right;
  return true;
}

bool TryPositiveProduct(c10::IntArrayRef sizes, int64_t* product) {
  if (product == nullptr) {
    return false;
  }
  int64_t value = 1;
  for (const int64_t size : sizes) {
    if (!TryMultiplyPositive(value, size, &value)) {
      return false;
    }
  }
  *product = value;
  return true;
}

bool TryCollapseLeadingDimensions(c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    int64_t leading_dimensions, int64_t* element_count,
    int64_t* flattened_stride) {
  if (sizes.size() != strides.size() || leading_dimensions < 0 ||
      leading_dimensions > static_cast<int64_t>(sizes.size()) ||
      element_count == nullptr || flattened_stride == nullptr ||
      !TryPositiveProduct(sizes.slice(0, leading_dimensions),
          element_count)) {
    return false;
  }
  bool found_non_unit_dimension = false;
  int64_t expected_outer_stride = 0;
  *flattened_stride = 0;
  for (int64_t axis = leading_dimensions - 1; axis >= 0; --axis) {
    const int64_t size = sizes[axis];
    if (size == 1) {
      continue;
    }
    const int64_t stride = strides[axis];
    if (!found_non_unit_dimension) {
      *flattened_stride = stride;
      if (stride > std::numeric_limits<int64_t>::max() / size) {
        return false;
      }
      expected_outer_stride = stride * size;
      found_non_unit_dimension = true;
      continue;
    }
    if (stride != expected_outer_stride ||
        expected_outer_stride >
            std::numeric_limits<int64_t>::max() / size) {
      return false;
    }
    expected_outer_stride *= size;
  }
  return true;
}

bool TryCollapseLeadingDimensions(const torch::Tensor& tensor,
    int64_t leading_dimensions, int64_t* element_count,
    int64_t* flattened_stride) {
  return TryCollapseLeadingDimensions(tensor.sizes(), tensor.strides(),
      leading_dimensions, element_count, flattened_stride);
}

bool HasSupportedLeftMatrixLayout(const torch::Tensor& tensor) {
  return tensor.stride(-1) == 1 &&
      tensor.stride(-2) >= tensor.size(-1);
}

bool HasSupportedLeftMatrixLayout(
    int64_t row_stride, int64_t column_stride, int64_t columns) {
  return column_stride == 1 && row_stride >= columns;
}

bool HasSupportedRightMatrixLayout(const torch::Tensor& tensor) {
  const bool row_major = tensor.stride(-1) == 1 &&
      tensor.stride(-2) >= tensor.size(-1);
  const bool transpose_view = tensor.stride(-2) == 1 &&
      tensor.stride(-1) >= tensor.size(-2);
  return row_major || transpose_view;
}

bool HasSupportedRightMatrixLayout(int64_t row_stride,
    int64_t column_stride, int64_t rows, int64_t columns) {
  const bool row_major = column_stride == 1 && row_stride >= columns;
  const bool transpose_view = row_stride == 1 && column_stride >= rows;
  return row_major || transpose_view;
}

bool IsRocmInferenceMatmulDataType(torch::ScalarType data_type) {
  return data_type == torch::kFloat16 ||
      data_type == torch::kBFloat16;
}

bool TryInferBroadcastBatchShape(c10::IntArrayRef left,
    c10::IntArrayRef right, std::vector<int64_t>* output) {
  if (output == nullptr) {
    return false;
  }
  const std::size_t dimensions = std::max(left.size(), right.size());
  output->assign(dimensions, 1);
  for (std::size_t offset = 0; offset < dimensions; ++offset) {
    const int64_t left_size = offset < left.size()
        ? left[left.size() - 1 - offset]
        : 1;
    const int64_t right_size = offset < right.size()
        ? right[right.size() - 1 - offset]
        : 1;
    if (left_size != right_size && left_size != 1 && right_size != 1) {
      return false;
    }
    (*output)[dimensions - 1 - offset] =
        std::max(left_size, right_size);
  }
  return true;
}

struct BatchedMatmulOperandPlan {
  torch::Tensor expanded;
  int64_t batch_count;
  int64_t rows;
  int64_t columns;
  int64_t batch_stride;
  int64_t row_stride;
  int64_t column_stride;
  bool requires_materialization;
};

bool TryPlanBatchedMatmulOperand(const torch::Tensor& tensor,
    c10::IntArrayRef output_batch_shape, int64_t batch_count,
    BatchedMatmulOperandPlan* plan) {
  if (plan == nullptr) {
    return false;
  }

  std::vector<int64_t> expanded_sizes(
      output_batch_shape.begin(), output_batch_shape.end());
  expanded_sizes.push_back(tensor.size(-2));
  expanded_sizes.push_back(tensor.size(-1));
  torch::Tensor expanded = tensor.expand(expanded_sizes);

  int64_t expanded_batch_count;
  int64_t expanded_batch_stride;
  if (TryCollapseLeadingDimensions(expanded, expanded.dim() - 2,
          &expanded_batch_count, &expanded_batch_stride)) {
    if (expanded_batch_count != batch_count) {
      return false;
    }
    *plan = BatchedMatmulOperandPlan{std::move(expanded), batch_count,
        tensor.size(-2), tensor.size(-1), expanded_batch_stride,
        tensor.stride(-2), tensor.stride(-1), false};
    return true;
  }

  int64_t matrix_elements;
  if (!TryMultiplyPositive(
          tensor.size(-2), tensor.size(-1), &matrix_elements)) {
    return false;
  }
  // Defer the same necessary materialization as PyTorch's batched matmul
  // until hipBLASLt has confirmed that the resulting geometry is supported.
  *plan = BatchedMatmulOperandPlan{std::move(expanded), batch_count,
      tensor.size(-2), tensor.size(-1), matrix_elements,
      tensor.size(-1), 1, true};
  return true;
}

torch::Tensor PrepareBatchedMatmulOperand(
    const BatchedMatmulOperandPlan& plan) {
  torch::Tensor prepared = plan.requires_materialization
      ? plan.expanded.contiguous().view(
            {plan.batch_count, plan.rows, plan.columns})
      : plan.expanded.as_strided(
            {plan.batch_count, plan.rows, plan.columns},
            {plan.batch_stride, plan.row_stride, plan.column_stride});
  TORCH_CHECK(prepared.stride(0) == plan.batch_stride &&
          prepared.stride(1) == plan.row_stride &&
          prepared.stride(2) == plan.column_stride,
      "direct ROCm batched matmul preparation changed its planned layout");
  return prepared;
}

FusionMatmulKey BuildRocmInferenceMatmulKey(
    const BatchedMatmulOperandPlan& left,
    const BatchedMatmulOperandPlan& right, c10::Device device,
    torch::ScalarType data_type, FusionMatmulEpilogue epilogue) {
  const bool right_row_major = right.column_stride == 1 &&
      right.row_stride >= right.columns;
  const int64_t right_leading_dimension = right_row_major
      ? right.row_stride
      : right.column_stride;
  int64_t output_batch_stride;
  TORCH_CHECK(TryMultiplyPositive(
          left.rows, right.columns, &output_batch_stride),
      "direct ROCm batched matmul output stride exceeds the supported range");
  return FusionMatmulKey{device.index(), data_type, left.batch_count,
      left.rows, right.columns, left.columns, left.row_stride,
      right_leading_dimension, right.columns, left.batch_stride,
      right.batch_stride, output_batch_stride, !right_row_major, epilogue};
}

bool SupportsRocmInferenceMatmulStructure(const torch::Tensor& left,
    const torch::Tensor& right, const torch::Tensor* bias) {
  if (!SupportsRocmInferenceTensor(left, 2) ||
      !SupportsRocmInferenceTensor(right, 2) ||
      left.device() != right.device() ||
      left.dim() != right.dim() ||
      (left.dim() != 2 && left.dim() != 3)) {
    return false;
  }
  const int64_t matrix_axis = left.dim() - 2;
  const int64_t batch_count = left.dim() == 3 ? left.size(0) : 1;
  const int64_t rows = left.size(matrix_axis);
  const int64_t reduction = left.size(matrix_axis + 1);
  const int64_t columns = right.size(matrix_axis + 1);
  int64_t output_matrix_elements;
  int64_t output_elements;
  if (right.size(matrix_axis) != reduction ||
      (left.dim() == 3 && right.size(0) != batch_count) ||
      batch_count > std::numeric_limits<int32_t>::max() ||
      !HasSupportedLeftMatrixLayout(left) ||
      !TryMultiplyPositive(rows, columns, &output_matrix_elements) ||
      !TryMultiplyPositive(
          batch_count, output_matrix_elements, &output_elements) ||
      output_elements <= 0) {
    return false;
  }
  if (!HasSupportedRightMatrixLayout(right)) {
    return false;
  }
  if (bias != nullptr &&
      (!SupportsRocmInferenceTensor(*bias, 1) ||
          bias->device() != left.device() ||
          bias->dim() != 1 || bias->size(0) != columns ||
          !bias->is_contiguous())) {
    return false;
  }
  return true;
}

bool SupportsRocmInferenceMatmul(const torch::Tensor& left,
    const torch::Tensor& right, const torch::Tensor* bias) {
  return CanUseDirectRocmInferenceMatmul() &&
      SupportsRocmInferenceMatmulStructure(left, right, bias) &&
      left.scalar_type() == right.scalar_type() &&
      (bias == nullptr || bias->scalar_type() == left.scalar_type()) &&
      IsRocmInferenceMatmulDataType(left.scalar_type());
}

torch::Tensor CastRocmAutocastTensor(
    const torch::Tensor& tensor, torch::ScalarType data_type) {
  return tensor.scalar_type() == data_type
      ? tensor
      : at::autocast::cached_cast(
            data_type, tensor, at::DeviceType::CUDA);
}

bool TryExecutePreparedRocmInferenceMatmul(torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor* bias) {
  if (!SupportsRocmInferenceMatmul(left, right, bias)) {
    return false;
  }
  c10::DeviceGuard device_guard(left.device());
  torch::Tensor result;
  const FusionMatmulEpilogue epilogue = bias == nullptr
      ? FusionMatmulEpilogue::kNone
      : FusionMatmulEpilogue::kBias;
  const FusionMatmulKey key =
      BuildRocmInferenceMatmulKey(left, right, epilogue);
  if (TryExecuteOnCurrentMatmulStream(
          key, result, left, right, bias, 0.0f) !=
      FusionMatmulExecutionResult::kExecuted) {
    return false;
  }
  output = std::move(result);
  return true;
}

bool TryPrepareLinearInput(const torch::Tensor& input,
    int64_t output_width, torch::Tensor* flattened,
    std::vector<int64_t>* output_shape) {
  if (flattened == nullptr || output_shape == nullptr ||
      input.stride(-1) != 1) {
    return false;
  }
  int64_t flattened_rows;
  int64_t flattened_row_stride;
  if (!TryCollapseLeadingDimensions(input, input.dim() - 1,
          &flattened_rows, &flattened_row_stride)) {
    return false;
  }
  if (flattened_rows == 1 && flattened_row_stride == 0) {
    flattened_row_stride = input.size(-1);
  }
  int64_t output_elements;
  if (flattened_row_stride < input.size(-1) ||
      !TryMultiplyPositive(
          flattened_rows, output_width, &output_elements) ||
      output_elements <= 0) {
    return false;
  }
  *flattened = input.as_strided(
      {flattened_rows, input.size(-1)},
      {flattened_row_stride, input.stride(-1)});
  *output_shape = input.sizes().vec();
  output_shape->back() = output_width;
  return true;
}

bool TryExecuteRocmInferenceMatmul(torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor* bias) {
  const bool matrix_product = left.dim() == 2 && right.dim() == 2;
  const bool batched_product = left.dim() >= 3 && right.dim() >= 3;
  if (!CanUseDirectRocmInferenceMatmul() ||
      !SupportsRocmInferenceTensor(left, 2) ||
      !SupportsRocmInferenceTensor(right, 2) ||
      (!matrix_product && !batched_product) ||
      left.device() != right.device() ||
      left.size(-1) != right.size(-2) ||
      (bias != nullptr &&
          (!SupportsRocmInferenceTensor(*bias, 1) ||
              bias->device() != left.device() ||
              bias->dim() != 1 ||
              bias->size(0) != right.size(-1) ||
              !bias->is_contiguous()))) {
    return false;
  }
  c10::DeviceGuard device_guard(left.device());
  torch::Tensor operation_left = left;
  torch::Tensor operation_right = right;
  torch::Tensor operation_bias;
  const torch::Tensor* operation_bias_ptr = bias;
  const bool autocast_enabled =
      at::autocast::is_autocast_enabled(at::DeviceType::CUDA);
  if (autocast_enabled) {
    const torch::ScalarType data_type =
        at::autocast::get_autocast_dtype(at::DeviceType::CUDA);
    if (data_type != torch::kFloat16 &&
        data_type != torch::kBFloat16) {
      return false;
    }
    operation_left = CastRocmAutocastTensor(left, data_type);
    operation_right = CastRocmAutocastTensor(right, data_type);
    if (bias != nullptr) {
      operation_bias = CastRocmAutocastTensor(*bias, data_type);
      operation_bias_ptr = &operation_bias;
    }
  }
  if (operation_left.scalar_type() != operation_right.scalar_type() ||
      (operation_bias_ptr != nullptr &&
          operation_bias_ptr->scalar_type() !=
              operation_left.scalar_type()) ||
      !IsRocmInferenceMatmulDataType(operation_left.scalar_type()) ||
      !HasSupportedLeftMatrixLayout(operation_left) ||
      !HasSupportedRightMatrixLayout(operation_right)) {
    return false;
  }
  if (matrix_product) {
    return TryExecutePreparedRocmInferenceMatmul(output,
        operation_left, operation_right, operation_bias_ptr);
  }

  std::vector<int64_t> output_batch_shape;
  if (!TryInferBroadcastBatchShape(
          operation_left.sizes().slice(0, operation_left.dim() - 2),
          operation_right.sizes().slice(0, operation_right.dim() - 2),
          &output_batch_shape)) {
    return false;
  }
  int64_t batch_count;
  int64_t output_matrix_elements;
  int64_t output_elements;
  if (!TryPositiveProduct(output_batch_shape, &batch_count) ||
      batch_count > std::numeric_limits<int32_t>::max() ||
      !TryMultiplyPositive(operation_left.size(-2),
          operation_right.size(-1), &output_matrix_elements) ||
      !TryMultiplyPositive(
          batch_count, output_matrix_elements, &output_elements) ||
      output_elements <= 0) {
    return false;
  }

  BatchedMatmulOperandPlan left_plan;
  BatchedMatmulOperandPlan right_plan;
  if (!TryPlanBatchedMatmulOperand(operation_left,
          output_batch_shape, batch_count, &left_plan) ||
      !TryPlanBatchedMatmulOperand(operation_right,
          output_batch_shape, batch_count, &right_plan) ||
      !HasSupportedLeftMatrixLayout(left_plan.row_stride,
          left_plan.column_stride, left_plan.columns) ||
      !HasSupportedRightMatrixLayout(right_plan.row_stride,
          right_plan.column_stride, right_plan.rows,
          right_plan.columns)) {
    return false;
  }

  const FusionMatmulEpilogue epilogue = operation_bias_ptr == nullptr
      ? FusionMatmulEpilogue::kNone
      : FusionMatmulEpilogue::kBias;
  const FusionMatmulKey key = BuildRocmInferenceMatmulKey(
      left_plan, right_plan, operation_left.device(),
      operation_left.scalar_type(), epilogue);
  FusionMatmulPlanLease lease = ResolveMatmulPlan(key);
  if (lease.result == FusionMatmulExecutionResult::kUnsupported) {
    return false;
  }
  TORCH_CHECK(lease.result == FusionMatmulExecutionResult::kExecuted &&
          lease.algorithm != nullptr,
      "direct ROCm matmul failed to reserve a supported algorithm");

  const torch::Tensor batched_left =
      PrepareBatchedMatmulOperand(left_plan);
  const torch::Tensor batched_right =
      PrepareBatchedMatmulOperand(right_plan);
  TORCH_CHECK(SupportsRocmInferenceMatmulStructure(
          batched_left, batched_right, operation_bias_ptr),
      "direct ROCm batched matmul preparation produced an invalid layout");
  TORCH_CHECK(BuildRocmInferenceMatmulKey(
          batched_left, batched_right, epilogue) == key,
      "direct ROCm batched matmul preparation changed its planned geometry");
  torch::Tensor batched_result;
  TORCH_CHECK(lease.context->ExecutePlanLocked(lease.stream, key,
          *lease.algorithm, *lease.descriptors, batched_result, batched_left,
          batched_right, operation_bias_ptr, 0.0f) ==
          FusionMatmulExecutionResult::kExecuted,
      "direct ROCm matmul stream changed while its context was leased");
  std::vector<int64_t> output_shape = output_batch_shape;
  output_shape.push_back(operation_left.size(-2));
  output_shape.push_back(operation_right.size(-1));
  output = batched_result.view(output_shape);
  return true;
}

bool TryExecuteRocmInferenceLinear(torch::Tensor& output,
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor* bias) {
  if (!CanUseDirectRocmInferenceMatmul() ||
      !SupportsRocmInferenceTensor(input, 2) ||
      !SupportsRocmInferenceTensor(weight, 2) ||
      weight.dim() != 2 ||
      input.device() != weight.device() ||
      input.size(-1) != weight.size(1) ||
      (bias != nullptr && input.dim() > 2 && !input.is_contiguous()) ||
      (bias != nullptr &&
          (!SupportsRocmInferenceTensor(*bias, 1) ||
              bias->device() != input.device() ||
              bias->dim() != 1 ||
              bias->size(0) != weight.size(0) ||
              !bias->is_contiguous()))) {
    return false;
  }
  c10::DeviceGuard device_guard(input.device());
  torch::Tensor operation_input = input;
  torch::Tensor operation_weight = weight;
  torch::Tensor operation_bias;
  const torch::Tensor* operation_bias_ptr = bias;
  const bool autocast_enabled =
      at::autocast::is_autocast_enabled(at::DeviceType::CUDA);
  if (autocast_enabled) {
    const torch::ScalarType data_type =
        at::autocast::get_autocast_dtype(at::DeviceType::CUDA);
    if (data_type != torch::kFloat16 &&
        data_type != torch::kBFloat16) {
      return false;
    }
    operation_input = CastRocmAutocastTensor(input, data_type);
    // Cast the parameter before creating its transpose so the transpose stays
    // a metadata-only view.
    operation_weight = CastRocmAutocastTensor(weight, data_type);
    if (bias != nullptr) {
      operation_bias = CastRocmAutocastTensor(*bias, data_type);
      operation_bias_ptr = &operation_bias;
    }
  }

  torch::Tensor flattened_input;
  std::vector<int64_t> output_shape;
  if (!TryPrepareLinearInput(operation_input, operation_weight.size(0),
          &flattened_input, &output_shape)) {
    return false;
  }
  const torch::Tensor transposed_weight =
      operation_weight.transpose(0, 1);
  torch::Tensor flattened_result;
  if (!TryExecutePreparedRocmInferenceMatmul(flattened_result,
          flattened_input, transposed_weight, operation_bias_ptr)) {
    return false;
  }
  output = flattened_result.view(output_shape);
  return true;
}

void ExecuteFusionMatmulImpl(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    FusionMatmulEpilogue epilogue, const torch::Tensor* bias,
    float beta) {
  TORCH_CHECK(plan != nullptr, "Fusion matmul plan must not be null");
  TORCH_CHECK(beta == 0.0f ||
          (beta == 1.0f && epilogue == FusionMatmulEpilogue::kNone),
      "Fusion matmul beta must be zero, or one without an epilogue");
  const FusionMatmulKey key =
      BuildFusionMatmulKey(output, left, right, epilogue, bias);
  TORCH_CHECK(workspace.TryExecute(
          key, output, left, right, bias, beta) ==
          FusionMatmulExecutionResult::kExecuted,
      "hipBLASLt has no direct Fusion GEMM algorithm for the requested geometry");
}

bool ExecuteFusionMatmul(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right) {
  ExecuteFusionMatmulImpl(plan, workspace, output, left, right,
      FusionMatmulEpilogue::kNone, nullptr, 0.0f);
  return true;
}

bool ExecuteFusionMatmulBias(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right,
    const torch::Tensor& bias) {
  ExecuteFusionMatmulImpl(plan, workspace, output, left, right,
      FusionMatmulEpilogue::kBias, &bias, 0.0f);
  return true;
}

bool ExecuteFusionMatmulAccumulate(
    const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace, torch::Tensor& output,
    const torch::Tensor& left, const torch::Tensor& right) {
  ExecuteFusionMatmulImpl(plan, workspace, output, left, right,
      FusionMatmulEpilogue::kNone, nullptr, 1.0f);
  return true;
}

bool ExecuteLinearBiasSilu(const std::shared_ptr<FusionMatmulPlan>& plan,
    FusionMatmulWorkspace& workspace,
    torch::Tensor& output, const torch::Tensor& input,
    const torch::Tensor& weight, const torch::Tensor& bias) {
#if defined(DJL_USE_HIPBLASLT_SWISH_BIAS)
  if (output.scalar_type() != torch::kFloat16) {
    return false;
  }
  ExecuteFusionMatmulImpl(plan, workspace, output, input, weight,
      FusionMatmulEpilogue::kSwishBias, &bias, 0.0f);
  return true;
#else
  return false;
#endif
}

#else

class FusionMatmulPlan {};
class FusionMatmulWorkspace {};

void PinCurrentRocmMatmulStreamContext(c10::Device) {}
void UnpinCurrentRocmMatmulStreamContext(c10::Device) {}

bool IsFusionMatmulSupported() {
  return false;
}

bool IsLinearBiasSiluSupported() {
  return false;
}

std::size_t GetFusionMatmulWorkspaceUpperBoundBytes() {
  return 0;
}

std::shared_ptr<FusionMatmulPlan> CreateFusionMatmulPlan() {
  return std::make_shared<FusionMatmulPlan>();
}

std::shared_ptr<FusionMatmulWorkspace> CreateFusionMatmulWorkspace() {
  return std::make_shared<FusionMatmulWorkspace>();
}

bool TryExecuteRocmInferenceMatmul(torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor*) {
  return false;
}

bool TryExecuteRocmInferenceLinear(torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor*) {
  return false;
}

bool ExecuteFusionMatmul(const std::shared_ptr<FusionMatmulPlan>&,
    FusionMatmulWorkspace&, torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&) {
  return false;
}

bool ExecuteFusionMatmulBias(const std::shared_ptr<FusionMatmulPlan>&,
    FusionMatmulWorkspace&, torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&) {
  return false;
}

bool ExecuteFusionMatmulAccumulate(
    const std::shared_ptr<FusionMatmulPlan>&,
    FusionMatmulWorkspace&, torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&) {
  return false;
}

bool ExecuteLinearBiasSilu(const std::shared_ptr<FusionMatmulPlan>&,
    FusionMatmulWorkspace&,
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&) {
  return false;
}

#endif

template <typename output_t>
void LaunchOutputPackTyped(const OutputPackArguments& arguments,
    int32_t source_count, uint32_t block_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL((output_pack_kernel<output_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream,
      arguments);
}

template <typename scalar_t>
__global__ void weighted_compact_reduce_kernel(const scalar_t* rows, const scalar_t* weights,
    const int64_t* indices, scalar_t* output, int64_t count, int64_t width, int64_t capacity) {
  const int64_t action = blockIdx.x;
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (indices[mid] < action * capacity) lo = mid + 1;
    else hi = mid;
  }
  for (int64_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
    scalar_t sum = static_cast<scalar_t>(0.0f);
    for (int64_t item = lo; item < count && indices[item] < (action + 1) * capacity; ++item) {
      const scalar_t product = static_cast<scalar_t>(static_cast<float>(rows[item * width + feature]) *
          static_cast<float>(weights[indices[item]]));
      sum = static_cast<scalar_t>(static_cast<float>(sum) + static_cast<float>(product));
    }
    output[action * width + feature] = sum;
  }
}

void LaunchWeightedCompactReduce(const torch::Tensor& rows, const torch::Tensor& weights,
    const torch::Tensor& indices, torch::Tensor& output, int64_t capacity) {
  if (output.numel() == 0) return;
  const auto stream = backend::GetCurrentStream(output.get_device());
#define LAUNCH_WEIGHTED_COMPACT(T) \
  DJL_FUSION_LAUNCH_KERNEL((weighted_compact_reduce_kernel<T>), dim3(output.size(0)), dim3(128), 0, stream, \
      rows.data_ptr<T>(), weights.data_ptr<T>(), indices.data_ptr<int64_t>(), output.data_ptr<T>(), \
      rows.size(0), rows.size(1), capacity)
  if (rows.scalar_type() == torch::kFloat32) { LAUNCH_WEIGHTED_COMPACT(float); }
  else if (rows.scalar_type() == torch::kBFloat16) { LAUNCH_WEIGHTED_COMPACT(c10::BFloat16); }
  else if (rows.scalar_type() == torch::kFloat16) { LAUNCH_WEIGHTED_COMPACT(c10::Half); }
  else TORCH_CHECK(false, "unsupported weighted compact reduction dtype");
#undef LAUNCH_WEIGHTED_COMPACT
  CheckKernelLaunch("weighted compact reduction");
}

void LaunchOutputPack(const OutputPackSource* sources, int32_t source_count,
    torch::Tensor& output, int64_t row_count, int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  OutputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion output-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      LaunchOutputPackTyped<c10::Half>(
          arguments, source_count, block_count, stream);
      break;
    case torch::kBFloat16:
      LaunchOutputPackTyped<c10::BFloat16>(
          arguments, source_count, block_count, stream);
      break;
    case torch::kFloat32:
      LaunchOutputPackTyped<float>(
          arguments, source_count, block_count, stream);
      break;
    default:
      TORCH_CHECK(false,
          "fusion output pack requires FLOAT16, BFLOAT16, or FLOAT32 output");
  }
  CheckKernelLaunch("fusion output pack");
}

template <typename scalar_t>
void LaunchSegmentedOutputPackTyped(
    const SegmentedOutputPackArguments& arguments, int32_t source_count,
    uint32_t block_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL((segmented_output_pack_kernel<scalar_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream,
      arguments);
}

void LaunchSegmentedOutputPack(const SegmentedOutputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  SegmentedOutputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    arguments.source_prefix_counts[source_index] = source.source_prefix_count;
    arguments.source_token_counts[source_index] = source.source_token_count;
    arguments.token_offsets[source_index] = source.token_offset;
    arguments.token_counts[source_index] = source.token_count;
    arguments.hidden_widths[source_index] = source.hidden_width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    const int64_t width = CheckedMultiply(
        CheckedMultiply(source.source_prefix_count, source.token_count,
            "fusion segmented output-pack source tokens"),
        source.hidden_width, "fusion segmented output-pack source width");
    maximum_width = std::max(maximum_width, width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion segmented output-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      LaunchSegmentedOutputPackTyped<c10::Half>(
          arguments, source_count, block_count, stream);
      break;
    case torch::kBFloat16:
      LaunchSegmentedOutputPackTyped<c10::BFloat16>(
          arguments, source_count, block_count, stream);
      break;
    case torch::kFloat32:
      LaunchSegmentedOutputPackTyped<float>(
          arguments, source_count, block_count, stream);
      break;
    default:
      TORCH_CHECK(false,
          "segmented output pack supports FLOAT16, BFLOAT16, and FLOAT32");
  }
  CheckKernelLaunch("fusion segmented output pack");
}

template <typename baseline_t, typename selected_t>
void LaunchBinaryBranchBlendTyped(const BinaryBranchBlendArguments& arguments,
    uint32_t block_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL((binary_branch_blend_kernel<baseline_t, selected_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchBinaryBranchBlend(const torch::Tensor& baseline_context,
    const torch::Tensor& selected_context,
    const torch::Tensor& selected_logit,
    const torch::Tensor& baseline_presence,
    const torch::Tensor& selected_presence, torch::Tensor& output,
    int64_t row_count, int64_t width) {
  if (row_count == 0) {
    return;
  }
  BinaryBranchBlendArguments arguments{};
  arguments.baseline_context = baseline_context.data_ptr();
  arguments.selected_context = selected_context.data_ptr();
  arguments.selected_logit = selected_logit.data_ptr();
  arguments.baseline_presence = baseline_presence.data_ptr();
  arguments.selected_presence = selected_presence.data_ptr();
  arguments.output = output.data_ptr<float>();
  arguments.row_count = row_count;
  arguments.width = width;
  arguments.selected_logit_data_type =
      static_cast<int32_t>(selected_logit.scalar_type());
  arguments.baseline_presence_data_type =
      static_cast<int32_t>(baseline_presence.scalar_type());
  arguments.selected_presence_data_type =
      static_cast<int32_t>(selected_presence.scalar_type());
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(row_count, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
#define LAUNCH_BINARY_BRANCH_BLEND(baseline_type, selected_type) \
  LaunchBinaryBranchBlendTyped<baseline_type, selected_type>( \
      arguments, block_count, stream)
  switch (baseline_context.scalar_type()) {
    case torch::kFloat16:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(c10::Half, float);
      }
      break;
    case torch::kBFloat16:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(c10::BFloat16, float);
      }
      break;
    case torch::kFloat32:
      if (selected_context.scalar_type() == torch::kFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(float, c10::Half);
      } else if (selected_context.scalar_type() == torch::kBFloat16) {
        LAUNCH_BINARY_BRANCH_BLEND(float, c10::BFloat16);
      } else {
        LAUNCH_BINARY_BRANCH_BLEND(float, float);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported binary branch baseline data type");
  }
#undef LAUNCH_BINARY_BRANCH_BLEND
  CheckKernelLaunch("fusion binary branch blend");
}

void LaunchAffineInputPack(const AffineInputPackSource* sources,
    int32_t source_count, torch::Tensor& output, int64_t row_count,
    int64_t output_width) {
  if (row_count == 0) {
    return;
  }
  AffineInputPackArguments arguments{};
  arguments.output = output.data_ptr();
  arguments.row_count = row_count;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      row_count, maximum_width, "fusion affine input-pack launch element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      DJL_FUSION_LAUNCH_KERNEL(affine_input_pack_kernel<c10::Half>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream, arguments);
      break;
    case torch::kBFloat16:
      DJL_FUSION_LAUNCH_KERNEL(affine_input_pack_kernel<c10::BFloat16>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream, arguments);
      break;
    case torch::kFloat32:
      DJL_FUSION_LAUNCH_KERNEL(affine_input_pack_kernel<float>,
          dim3(block_count, source_count), dim3(kThreadsPerBlock), 0,
          stream, arguments);
      break;
    default:
      TORCH_CHECK(false, "unsupported affine input-pack data type");
  }
  CheckKernelLaunch("fusion affine input pack");
}

void LaunchAffineFinalize(const AffineSumSource* sources, int32_t source_count,
    const void* bias, torch::Tensor& output, int64_t batch_count,
    const int64_t* output_prefix, int32_t output_prefix_rank,
    int64_t output_width, AffineActivation activation) {
  if (batch_count == 0) {
    return;
  }
  AffineFinalizeArguments arguments{};
  arguments.bias = bias;
  arguments.output = output.data_ptr();
  arguments.batch_count = batch_count;
  arguments.output_width = output_width;
  arguments.source_count = source_count;
  arguments.output_prefix_rank = output_prefix_rank;
  arguments.activation = static_cast<int32_t>(activation);
  arguments.output_prefix_rows = 1;
  for (int32_t axis = 0; axis < output_prefix_rank; ++axis) {
    arguments.output_prefix[axis] = output_prefix[axis];
    arguments.output_prefix_rows = CheckedMultiply(arguments.output_prefix_rows,
        output_prefix[axis], "fusion affine output prefix size");
  }
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    arguments.sources[source_index] = sources[source_index].data;
    arguments.source_rows_per_batch[source_index] =
        sources[source_index].rows_per_batch;
    for (int32_t axis = 0; axis < output_prefix_rank; ++axis) {
      arguments.source_strides[source_index][axis] =
          sources[source_index].output_strides[axis];
    }
  }

  const int64_t output_rows = CheckedMultiply(batch_count,
      arguments.output_prefix_rows, "fusion affine output row count");
  const int64_t element_count = CheckedMultiply(
      output_rows, output_width, "fusion affine launch element count");
  const int64_t required_blocks = RequiredBlockCount(element_count);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
  switch (output.scalar_type()) {
    case torch::kFloat16:
      DJL_FUSION_LAUNCH_KERNEL(affine_finalize_kernel<c10::Half>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream, arguments);
      break;
    case torch::kBFloat16:
      DJL_FUSION_LAUNCH_KERNEL(affine_finalize_kernel<c10::BFloat16>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream, arguments);
      break;
    case torch::kFloat32:
      DJL_FUSION_LAUNCH_KERNEL(affine_finalize_kernel<float>, dim3(block_count),
          dim3(kThreadsPerBlock), 0, stream, arguments);
      break;
    default:
      TORCH_CHECK(false, "unsupported affine finalize data type");
  }
  CheckKernelLaunch("fusion affine finalize");
}

void LaunchProjectedResidualMlpPrepare(torch::Tensor& combined,
    torch::Tensor& activated, torch::Tensor& residual,
    int64_t row_count, int64_t output_width, int64_t hidden_width) {
  TORCH_CHECK(combined.scalar_type() == activated.scalar_type() &&
          combined.scalar_type() == residual.scalar_type(),
      "projected-residual MLP tensors must use one data type");
  TORCH_CHECK(output_width > 0 && hidden_width > 0 &&
          combined.size(-1) == output_width + hidden_width,
      "projected-residual MLP widths are invalid");
  const int64_t combined_width = combined.size(-1);
  const int64_t element_count = CheckedMultiply(
      row_count, combined_width, "projected-residual MLP element count");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = backend::GetCurrentStream(combined.get_device());
#define LAUNCH_PROJECTED_RESIDUAL_MLP_PREPARE(scalar_type) \
  DJL_FUSION_LAUNCH_KERNEL( \
      (projected_residual_mlp_prepare_kernel<scalar_type>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, \
      combined.data_ptr<scalar_type>(), \
      activated.data_ptr<scalar_type>(), residual.data_ptr<scalar_type>(), \
      row_count, output_width, hidden_width)
  switch (combined.scalar_type()) {
    case torch::kFloat16:
      LAUNCH_PROJECTED_RESIDUAL_MLP_PREPARE(c10::Half);
      break;
    case torch::kBFloat16:
      LAUNCH_PROJECTED_RESIDUAL_MLP_PREPARE(c10::BFloat16);
      break;
    case torch::kFloat32:
      LAUNCH_PROJECTED_RESIDUAL_MLP_PREPARE(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported projected-residual MLP data type");
  }
#undef LAUNCH_PROJECTED_RESIDUAL_MLP_PREPARE
  CheckKernelLaunch("fusion projected-residual MLP prepare");
}

template <typename output_t, typename index_t>
void LaunchIndexedAffineInputPackTyped(const IndexedAffineInputPackArguments& arguments,
    uint32_t block_count, int32_t source_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL((indexed_affine_input_pack_kernel<output_t, index_t>),
      dim3(block_count, source_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchIndexedAffineInputPack(const IndexedAffineSource* sources,
    int32_t source_count, const void* indices, torch::ScalarType index_data_type,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t output_width) {
  if (active_rows == 0) {
    return;
  }
  IndexedAffineInputPackArguments arguments{};
  arguments.indices = indices;
  arguments.output = output.data_ptr();
  arguments.active_rows = active_rows;
  arguments.destination_rows = destination_rows;
  arguments.output_width = output_width;
  int64_t maximum_width = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const auto& source = sources[source_index];
    arguments.sources[source_index] = source.data;
    arguments.widths[source_index] = source.width;
    arguments.destination_offsets[source_index] = source.destination_offset;
    arguments.index_divisors[source_index] = source.index_divisor;
    arguments.source_rows[source_index] = source.row_count;
    arguments.data_types[source_index] = static_cast<int32_t>(source.data_type);
    maximum_width = std::max(maximum_width, source.width);
  }

  const int64_t maximum_elements = CheckedMultiply(
      active_rows, maximum_width, "fusion indexed-affine input-pack element count");
  const int64_t required_blocks = RequiredBlockCount(maximum_elements);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
  const auto stream = backend::GetCurrentStream(output.get_device());
#define LAUNCH_INDEXED_PACK(output_type, index_type) \
  LaunchIndexedAffineInputPackTyped<output_type, index_type>( \
      arguments, block_count, source_count, stream)
  switch (output.scalar_type()) {
    case torch::kFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(c10::Half, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(c10::Half, int64_t);
      }
      break;
    case torch::kBFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(c10::BFloat16, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(c10::BFloat16, int64_t);
      }
      break;
    case torch::kFloat32:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_PACK(float, int32_t);
      } else {
        LAUNCH_INDEXED_PACK(float, int64_t);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported indexed-affine input-pack data type");
  }
#undef LAUNCH_INDEXED_PACK
  CheckKernelLaunch("fusion indexed-affine input pack");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedAffineFinalizeTyped(const IndexedAffineFinalizeArguments& arguments,
    uint32_t block_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL((indexed_affine_finalize_kernel<scalar_t, index_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, arguments);
}

void LaunchIndexedAffineFinalize(const void* indices,
    torch::ScalarType index_data_type, const torch::Tensor& hidden,
    const void* hidden_bias, const void* output_weight, const void* output_bias,
    torch::Tensor& output, int64_t active_rows, int64_t destination_rows,
    int64_t hidden_width, int64_t output_width, AffineActivation activation) {
  const auto stream = backend::GetCurrentStream(output.get_device());
  const int64_t output_elements = CheckedMultiply(destination_rows, output_width,
      "fusion indexed-affine output element count");
  const int64_t output_bytes = CheckedMultiply(output_elements,
      static_cast<int64_t>(output.element_size()),
      "fusion indexed-affine output byte count");
  if (output_bytes != 0) {
    backend::MemsetAsync(output.data_ptr(), 0,
        static_cast<std::size_t>(output_bytes), stream,
        "fusion indexed-affine output clear");
  }
  if (active_rows == 0) {
    return;
  }

  IndexedAffineFinalizeArguments arguments{};
  arguments.indices = indices;
  arguments.hidden = hidden.data_ptr();
  arguments.hidden_bias = hidden_bias;
  arguments.output_weight = output_weight;
  arguments.output_bias = output_bias;
  arguments.output = output.data_ptr();
  arguments.active_rows = active_rows;
  arguments.destination_rows = destination_rows;
  arguments.hidden_width = hidden_width;
  arguments.output_width = output_width;
  arguments.activation = static_cast<int32_t>(activation);
  const int64_t active_elements = CheckedMultiply(active_rows, output_width,
      "fusion indexed-affine active output element count");
  constexpr int64_t maximum_wavefront_size = 64;
  const int64_t minimum_waves_per_block = kThreadsPerBlock / maximum_wavefront_size;
  const int64_t required_blocks = active_elements / minimum_waves_per_block +
      (active_elements % minimum_waves_per_block == 0 ? 0 : 1);
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>(required_blocks, 65535));
#define LAUNCH_INDEXED_FINALIZE(scalar_type, index_type) \
  LaunchIndexedAffineFinalizeTyped<scalar_type, index_type>( \
      arguments, block_count, stream)
  switch (output.scalar_type()) {
    case torch::kFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(c10::Half, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(c10::Half, int64_t);
      }
      break;
    case torch::kBFloat16:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(c10::BFloat16, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(c10::BFloat16, int64_t);
      }
      break;
    case torch::kFloat32:
      if (index_data_type == torch::kInt32) {
        LAUNCH_INDEXED_FINALIZE(float, int32_t);
      } else {
        LAUNCH_INDEXED_FINALIZE(float, int64_t);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported indexed-affine finalize data type");
  }
#undef LAUNCH_INDEXED_FINALIZE
  CheckKernelLaunch("fusion indexed-affine finalize");
}

template <typename scalar_t, typename norm_t>
void LaunchTransformerCopyAndLayerNormTyped(const torch::Tensor& input, torch::Tensor& state, torch::Tensor& normalized,
    const torch::Tensor& weight, const torch::Tensor& bias, int64_t active_rows, int64_t hidden_width, float epsilon,
    bool copy_input, backend::Stream stream) {
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>((active_rows - 1) / maximum_waves_per_block + 1, 65535));
#define LAUNCH_TRANSFORMER_COPY_LAYER_NORM_WIDTH(width, cache_values)                                                \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (transformer_copy_layer_norm_kernel<scalar_t, norm_t, width, cache_values>), dim3(block_count),                \
      dim3(kThreadsPerBlock), 0, stream, input.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),                     \
      normalized.data_ptr<scalar_t>(), weight.data_ptr<norm_t>(), bias.data_ptr<norm_t>(), active_rows, epsilon,     \
      copy_input, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_TRANSFORMER_COPY_LAYER_NORM_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_TRANSFORMER_COPY_LAYER_NORM_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, input.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_TRANSFORMER_COPY_LAYER_NORM_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL((transformer_dynamic_layer_norm_kernel<scalar_t, norm_t, false>), dim3(block_count),
            dim3(kThreadsPerBlock), 0, stream, input.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(), nullptr, weight.data_ptr<norm_t>(), bias.data_ptr<norm_t>(), active_rows,
            hidden_width, epsilon, copy_input, false);
      }
    });
  }
#undef LAUNCH_TRANSFORMER_COPY_LAYER_NORM_WIDTH
}

void LaunchTransformerCopyAndLayerNorm(const torch::Tensor& input,
    torch::Tensor& state, torch::Tensor& normalized,
    const torch::Tensor& weight, const torch::Tensor& bias,
    int64_t batch_count, int64_t token_count, int64_t hidden_width,
    float epsilon, bool copy_input) {
  TORCH_CHECK(hidden_width > 0, "transformer encoder hidden width must be positive");
  const int64_t active_rows = CheckedMultiply(
      batch_count, token_count, "transformer LayerNorm row count");
  if (active_rows == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(state.get_device());
#define LAUNCH_TRANSFORMER_COPY_LAYER_NORM(scalar_type, norm_type) \
  LaunchTransformerCopyAndLayerNormTyped<scalar_type, norm_type>(  \
      input, state, normalized, weight, bias, active_rows, hidden_width, epsilon, copy_input, stream)
  if (input.scalar_type() == torch::kFloat16) {
    if (weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::Half, float);
    } else {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::Half, c10::Half);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_TRANSFORMER_COPY_LAYER_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_COPY_LAYER_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_COPY_LAYER_NORM
  CheckKernelLaunch("fusion transformer copy LayerNorm");
}

template <typename scalar_t>
void LaunchTransformerAttentionTyped(const torch::Tensor& query_key_value, torch::Tensor& context, int64_t batch_count,
    int64_t token_count, int64_t attention_heads, int64_t attention_width, backend::Stream stream) {
  if (token_count <= kTransformerMaximumTokens && attention_heads == kTransformerAttentionHeads &&
      attention_width == kTransformerAttentionWidth && batch_count <= 65535) {
    DJL_FUSION_LAUNCH_KERNEL((transformer_attention_kernel<scalar_t>), dim3(batch_count), dim3(kThreadsPerBlock), 0,
        stream, query_key_value.data_ptr<scalar_t>(), context.data_ptr<scalar_t>(), batch_count, token_count);
  } else {
    const int64_t attention_rows =
        CheckedMultiply(CheckedMultiply(batch_count, attention_heads, "transformer attention row count"), token_count,
            "transformer attention row count");
    const uint32_t block_count = static_cast<uint32_t>(std::min<int64_t>(attention_rows, 65535));
    DJL_FUSION_LAUNCH_KERNEL((transformer_dynamic_attention_kernel<scalar_t>), dim3(block_count),
        dim3(kThreadsPerBlock), token_count * sizeof(float), stream, query_key_value.data_ptr<scalar_t>(),
        context.data_ptr<scalar_t>(), attention_rows, token_count, attention_heads, attention_width,
        attention_width / attention_heads);
  }
}

void LaunchTransformerAttention(const torch::Tensor& query_key_value,
    torch::Tensor& context, int64_t batch_count, int64_t token_count,
    int64_t attention_heads, int64_t attention_width, int64_t hidden_width) {
  // Keep dynamic shared memory within the default per-block budget on both backends.
  constexpr int64_t shared_memory_budget = kFusionAttentionSharedMemoryBytes;
  TORCH_CHECK(token_count > 0 && token_count <= shared_memory_budget / sizeof(float),
      "transformer attention exceeds the 48 KiB shared-memory budget");
  TORCH_CHECK(hidden_width > 0 && attention_heads > 0 && attention_width > 0 && attention_width % attention_heads == 0,
      "transformer encoder requires positive widths and attention width divisible by heads");
  if (batch_count == 0) {
    return;
  }
  const auto stream =
      backend::GetCurrentStream(query_key_value.get_device());
#define LAUNCH_TRANSFORMER_ATTENTION(scalar_type) \
  LaunchTransformerAttentionTyped<scalar_type>(   \
      query_key_value, context, batch_count, token_count, attention_heads, attention_width, stream)
  if (query_key_value.scalar_type() == torch::kFloat16) {
    LAUNCH_TRANSFORMER_ATTENTION(c10::Half);
  } else if (query_key_value.scalar_type() == torch::kBFloat16) {
    LAUNCH_TRANSFORMER_ATTENTION(c10::BFloat16);
  } else if (query_key_value.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_ATTENTION(float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_ATTENTION
  CheckKernelLaunch("fusion transformer attention");
}

template <typename scalar_t>
void LaunchIndexedRelationAttentionBiasTyped(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_ids,
    const torch::Tensor& pair_bias, torch::Tensor& attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale,
    uint32_t block_count, backend::Stream stream) {
  DJL_FUSION_LAUNCH_KERNEL(
      (indexed_relation_attention_bias_kernel<scalar_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      relation_logits.data_ptr<scalar_t>(), relation_ids.data_ptr<int16_t>(),
      pair_bias.data_ptr<scalar_t>(), attention_bias.data_ptr<scalar_t>(),
      batch_count, attention_heads, token_count, relation_count,
      padded_token_count, scale);
}

void LaunchIndexedRelationAttentionBias(
    const torch::Tensor& relation_logits,
    const torch::Tensor& relation_ids,
    const torch::Tensor& pair_bias, torch::Tensor& attention_bias,
    int64_t batch_count, int64_t attention_heads, int64_t token_count,
    int64_t relation_count, int64_t padded_token_count, float scale) {
  TORCH_CHECK(relation_ids.scalar_type() == torch::kInt16,
      "indexed-relation IDs must use INT16");
  TORCH_CHECK(relation_logits.scalar_type() == pair_bias.scalar_type() &&
          relation_logits.scalar_type() == attention_bias.scalar_type(),
      "indexed-relation attention tensors must use one data type");
  const int64_t element_count = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(batch_count, attention_heads,
              "indexed-relation attention bias elements"),
          token_count, "indexed-relation attention bias elements"),
      padded_token_count, "indexed-relation attention bias elements");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = backend::GetCurrentStream(attention_bias.get_device());
#define LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(scalar_type) \
  LaunchIndexedRelationAttentionBiasTyped<scalar_type>(relation_logits, \
      relation_ids, pair_bias, attention_bias, batch_count, attention_heads, \
      token_count, relation_count, padded_token_count, scale, block_count, \
      stream)
  if (attention_bias.scalar_type() == torch::kFloat16) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(c10::Half);
  } else if (attention_bias.scalar_type() == torch::kBFloat16) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(c10::BFloat16);
  } else if (attention_bias.scalar_type() == torch::kFloat32) {
    LAUNCH_INDEXED_RELATION_ATTENTION_BIAS(float);
  } else {
    TORCH_CHECK(false, "unsupported indexed-relation attention data type");
  }
#undef LAUNCH_INDEXED_RELATION_ATTENTION_BIAS
  CheckKernelLaunch("fusion indexed-relation attention bias");
}

void LaunchTransformerBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t active_rows, int64_t width,
    bool round_bias_before_activation) {
  const int64_t element_count = CheckedMultiply(
      active_rows, width, "transformer SiLU element count");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = backend::GetCurrentStream(values.get_device());
#define LAUNCH_TRANSFORMER_SILU(scalar_type) \
  DJL_FUSION_LAUNCH_KERNEL((transformer_bias_silu_kernel<scalar_type>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, \
      values.data_ptr<scalar_type>(), bias.data_ptr<scalar_type>(), \
      element_count, width, round_bias_before_activation)
  if (values.scalar_type() == torch::kFloat16) {
    LAUNCH_TRANSFORMER_SILU(c10::Half);
  } else if (values.scalar_type() == torch::kBFloat16) {
    LAUNCH_TRANSFORMER_SILU(c10::BFloat16);
  } else if (values.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_SILU(float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_SILU
  CheckKernelLaunch("fusion transformer bias SiLU");
}

template <typename scalar_t, typename norm_t>
void LaunchTransformerResidualLayerNormTyped(torch::Tensor& state, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, int64_t hidden_width, float epsilon,
    bool round_update_before_residual, backend::Stream stream) {
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>(std::min<int64_t>((active_rows - 1) / maximum_waves_per_block + 1, 65535));
#define LAUNCH_TRANSFORMER_RESIDUAL_LAYER_NORM_WIDTH(width, cache_values)                                            \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (transformer_residual_layer_norm_kernel<scalar_t, norm_t, width, cache_values>), dim3(block_count),            \
      dim3(kThreadsPerBlock), 0, stream, state.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),                    \
      update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(), norm_bias.data_ptr<norm_t>(),                \
      normalized.data_ptr<scalar_t>(), active_rows, epsilon, round_update_before_residual, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_TRANSFORMER_RESIDUAL_LAYER_NORM_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_TRANSFORMER_RESIDUAL_LAYER_NORM_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, state.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_TRANSFORMER_RESIDUAL_LAYER_NORM_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL((transformer_dynamic_layer_norm_kernel<scalar_t, norm_t, true>), dim3(block_count),
            dim3(kThreadsPerBlock), 0, stream, update.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(),
            norm_bias.data_ptr<norm_t>(), active_rows, hidden_width, epsilon, false, round_update_before_residual);
      }
    });
  }
#undef LAUNCH_TRANSFORMER_RESIDUAL_LAYER_NORM_WIDTH
}

void LaunchTransformerResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& normalized, int64_t active_rows, int64_t hidden_width,
    float epsilon, bool round_update_before_residual) {
  TORCH_CHECK(hidden_width > 0, "transformer encoder hidden width must be positive");
  if (active_rows == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(state.get_device());
#define LAUNCH_TRANSFORMER_RESIDUAL_NORM(scalar_type, norm_type)                                                      \
  LaunchTransformerResidualLayerNormTyped<scalar_type, norm_type>(state, update, update_bias, norm_weight, norm_bias, \
      normalized, active_rows, hidden_width, epsilon, round_update_before_residual, stream)
  if (state.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::Half, float);
    } else {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::Half, c10::Half);
    }
  } else if (state.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_TRANSFORMER_RESIDUAL_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (state.scalar_type() == torch::kFloat32) {
    LAUNCH_TRANSFORMER_RESIDUAL_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported transformer encoder data type");
  }
#undef LAUNCH_TRANSFORMER_RESIDUAL_NORM
  CheckKernelLaunch("fusion transformer residual LayerNorm");
}

template <typename memory_t, typename scalar_t, typename mask_t>
void LaunchSingleQueryReadoutSeedInputTyped(const torch::Tensor& memory, const torch::Tensor& query_source,
    const torch::Tensor& valid_mask, torch::Tensor& seed_input, int64_t batch_count, int64_t token_count,
    int64_t hidden_width, int64_t query_token_count, int64_t query_index, backend::Stream stream) {
#define LAUNCH_SINGLE_QUERY_SEED_WIDTH(specialized)                                                                   \
  DJL_FUSION_LAUNCH_KERNEL((single_query_seed_input_kernel<memory_t, scalar_t, mask_t, specialized>),                 \
      dim3(batch_count), dim3(kThreadsPerBlock), 0, stream, memory.data_ptr<memory_t>(),                              \
      query_source.data_ptr<memory_t>(), valid_mask.data_ptr<mask_t>(), seed_input.data_ptr<scalar_t>(), batch_count, \
      token_count, hidden_width, query_token_count, query_index)
  if (hidden_width == kSingleQuerySpecializedHiddenWidth) {
    LAUNCH_SINGLE_QUERY_SEED_WIDTH(true);
  } else {
    LAUNCH_SINGLE_QUERY_SEED_WIDTH(false);
  }
#undef LAUNCH_SINGLE_QUERY_SEED_WIDTH
}

void LaunchSingleQueryReadoutSeedInput(const torch::Tensor& memory,
    const torch::Tensor& query_source, const torch::Tensor& valid_mask,
    torch::Tensor& seed_input, int64_t batch_count, int64_t token_count,
    int64_t hidden_width, int64_t query_token_count, int64_t query_index) {
  TORCH_CHECK(hidden_width > 0 && token_count > 0 && query_token_count > 0 && query_index >= 0 &&
                  query_index < query_token_count,
      "single-query readout seed metadata is unsupported");
  if (batch_count == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(memory.get_device());
#define LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, mask_type)                                           \
  LaunchSingleQueryReadoutSeedInputTyped<memory_type, compute_t, mask_type>(memory, query_source, valid_mask, \
      seed_input, batch_count, token_count, hidden_width, query_token_count, query_index, stream)
#define DISPATCH_SINGLE_QUERY_SEED_MASK(memory_type, compute_t) \
  switch (valid_mask.scalar_type()) { \
    case torch::kBool: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, bool); \
      break; \
    case torch::kUInt8: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, uint8_t); \
      break; \
    case torch::kFloat16: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_SINGLE_QUERY_SEED(memory_type, compute_t, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout mask data type"); \
  }
#define DISPATCH_SINGLE_QUERY_SEED_MEMORY(compute_type) \
  switch (memory.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(c10::Half, compute_type); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(c10::BFloat16, compute_type); \
      break; \
    case torch::kFloat32: \
      DISPATCH_SINGLE_QUERY_SEED_MASK(float, compute_type); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout memory data type"); \
  }
  if (seed_input.scalar_type() == torch::kFloat16) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(c10::Half);
  } else if (seed_input.scalar_type() == torch::kBFloat16) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(c10::BFloat16);
  } else if (seed_input.scalar_type() == torch::kFloat32) {
    DISPATCH_SINGLE_QUERY_SEED_MEMORY(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout compute data type");
  }
#undef DISPATCH_SINGLE_QUERY_SEED_MEMORY
#undef DISPATCH_SINGLE_QUERY_SEED_MASK
#undef LAUNCH_SINGLE_QUERY_SEED
  CheckKernelLaunch("fusion single-query readout seed input");
}

template <typename memory_t, typename scalar_t, typename mask_t>
void LaunchSingleQueryReadoutAttentionTyped(const torch::Tensor& seed_products, const torch::Tensor& seed_bias,
    const torch::Tensor& memory, const torch::Tensor& valid_mask, const torch::Tensor& query_weight,
    const torch::Tensor& query_bias, const torch::Tensor& key_value_weight, torch::Tensor& state,
    torch::Tensor& context, int64_t batch_count, int64_t readout_count, int64_t token_count, int64_t hidden_width,
    int64_t attention_heads, int64_t attention_width, backend::Stream stream) {
  const bool specialized = hidden_width == kSingleQuerySpecializedHiddenWidth &&
                           token_count <= kSingleQuerySpecializedTokens &&
                           attention_width / attention_heads <= kSingleQuerySpecializedHeadWidth;
  const std::size_t shared_bytes =
      specialized ? sizeof(float) * (2 * kSingleQuerySpecializedHiddenWidth + kSingleQuerySpecializedHeadWidth +
                                        kSingleQuerySpecializedTokens) +
                        sizeof(uint8_t) * kSingleQuerySpecializedTokens
                  : sizeof(float) * (2 * hidden_width + attention_width / attention_heads + 2 * token_count);
#define LAUNCH_SINGLE_QUERY_ATTENTION_SHAPE(specialized)                                                            \
  DJL_FUSION_LAUNCH_KERNEL((single_query_attention_kernel<memory_t, scalar_t, mask_t, specialized>),                \
      dim3(readout_count * attention_heads, batch_count), dim3(128), shared_bytes, stream,              \
      seed_products.data_ptr<scalar_t>(), seed_bias.data_ptr<scalar_t>(), memory.data_ptr<memory_t>(),              \
      valid_mask.data_ptr<mask_t>(), query_weight.data_ptr<scalar_t>(), query_bias.data_ptr<scalar_t>(),            \
      key_value_weight.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(), context.data_ptr<scalar_t>(), batch_count, \
      readout_count, token_count, hidden_width, attention_heads, attention_width)
  if (specialized) {
    LAUNCH_SINGLE_QUERY_ATTENTION_SHAPE(true);
  } else {
    LAUNCH_SINGLE_QUERY_ATTENTION_SHAPE(false);
  }
#undef LAUNCH_SINGLE_QUERY_ATTENTION_SHAPE
}

void LaunchSingleQueryReadoutAttention(const torch::Tensor& seed_products,
    const torch::Tensor& seed_bias, const torch::Tensor& memory,
    const torch::Tensor& valid_mask, const torch::Tensor& query_weight,
    const torch::Tensor& query_bias, const torch::Tensor& key_value_weight,
    torch::Tensor& state, torch::Tensor& context, int64_t batch_count,
    int64_t readout_count, int64_t token_count, int64_t hidden_width,
    int64_t attention_heads, int64_t attention_width) {
  TORCH_CHECK(hidden_width > 0 && token_count > 0 && attention_heads > 0 && attention_width > 0 &&
                  attention_width % attention_heads == 0,
      "single-query readout attention metadata is unsupported");
  TORCH_CHECK(readout_count > 0 && readout_count <= 65535 && attention_heads <= 65535 / readout_count,
      "single-query readout attention exceeds the grid-y limit");
  constexpr int64_t maximum_shared_values = kSingleQuerySharedMemoryBudgetBytes / sizeof(float);
  const int64_t head_width = attention_width / attention_heads;
  TORCH_CHECK(hidden_width <= maximum_shared_values && head_width <= maximum_shared_values &&
                  token_count <= maximum_shared_values &&
                  2 * hidden_width + head_width + 2 * token_count <= maximum_shared_values,
      "single-query readout attention exceeds the shared-memory budget");
  if (batch_count == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(memory.get_device());
#define LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, mask_type)                                               \
  LaunchSingleQueryReadoutAttentionTyped<memory_type, compute_t, mask_type>(seed_products, seed_bias, memory,          \
      valid_mask, query_weight, query_bias, key_value_weight, state, context, batch_count, readout_count, token_count, \
      hidden_width, attention_heads, attention_width, stream)
#define DISPATCH_SINGLE_QUERY_ATTENTION_MASK(memory_type, compute_t) \
  switch (valid_mask.scalar_type()) { \
    case torch::kBool: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, bool); \
      break; \
    case torch::kUInt8: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, uint8_t); \
      break; \
    case torch::kFloat16: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_SINGLE_QUERY_ATTENTION(memory_type, compute_t, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout mask data type"); \
  }
#define DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(compute_type) \
  switch (memory.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(c10::Half, compute_type); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(c10::BFloat16, compute_type); \
      break; \
    case torch::kFloat32: \
      DISPATCH_SINGLE_QUERY_ATTENTION_MASK(float, compute_type); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported single-query readout memory data type"); \
  }
  if (state.scalar_type() == torch::kFloat16) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(c10::Half);
  } else if (state.scalar_type() == torch::kBFloat16) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(c10::BFloat16);
  } else if (state.scalar_type() == torch::kFloat32) {
    DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout compute data type");
  }
#undef DISPATCH_SINGLE_QUERY_ATTENTION_MEMORY
#undef DISPATCH_SINGLE_QUERY_ATTENTION_MASK
#undef LAUNCH_SINGLE_QUERY_ATTENTION
  CheckKernelLaunch("fusion single-query readout attention");
}

template <typename scalar_t, typename norm_t>
void LaunchSingleQueryReadoutResidualLayerNormTyped(torch::Tensor& state, const torch::Tensor& update,
    const torch::Tensor* state_bias, const torch::Tensor& update_bias, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    int64_t hidden_width, int64_t output_batch_stride, float epsilon, bool batch_major_output, backend::Stream stream) {
  const int64_t active_rows = batch_count * readout_count;
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
#define LAUNCH_SINGLE_QUERY_RESIDUAL_NORM_WIDTH(specialized, cache_values)                                           \
  DJL_FUSION_LAUNCH_KERNEL((single_query_residual_layer_norm_kernel<scalar_t, norm_t, specialized, cache_values>),   \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, state.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(), \
      state_bias == nullptr ? nullptr : state_bias->data_ptr<scalar_t>(), update_bias.data_ptr<scalar_t>(),          \
      norm_weight.data_ptr<norm_t>(), norm_bias.data_ptr<norm_t>(), output.data_ptr<scalar_t>(), batch_count,        \
      readout_count, hidden_width, output_batch_stride, epsilon, batch_major_output)
  if (hidden_width == kSingleQuerySpecializedHiddenWidth) {
    LAUNCH_SINGLE_QUERY_RESIDUAL_NORM_WIDTH(true, kSingleQueryCachedValuesPerLane);
  } else {
    DispatchNormalizationCache(hidden_width, state.get_device(), [&](auto cache_width) {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM_WIDTH(false, decltype(cache_width)::value);
    });
  }
#undef LAUNCH_SINGLE_QUERY_RESIDUAL_NORM_WIDTH
}

void LaunchSingleQueryReadoutResidualLayerNorm(torch::Tensor& state,
    const torch::Tensor& update, const torch::Tensor* state_bias,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output,
    int64_t batch_count, int64_t readout_count, int64_t hidden_width,
    int64_t output_batch_stride, float epsilon, bool batch_major_output) {
  TORCH_CHECK(hidden_width > 0, "single-query readout residual LayerNorm requires a positive hidden width");
  if (batch_count == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(state.get_device());
#define LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(scalar_type, norm_type)                                                \
  LaunchSingleQueryReadoutResidualLayerNormTyped<scalar_type, norm_type>(state, update, state_bias, update_bias, \
      norm_weight, norm_bias, output, batch_count, readout_count, hidden_width, output_batch_stride, epsilon,    \
      batch_major_output, stream)
  if (state.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::Half, float);
    } else {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::Half, c10::Half);
    }
  } else if (state.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (state.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_RESIDUAL_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_RESIDUAL_NORM
  CheckKernelLaunch("fusion single-query readout residual LayerNorm");
}

template <typename scalar_t, typename norm_t>
void LaunchSingleQueryReadoutLayerNormTyped(const torch::Tensor& input, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    int64_t hidden_width, float epsilon, backend::Stream stream) {
  const int64_t active_rows = batch_count * readout_count;
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count = static_cast<uint32_t>(
      (active_rows + maximum_waves_per_block - 1) /
      maximum_waves_per_block);
#define LAUNCH_SINGLE_QUERY_NORM_WIDTH(specialized, cache_values)                                                    \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (single_query_layer_norm_kernel<scalar_t, norm_t, specialized, cache_values>), dim3(block_count),              \
      dim3(kThreadsPerBlock), 0, stream, input.data_ptr<scalar_t>(), norm_weight.data_ptr<norm_t>(),                 \
      norm_bias.data_ptr<norm_t>(), output.data_ptr<scalar_t>(), batch_count, readout_count, hidden_width, epsilon)
  if (hidden_width == kSingleQuerySpecializedHiddenWidth) {
    LAUNCH_SINGLE_QUERY_NORM_WIDTH(true, kSingleQueryCachedValuesPerLane);
  } else {
    DispatchNormalizationCache(hidden_width, input.get_device(), [&](auto cache_width) {
      LAUNCH_SINGLE_QUERY_NORM_WIDTH(false, decltype(cache_width)::value);
    });
  }
#undef LAUNCH_SINGLE_QUERY_NORM_WIDTH
}

void LaunchSingleQueryReadoutLayerNorm(const torch::Tensor& input,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    torch::Tensor& output, int64_t batch_count, int64_t readout_count,
    int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width > 0, "single-query readout LayerNorm requires a positive hidden width");
  if (batch_count == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(input.get_device());
#define LAUNCH_SINGLE_QUERY_NORM(scalar_type, norm_type)          \
  LaunchSingleQueryReadoutLayerNormTyped<scalar_type, norm_type>( \
      input, norm_weight, norm_bias, output, batch_count, readout_count, hidden_width, epsilon, stream)
  if (input.scalar_type() == torch::kFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_NORM(c10::Half, float);
    } else {
      LAUNCH_SINGLE_QUERY_NORM(c10::Half, c10::Half);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (norm_weight.scalar_type() == torch::kFloat32) {
      LAUNCH_SINGLE_QUERY_NORM(c10::BFloat16, float);
    } else {
      LAUNCH_SINGLE_QUERY_NORM(c10::BFloat16, c10::BFloat16);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_NORM(float, float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_NORM
  CheckKernelLaunch("fusion single-query readout LayerNorm");
}

void LaunchSingleQueryReadoutBiasSilu(torch::Tensor& values,
    const torch::Tensor& bias, int64_t batch_count, int64_t readout_count,
    int64_t width) {
  const int64_t element_count = CheckedMultiply(
      CheckedMultiply(batch_count, readout_count,
          "single-query readout SiLU row count"),
      width, "single-query readout SiLU element count");
  if (element_count == 0) {
    return;
  }
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(RequiredBlockCount(element_count), 65535));
  const auto stream = backend::GetCurrentStream(values.get_device());
#define LAUNCH_SINGLE_QUERY_SILU(scalar_type) \
  DJL_FUSION_LAUNCH_KERNEL((single_query_bias_silu_kernel<scalar_type>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, \
      values.data_ptr<scalar_type>(), bias.data_ptr<scalar_type>(), \
      batch_count, readout_count, width)
  if (values.scalar_type() == torch::kFloat16) {
    LAUNCH_SINGLE_QUERY_SILU(c10::Half);
  } else if (values.scalar_type() == torch::kBFloat16) {
    LAUNCH_SINGLE_QUERY_SILU(c10::BFloat16);
  } else if (values.scalar_type() == torch::kFloat32) {
    LAUNCH_SINGLE_QUERY_SILU(float);
  } else {
    TORCH_CHECK(false, "unsupported single-query readout data type");
  }
#undef LAUNCH_SINGLE_QUERY_SILU
  CheckKernelLaunch("fusion single-query readout bias SiLU");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalGatherNormalizeTyped(const torch::Tensor& input, const torch::Tensor& indices,
    const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t hidden_width,
    float epsilon, uint32_t block_count, backend::Stream stream) {
  const IndexedLocalDenseInput<scalar_t> dense_input{input.data_ptr<scalar_t>(), hidden_width};
#define LAUNCH_INDEXED_LOCAL_GATHER_NORM_WIDTH(width, cache_values)                                                  \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (indexed_local_gather_normalize_kernel<scalar_t, index_t, IndexedLocalDenseInput<scalar_t>, width, cache_values>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, dense_input, indices.data_ptr<index_t>(),                \
      input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()), \
      attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),                                              \
      static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),                        \
      normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, epsilon, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_INDEXED_LOCAL_GATHER_NORM_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_INDEXED_LOCAL_GATHER_NORM_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, input.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_INDEXED_LOCAL_GATHER_NORM_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL(
            (indexed_local_gather_normalize_dynamic_kernel<scalar_t, index_t, IndexedLocalDenseInput<scalar_t>>),
            dim3(block_count), dim3(kThreadsPerBlock), 0, stream, dense_input, indices.data_ptr<index_t>(),
            input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()),
            attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),
            static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, hidden_width, epsilon);
      }
    });
  }
#undef LAUNCH_INDEXED_LOCAL_GATHER_NORM_WIDTH
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalGatherNormalizeSegmentsTyped(const IndexedLocalTransformerInputSegment* input_segments,
    int32_t input_segment_count, const torch::Tensor& indices, const torch::Tensor& input_norm_weight,
    const torch::Tensor& input_norm_bias, const torch::Tensor& attention_norm_weight,
    const torch::Tensor& attention_norm_bias, torch::Tensor& output, torch::Tensor& normalized, int64_t active_offset,
    int64_t active_rows, int64_t dense_rows, int64_t token_count, int64_t hidden_width, float epsilon,
    uint32_t block_count, backend::Stream stream) {
  IndexedLocalSegmentedInput<scalar_t> segmented_input{};
  segmented_input.segment_count = input_segment_count;
  segmented_input.token_count = token_count;
  segmented_input.hidden_width = hidden_width;
  for (int32_t segment = 0; segment < input_segment_count; ++segment) {
    segmented_input.data[segment] = input_segments[segment].data;
    segmented_input.token_offsets[segment] = input_segments[segment].token_offset;
    segmented_input.token_counts[segment] = input_segments[segment].token_count;
  }
#define LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER_NORM_WIDTH(width, cache_values)                                        \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (indexed_local_gather_normalize_kernel<scalar_t, index_t, IndexedLocalSegmentedInput<scalar_t>, width, cache_values>), \
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream, segmented_input, indices.data_ptr<index_t>(),            \
      input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()), \
      attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),                                              \
      static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),                        \
      normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, epsilon, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER_NORM_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER_NORM_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, output.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER_NORM_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL(
            (indexed_local_gather_normalize_dynamic_kernel<scalar_t, index_t, IndexedLocalSegmentedInput<scalar_t>>),
            dim3(block_count), dim3(kThreadsPerBlock), 0, stream, segmented_input, indices.data_ptr<index_t>(),
            input_norm_weight.data_ptr(), input_norm_bias.data_ptr(), static_cast<int32_t>(input_norm_weight.scalar_type()),
            attention_norm_weight.data_ptr(), attention_norm_bias.data_ptr(),
            static_cast<int32_t>(attention_norm_weight.scalar_type()), output.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(), active_offset, active_rows, dense_rows, hidden_width, epsilon);
      }
    });
  }
#undef LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER_NORM_WIDTH
}

void LaunchIndexedLocalTransformerClear(torch::Tensor& output, int64_t dense_rows, int64_t hidden_width) {
  const auto stream = backend::GetCurrentStream(output.get_device());
  const int64_t output_bytes =
      CheckedMultiply(CheckedMultiply(dense_rows, hidden_width, "indexed local transformer output elements"),
          static_cast<int64_t>(output.element_size()), "indexed local transformer output bytes");
  backend::MemsetAsync(output.data_ptr(), 0,
      static_cast<std::size_t>(output_bytes), stream,
      "indexed local transformer output clear");
}

void LaunchIndexedLocalTransformerGatherNormalize(const torch::Tensor& input, const torch::Tensor& indices,
    torch::ScalarType index_data_type, const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t hidden_width,
    float epsilon) {
  TORCH_CHECK(hidden_width > 0, "indexed local transformer requires a positive hidden width");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = backend::GetCurrentStream(input.get_device());
#define LAUNCH_INDEXED_LOCAL_GATHER(scalar_type, index_type)                                                          \
  LaunchIndexedLocalGatherNormalizeTyped<scalar_type, index_type>(input, indices, input_norm_weight, input_norm_bias, \
      attention_norm_weight, attention_norm_bias, output, normalized, active_offset, active_rows, dense_rows,         \
      hidden_width, epsilon, block_count, stream)
  if (input.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::Half, int64_t);
    }
  } else if (input.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(c10::BFloat16, int64_t);
    }
  } else if (input.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_GATHER(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_GATHER(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_GATHER
  CheckKernelLaunch("fusion indexed local transformer gather LayerNorm");
}

void LaunchIndexedLocalTransformerGatherNormalizeSegments(
    const IndexedLocalTransformerInputSegment* input_segments, int32_t input_segment_count,
    torch::ScalarType input_data_type, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& input_norm_weight, const torch::Tensor& input_norm_bias,
    const torch::Tensor& attention_norm_weight, const torch::Tensor& attention_norm_bias, torch::Tensor& output,
    torch::Tensor& normalized, int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t group_count,
    int64_t token_count, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(input_segment_count > 1 && input_segment_count <= kMaximumIndexedLocalTransformerSegments,
      "segmented indexed local transformer requires two to eight input segments");
  TORCH_CHECK(group_count > 0 && token_count > 0 && hidden_width > 0,
      "segmented indexed local transformer metadata is invalid");
  int64_t next_token_offset = 0;
  for (int32_t segment = 0; segment < input_segment_count; ++segment) {
    TORCH_CHECK(input_segments[segment].data != nullptr &&
            input_segments[segment].token_offset == next_token_offset &&
            input_segments[segment].token_count > 0,
        "segmented indexed local transformer input metadata is invalid");
    next_token_offset += input_segments[segment].token_count;
  }
  TORCH_CHECK(next_token_offset == token_count,
      "segmented indexed local transformer token count does not match its inputs");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = backend::GetCurrentStream(output.get_device());
#define LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(scalar_type, index_type)                                             \
  LaunchIndexedLocalGatherNormalizeSegmentsTyped<scalar_type, index_type>(input_segments, input_segment_count,     \
      indices, input_norm_weight, input_norm_bias, attention_norm_weight, attention_norm_bias, output, normalized, \
      active_offset, active_rows, dense_rows, token_count, hidden_width, epsilon, block_count, stream)
  if (input_data_type == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::Half, int64_t);
    }
  } else if (input_data_type == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(c10::BFloat16, int64_t);
    }
  } else if (input_data_type == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported segmented indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_SEGMENTED_GATHER
  CheckKernelLaunch("fusion segmented indexed local transformer gather LayerNorm");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalAttentionTyped(torch::Tensor& query_key_value, const torch::Tensor& indices, int64_t active_rows,
    int64_t dense_rows, int64_t token_count, int64_t attention_heads, int64_t attention_width, torch::Tensor* context,
    uint32_t block_count, backend::Stream stream) {
  if (UsesIndexedLocalAttentionInPlace(token_count, attention_heads, attention_width)) {
    DJL_FUSION_LAUNCH_KERNEL((indexed_local_attention_kernel<scalar_t, index_t>), dim3(block_count),
        dim3(kThreadsPerBlock), 0, stream, query_key_value.data_ptr<scalar_t>(), indices.data_ptr<index_t>(),
        active_rows, dense_rows, token_count);
  } else {
    DJL_FUSION_LAUNCH_KERNEL((indexed_local_attention_dynamic_kernel<scalar_t, index_t>), dim3(block_count),
        dim3(kThreadsPerBlock), 0, stream, query_key_value.data_ptr<scalar_t>(), indices.data_ptr<index_t>(),
        context->data_ptr<scalar_t>(), active_rows, token_count, attention_heads, attention_width);
  }
}

void LaunchIndexedLocalTransformerAttention(torch::Tensor& query_key_value, const torch::Tensor& indices,
    torch::ScalarType index_data_type, int64_t active_rows, int64_t dense_rows, int64_t token_count,
    int64_t attention_heads, int64_t attention_width, torch::Tensor* context) {
  TORCH_CHECK(token_count > 0 && attention_heads > 0 && attention_width > 0 && attention_width % attention_heads == 0,
      "indexed local transformer attention requires positive dimensions and a width divisible by its head count");
  if (active_rows == 0) {
    return;
  }
  const bool in_place = UsesIndexedLocalAttentionInPlace(token_count, attention_heads, attention_width);
  TORCH_CHECK(in_place || context != nullptr, "indexed local transformer attention requires context storage");
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const int64_t query_heads = CheckedMultiply(active_rows, attention_heads, "indexed local transformer query heads");
  const uint32_t block_count = static_cast<uint32_t>(
      in_place ? dense_rows / token_count : std::min<int64_t>((query_heads - 1) / maximum_waves_per_block + 1, 65535));
  const auto stream = backend::GetCurrentStream(query_key_value.get_device());
#define LAUNCH_INDEXED_LOCAL_ATTENTION(scalar_type, index_type)                                                \
  LaunchIndexedLocalAttentionTyped<scalar_type, index_type>(query_key_value, indices, active_rows, dense_rows, \
      token_count, attention_heads, attention_width, context, block_count, stream)
  if (query_key_value.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::Half, int64_t);
    }
  } else if (query_key_value.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(c10::BFloat16, int64_t);
    }
  } else if (query_key_value.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_ATTENTION(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_ATTENTION(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_ATTENTION
  CheckKernelLaunch("fusion indexed local transformer attention");
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalResidualNormalizeTyped(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& indices, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, torch::Tensor& normalized, int64_t active_offset, int64_t active_rows,
    int64_t dense_rows, int64_t hidden_width, float epsilon, uint32_t block_count, backend::Stream stream) {
#define LAUNCH_INDEXED_LOCAL_RESIDUAL_NORM_WIDTH(width, cache_values)                                                \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (indexed_local_residual_normalize_kernel<scalar_t, index_t, width, cache_values>), dim3(block_count),          \
      dim3(kThreadsPerBlock), 0, stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),                   \
      indices.data_ptr<index_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),   \
      static_cast<int32_t>(norm_weight.scalar_type()), normalized.data_ptr<scalar_t>(), active_offset, active_rows,  \
      dense_rows, epsilon, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_INDEXED_LOCAL_RESIDUAL_NORM_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_INDEXED_LOCAL_RESIDUAL_NORM_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, output.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_INDEXED_LOCAL_RESIDUAL_NORM_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL((indexed_local_residual_normalize_dynamic_kernel<scalar_t, index_t, false>),
            dim3(block_count), dim3(kThreadsPerBlock), 0, stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
            indices.data_ptr<index_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),
            static_cast<int32_t>(norm_weight.scalar_type()), normalized.data_ptr<scalar_t>(), active_offset, active_rows,
            dense_rows, hidden_width, epsilon);
      }
    });
  }
#undef LAUNCH_INDEXED_LOCAL_RESIDUAL_NORM_WIDTH
}

void LaunchIndexedLocalTransformerResidualLayerNorm(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& update_bias, const torch::Tensor& indices, torch::ScalarType index_data_type,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, torch::Tensor& normalized, int64_t active_offset,
    int64_t active_rows, int64_t dense_rows, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width > 0, "indexed local transformer requires a positive hidden width");
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
  const auto stream = backend::GetCurrentStream(output.get_device());
#define LAUNCH_INDEXED_LOCAL_RESIDUAL(scalar_type, index_type)                                                         \
  LaunchIndexedLocalResidualNormalizeTyped<scalar_type, index_type>(output, update, update_bias, indices, norm_weight, \
      norm_bias, normalized, active_offset, active_rows, dense_rows, hidden_width, epsilon, block_count, stream)
  if (output.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::Half, int64_t);
    }
  } else if (output.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(c10::BFloat16, int64_t);
    }
  } else if (output.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_RESIDUAL(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_RESIDUAL
  CheckKernelLaunch("fusion indexed local transformer residual LayerNorm");
}

void LaunchIndexedLocalTransformerBiasSilu(
    torch::Tensor& values, const torch::Tensor& bias, int64_t active_rows, int64_t width) {
  LaunchTransformerBiasSilu(values, bias, active_rows, width);
}

template <typename scalar_t, typename index_t>
void LaunchIndexedLocalFinalizeTyped(torch::Tensor& output, const torch::Tensor& update, const torch::Tensor& indices,
    const torch::Tensor& update_bias, const torch::Tensor& norm_weight, const torch::Tensor& norm_bias,
    int64_t active_offset, int64_t active_rows, int64_t dense_rows, int64_t hidden_width, float epsilon,
    uint32_t block_count, backend::Stream stream) {
#define LAUNCH_INDEXED_LOCAL_FINALIZE_WIDTH(width, cache_values)                                                     \
  DJL_FUSION_LAUNCH_KERNEL(                                                                                          \
      (indexed_local_finalize_kernel<scalar_t, index_t, width, cache_values>), dim3(block_count),                    \
      dim3(kThreadsPerBlock), 0, stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),                   \
      indices.data_ptr<index_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),   \
      static_cast<int32_t>(norm_weight.scalar_type()), active_offset, active_rows, dense_rows, epsilon, hidden_width)
  if (hidden_width == 128) {
    LAUNCH_INDEXED_LOCAL_FINALIZE_WIDTH(128, 4);
  } else if (hidden_width == 256) {
    LAUNCH_INDEXED_LOCAL_FINALIZE_WIDTH(256, 8);
  } else {
    DispatchNormalizationCache(hidden_width, output.get_device(), [&](auto cache_width) {
      constexpr int cache_values = decltype(cache_width)::value;
      if constexpr (cache_values > 0) {
        LAUNCH_INDEXED_LOCAL_FINALIZE_WIDTH(0, cache_values);
      } else {
        DJL_FUSION_LAUNCH_KERNEL((indexed_local_residual_normalize_dynamic_kernel<scalar_t, index_t, true>),
            dim3(block_count), dim3(kThreadsPerBlock), 0, stream, output.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
            indices.data_ptr<index_t>(), update_bias.data_ptr<scalar_t>(), norm_weight.data_ptr(), norm_bias.data_ptr(),
            static_cast<int32_t>(norm_weight.scalar_type()), static_cast<scalar_t*>(nullptr), active_offset, active_rows,
            dense_rows, hidden_width, epsilon);
      }
    });
  }
#undef LAUNCH_INDEXED_LOCAL_FINALIZE_WIDTH
}

void LaunchIndexedLocalTransformerFinalize(torch::Tensor& output, const torch::Tensor& update,
    const torch::Tensor& indices, torch::ScalarType index_data_type, const torch::Tensor& update_bias,
    const torch::Tensor& norm_weight, const torch::Tensor& norm_bias, int64_t active_offset, int64_t active_rows,
    int64_t dense_rows, int64_t hidden_width, float epsilon) {
  TORCH_CHECK(hidden_width > 0, "indexed local transformer requires a positive hidden width");
  const auto stream = backend::GetCurrentStream(output.get_device());
  if (active_rows == 0) {
    return;
  }
  constexpr int64_t maximum_waves_per_block = kThreadsPerBlock / 32;
  const uint32_t block_count =
      static_cast<uint32_t>((active_rows + maximum_waves_per_block - 1) / maximum_waves_per_block);
#define LAUNCH_INDEXED_LOCAL_FINALIZE(scalar_type, index_type)                                                \
  LaunchIndexedLocalFinalizeTyped<scalar_type, index_type>(output, update, indices, update_bias, norm_weight, \
      norm_bias, active_offset, active_rows, dense_rows, hidden_width, epsilon, block_count, stream)
  if (output.scalar_type() == torch::kFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::Half, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::Half, int64_t);
    }
  } else if (output.scalar_type() == torch::kBFloat16) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::BFloat16, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(c10::BFloat16, int64_t);
    }
  } else if (output.scalar_type() == torch::kFloat32) {
    if (index_data_type == torch::kInt32) {
      LAUNCH_INDEXED_LOCAL_FINALIZE(float, int32_t);
    } else {
      LAUNCH_INDEXED_LOCAL_FINALIZE(float, int64_t);
    }
  } else {
    TORCH_CHECK(false, "unsupported indexed local transformer data type");
  }
#undef LAUNCH_INDEXED_LOCAL_FINALIZE
  CheckKernelLaunch("fusion indexed local transformer finalize");
}

template <typename score_t, typename mask_t, typename value_t>
void LaunchMappedGroupedMaskedSoftmaxPoolTyped(
    const torch::Tensor& scores, const torch::Tensor& masks,
    const torch::Tensor& values, const torch::Tensor& destination_metadata,
    torch::Tensor& contexts, torch::Tensor& presence, int64_t batch_count,
    int64_t output_batch_capacity, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count,
    backend::Stream stream) {
  const int64_t row_count = CheckedMultiply(
      batch_count, destination_count, "mapped grouped pool row count");
  const uint32_t block_count = static_cast<uint32_t>(
      std::min<int64_t>(row_count, 65535));
  DJL_FUSION_LAUNCH_KERNEL(
      (mapped_grouped_masked_softmax_pool_kernel<
          score_t, mask_t, value_t>),
      dim3(block_count), dim3(kThreadsPerBlock), 0, stream,
      scores.data_ptr<score_t>(), masks.data_ptr<mask_t>(),
      values.data_ptr<value_t>(), destination_metadata.data_ptr<int32_t>(),
      contexts.data_ptr<float>(), presence.data_ptr<mask_t>(), batch_count,
      candidate_count, group_count, width, destination_count,
      output_batch_capacity);
}

void LaunchMappedGroupedMaskedSoftmaxPool(const torch::Tensor& scores,
    const torch::Tensor& masks, const torch::Tensor& values,
    const torch::Tensor& destination_metadata, torch::Tensor& contexts,
    torch::Tensor& presence, int64_t batch_count,
    int64_t output_batch_capacity, int64_t candidate_count,
    int64_t group_count, int64_t width, int64_t destination_count) {
  TORCH_CHECK(candidate_count > 0 && candidate_count <= 256 &&
          output_batch_capacity > 0 && batch_count >= 0 &&
          batch_count <= output_batch_capacity && group_count > 0 &&
          width > 0 && destination_count > 0,
      "mapped grouped pool metadata is unsupported");
  TORCH_CHECK(destination_metadata.scalar_type() == torch::kInt32 &&
          destination_metadata.is_contiguous() &&
          destination_metadata.dim() == 2 &&
          destination_metadata.size(0) == destination_count &&
          destination_metadata.size(1) == 4,
      "mapped grouped pool destination metadata is invalid");
  TORCH_CHECK(contexts.scalar_type() == torch::kFloat32 &&
          presence.scalar_type() == masks.scalar_type(),
      "mapped grouped pool result data types are invalid");
  if (batch_count == 0) {
    return;
  }
  const auto stream = backend::GetCurrentStream(scores.get_device());
#define LAUNCH_MAPPED_POOL(score_type, mask_type, value_type) \
  LaunchMappedGroupedMaskedSoftmaxPoolTyped<score_type, mask_type, value_type>( \
      scores, masks, values, destination_metadata, contexts, presence, \
      batch_count, output_batch_capacity, candidate_count, group_count, width, \
      destination_count, stream)
#define DISPATCH_MAPPED_POOL_VALUE(score_type, mask_type) \
  switch (values.scalar_type()) { \
    case torch::kFloat16: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, c10::Half); \
      break; \
    case torch::kBFloat16: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      LAUNCH_MAPPED_POOL(score_type, mask_type, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported mapped grouped pool value data type"); \
  }
#define DISPATCH_MAPPED_POOL_MASK(score_type) \
  switch (masks.scalar_type()) { \
    case torch::kFloat16: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, c10::Half); \
      break; \
    case torch::kBFloat16: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, c10::BFloat16); \
      break; \
    case torch::kFloat32: \
      DISPATCH_MAPPED_POOL_VALUE(score_type, float); \
      break; \
    default: \
      TORCH_CHECK(false, "unsupported mapped grouped pool mask data type"); \
  }
  switch (scores.scalar_type()) {
    case torch::kFloat16:
      DISPATCH_MAPPED_POOL_MASK(c10::Half);
      break;
    case torch::kBFloat16:
      DISPATCH_MAPPED_POOL_MASK(c10::BFloat16);
      break;
    case torch::kFloat32:
      DISPATCH_MAPPED_POOL_MASK(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported mapped grouped pool score data type");
  }
#undef DISPATCH_MAPPED_POOL_MASK
#undef DISPATCH_MAPPED_POOL_VALUE
#undef LAUNCH_MAPPED_POOL
  CheckKernelLaunch("fusion mapped grouped masked-softmax pool");
}

}  // namespace djl::pytorch::fusion
