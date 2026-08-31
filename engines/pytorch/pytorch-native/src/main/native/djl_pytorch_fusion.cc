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
#include "djl_pytorch_fusion.h"

#include <ATen/ops/mm.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "djl_pytorch_fusion_kernels.h"

namespace djl::pytorch::fusion {
namespace {

// This wire schema mirrors PtFusionDescriptor. Stable codes must stay explicit on both sides;
// incompatible record changes require a descriptor version bump.
constexpr int64_t kDescriptorMagic = 0x444a4c5f46555332LL;
constexpr int64_t kDescriptorVersion = 2;
constexpr std::size_t kHeaderWords = 16;
constexpr std::size_t kDimensionRecordWords = 3;
constexpr std::size_t kValueRecordHeaderWords = 7;
constexpr std::size_t kBindingRecordWords = 2;
constexpr std::size_t kCommandRecordHeaderWords = 6;
constexpr std::size_t kOutputRecordWords = 2;

constexpr int64_t kDTypeFloat16 = 1;
constexpr int64_t kDTypeBFloat16 = 2;
constexpr int64_t kDTypeFloat32 = 3;
constexpr int64_t kDTypeBoolean = 4;
constexpr int64_t kDTypeUInt8 = 5;
constexpr int64_t kDTypeInt8 = 6;
constexpr int64_t kDTypeInt16 = 7;
constexpr int64_t kDTypeInt32 = 8;
constexpr int64_t kDTypeInt64 = 9;
constexpr int64_t kDTypeFloat64 = 10;

constexpr int64_t kOutputPackV1 = 1;
constexpr int64_t kAffineSumV1 = 2;
constexpr int64_t kIndexedAffineV1 = 3;
constexpr int64_t kTransformerEncoderStackV1 = 4;
constexpr int64_t kDimensionPrefixExtent = 1;
constexpr int64_t kLayoutContiguous = 1;

constexpr int64_t kAttributeInt64 = 1;
constexpr int64_t kAttributeFloat64Bits = 2;
constexpr int64_t kAttributeBoolean = 3;
constexpr std::size_t kAttributeRecordHeaderWords = 4;

constexpr int64_t kAffineTermCount = 1;
constexpr int64_t kAffineActivation = 2;
constexpr int64_t kAffineHasBias = 3;
constexpr int64_t kIndexedSourceCount = 1;
constexpr int64_t kIndexedActivation = 2;
constexpr int64_t kIndexedHasHiddenBias = 3;
constexpr int64_t kIndexedHasOutputBias = 4;
constexpr int64_t kIndexedSourceDivisors = 5;
constexpr int64_t kTransformerBlockCount = 1;
constexpr int64_t kTransformerAttentionHeads = 2;
constexpr int64_t kTransformerAttentionWidth = 3;
constexpr int64_t kTransformerFeedForwardWidth = 4;
constexpr int64_t kTransformerEpsilon = 5;
constexpr int64_t kActivationNone = 0;
constexpr int64_t kActivationSilu = 1;

enum class ValueKind {
  kUnbound,
  kInput,
  kConstant,
  kComputed,
};

struct DimensionSpec {
  int64_t maximum_extent;
};

struct ValueSpec {
  torch::ScalarType data_type;
  int32_t dimension_index;
  std::vector<int64_t> inner_shape;
  std::vector<int64_t> maximum_shape;
  ValueKind kind = ValueKind::kUnbound;
  int32_t binding_index = -1;
  int32_t producer_index = -1;
  int32_t storage_index = -1;
};

struct StorageSpec {
  torch::ScalarType data_type;
  std::vector<int64_t> maximum_shape;
};

struct OutputSpec {
  int32_t value_index;
  int32_t storage_index;
};

struct OutputPackSourceSpec {
  int32_t value_index;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
};

struct OutputPackCommandSpec {
  int32_t result_value_index;
  int32_t result_storage_index;
  int32_t extent_index;
  int64_t maximum_rows;
  int64_t output_width;
  std::vector<int32_t> operand_value_indices;
  std::vector<OutputPackSourceSpec> sources;
};

struct AffineTermSpec {
  int32_t input_value_index;
  int32_t weight_value_index;
  int32_t weight_binding_index;
  torch::ScalarType input_data_type;
  int64_t input_width;
};

struct AffineGroupSpec {
  std::vector<int64_t> prefix_shape;
  bool dynamic_leading = false;
  bool precompute_at_bind = false;
  int64_t rows_per_batch = 0;
  int64_t packed_input_width = 0;
  int32_t packed_input_storage_index = -1;
  int32_t product_storage_index = -1;
  bool requires_input_pack = false;
  std::array<int64_t, kMaximumAffinePrefixRank> output_strides{};
  std::vector<AffineTermSpec> terms;
};

struct AffineSumCommandSpec {
  int32_t result_value_index;
  int32_t result_storage_index;
  int32_t extent_index;
  int32_t bias_value_index;
  int64_t maximum_batches;
  int64_t output_width;
  int64_t output_rows_per_batch;
  AffineActivation activation;
  bool needs_finalize;
  std::vector<int64_t> output_prefix_shape;
  std::vector<int32_t> operand_value_indices;
  std::vector<AffineGroupSpec> groups;
};

struct IndexedAffineSourceSpec {
  int32_t value_index;
  int32_t extent_index;
  torch::ScalarType data_type;
  int64_t width;
  int64_t destination_offset;
  int64_t index_divisor;
};

struct IndexedAffineCommandSpec {
  int32_t result_value_index;
  int32_t result_storage_index;
  int32_t active_extent_index;
  int32_t destination_extent_index;
  int32_t indices_value_index;
  int32_t hidden_weight_binding_index;
  int32_t hidden_bias_value_index;
  int32_t output_weight_value_index;
  int32_t output_bias_value_index;
  int32_t packed_input_storage_index;
  int32_t hidden_storage_index;
  torch::ScalarType index_data_type;
  int64_t maximum_active_rows;
  int64_t maximum_destination_rows;
  int64_t input_width;
  int64_t hidden_width;
  int64_t output_width;
  AffineActivation activation;
  std::vector<int32_t> operand_value_indices;
  std::vector<IndexedAffineSourceSpec> sources;
};

struct TransformerEncoderBlockSpec {
  std::array<int32_t, 13> value_indices;
  int32_t query_key_value_weight_binding_index;
  int32_t attention_output_weight_binding_index;
  int32_t feed_forward_expansion_weight_binding_index;
  int32_t feed_forward_projection_weight_binding_index;
};

struct TransformerEncoderStackCommandSpec {
  int32_t result_value_index;
  int32_t result_storage_index;
  int32_t input_value_index;
  int32_t extent_index;
  int32_t normalized_storage_index;
  int32_t query_key_value_storage_index;
  int32_t expanded_storage_index;
  int64_t maximum_batches;
  int64_t token_count;
  int64_t hidden_width;
  int64_t attention_heads;
  int64_t attention_width;
  int64_t feed_forward_width;
  float epsilon;
  std::vector<int32_t> operand_value_indices;
  std::vector<TransformerEncoderBlockSpec> blocks;
};

using FusionCommand = std::variant<OutputPackCommandSpec, AffineSumCommandSpec,
    IndexedAffineCommandSpec, TransformerEncoderStackCommandSpec>;

struct FusionPlanData {
  explicit FusionPlanData(c10::Device device) : device(device) {}

  c10::Device device;
  int32_t input_count;
  int32_t dimension_count;
  int32_t constant_count;
  std::vector<DimensionSpec> dimensions;
  std::vector<ValueSpec> values;
  std::vector<int32_t> input_value_indices;
  std::vector<int32_t> constant_value_indices;
  std::vector<StorageSpec> storages;
  std::vector<FusionCommand> commands;
  std::vector<OutputSpec> outputs;
};

struct FusionExecutableData {
  std::shared_ptr<const FusionPlanData> plan;
  std::vector<torch::Tensor> constants;
  std::vector<std::vector<torch::Tensor>> affine_weights;
  std::vector<std::vector<torch::Tensor>> affine_products;
  std::vector<torch::Tensor> indexed_affine_weights;
  std::vector<std::vector<std::array<torch::Tensor, 4>>>
      transformer_encoder_weights;
  std::unique_ptr<c10::Event> binding_ready;
};

struct CommandAttribute {
  int64_t type;
  std::vector<int64_t> payload;
};

using CommandAttributes = std::unordered_map<int64_t, CommandAttribute>;

class DescriptorReader {
 public:
  DescriptorReader(const int64_t* begin, const int64_t* end)
      : current_(begin), end_(end) {}

  int64_t Read() {
    TORCH_CHECK(current_ != end_, "truncated fusion descriptor record");
    return *current_++;
  }

  std::size_t Remaining() const {
    return static_cast<std::size_t>(end_ - current_);
  }

  const int64_t* Current() const {
    return current_;
  }

  bool IsComplete() const {
    return current_ == end_;
  }

 private:
  const int64_t* current_;
  const int64_t* end_;
};

std::size_t CheckedSize(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0, name, " must not be negative");
  const auto converted = static_cast<uint64_t>(value);
  TORCH_CHECK(converted <= std::numeric_limits<std::size_t>::max(),
      name, " exceeds the native size range");
  return static_cast<std::size_t>(converted);
}

int32_t CheckedCount(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int32_t>::max(),
      "invalid ", name);
  return static_cast<int32_t>(value);
}

int32_t CheckedIndex(int64_t value, std::size_t upper_bound, const char* name) {
  TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) < upper_bound,
      name, " is outside the descriptor range");
  return static_cast<int32_t>(value);
}

std::size_t CheckedFixedTableEnd(std::size_t offset, std::size_t count,
    std::size_t record_words, std::size_t total_words, const char* name) {
  TORCH_CHECK(offset <= total_words, name, " offset exceeds the descriptor");
  TORCH_CHECK(count <= (total_words - offset) / record_words,
      name, " exceeds the descriptor");
  return offset + count * record_words;
}

torch::ScalarType ScalarTypeFromWireCode(int64_t code) {
  switch (code) {
    case kDTypeFloat16:
      return torch::kFloat16;
    case kDTypeBFloat16:
      return torch::kBFloat16;
    case kDTypeFloat32:
      return torch::kFloat32;
    case kDTypeBoolean:
      return torch::kBool;
    case kDTypeUInt8:
      return torch::kUInt8;
    case kDTypeInt8:
      return torch::kInt8;
    case kDTypeInt16:
      return torch::kInt16;
    case kDTypeInt32:
      return torch::kInt32;
    case kDTypeInt64:
      return torch::kInt64;
    case kDTypeFloat64:
      return torch::kFloat64;
    default:
      TORCH_CHECK(false, "unsupported fusion descriptor data type code: ", code);
  }
}

bool IsFusionFloatingDataType(torch::ScalarType data_type) {
  return data_type == torch::kFloat16 || data_type == torch::kBFloat16 ||
      data_type == torch::kFloat32;
}

void RecordCurrentStream(const torch::Tensor& tensor) {
  c10::DeviceGuard device_guard(tensor.device());
  c10::impl::VirtualGuardImpl guard_impl(tensor.device().type());
  guard_impl.recordDataPtrOnStream(
      tensor.storage().data_ptr(), guard_impl.getStream(tensor.device()));
}

CommandAttributes ReadAttributeRecords(
    DescriptorReader& reader, int32_t attribute_count) {
  TORCH_CHECK(static_cast<std::size_t>(attribute_count) <=
          reader.Remaining() / kAttributeRecordHeaderWords,
      "fusion command attribute count exceeds its record");
  CommandAttributes attributes;
  attributes.reserve(static_cast<std::size_t>(attribute_count));
  for (int32_t index = 0; index < attribute_count; ++index) {
    TORCH_CHECK(reader.Remaining() >= kAttributeRecordHeaderWords,
        "truncated fusion command attribute");
    const std::size_t record_words = CheckedSize(reader.Read(),
        "fusion command attribute record size");
    TORCH_CHECK(record_words >= kAttributeRecordHeaderWords &&
            record_words <= reader.Remaining() + 1,
        "invalid fusion command attribute record size");
    const int64_t key = reader.Read();
    const int64_t type = reader.Read();
    const std::size_t element_count = CheckedSize(reader.Read(),
        "fusion command attribute element count");
    TORCH_CHECK(element_count == record_words - kAttributeRecordHeaderWords,
        "fusion command attribute size does not match its payload");
    TORCH_CHECK(attributes.find(key) == attributes.end(),
        "duplicate fusion command attribute key");
    TORCH_CHECK(type == kAttributeInt64 || type == kAttributeFloat64Bits ||
            type == kAttributeBoolean,
        "unsupported fusion command attribute type");
    TORCH_CHECK(element_count <= reader.Remaining(),
        "truncated fusion command attribute payload");
    CommandAttribute attribute;
    attribute.type = type;
    attribute.payload.reserve(element_count);
    for (std::size_t element = 0; element < element_count; ++element) {
      const int64_t payload = reader.Read();
      TORCH_CHECK(type != kAttributeBoolean || payload == 0 || payload == 1,
          "fusion boolean command attribute must contain zero or one");
      attribute.payload.push_back(payload);
    }
    attributes.emplace(key, std::move(attribute));
  }
  return attributes;
}

int64_t GetRequiredInt64Scalar(const CommandAttributes& attributes,
    int64_t key, const char* name) {
  const auto found = attributes.find(key);
  TORCH_CHECK(found != attributes.end(), "missing ", name, " attribute");
  TORCH_CHECK(found->second.type == kAttributeInt64 &&
          found->second.payload.size() == 1,
      name, " attribute must be one int64 value");
  return found->second.payload[0];
}

double GetRequiredFloat64Scalar(const CommandAttributes& attributes,
    int64_t key, const char* name) {
  const auto found = attributes.find(key);
  TORCH_CHECK(found != attributes.end(), "missing ", name, " attribute");
  TORCH_CHECK(found->second.type == kAttributeFloat64Bits &&
          found->second.payload.size() == 1,
      name, " attribute must be one float64 value");
  double value;
  const int64_t bits = found->second.payload[0];
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

const std::vector<int64_t>& GetRequiredInt64Vector(
    const CommandAttributes& attributes, int64_t key, std::size_t size,
    const char* name) {
  const auto found = attributes.find(key);
  TORCH_CHECK(found != attributes.end(), "missing ", name, " attribute");
  TORCH_CHECK(found->second.type == kAttributeInt64 &&
          found->second.payload.size() == size,
      name, " attribute must have the required int64 element count");
  return found->second.payload;
}

void ValidateTensorMetadata(const torch::Tensor& tensor, const FusionPlanData& plan,
    const ValueSpec& spec, const int64_t* dimensions, bool require_maximum_shape,
    const char* kind) {
  TORCH_CHECK(tensor.defined(), "fusion ", kind, " must be defined");
  TORCH_CHECK(tensor.device() == plan.device,
      "fusion ", kind, " must use the prepared device");
  TORCH_CHECK(tensor.layout() == c10::kStrided && tensor.is_contiguous(),
      "fusion ", kind, " must be a contiguous strided tensor");
  TORCH_CHECK(tensor.scalar_type() == spec.data_type,
      "fusion ", kind, " data type does not match the recipe");
  TORCH_CHECK(static_cast<std::size_t>(tensor.dim()) == spec.maximum_shape.size(),
      "fusion ", kind, " rank does not match the recipe");
  if (spec.dimension_index < 0 || require_maximum_shape) {
    for (std::size_t axis = 0; axis < spec.maximum_shape.size(); ++axis) {
      TORCH_CHECK(tensor.size(static_cast<int64_t>(axis)) == spec.maximum_shape[axis],
          "fusion ", kind, " shape does not match the recipe");
    }
    return;
  }
  const int64_t active_extent = dimensions[spec.dimension_index];
  TORCH_CHECK(tensor.size(0) <= spec.maximum_shape[0],
      "fusion ", kind, " row capacity exceeds the prepared range");
  TORCH_CHECK(active_extent <= tensor.size(0),
      "fusion active extent exceeds the ", kind, " row count");
  for (std::size_t axis = 0; axis < spec.inner_shape.size(); ++axis) {
    TORCH_CHECK(tensor.size(static_cast<int64_t>(axis + 1)) == spec.inner_shape[axis],
        "fusion ", kind, " shape does not match the recipe");
  }
}

int32_t NextStorageIndex(const FusionPlanData& plan) {
  TORCH_CHECK(plan.storages.size() <=
          static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
      "fusion storage count exceeds the supported range");
  return static_cast<int32_t>(plan.storages.size());
}

OutputPackCommandSpec BuildOutputPackCommand(FusionPlanData& plan,
    int32_t command_index, const std::vector<int32_t>& results,
    const std::vector<int32_t>& operands, int64_t flags,
    const CommandAttributes& attributes) {
  TORCH_CHECK(flags == 0, "OUTPUT_PACK_V1 does not support command flags");
  TORCH_CHECK(results.size() == 1,
      "OUTPUT_PACK_V1 requires exactly one result");
  TORCH_CHECK(!operands.empty() && operands.size() <= kMaximumOutputPackSources,
      "OUTPUT_PACK_V1 operand count exceeds the native limit");
  TORCH_CHECK(attributes.empty(), "OUTPUT_PACK_V1 does not support attributes");

  const int32_t result_index = results[0];
  ValueSpec& result = plan.values[result_index];
  TORCH_CHECK(result.kind == ValueKind::kUnbound,
      "fusion command result already has a producer or binding");
  TORCH_CHECK(result.data_type == torch::kFloat32,
      "OUTPUT_PACK_V1 result must use FLOAT32");
  TORCH_CHECK(result.dimension_index >= 0 && result.inner_shape.size() == 1,
      "OUTPUT_PACK_V1 result must have one leading and one inner dimension");

  OutputPackCommandSpec command;
  command.result_value_index = result_index;
  command.result_storage_index = NextStorageIndex(plan);
  command.extent_index = result.dimension_index;
  command.maximum_rows = plan.dimensions[result.dimension_index].maximum_extent;
  command.output_width = result.inner_shape[0];
  command.operand_value_indices = operands;
  command.sources.reserve(operands.size());

  int64_t destination_offset = 0;
  for (int32_t operand_index : operands) {
    const ValueSpec& operand = plan.values[operand_index];
    TORCH_CHECK(operand.kind != ValueKind::kUnbound,
        "fusion command operand is not topologically available");
    TORCH_CHECK(IsFusionFloatingDataType(operand.data_type),
        "OUTPUT_PACK_V1 supports FLOAT16, BFLOAT16, and FLOAT32 operands");
    TORCH_CHECK(operand.dimension_index == command.extent_index &&
            operand.inner_shape.size() == 1,
        "OUTPUT_PACK_V1 operands must share the result leading dimension and rank");
    const int64_t width = operand.inner_shape[0];
    TORCH_CHECK(width <= std::numeric_limits<int64_t>::max() - destination_offset,
        "OUTPUT_PACK_V1 width exceeds the supported range");
    command.sources.push_back(OutputPackSourceSpec{
        operand_index, operand.data_type, width, destination_offset});
    destination_offset += width;
  }
  TORCH_CHECK(destination_offset == command.output_width,
      "OUTPUT_PACK_V1 operand widths do not match the result width");

  result.kind = ValueKind::kComputed;
  result.producer_index = command_index;
  result.storage_index = command.result_storage_index;
  plan.storages.push_back(StorageSpec{result.data_type, result.maximum_shape});
  return command;
}

int64_t CheckedMultiply(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " factors must not be negative");
  TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
      name, " exceeds the supported range");
  return left * right;
}

int64_t Product(const std::vector<int64_t>& shape, const char* name) {
  int64_t product = 1;
  for (int64_t extent : shape) {
    product = CheckedMultiply(product, extent, name);
  }
  return product;
}

int64_t TensorPayloadBytes(const std::vector<int64_t>& shape,
    torch::ScalarType data_type, const char* name) {
  const std::size_t element_size = c10::elementSize(data_type);
  TORCH_CHECK(element_size <=
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
      name, " element size exceeds the supported range");
  return CheckedMultiply(Product(shape, name),
      static_cast<int64_t>(element_size), name);
}

std::vector<int64_t> BroadcastPrefix(
    const std::vector<int64_t>& left, const std::vector<int64_t>& right) {
  const std::size_t rank = std::max(left.size(), right.size());
  std::vector<int64_t> result(rank, 1);
  for (std::size_t result_axis = 0; result_axis < rank; ++result_axis) {
    const int64_t left_axis = static_cast<int64_t>(result_axis) -
        static_cast<int64_t>(rank - left.size());
    const int64_t right_axis = static_cast<int64_t>(result_axis) -
        static_cast<int64_t>(rank - right.size());
    const int64_t left_extent = left_axis < 0 ? 1 : left[left_axis];
    const int64_t right_extent = right_axis < 0 ? 1 : right[right_axis];
    TORCH_CHECK(left_extent == right_extent || left_extent == 1 || right_extent == 1,
        "AFFINE_SUM_V1 input prefixes are not broadcast-compatible");
    result[result_axis] = std::max(left_extent, right_extent);
  }
  return result;
}

AffineActivation AffineActivationFromWireCode(int64_t code) {
  switch (code) {
    case kActivationNone:
      return AffineActivation::kNone;
    case kActivationSilu:
      return AffineActivation::kSilu;
    default:
      TORCH_CHECK(false, "unsupported AFFINE_SUM_V1 activation code: ", code);
  }
}

AffineGroupSpec* FindAffineGroup(
    std::vector<AffineGroupSpec>& groups, const std::vector<int64_t>& prefix,
    bool dynamic_leading, bool precompute_at_bind) {
  for (auto& group : groups) {
    if (group.dynamic_leading == dynamic_leading &&
        group.precompute_at_bind == precompute_at_bind &&
        group.prefix_shape == prefix) {
      return &group;
    }
  }
  return nullptr;
}

AffineSumCommandSpec BuildAffineSumCommand(FusionPlanData& plan,
    int32_t command_index, const std::vector<int32_t>& results,
    const std::vector<int32_t>& operands, int64_t flags,
    const CommandAttributes& attributes) {
  TORCH_CHECK(flags == 0, "AFFINE_SUM_V1 does not support command flags");
  TORCH_CHECK(results.size() == 1,
      "AFFINE_SUM_V1 requires exactly one result");
  TORCH_CHECK(attributes.size() == 3,
      "AFFINE_SUM_V1 requires exactly three attributes");
  const int64_t term_count_wire = GetRequiredInt64Scalar(
      attributes, kAffineTermCount, "AFFINE_SUM_V1 term count");
  TORCH_CHECK(term_count_wire > 0 && term_count_wire <= kMaximumAffineTerms,
      "AFFINE_SUM_V1 term count exceeds the native limit");
  const int32_t term_count = static_cast<int32_t>(term_count_wire);
  const int64_t has_bias_wire = GetRequiredInt64Scalar(
      attributes, kAffineHasBias, "AFFINE_SUM_V1 bias presence");
  TORCH_CHECK(has_bias_wire == 0 || has_bias_wire == 1,
      "AFFINE_SUM_V1 bias presence must be zero or one");
  const bool has_bias = has_bias_wire != 0;
  const std::size_t expected_operands =
      static_cast<std::size_t>(term_count) * 2 + (has_bias ? 1 : 0);
  TORCH_CHECK(operands.size() == expected_operands,
      "AFFINE_SUM_V1 operand count does not match its attributes");

  const int32_t result_index = results[0];
  ValueSpec& result = plan.values[result_index];
  TORCH_CHECK(result.kind == ValueKind::kUnbound,
      "fusion command result already has a producer or binding");
  TORCH_CHECK(IsFusionFloatingDataType(result.data_type),
      "AFFINE_SUM_V1 supports FLOAT16, BFLOAT16, and FLOAT32");
  TORCH_CHECK(result.dimension_index >= 0 && !result.inner_shape.empty(),
      "AFFINE_SUM_V1 result requires a leading dimension and feature axis");
  TORCH_CHECK(result.inner_shape.size() - 1 <= kMaximumAffinePrefixRank,
      "AFFINE_SUM_V1 result prefix rank exceeds the native limit");

  AffineSumCommandSpec command;
  command.result_value_index = result_index;
  command.result_storage_index = NextStorageIndex(plan);
  command.extent_index = result.dimension_index;
  command.bias_value_index = -1;
  command.maximum_batches = plan.dimensions[result.dimension_index].maximum_extent;
  command.output_width = result.inner_shape.back();
  command.output_prefix_shape.assign(
      result.inner_shape.begin(), result.inner_shape.end() - 1);
  command.output_rows_per_batch = Product(
      command.output_prefix_shape, "AFFINE_SUM_V1 output prefix size");
  command.activation = AffineActivationFromWireCode(GetRequiredInt64Scalar(
      attributes, kAffineActivation, "AFFINE_SUM_V1 activation"));
  command.operand_value_indices = operands;
  command.groups.reserve(static_cast<std::size_t>(term_count));

  std::vector<int64_t> inferred_output_prefix;
  bool has_dynamic_input = false;
  for (int32_t term_index = 0; term_index < term_count; ++term_index) {
    const int32_t input_index = operands[static_cast<std::size_t>(term_index) * 2];
    const int32_t weight_index = operands[static_cast<std::size_t>(term_index) * 2 + 1];
    const ValueSpec& input = plan.values[input_index];
    const ValueSpec& weight = plan.values[weight_index];
    TORCH_CHECK(input.kind != ValueKind::kUnbound,
        "AFFINE_SUM_V1 input is not topologically available");
    const bool dynamic_leading = input.dimension_index == command.extent_index;
    has_dynamic_input = has_dynamic_input || dynamic_leading;
    const bool fixed_leading = input.dimension_index < 0 &&
        input.inner_shape.size() >= 2 && input.inner_shape[0] == 1;
    TORCH_CHECK(IsFusionFloatingDataType(input.data_type) &&
            (dynamic_leading || fixed_leading) && !input.inner_shape.empty(),
        "AFFINE_SUM_V1 input metadata does not match the result");
    TORCH_CHECK(weight.kind == ValueKind::kConstant &&
            weight.dimension_index < 0 && weight.inner_shape.size() == 2 &&
            weight.data_type == result.data_type &&
            weight.inner_shape[0] == command.output_width &&
            weight.inner_shape[1] == input.inner_shape.back(),
        "AFFINE_SUM_V1 weight metadata does not match its input");

    const auto prefix_begin = input.inner_shape.begin() + (fixed_leading ? 1 : 0);
    std::vector<int64_t> prefix(prefix_begin, input.inner_shape.end() - 1);
    inferred_output_prefix = BroadcastPrefix(inferred_output_prefix, prefix);
    const bool precompute_at_bind =
        !dynamic_leading && input.kind == ValueKind::kConstant;
    AffineGroupSpec* group = FindAffineGroup(
        command.groups, prefix, dynamic_leading, precompute_at_bind);
    if (group == nullptr) {
      TORCH_CHECK(command.groups.size() < kMaximumAffineGroups,
          "AFFINE_SUM_V1 group count exceeds the native limit");
      command.groups.emplace_back();
      group = &command.groups.back();
      group->prefix_shape = std::move(prefix);
      group->dynamic_leading = dynamic_leading;
      group->precompute_at_bind = precompute_at_bind;
      group->rows_per_batch = Product(
          group->prefix_shape, "AFFINE_SUM_V1 group prefix size");
    }
    const int64_t input_width = input.inner_shape.back();
    TORCH_CHECK(input_width <= std::numeric_limits<int64_t>::max() -
            group->packed_input_width,
        "AFFINE_SUM_V1 packed input width exceeds the supported range");
    if (!group->terms.empty() || input.data_type != result.data_type) {
      group->requires_input_pack = true;
    }
    group->terms.push_back(AffineTermSpec{input_index, weight_index,
        weight.binding_index, input.data_type, input_width});
    group->packed_input_width += input_width;
  }
  TORCH_CHECK(has_dynamic_input,
      "AFFINE_SUM_V1 requires at least one dynamically bounded input");
  TORCH_CHECK(inferred_output_prefix == command.output_prefix_shape,
      "AFFINE_SUM_V1 result prefix does not match its broadcast inputs");
  for (const auto& group : command.groups) {
    TORCH_CHECK(group.dynamic_leading ||
            group.prefix_shape.size() == command.output_prefix_shape.size(),
        "AFFINE_SUM_V1 fixed inputs must match the result rank");
  }

  if (has_bias) {
    command.bias_value_index = operands.back();
    const ValueSpec& bias = plan.values[command.bias_value_index];
    TORCH_CHECK(bias.kind == ValueKind::kConstant &&
            bias.dimension_index < 0 && bias.inner_shape.size() == 1 &&
            bias.data_type == result.data_type &&
            bias.inner_shape[0] == command.output_width,
        "AFFINE_SUM_V1 bias metadata does not match the result");
  }

  result.kind = ValueKind::kComputed;
  result.producer_index = command_index;
  result.storage_index = command.result_storage_index;
  plan.storages.push_back(StorageSpec{result.data_type, result.maximum_shape});

  bool direct_group_assigned = false;
  for (auto& group : command.groups) {
    if (!group.precompute_at_bind) {
      std::vector<int64_t> group_shape;
      group_shape.reserve(group.prefix_shape.size() + 2);
      group_shape.push_back(group.dynamic_leading ? command.maximum_batches : 1);
      group_shape.insert(
          group_shape.end(), group.prefix_shape.begin(), group.prefix_shape.end());
      if (group.requires_input_pack) {
        std::vector<int64_t> packed_shape = group_shape;
        packed_shape.push_back(group.packed_input_width);
        TensorPayloadBytes(packed_shape, result.data_type,
            "AFFINE_SUM_V1 packed input payload size");
        group.packed_input_storage_index = NextStorageIndex(plan);
        plan.storages.push_back(StorageSpec{result.data_type, std::move(packed_shape)});
      }
      if (!direct_group_assigned && group.dynamic_leading &&
          group.prefix_shape == command.output_prefix_shape) {
        group.product_storage_index = command.result_storage_index;
        direct_group_assigned = true;
      } else {
        std::vector<int64_t> product_shape = group_shape;
        product_shape.push_back(command.output_width);
        TensorPayloadBytes(product_shape, result.data_type,
            "AFFINE_SUM_V1 product payload size");
        group.product_storage_index = NextStorageIndex(plan);
        plan.storages.push_back(StorageSpec{result.data_type, std::move(product_shape)});
      }
    }
    const int64_t packed_weight_elements = CheckedMultiply(
        group.packed_input_width, command.output_width,
        "AFFINE_SUM_V1 packed weight element count");
    CheckedMultiply(packed_weight_elements,
        static_cast<int64_t>(c10::elementSize(result.data_type)),
        "AFFINE_SUM_V1 packed weight payload size");

    const std::size_t output_rank = command.output_prefix_shape.size();
    const std::size_t source_rank = group.prefix_shape.size();
    int64_t source_stride = 1;
    for (std::size_t reverse_axis = 0; reverse_axis < output_rank; ++reverse_axis) {
      const std::size_t output_axis = output_rank - reverse_axis - 1;
      if (reverse_axis < source_rank) {
        const std::size_t source_axis = source_rank - reverse_axis - 1;
        const int64_t source_extent = group.prefix_shape[source_axis];
        const int64_t output_extent = command.output_prefix_shape[output_axis];
        TORCH_CHECK(source_extent == output_extent || source_extent == 1,
            "AFFINE_SUM_V1 group cannot broadcast to its result");
        group.output_strides[output_axis] = source_extent == 1 ? 0 : source_stride;
        source_stride = CheckedMultiply(source_stride, source_extent,
            "AFFINE_SUM_V1 source prefix stride");
      } else {
        group.output_strides[output_axis] = 0;
      }
    }
  }
  command.needs_finalize = !direct_group_assigned || command.groups.size() != 1 ||
      command.bias_value_index >= 0 || command.activation != AffineActivation::kNone;
  return command;
}

AffineActivation IndexedAffineActivationFromWireCode(int64_t code) {
  switch (code) {
    case kActivationNone:
      return AffineActivation::kNone;
    case kActivationSilu:
      return AffineActivation::kSilu;
    default:
      TORCH_CHECK(false, "unsupported INDEXED_AFFINE_V1 activation code: ", code);
  }
}

IndexedAffineCommandSpec BuildIndexedAffineCommand(FusionPlanData& plan,
    int32_t command_index, const std::vector<int32_t>& results,
    const std::vector<int32_t>& operands, int64_t flags,
    const CommandAttributes& attributes) {
  TORCH_CHECK(flags == 0, "INDEXED_AFFINE_V1 does not support command flags");
  TORCH_CHECK(results.size() == 1,
      "INDEXED_AFFINE_V1 requires exactly one result");
  TORCH_CHECK(attributes.size() == 5,
      "INDEXED_AFFINE_V1 requires exactly five attributes");
  const int64_t source_count_wire = GetRequiredInt64Scalar(
      attributes, kIndexedSourceCount, "INDEXED_AFFINE_V1 source count");
  TORCH_CHECK(source_count_wire > 0 &&
          source_count_wire <= kMaximumIndexedAffineSources,
      "INDEXED_AFFINE_V1 source count exceeds the native limit");
  const int32_t source_count = static_cast<int32_t>(source_count_wire);
  const int64_t has_hidden_bias_wire = GetRequiredInt64Scalar(attributes,
      kIndexedHasHiddenBias, "INDEXED_AFFINE_V1 hidden bias presence");
  const int64_t has_output_bias_wire = GetRequiredInt64Scalar(attributes,
      kIndexedHasOutputBias, "INDEXED_AFFINE_V1 output bias presence");
  TORCH_CHECK((has_hidden_bias_wire == 0 || has_hidden_bias_wire == 1) &&
          (has_output_bias_wire == 0 || has_output_bias_wire == 1),
      "INDEXED_AFFINE_V1 bias presence must be zero or one");
  const bool has_hidden_bias = has_hidden_bias_wire != 0;
  const bool has_output_bias = has_output_bias_wire != 0;
  const std::size_t expected_operands = static_cast<std::size_t>(source_count) + 3 +
      (has_hidden_bias ? 1 : 0) + (has_output_bias ? 1 : 0);
  TORCH_CHECK(operands.size() == expected_operands,
      "INDEXED_AFFINE_V1 operand count does not match its attributes");
  const auto& source_divisors = GetRequiredInt64Vector(attributes,
      kIndexedSourceDivisors, static_cast<std::size_t>(source_count),
      "INDEXED_AFFINE_V1 source divisors");

  const int32_t result_index = results[0];
  ValueSpec& result = plan.values[result_index];
  TORCH_CHECK(result.kind == ValueKind::kUnbound,
      "fusion command result already has a producer or binding");
  TORCH_CHECK(IsFusionFloatingDataType(result.data_type) &&
          result.dimension_index >= 0 && result.inner_shape.size() == 1,
      "INDEXED_AFFINE_V1 result must be a dynamically bounded floating-point matrix");
  TORCH_CHECK(result.inner_shape[0] <= kMaximumIndexedAffineOutputWidth,
      "INDEXED_AFFINE_V1 output width exceeds the native limit");

  IndexedAffineCommandSpec command;
  command.result_value_index = result_index;
  command.result_storage_index = NextStorageIndex(plan);
  command.destination_extent_index = result.dimension_index;
  command.maximum_destination_rows =
      plan.dimensions[result.dimension_index].maximum_extent;
  command.output_width = result.inner_shape[0];
  command.hidden_bias_value_index = -1;
  command.output_bias_value_index = -1;
  command.operand_value_indices = operands;
  command.sources.reserve(static_cast<std::size_t>(source_count));
  command.activation = IndexedAffineActivationFromWireCode(GetRequiredInt64Scalar(
      attributes, kIndexedActivation, "INDEXED_AFFINE_V1 activation"));

  command.indices_value_index = operands[0];
  const ValueSpec& indices = plan.values[command.indices_value_index];
  TORCH_CHECK(indices.kind != ValueKind::kUnbound && indices.dimension_index >= 0 &&
          indices.inner_shape.empty() &&
          (indices.data_type == torch::kInt32 || indices.data_type == torch::kInt64),
      "INDEXED_AFFINE_V1 indices must be a dynamically bounded INT32 or INT64 vector");
  command.active_extent_index = indices.dimension_index;
  command.maximum_active_rows =
      plan.dimensions[indices.dimension_index].maximum_extent;
  command.index_data_type = indices.data_type;
  TORCH_CHECK(command.maximum_active_rows <= command.maximum_destination_rows,
      "INDEXED_AFFINE_V1 active capacity exceeds destination capacity");

  int64_t destination_offset = 0;
  for (int32_t source_index = 0; source_index < source_count; ++source_index) {
    const int32_t value_index = operands[static_cast<std::size_t>(source_index) + 1];
    const ValueSpec& source = plan.values[value_index];
    const int64_t index_divisor = source_divisors[source_index];
    TORCH_CHECK(source.kind != ValueKind::kUnbound && source.dimension_index >= 0 &&
            source.inner_shape.size() == 1 &&
            IsFusionFloatingDataType(source.data_type),
        "INDEXED_AFFINE_V1 sources must be dynamically bounded floating-point matrices");
    TORCH_CHECK(index_divisor > 0,
        "INDEXED_AFFINE_V1 source divisors must be positive");
    const int64_t required_source_rows =
        (command.maximum_destination_rows - 1) / index_divisor + 1;
    TORCH_CHECK(plan.dimensions[source.dimension_index].maximum_extent >=
            required_source_rows,
        "INDEXED_AFFINE_V1 source capacity does not cover divided destination indices");
    const int64_t width = source.inner_shape[0];
    TORCH_CHECK(width <= std::numeric_limits<int64_t>::max() - destination_offset,
        "INDEXED_AFFINE_V1 concatenated source width exceeds the supported range");
    command.sources.push_back(IndexedAffineSourceSpec{value_index,
        source.dimension_index, source.data_type, width, destination_offset,
        index_divisor});
    destination_offset += width;
  }
  command.input_width = destination_offset;

  std::size_t operand_offset = static_cast<std::size_t>(source_count) + 1;
  const int32_t hidden_weight_index = operands[operand_offset++];
  const ValueSpec& hidden_weight = plan.values[hidden_weight_index];
  TORCH_CHECK(hidden_weight.kind == ValueKind::kConstant &&
          hidden_weight.dimension_index < 0 && hidden_weight.inner_shape.size() == 2 &&
          hidden_weight.data_type == result.data_type &&
          hidden_weight.inner_shape[1] == command.input_width,
      "INDEXED_AFFINE_V1 hidden weight metadata does not match its sources");
  command.hidden_weight_binding_index = hidden_weight.binding_index;
  command.hidden_width = hidden_weight.inner_shape[0];

  if (has_hidden_bias) {
    command.hidden_bias_value_index = operands[operand_offset++];
    const ValueSpec& hidden_bias = plan.values[command.hidden_bias_value_index];
    TORCH_CHECK(hidden_bias.kind == ValueKind::kConstant &&
            hidden_bias.dimension_index < 0 && hidden_bias.inner_shape.size() == 1 &&
            hidden_bias.data_type == result.data_type &&
            hidden_bias.inner_shape[0] == command.hidden_width,
        "INDEXED_AFFINE_V1 hidden bias metadata does not match its hidden width");
  }

  command.output_weight_value_index = operands[operand_offset++];
  const ValueSpec& output_weight = plan.values[command.output_weight_value_index];
  TORCH_CHECK(output_weight.kind == ValueKind::kConstant &&
          output_weight.dimension_index < 0 && output_weight.inner_shape.size() == 2 &&
          output_weight.data_type == result.data_type &&
          output_weight.inner_shape[0] == command.output_width &&
          output_weight.inner_shape[1] == command.hidden_width,
      "INDEXED_AFFINE_V1 output weight metadata does not match its result");

  if (has_output_bias) {
    command.output_bias_value_index = operands[operand_offset++];
    const ValueSpec& output_bias = plan.values[command.output_bias_value_index];
    TORCH_CHECK(output_bias.kind == ValueKind::kConstant &&
            output_bias.dimension_index < 0 && output_bias.inner_shape.size() == 1 &&
            output_bias.data_type == result.data_type &&
            output_bias.inner_shape[0] == command.output_width,
        "INDEXED_AFFINE_V1 output bias metadata does not match its result");
  }
  TORCH_CHECK(operand_offset == operands.size(),
      "INDEXED_AFFINE_V1 operand parsing is inconsistent");

  result.kind = ValueKind::kComputed;
  result.producer_index = command_index;
  result.storage_index = command.result_storage_index;
  plan.storages.push_back(StorageSpec{result.data_type, result.maximum_shape});
  command.packed_input_storage_index = NextStorageIndex(plan);
  std::vector<int64_t> packed_input_shape{
      command.maximum_active_rows, command.input_width};
  TensorPayloadBytes(packed_input_shape, result.data_type,
      "INDEXED_AFFINE_V1 packed input payload size");
  plan.storages.push_back(
      StorageSpec{result.data_type, std::move(packed_input_shape)});
  command.hidden_storage_index = NextStorageIndex(plan);
  std::vector<int64_t> hidden_shape{
      command.maximum_active_rows, command.hidden_width};
  TensorPayloadBytes(hidden_shape, result.data_type,
      "INDEXED_AFFINE_V1 hidden payload size");
  plan.storages.push_back(StorageSpec{result.data_type, std::move(hidden_shape)});
  return command;
}

TransformerEncoderStackCommandSpec BuildTransformerEncoderStackCommand(
    FusionPlanData& plan, int32_t command_index,
    const std::vector<int32_t>& results,
    const std::vector<int32_t>& operands, int64_t flags,
    const CommandAttributes& attributes) {
  TORCH_CHECK(flags == 0,
      "TRANSFORMER_ENCODER_STACK_V1 does not support command flags");
  TORCH_CHECK(results.size() == 1,
      "TRANSFORMER_ENCODER_STACK_V1 requires exactly one result");
  TORCH_CHECK(attributes.size() == 5,
      "TRANSFORMER_ENCODER_STACK_V1 requires exactly five attributes");
  const int64_t block_count = GetRequiredInt64Scalar(attributes,
      kTransformerBlockCount, "TRANSFORMER_ENCODER_STACK_V1 block count");
  TORCH_CHECK(block_count > 0 && block_count <= 16,
      "TRANSFORMER_ENCODER_STACK_V1 block count exceeds the native limit");
  TORCH_CHECK(operands.size() ==
          static_cast<std::size_t>(1 + 13 * block_count),
      "TRANSFORMER_ENCODER_STACK_V1 operand count does not match its blocks");

  TransformerEncoderStackCommandSpec command;
  command.result_value_index = results[0];
  command.input_value_index = operands[0];
  command.attention_heads = GetRequiredInt64Scalar(attributes,
      kTransformerAttentionHeads,
      "TRANSFORMER_ENCODER_STACK_V1 attention heads");
  command.attention_width = GetRequiredInt64Scalar(attributes,
      kTransformerAttentionWidth,
      "TRANSFORMER_ENCODER_STACK_V1 attention width");
  command.feed_forward_width = GetRequiredInt64Scalar(attributes,
      kTransformerFeedForwardWidth,
      "TRANSFORMER_ENCODER_STACK_V1 feed-forward width");
  const double epsilon = GetRequiredFloat64Scalar(attributes,
      kTransformerEpsilon, "TRANSFORMER_ENCODER_STACK_V1 epsilon");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0 &&
          epsilon <= std::numeric_limits<float>::max(),
      "TRANSFORMER_ENCODER_STACK_V1 epsilon must be finite and positive");
  command.epsilon = static_cast<float>(epsilon);
  command.operand_value_indices = operands;

  ValueSpec& result = plan.values[command.result_value_index];
  const ValueSpec& input = plan.values[command.input_value_index];
  TORCH_CHECK(result.kind == ValueKind::kUnbound &&
          input.kind != ValueKind::kUnbound && input.dimension_index >= 0 &&
          IsFusionFloatingDataType(input.data_type) &&
          input.inner_shape.size() == 2 && result.data_type == input.data_type &&
          result.dimension_index == input.dimension_index &&
          result.inner_shape == input.inner_shape,
      "TRANSFORMER_ENCODER_STACK_V1 input and result metadata must match");
  command.extent_index = input.dimension_index;
  command.maximum_batches = plan.dimensions[input.dimension_index].maximum_extent;
  command.token_count = input.inner_shape[0];
  command.hidden_width = input.inner_shape[1];
  TORCH_CHECK(command.token_count > 0 && command.token_count <= 8 &&
          command.hidden_width == 256 && command.attention_heads == 4 &&
          command.attention_width == 128 &&
          command.feed_forward_width > 0,
      "TRANSFORMER_ENCODER_STACK_V1 native lowering requires tokens<=8, "
      "hidden=256, heads=4, and attention width=128");
  TORCH_CHECK(command.maximum_batches <=
          std::numeric_limits<uint32_t>::max() / command.token_count,
      "TRANSFORMER_ENCODER_STACK_V1 maximum batch exceeds the native grid limit");

  const auto require_vector = [&](int32_t value_index, int64_t width,
                                  bool allow_float32, const char* name) {
    const ValueSpec& value = plan.values[value_index];
    TORCH_CHECK(value.kind == ValueKind::kConstant &&
            value.dimension_index < 0 && value.inner_shape.size() == 1 &&
            value.inner_shape[0] == width &&
            (value.data_type == result.data_type ||
                (allow_float32 && value.data_type == torch::kFloat32)),
        "TRANSFORMER_ENCODER_STACK_V1 ", name,
        " metadata does not match the stack");
  };
  const auto require_projection = [&](int32_t value_index, int64_t rows,
                                      int64_t columns, const char* name) {
    const ValueSpec& value = plan.values[value_index];
    TORCH_CHECK(value.kind == ValueKind::kConstant &&
            value.dimension_index < 0 && value.inner_shape.size() == 2 &&
            value.inner_shape[0] == rows && value.inner_shape[1] == columns &&
            value.data_type == result.data_type,
        "TRANSFORMER_ENCODER_STACK_V1 ", name,
        " metadata does not match the stack");
  };

  command.blocks.reserve(static_cast<std::size_t>(block_count));
  std::size_t operand_offset = 1;
  for (int64_t block_index = 0; block_index < block_count; ++block_index) {
    TransformerEncoderBlockSpec block;
    for (int32_t constant_index = 0; constant_index < 13; ++constant_index) {
      block.value_indices[constant_index] = operands[operand_offset++];
    }
    require_vector(block.value_indices[0], command.hidden_width, true,
        "attention-input norm weight");
    require_vector(block.value_indices[1], command.hidden_width, true,
        "attention-input norm bias");
    TORCH_CHECK(plan.values[block.value_indices[0]].data_type ==
            plan.values[block.value_indices[1]].data_type,
        "TRANSFORMER_ENCODER_STACK_V1 attention-input norm types must match");
    require_projection(block.value_indices[2], 3 * command.attention_width,
        command.hidden_width, "QKV weight");
    require_projection(block.value_indices[3], command.hidden_width,
        command.attention_width, "attention output weight");
    require_vector(block.value_indices[4], command.hidden_width, false,
        "attention output bias");
    require_vector(block.value_indices[5], command.hidden_width, true,
        "feed-forward-input norm weight");
    require_vector(block.value_indices[6], command.hidden_width, true,
        "feed-forward-input norm bias");
    TORCH_CHECK(plan.values[block.value_indices[5]].data_type ==
            plan.values[block.value_indices[6]].data_type,
        "TRANSFORMER_ENCODER_STACK_V1 feed-forward-input norm types must match");
    require_projection(block.value_indices[7], command.feed_forward_width,
        command.hidden_width, "feed-forward expansion weight");
    require_vector(block.value_indices[8], command.feed_forward_width, false,
        "feed-forward expansion bias");
    require_projection(block.value_indices[9], command.hidden_width,
        command.feed_forward_width, "feed-forward projection weight");
    require_vector(block.value_indices[10], command.hidden_width, false,
        "feed-forward projection bias");
    require_vector(block.value_indices[11], command.hidden_width, true,
        "output norm weight");
    require_vector(block.value_indices[12], command.hidden_width, true,
        "output norm bias");
    TORCH_CHECK(plan.values[block.value_indices[11]].data_type ==
            plan.values[block.value_indices[12]].data_type,
        "TRANSFORMER_ENCODER_STACK_V1 output norm types must match");
    block.query_key_value_weight_binding_index =
        plan.values[block.value_indices[2]].binding_index;
    block.attention_output_weight_binding_index =
        plan.values[block.value_indices[3]].binding_index;
    block.feed_forward_expansion_weight_binding_index =
        plan.values[block.value_indices[7]].binding_index;
    block.feed_forward_projection_weight_binding_index =
        plan.values[block.value_indices[9]].binding_index;
    command.blocks.push_back(block);
  }

  result.kind = ValueKind::kComputed;
  result.producer_index = command_index;
  command.result_storage_index = NextStorageIndex(plan);
  result.storage_index = command.result_storage_index;
  plan.storages.push_back(StorageSpec{result.data_type, result.maximum_shape});

  command.normalized_storage_index = NextStorageIndex(plan);
  plan.storages.push_back(StorageSpec{result.data_type, result.maximum_shape});
  command.query_key_value_storage_index = NextStorageIndex(plan);
  plan.storages.push_back(StorageSpec{result.data_type,
      {command.maximum_batches, command.token_count,
          3 * command.attention_width}});
  command.expanded_storage_index = NextStorageIndex(plan);
  plan.storages.push_back(StorageSpec{result.data_type,
      {command.maximum_batches, command.token_count,
          command.feed_forward_width}});
  return command;
}

void ValidateOutputReachability(const FusionPlanData& plan) {
  std::vector<bool> reachable_values(plan.values.size(), false);
  std::vector<bool> reachable_commands(plan.commands.size(), false);
  std::vector<int32_t> work;
  work.reserve(plan.values.size());
  for (const auto& output : plan.outputs) {
    work.push_back(output.value_index);
  }
  while (!work.empty()) {
    const int32_t value_index = work.back();
    work.pop_back();
    if (reachable_values[value_index]) {
      continue;
    }
    reachable_values[value_index] = true;
    const ValueSpec& value = plan.values[value_index];
    if (value.kind != ValueKind::kComputed) {
      continue;
    }
    TORCH_CHECK(value.producer_index >= 0,
        "fusion computed value does not have a producer");
    const int32_t producer_index = value.producer_index;
    if (reachable_commands[producer_index]) {
      continue;
    }
    reachable_commands[producer_index] = true;
    std::visit(
        [&work](const auto& command) {
          for (int32_t operand : command.operand_value_indices) {
            work.push_back(operand);
          }
        },
        plan.commands[producer_index]);
  }
  for (bool reachable : reachable_commands) {
    TORCH_CHECK(reachable,
        "fusion descriptor contains a command that cannot reach an output");
  }
}

std::shared_ptr<const FusionPlanData> ParsePlan(
    c10::Device device, const int64_t* descriptor, std::size_t descriptor_size) {
  TORCH_CHECK(descriptor != nullptr, "fusion descriptor must not be null");
  TORCH_CHECK(descriptor_size >= kHeaderWords, "truncated fusion descriptor header");
  TORCH_CHECK(descriptor[0] == kDescriptorMagic, "invalid fusion descriptor magic");
  TORCH_CHECK(descriptor[1] == kDescriptorVersion,
      "unsupported fusion descriptor version");
  const std::size_t total_words = CheckedSize(descriptor[2],
      "fusion descriptor total size");
  TORCH_CHECK(total_words == descriptor_size,
      "fusion descriptor size does not match its header");
  TORCH_CHECK(descriptor[3] == 0, "unsupported fusion descriptor flags");

  const int32_t dimension_count = CheckedCount(descriptor[4], "fusion dimension count");
  const int32_t value_count = CheckedCount(descriptor[5], "fusion value count");
  const int32_t input_count = CheckedCount(descriptor[6], "fusion input count");
  const int32_t constant_count = CheckedCount(descriptor[7], "fusion constant count");
  const int32_t command_count = CheckedCount(descriptor[8], "fusion command count");
  const int32_t output_count = CheckedCount(descriptor[9], "fusion output count");
  TORCH_CHECK(value_count > 0, "fusion descriptor must contain values");
  TORCH_CHECK(command_count > 0, "fusion descriptor must contain commands");
  TORCH_CHECK(output_count > 0, "fusion descriptor must contain outputs");

  const std::size_t dimension_offset = CheckedSize(descriptor[10],
      "fusion dimension table offset");
  const std::size_t value_offset = CheckedSize(descriptor[11],
      "fusion value table offset");
  const std::size_t input_offset = CheckedSize(descriptor[12],
      "fusion input table offset");
  const std::size_t constant_offset = CheckedSize(descriptor[13],
      "fusion constant table offset");
  const std::size_t command_offset = CheckedSize(descriptor[14],
      "fusion command table offset");
  const std::size_t output_offset = CheckedSize(descriptor[15],
      "fusion output table offset");
  TORCH_CHECK(dimension_offset == kHeaderWords,
      "fusion dimension table must immediately follow the header");
  TORCH_CHECK(CheckedFixedTableEnd(dimension_offset, dimension_count,
          kDimensionRecordWords, total_words, "fusion dimension table") == value_offset,
      "fusion dimension table has gaps or overlaps");
  TORCH_CHECK(value_offset <= input_offset && input_offset <= constant_offset &&
          constant_offset <= command_offset && command_offset <= output_offset &&
          output_offset <= total_words,
      "fusion descriptor table offsets are not ordered");
  TORCH_CHECK(CheckedFixedTableEnd(input_offset, input_count, kBindingRecordWords,
          total_words, "fusion input table") == constant_offset,
      "fusion input table has gaps or overlaps");
  TORCH_CHECK(CheckedFixedTableEnd(constant_offset, constant_count, kBindingRecordWords,
          total_words, "fusion constant table") == command_offset,
      "fusion constant table has gaps or overlaps");
  TORCH_CHECK(CheckedFixedTableEnd(output_offset, output_count, kOutputRecordWords,
          total_words, "fusion output table") == total_words,
      "fusion output table is truncated or has trailing data");
  TORCH_CHECK(static_cast<std::size_t>(value_count) <=
          (input_offset - value_offset) / kValueRecordHeaderWords,
      "fusion value count exceeds the value table capacity");
  TORCH_CHECK(static_cast<std::size_t>(command_count) <=
          (output_offset - command_offset) / kCommandRecordHeaderWords,
      "fusion command count exceeds the command table capacity");

  auto plan = std::make_shared<FusionPlanData>(device);
  plan->input_count = input_count;
  plan->dimension_count = dimension_count;
  plan->constant_count = constant_count;
  plan->dimensions.resize(static_cast<std::size_t>(dimension_count));
  plan->values.resize(static_cast<std::size_t>(value_count));
  plan->input_value_indices.resize(static_cast<std::size_t>(input_count), -1);
  plan->constant_value_indices.resize(static_cast<std::size_t>(constant_count), -1);
  plan->commands.reserve(static_cast<std::size_t>(command_count));
  plan->storages.reserve(static_cast<std::size_t>(command_count));
  plan->outputs.resize(static_cast<std::size_t>(output_count), OutputSpec{-1, -1});

  DescriptorReader dimension_reader(
      descriptor + dimension_offset, descriptor + value_offset);
  for (int32_t record = 0; record < dimension_count; ++record) {
    TORCH_CHECK(dimension_reader.Read() == static_cast<int64_t>(kDimensionRecordWords),
        "invalid fusion dimension record size");
    TORCH_CHECK(dimension_reader.Read() == kDimensionPrefixExtent,
        "unsupported fusion dimension kind");
    const int64_t maximum_extent = dimension_reader.Read();
    TORCH_CHECK(maximum_extent > 0, "fusion dimension maximum must be positive");
    plan->dimensions[record] = DimensionSpec{maximum_extent};
  }
  TORCH_CHECK(dimension_reader.IsComplete(), "invalid fusion dimension table size");

  std::vector<bool> assigned_values(static_cast<std::size_t>(value_count), false);
  DescriptorReader value_reader(descriptor + value_offset, descriptor + input_offset);
  for (int32_t record = 0; record < value_count; ++record) {
    TORCH_CHECK(value_reader.Remaining() >= kValueRecordHeaderWords,
        "truncated fusion value record");
    const std::size_t record_words = CheckedSize(value_reader.Read(),
        "fusion value record size");
    TORCH_CHECK(record_words >= kValueRecordHeaderWords &&
            record_words <= value_reader.Remaining() + 1,
        "invalid fusion value record size");
    const int32_t index = CheckedIndex(value_reader.Read(), plan->values.size(),
        "fusion value index");
    ValueSpec value;
    value.data_type = ScalarTypeFromWireCode(value_reader.Read());
    TORCH_CHECK(value_reader.Read() == kLayoutContiguous,
        "unsupported fusion value layout");
    const int64_t dimension_index = value_reader.Read();
    if (dimension_index == -1) {
      value.dimension_index = -1;
    } else {
      value.dimension_index = CheckedIndex(dimension_index, plan->dimensions.size(),
          "fusion value dimension index");
    }
    const std::size_t inner_rank = CheckedSize(value_reader.Read(),
        "fusion value inner rank");
    TORCH_CHECK(value_reader.Read() == 0, "unsupported fusion value flags");
    TORCH_CHECK(inner_rank == record_words - kValueRecordHeaderWords,
        "fusion value record size does not match its rank");
    TORCH_CHECK(inner_rank <= value_reader.Remaining(), "truncated fusion value shape");
    value.inner_shape.reserve(inner_rank);
    value.maximum_shape.reserve(inner_rank + (value.dimension_index >= 0 ? 1 : 0));
    if (value.dimension_index >= 0) {
      value.maximum_shape.push_back(plan->dimensions[value.dimension_index].maximum_extent);
    }
    for (std::size_t axis = 0; axis < inner_rank; ++axis) {
      const int64_t extent = value_reader.Read();
      TORCH_CHECK(extent > 0, "fusion value dimensions must be positive");
      value.inner_shape.push_back(extent);
      value.maximum_shape.push_back(extent);
    }
    TensorPayloadBytes(value.maximum_shape, value.data_type,
        "fusion value maximum payload size");
    TORCH_CHECK(!assigned_values[index], "duplicate fusion value index");
    plan->values[index] = std::move(value);
    assigned_values[index] = true;
  }
  TORCH_CHECK(value_reader.IsComplete(), "invalid fusion value table size");

  std::vector<bool> assigned_inputs(static_cast<std::size_t>(input_count), false);
  DescriptorReader input_reader(descriptor + input_offset, descriptor + constant_offset);
  for (int32_t record = 0; record < input_count; ++record) {
    const int32_t binding_index = CheckedIndex(input_reader.Read(),
        plan->input_value_indices.size(), "fusion input binding index");
    const int32_t value_index = CheckedIndex(input_reader.Read(), plan->values.size(),
        "fusion input value index");
    TORCH_CHECK(!assigned_inputs[binding_index], "duplicate fusion input binding index");
    ValueSpec& value = plan->values[value_index];
    TORCH_CHECK(value.kind == ValueKind::kUnbound,
        "fusion value has more than one binding or producer");
    value.kind = ValueKind::kInput;
    value.binding_index = binding_index;
    plan->input_value_indices[binding_index] = value_index;
    assigned_inputs[binding_index] = true;
  }
  TORCH_CHECK(input_reader.IsComplete(), "invalid fusion input table size");

  std::vector<bool> assigned_constants(static_cast<std::size_t>(constant_count), false);
  DescriptorReader constant_reader(descriptor + constant_offset, descriptor + command_offset);
  for (int32_t record = 0; record < constant_count; ++record) {
    const int32_t binding_index = CheckedIndex(constant_reader.Read(),
        plan->constant_value_indices.size(), "fusion constant binding index");
    const int32_t value_index = CheckedIndex(constant_reader.Read(), plan->values.size(),
        "fusion constant value index");
    TORCH_CHECK(!assigned_constants[binding_index],
        "duplicate fusion constant binding index");
    ValueSpec& value = plan->values[value_index];
    TORCH_CHECK(value.kind == ValueKind::kUnbound,
        "fusion value has more than one binding or producer");
    value.kind = ValueKind::kConstant;
    value.binding_index = binding_index;
    plan->constant_value_indices[binding_index] = value_index;
    assigned_constants[binding_index] = true;
  }
  TORCH_CHECK(constant_reader.IsComplete(), "invalid fusion constant table size");

  DescriptorReader command_reader(descriptor + command_offset, descriptor + output_offset);
  for (int32_t command_index = 0; command_index < command_count; ++command_index) {
    TORCH_CHECK(command_reader.Remaining() >= kCommandRecordHeaderWords,
        "truncated fusion command record");
    const std::size_t record_words = CheckedSize(command_reader.Read(),
        "fusion command record size");
    TORCH_CHECK(record_words >= kCommandRecordHeaderWords &&
            record_words <= command_reader.Remaining() + 1,
        "invalid fusion command record size");
    DescriptorReader record_reader(
        command_reader.Current(), command_reader.Current() + record_words - 1);
    const int64_t opcode = record_reader.Read();
    const int64_t flags = record_reader.Read();
    const int32_t result_count = CheckedCount(record_reader.Read(),
        "fusion command result count");
    const int32_t operand_count = CheckedCount(record_reader.Read(),
        "fusion command operand count");
    const int32_t attribute_count = CheckedCount(record_reader.Read(),
        "fusion command attribute count");
    TORCH_CHECK(static_cast<std::size_t>(result_count) + operand_count <=
            record_reader.Remaining(),
        "truncated fusion command values");
    std::vector<int32_t> results;
    results.reserve(static_cast<std::size_t>(result_count));
    std::unordered_set<int32_t> unique_results;
    for (int32_t index = 0; index < result_count; ++index) {
      const int32_t value_index = CheckedIndex(record_reader.Read(), plan->values.size(),
          "fusion command result value index");
      TORCH_CHECK(unique_results.insert(value_index).second,
          "duplicate result in a fusion command");
      TORCH_CHECK(plan->values[value_index].kind == ValueKind::kUnbound,
          "fusion command result already has a producer or binding");
      results.push_back(value_index);
    }
    std::vector<int32_t> operands;
    operands.reserve(static_cast<std::size_t>(operand_count));
    for (int32_t index = 0; index < operand_count; ++index) {
      const int32_t value_index = CheckedIndex(record_reader.Read(), plan->values.size(),
          "fusion command operand value index");
      TORCH_CHECK(plan->values[value_index].kind != ValueKind::kUnbound,
          "fusion command operand is not topologically available");
      operands.push_back(value_index);
    }
    CommandAttributes attributes = ReadAttributeRecords(record_reader, attribute_count);
    TORCH_CHECK(record_reader.IsComplete(),
        "fusion command record size does not match its payload");
    command_reader = DescriptorReader(
        record_reader.Current(), descriptor + output_offset);

    switch (opcode) {
      case kOutputPackV1:
        plan->commands.emplace_back(BuildOutputPackCommand(
            *plan, command_index, results, operands, flags, attributes));
        break;
      case kAffineSumV1:
        plan->commands.emplace_back(BuildAffineSumCommand(
            *plan, command_index, results, operands, flags, attributes));
        break;
      case kIndexedAffineV1:
        plan->commands.emplace_back(BuildIndexedAffineCommand(
            *plan, command_index, results, operands, flags, attributes));
        break;
      case kTransformerEncoderStackV1:
        plan->commands.emplace_back(BuildTransformerEncoderStackCommand(
            *plan, command_index, results, operands, flags, attributes));
        break;
      default:
        TORCH_CHECK(false, "unsupported fusion command opcode: ", opcode);
    }
  }
  TORCH_CHECK(command_reader.IsComplete(), "invalid fusion command table size");
  for (const auto& value : plan->values) {
    TORCH_CHECK(value.kind != ValueKind::kUnbound,
        "fusion value does not have a binding or producer");
  }

  std::vector<bool> assigned_outputs(static_cast<std::size_t>(output_count), false);
  DescriptorReader output_reader(descriptor + output_offset, descriptor + total_words);
  for (int32_t record = 0; record < output_count; ++record) {
    const int32_t output_index = CheckedIndex(output_reader.Read(), plan->outputs.size(),
        "fusion output index");
    const int32_t value_index = CheckedIndex(output_reader.Read(), plan->values.size(),
        "fusion output value index");
    TORCH_CHECK(!assigned_outputs[output_index], "duplicate fusion output index");
    const ValueSpec& value = plan->values[value_index];
    TORCH_CHECK(value.kind == ValueKind::kComputed && value.storage_index >= 0,
        "fusion outputs must export a computed persistent value");
    plan->outputs[output_index] = OutputSpec{value_index, value.storage_index};
    assigned_outputs[output_index] = true;
  }
  TORCH_CHECK(output_reader.IsComplete(), "invalid fusion output table size");
  ValidateOutputReachability(*plan);
  return plan;
}

const torch::Tensor& ResolveValue(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, int32_t value_index);

}  // namespace

struct FusionPlan {
  explicit FusionPlan(std::shared_ptr<const FusionPlanData> data) : data(std::move(data)) {}

  std::shared_ptr<const FusionPlanData> data;
};

struct FusionExecutable {
  explicit FusionExecutable(std::shared_ptr<const FusionExecutableData> data)
      : data(std::move(data)) {}

  std::shared_ptr<const FusionExecutableData> data;
};

struct FusionSession {
  FusionSession(std::shared_ptr<const FusionExecutableData> executable, int32_t buffer_count)
      : executable(std::move(executable)), storages(static_cast<std::size_t>(buffer_count)),
        allocation_waited(static_cast<std::size_t>(buffer_count), false),
        binding_waited(static_cast<std::size_t>(buffer_count), false),
        submission_streams(static_cast<std::size_t>(buffer_count)) {
    const auto& plan = this->executable->plan;
    c10::DeviceGuard device_guard(plan->device);
    const torch::TensorOptions options = torch::TensorOptions().device(plan->device);
    for (auto& buffer : storages) {
      buffer.reserve(plan->storages.size());
      for (const auto& storage : plan->storages) {
        buffer.emplace_back(torch::empty(
            storage.maximum_shape, options.dtype(storage.data_type)));
      }
    }
    c10::impl::VirtualGuardImpl guard_impl(plan->device.type());
    allocation_ready = std::make_unique<c10::Event>(plan->device.type());
    allocation_ready->record(guard_impl.getStream(plan->device));
    retained_inputs.reserve(static_cast<std::size_t>(plan->input_count));
  }

  void WaitForAllocation(int32_t buffer_index, const c10::Stream& stream) {
    if (!allocation_waited[buffer_index]) {
      allocation_ready->block(stream);
      allocation_waited[buffer_index] = true;
    }
  }

  void WaitForBinding(int32_t buffer_index, const c10::Stream& stream) {
    if (!binding_waited[buffer_index]) {
      executable->binding_ready->block(stream);
      binding_waited[buffer_index] = true;
    }
  }

  void SetSubmissionStream(int32_t buffer_index, const c10::Stream& stream) {
    submission_streams[buffer_index] = stream;
  }

  void Synchronize(int32_t buffer_index) {
    TORCH_CHECK(submission_streams[buffer_index].has_value(),
        "fusion output slot has not been submitted");
    c10::Event completion(executable->plan->device.type());
    completion.record(*submission_streams[buffer_index]);
    completion.synchronize();
  }

  void RetainIncompleteSubmission(
      const c10::Stream& stream, const int64_t* input_handles, std::size_t input_count) {
    retained_inputs.clear();
    for (std::size_t index = 0; index < input_count; ++index) {
      const auto* input = reinterpret_cast<const torch::Tensor*>(input_handles[index]);
      retained_inputs.emplace_back(*input);
    }
    incomplete_submission_stream = stream;
    poisoned = true;
  }

  void DrainIncompleteSubmission() {
    TORCH_CHECK(incomplete_submission_stream.has_value(),
        "poisoned fusion session has no incomplete submission stream");
    incomplete_submission_stream->synchronize();
    retained_inputs.clear();
    incomplete_submission_stream.reset();
    poisoned = false;
  }

  std::shared_ptr<const FusionExecutableData> executable;
  std::vector<std::vector<torch::Tensor>> storages;
  std::unique_ptr<c10::Event> allocation_ready;
  std::vector<bool> allocation_waited;
  std::vector<bool> binding_waited;
  std::vector<std::optional<c10::Stream>> submission_streams;
  std::vector<torch::Tensor> retained_inputs;
  std::optional<c10::Stream> incomplete_submission_stream;
  bool poisoned = false;
};

namespace {

const torch::Tensor& ResolveValue(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, int32_t value_index) {
  const auto& plan = *session.executable->plan;
  const ValueSpec& value = plan.values[value_index];
  switch (value.kind) {
    case ValueKind::kInput: {
      const auto* input = reinterpret_cast<const torch::Tensor*>(
          input_handles[value.binding_index]);
      TORCH_CHECK(input != nullptr, "fusion input handle must not be null");
      return *input;
    }
    case ValueKind::kConstant:
      return session.executable->constants[value.binding_index];
    case ValueKind::kComputed:
      return session.storages[buffer_index][value.storage_index];
    case ValueKind::kUnbound:
      TORCH_CHECK(false, "fusion value is not bound or computed");
  }
  std::terminate();
}

void ValidateCommandSubmission(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const OutputPackCommandSpec& command) {
  const auto& plan = *session.executable->plan;
  for (const auto& source : command.sources) {
    const torch::Tensor& tensor = ResolveValue(
        session, buffer_index, input_handles, source.value_index);
    const ValueSpec& source_spec = plan.values[source.value_index];
    ValidateTensorMetadata(tensor, plan, source_spec, dimensions,
        source_spec.kind == ValueKind::kComputed, "operand");
  }
  const torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  ValidateTensorMetadata(result, plan, plan.values[command.result_value_index],
      dimensions, true, "result storage");
}

void ValidateCommandSubmission(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const AffineSumCommandSpec& command) {
  const auto& plan = *session.executable->plan;
  for (const auto& group : command.groups) {
    for (const auto& term : group.terms) {
      const torch::Tensor& input = ResolveValue(
          session, buffer_index, input_handles, term.input_value_index);
      const ValueSpec& input_spec = plan.values[term.input_value_index];
      ValidateTensorMetadata(input, plan, input_spec, dimensions,
          input_spec.kind == ValueKind::kComputed, "affine input");
    }
  }
  const torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  ValidateTensorMetadata(result, plan, plan.values[command.result_value_index],
      dimensions, true, "affine result storage");
}

void ValidateCommandSubmission(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const IndexedAffineCommandSpec& command) {
  const auto& plan = *session.executable->plan;
  const torch::Tensor& indices = ResolveValue(
      session, buffer_index, input_handles, command.indices_value_index);
  ValidateTensorMetadata(indices, plan, plan.values[command.indices_value_index],
      dimensions, false, "indexed-affine indices");
  for (const auto& source : command.sources) {
    const torch::Tensor& input = ResolveValue(
        session, buffer_index, input_handles, source.value_index);
    ValidateTensorMetadata(input, plan, plan.values[source.value_index],
        dimensions, plan.values[source.value_index].kind == ValueKind::kComputed,
        "indexed-affine source");
  }
  const torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  ValidateTensorMetadata(result, plan, plan.values[command.result_value_index],
      dimensions, true, "indexed-affine result storage");
}

void ValidateCommandSubmission(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const TransformerEncoderStackCommandSpec& command) {
  const auto& plan = *session.executable->plan;
  const torch::Tensor& input = ResolveValue(
      session, buffer_index, input_handles, command.input_value_index);
  ValidateTensorMetadata(input, plan, plan.values[command.input_value_index],
      dimensions, false, "transformer encoder input");
  const torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  ValidateTensorMetadata(result, plan, plan.values[command.result_value_index],
      dimensions, true, "transformer encoder result storage");
}

void ValidateSubmission(const FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions) {
  const auto& plan = *session.executable->plan;
  for (std::size_t index = 0; index < plan.dimensions.size(); ++index) {
    TORCH_CHECK(dimensions[index] >= 0 &&
            dimensions[index] <= plan.dimensions[index].maximum_extent,
        "fusion active extent is outside the prepared range");
  }
  for (std::size_t index = 0; index < plan.input_value_indices.size(); ++index) {
    const auto* input = reinterpret_cast<const torch::Tensor*>(input_handles[index]);
    TORCH_CHECK(input != nullptr, "fusion input handle must not be null");
    ValidateTensorMetadata(*input, plan,
        plan.values[plan.input_value_indices[index]], dimensions, false, "input");
  }
  for (const auto& fusion_command : plan.commands) {
    std::visit(
        [&](const auto& command) {
          ValidateCommandSubmission(
              session, buffer_index, input_handles, dimensions, command);
        },
        fusion_command);
  }
}

void ExecuteCommand(FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const OutputPackCommandSpec& command, bool& work_submitted) {
  std::array<OutputPackSource, kMaximumOutputPackSources> launch_sources;
  const int64_t row_count = dimensions[command.extent_index];
  torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  for (std::size_t source_index = 0; source_index < command.sources.size(); ++source_index) {
    const auto& source = command.sources[source_index];
    const torch::Tensor& tensor = ResolveValue(
        session, buffer_index, input_handles, source.value_index);
    launch_sources[source_index] = OutputPackSource{tensor.data_ptr(),
        source.data_type, source.width, source.destination_offset};
    RecordCurrentStream(tensor);
  }
  RecordCurrentStream(result);
  if (row_count != 0) {
    work_submitted = true;
  }
  LaunchOutputPack(launch_sources.data(), static_cast<int32_t>(command.sources.size()),
      result, row_count, command.output_width);
}

torch::Tensor ActiveAffineMatrix(const torch::Tensor& tensor, int64_t batch_count,
    int64_t rows_per_batch, int64_t width) {
  return tensor.narrow(0, 0, batch_count).view(
      {CheckedMultiply(batch_count, rows_per_batch,
           "AFFINE_SUM_V1 active matrix rows"),
       width});
}

void ExecuteCommand(FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const AffineSumCommandSpec& command, std::size_t command_index,
    bool& work_submitted) {
  const int64_t batch_count = dimensions[command.extent_index];
  if (batch_count == 0) {
    return;
  }
  std::array<AffineInputPackSource, kMaximumAffineTerms> pack_sources;
  std::array<AffineSumSource, kMaximumAffineGroups> sum_sources;
  const auto& packed_weights = session.executable->affine_weights[command_index];
  TORCH_CHECK(packed_weights.size() == command.groups.size(),
      "AFFINE_SUM_V1 executable weight groups are inconsistent");

  for (std::size_t group_index = 0;
       group_index < command.groups.size(); ++group_index) {
    const auto& group = command.groups[group_index];
    if (group.precompute_at_bind) {
      const torch::Tensor& product =
          session.executable->affine_products[command_index][group_index];
      TORCH_CHECK(product.defined(),
          "AFFINE_SUM_V1 precomputed product is not bound");
      RecordCurrentStream(product);
      sum_sources[group_index].data = product.data_ptr();
      sum_sources[group_index].rows_per_batch = 0;
      std::copy(group.output_strides.begin(), group.output_strides.end(),
          sum_sources[group_index].output_strides);
      continue;
    }
    const int64_t matrix_batch_count = group.dynamic_leading ? batch_count : 1;
    const torch::Tensor* group_input;
    if (!group.requires_input_pack) {
      TORCH_CHECK(group.terms.size() == 1,
          "AFFINE_SUM_V1 unpacked group must contain exactly one term");
      group_input = &ResolveValue(session, buffer_index, input_handles,
          group.terms[0].input_value_index);
      RecordCurrentStream(*group_input);
    } else {
      torch::Tensor& packed_input = session.storages[buffer_index]
          [group.packed_input_storage_index];
      int64_t destination_offset = 0;
      for (std::size_t term_index = 0;
           term_index < group.terms.size(); ++term_index) {
        const auto& term = group.terms[term_index];
        const torch::Tensor& input = ResolveValue(
            session, buffer_index, input_handles, term.input_value_index);
        pack_sources[term_index] = AffineInputPackSource{input.data_ptr(),
            term.input_data_type, term.input_width, destination_offset};
        destination_offset += term.input_width;
        RecordCurrentStream(input);
      }
      RecordCurrentStream(packed_input);
      work_submitted = true;
      LaunchAffineInputPack(pack_sources.data(),
          static_cast<int32_t>(group.terms.size()), packed_input,
          CheckedMultiply(matrix_batch_count, group.rows_per_batch,
              "AFFINE_SUM_V1 packed input rows"),
          group.packed_input_width);
      group_input = &packed_input;
    }

    torch::Tensor& product = session.storages[buffer_index]
        [group.product_storage_index];
    const torch::Tensor& packed_weight = packed_weights[group_index];
    RecordCurrentStream(product);
    RecordCurrentStream(packed_weight);
    torch::Tensor input_matrix = ActiveAffineMatrix(
        *group_input, matrix_batch_count, group.rows_per_batch,
        group.packed_input_width);
    torch::Tensor product_matrix = ActiveAffineMatrix(
        product, matrix_batch_count, group.rows_per_batch, command.output_width);
    work_submitted = true;
    at::mm_out(product_matrix, input_matrix, packed_weight);

    sum_sources[group_index].data = product.data_ptr();
    sum_sources[group_index].rows_per_batch =
        group.dynamic_leading ? group.rows_per_batch : 0;
    std::copy(group.output_strides.begin(), group.output_strides.end(),
        sum_sources[group_index].output_strides);
  }

  if (command.needs_finalize) {
    const void* bias = nullptr;
    if (command.bias_value_index >= 0) {
      const torch::Tensor& bias_tensor = ResolveValue(
          session, buffer_index, input_handles, command.bias_value_index);
      bias = bias_tensor.data_ptr();
      RecordCurrentStream(bias_tensor);
    }
    torch::Tensor& result = session.storages[buffer_index]
        [command.result_storage_index];
    RecordCurrentStream(result);
    work_submitted = true;
    LaunchAffineFinalize(sum_sources.data(),
        static_cast<int32_t>(command.groups.size()), bias, result, batch_count,
        command.output_prefix_shape.data(),
        static_cast<int32_t>(command.output_prefix_shape.size()),
        command.output_width, command.activation);
  }
}

void ExecuteCommand(FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const IndexedAffineCommandSpec& command, std::size_t command_index,
    bool& work_submitted) {
  const int64_t active_rows = dimensions[command.active_extent_index];
  const int64_t destination_rows = dimensions[command.destination_extent_index];
  TORCH_CHECK(active_rows <= destination_rows,
      "INDEXED_AFFINE_V1 active row count exceeds destination rows");
  const torch::Tensor& indices = ResolveValue(
      session, buffer_index, input_handles, command.indices_value_index);
  torch::Tensor& packed_input = session.storages[buffer_index]
      [command.packed_input_storage_index];
  torch::Tensor& hidden = session.storages[buffer_index]
      [command.hidden_storage_index];
  torch::Tensor& result = session.storages[buffer_index]
      [command.result_storage_index];
  RecordCurrentStream(result);

  if (active_rows != 0) {
    std::array<IndexedAffineSource, kMaximumIndexedAffineSources> launch_sources;
    for (std::size_t source_index = 0;
         source_index < command.sources.size(); ++source_index) {
      const auto& source = command.sources[source_index];
      const torch::Tensor& input = ResolveValue(
          session, buffer_index, input_handles, source.value_index);
      const int64_t source_rows = dimensions[source.extent_index];
      TORCH_CHECK(destination_rows == 0 ||
              source_rows >= (destination_rows - 1) / source.index_divisor + 1,
          "INDEXED_AFFINE_V1 active source extent does not cover destination rows");
      launch_sources[source_index] = IndexedAffineSource{input.data_ptr(),
          source.data_type, source.width, source.destination_offset,
          source.index_divisor, source_rows};
      RecordCurrentStream(input);
    }
    RecordCurrentStream(indices);
    RecordCurrentStream(packed_input);
    work_submitted = true;
    LaunchIndexedAffineInputPack(launch_sources.data(),
        static_cast<int32_t>(command.sources.size()), indices.data_ptr(),
        command.index_data_type, packed_input, active_rows, destination_rows,
        command.input_width);

    const torch::Tensor& packed_weight =
        session.executable->indexed_affine_weights[command_index];
    TORCH_CHECK(packed_weight.defined(),
        "INDEXED_AFFINE_V1 packed hidden weight is not bound");
    RecordCurrentStream(packed_weight);
    RecordCurrentStream(hidden);
    torch::Tensor input_matrix = packed_input.narrow(0, 0, active_rows);
    torch::Tensor hidden_matrix = hidden.narrow(0, 0, active_rows);
    at::mm_out(hidden_matrix, input_matrix, packed_weight);
  }

  const void* hidden_bias = nullptr;
  if (command.hidden_bias_value_index >= 0) {
    const torch::Tensor& bias = ResolveValue(
        session, buffer_index, input_handles, command.hidden_bias_value_index);
    hidden_bias = bias.data_ptr();
    RecordCurrentStream(bias);
  }
  const torch::Tensor& output_weight = ResolveValue(
      session, buffer_index, input_handles, command.output_weight_value_index);
  RecordCurrentStream(output_weight);
  const void* output_bias = nullptr;
  if (command.output_bias_value_index >= 0) {
    const torch::Tensor& bias = ResolveValue(
        session, buffer_index, input_handles, command.output_bias_value_index);
    output_bias = bias.data_ptr();
    RecordCurrentStream(bias);
  }
  if (active_rows != 0) {
    RecordCurrentStream(hidden);
    RecordCurrentStream(indices);
  }
  if (destination_rows != 0) {
    work_submitted = true;
  }
  LaunchIndexedAffineFinalize(indices.data_ptr(), command.index_data_type, hidden,
      hidden_bias, output_weight.data_ptr(), output_bias, result, active_rows,
      destination_rows, command.hidden_width, command.output_width,
      command.activation);
}

void ExecuteCommand(FusionSession& session, int32_t buffer_index,
    const int64_t* input_handles, const int64_t* dimensions,
    const TransformerEncoderStackCommandSpec& command,
    std::size_t command_index, bool& work_submitted) {
  const int64_t batch_count = dimensions[command.extent_index];
  if (batch_count == 0) {
    return;
  }
  work_submitted = true;
  const torch::Tensor& input = ResolveValue(
      session, buffer_index, input_handles, command.input_value_index);
  torch::Tensor& state = session.storages[buffer_index]
      [command.result_storage_index];
  torch::Tensor& normalized = session.storages[buffer_index]
      [command.normalized_storage_index];
  torch::Tensor& query_key_value = session.storages[buffer_index]
      [command.query_key_value_storage_index];
  torch::Tensor& expanded = session.storages[buffer_index]
      [command.expanded_storage_index];
  const auto& executable_blocks =
      session.executable->transformer_encoder_weights[command_index];
  TORCH_CHECK(executable_blocks.size() == command.blocks.size(),
      "TRANSFORMER_ENCODER_STACK_V1 executable blocks are inconsistent");

  RecordCurrentStream(input);
  RecordCurrentStream(state);
  RecordCurrentStream(normalized);
  RecordCurrentStream(query_key_value);
  RecordCurrentStream(expanded);
  const int64_t active_rows = CheckedMultiply(batch_count,
      command.token_count, "TRANSFORMER_ENCODER_STACK_V1 active rows");
  torch::Tensor normalized_matrix = normalized.narrow(0, 0, batch_count).view(
      {active_rows, command.hidden_width});
  torch::Tensor query_key_value_matrix =
      query_key_value.narrow(0, 0, batch_count).view(
          {active_rows, 3 * command.attention_width});
  torch::Tensor attention_context_matrix = normalized.narrow(0, 0, batch_count)
      .view({active_rows * command.hidden_width})
      .narrow(0, 0, active_rows * command.attention_width)
      .view({active_rows, command.attention_width});
  torch::Tensor attention_output_matrix =
      query_key_value.narrow(0, 0, batch_count)
          .view({active_rows * 3 * command.attention_width})
          .narrow(0, 0, active_rows * command.hidden_width)
          .view({active_rows, command.hidden_width});
  torch::Tensor expanded_matrix = expanded.narrow(0, 0, batch_count).view(
      {active_rows, command.feed_forward_width});

  for (std::size_t block_index = 0;
       block_index < command.blocks.size(); ++block_index) {
    const auto& block = command.blocks[block_index];
    const auto& weights = executable_blocks[block_index];
    for (const torch::Tensor& weight : weights) {
      RecordCurrentStream(weight);
    }
    const torch::Tensor& attention_input_weight = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[0]);
    const torch::Tensor& attention_input_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[1]);
    const torch::Tensor& attention_output_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[4]);
    const torch::Tensor& feed_forward_input_weight = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[5]);
    const torch::Tensor& feed_forward_input_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[6]);
    const torch::Tensor& expansion_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[8]);
    const torch::Tensor& projection_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[10]);
    const torch::Tensor& output_weight = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[11]);
    const torch::Tensor& output_bias = ResolveValue(session,
        buffer_index, input_handles, block.value_indices[12]);
    RecordCurrentStream(attention_input_weight);
    RecordCurrentStream(attention_input_bias);
    RecordCurrentStream(attention_output_bias);
    RecordCurrentStream(feed_forward_input_weight);
    RecordCurrentStream(feed_forward_input_bias);
    RecordCurrentStream(expansion_bias);
    RecordCurrentStream(projection_bias);
    RecordCurrentStream(output_weight);
    RecordCurrentStream(output_bias);

    const torch::Tensor& layer_input = block_index == 0 ? input : state;
    LaunchTransformerCopyAndLayerNorm(layer_input, state, normalized,
        attention_input_weight, attention_input_bias, batch_count,
        command.token_count, command.hidden_width, command.epsilon,
        block_index == 0);
    at::mm_out(query_key_value_matrix, normalized_matrix, weights[0]);
    LaunchTransformerAttention(query_key_value, normalized, batch_count,
        command.token_count, command.attention_heads, command.attention_width,
        command.hidden_width);
    at::mm_out(
        attention_output_matrix, attention_context_matrix, weights[1]);
    LaunchTransformerResidualLayerNorm(state, attention_output_matrix,
        attention_output_bias, feed_forward_input_weight,
        feed_forward_input_bias, normalized, active_rows,
        command.hidden_width, command.epsilon);
    at::mm_out(expanded_matrix, normalized_matrix, weights[2]);
    LaunchTransformerBiasSilu(
        expanded, expansion_bias, active_rows, command.feed_forward_width);
    at::mm_out(normalized_matrix, expanded_matrix, weights[3]);
    LaunchTransformerResidualLayerNorm(state, normalized,
        projection_bias, output_weight, output_bias, state, active_rows,
        command.hidden_width, command.epsilon);
  }
}

}  // namespace

FusionPlan* PrepareFusionPlan(
    c10::Device device, const int64_t* descriptor, std::size_t descriptor_size) {
#if defined(DJL_USE_ROCM_KERNELS)
  TORCH_CHECK(device.is_cuda(), "PyTorch fusion requires a ROCm device");
  c10::DeviceGuard device_guard(device);
  return new FusionPlan(ParsePlan(device, descriptor, descriptor_size));
#else
  TORCH_CHECK(false, "PyTorch fusion requires a ROCm build");
#endif
}

FusionExecutable* BindFusionPlan(const FusionPlan* plan,
    const int64_t* constant_handles, std::size_t constant_count) {
  TORCH_CHECK(plan != nullptr, "fusion plan must not be null");
  TORCH_CHECK(constant_count == static_cast<std::size_t>(plan->data->constant_count),
      "fusion constant count does not match the prepared plan");
  c10::DeviceGuard device_guard(plan->data->device);
  c10::InferenceMode inference_mode;
  auto data = std::make_shared<FusionExecutableData>();
  data->plan = plan->data;
  data->constants.reserve(constant_count);
  for (std::size_t index = 0; index < constant_count; ++index) {
    const auto* constant = reinterpret_cast<const torch::Tensor*>(constant_handles[index]);
    TORCH_CHECK(constant != nullptr, "fusion constant handle must not be null");
    const ValueSpec& spec = plan->data->values[plan->data->constant_value_indices[index]];
    ValidateTensorMetadata(*constant, *plan->data, spec, nullptr, true, "constant");
    data->constants.push_back(*constant);
  }
  data->affine_weights.resize(plan->data->commands.size());
  data->affine_products.resize(plan->data->commands.size());
  data->indexed_affine_weights.resize(plan->data->commands.size());
  data->transformer_encoder_weights.resize(plan->data->commands.size());
  for (std::size_t command_index = 0;
       command_index < plan->data->commands.size(); ++command_index) {
    const auto* command = std::get_if<AffineSumCommandSpec>(
        &plan->data->commands[command_index]);
    if (command == nullptr) {
      const auto* indexed_command = std::get_if<IndexedAffineCommandSpec>(
          &plan->data->commands[command_index]);
      if (indexed_command != nullptr) {
        const torch::Tensor& weight =
            data->constants[indexed_command->hidden_weight_binding_index];
        RecordCurrentStream(weight);
        data->indexed_affine_weights[command_index] =
            weight.transpose(0, 1).contiguous();
        RecordCurrentStream(data->indexed_affine_weights[command_index]);
      }
      const auto* transformer_command =
          std::get_if<TransformerEncoderStackCommandSpec>(
              &plan->data->commands[command_index]);
      if (transformer_command != nullptr) {
        auto& executable_blocks =
            data->transformer_encoder_weights[command_index];
        executable_blocks.resize(transformer_command->blocks.size());
        for (std::size_t block_index = 0;
             block_index < transformer_command->blocks.size(); ++block_index) {
          const auto& block = transformer_command->blocks[block_index];
          const std::array<int32_t, 4> binding_indices{
              block.query_key_value_weight_binding_index,
              block.attention_output_weight_binding_index,
              block.feed_forward_expansion_weight_binding_index,
              block.feed_forward_projection_weight_binding_index};
          for (std::size_t weight_index = 0;
               weight_index < binding_indices.size(); ++weight_index) {
            const torch::Tensor& weight =
                data->constants[binding_indices[weight_index]];
            RecordCurrentStream(weight);
            executable_blocks[block_index][weight_index] =
                weight.transpose(0, 1).contiguous();
            RecordCurrentStream(executable_blocks[block_index][weight_index]);
          }
        }
      }
      continue;
    }
    auto& packed_groups = data->affine_weights[command_index];
    auto& product_groups = data->affine_products[command_index];
    packed_groups.resize(command->groups.size());
    product_groups.resize(command->groups.size());
    const torch::ScalarType projection_data_type =
        plan->data->values[command->result_value_index].data_type;
    for (std::size_t group_index = 0;
         group_index < command->groups.size(); ++group_index) {
      const auto& group = command->groups[group_index];
      std::vector<torch::Tensor> transposed_weights;
      transposed_weights.reserve(group.terms.size());
      for (const auto& term : group.terms) {
        const torch::Tensor& weight = data->constants[term.weight_binding_index];
        RecordCurrentStream(weight);
        transposed_weights.push_back(weight.transpose(0, 1));
      }
      torch::Tensor packed_weight = transposed_weights.size() == 1
          ? transposed_weights[0].contiguous()
          : torch::cat(transposed_weights, 0);
      RecordCurrentStream(packed_weight);
      if (!group.precompute_at_bind) {
        packed_groups[group_index] = std::move(packed_weight);
        continue;
      }

      std::vector<torch::Tensor> constant_inputs;
      constant_inputs.reserve(group.terms.size());
      for (const auto& term : group.terms) {
        const ValueSpec& input_spec = plan->data->values[term.input_value_index];
        TORCH_CHECK(input_spec.kind == ValueKind::kConstant,
            "AFFINE_SUM_V1 precomputed input must be constant");
        const torch::Tensor& input = data->constants[input_spec.binding_index];
        RecordCurrentStream(input);
        torch::Tensor converted_input = input.scalar_type() == projection_data_type
            ? input
            : input.to(projection_data_type);
        RecordCurrentStream(converted_input);
        constant_inputs.push_back(std::move(converted_input));
      }
      torch::Tensor packed_input = constant_inputs.size() == 1
          ? constant_inputs[0]
          : torch::cat(constant_inputs, -1);
      const torch::Tensor input_matrix = packed_input.view(
          {group.rows_per_batch, group.packed_input_width});
      torch::Tensor product = at::mm(input_matrix, packed_weight);
      std::vector<int64_t> product_shape;
      product_shape.reserve(group.prefix_shape.size() + 2);
      product_shape.push_back(1);
      product_shape.insert(
          product_shape.end(), group.prefix_shape.begin(), group.prefix_shape.end());
      product_shape.push_back(command->output_width);
      product_groups[group_index] = product.view(product_shape);
      RecordCurrentStream(product_groups[group_index]);
    }
  }
  c10::impl::VirtualGuardImpl guard_impl(plan->data->device.type());
  data->binding_ready = std::make_unique<c10::Event>(plan->data->device.type());
  data->binding_ready->record(guard_impl.getStream(plan->data->device));
  return new FusionExecutable(std::move(data));
}

FusionSession* NewFusionSession(const FusionExecutable* executable, int32_t buffer_count) {
  TORCH_CHECK(executable != nullptr, "fusion executable must not be null");
  TORCH_CHECK(buffer_count > 0, "fusion session buffer count must be positive");
  return new FusionSession(executable->data, buffer_count);
}

torch::Tensor GetFusionSessionOutput(
    const FusionSession* session, int32_t buffer_index, int32_t output_index) {
  TORCH_CHECK(session != nullptr, "fusion session must not be null");
  TORCH_CHECK(buffer_index >= 0 &&
          static_cast<std::size_t>(buffer_index) < session->storages.size(),
      "fusion output buffer index is outside the session range");
  TORCH_CHECK(output_index >= 0 &&
          static_cast<std::size_t>(output_index) < session->executable->plan->outputs.size(),
      "fusion output index is outside the plan range");
  const int32_t storage_index =
      session->executable->plan->outputs[output_index].storage_index;
  return session->storages[buffer_index][storage_index];
}

void SubmitFusion(FusionSession* session, int32_t buffer_index,
    const int64_t* input_handles, std::size_t input_count,
    const int64_t* dimensions, std::size_t dimension_count) {
  TORCH_CHECK(session != nullptr, "fusion session must not be null");
  TORCH_CHECK(buffer_index >= 0 &&
          static_cast<std::size_t>(buffer_index) < session->storages.size(),
      "fusion output buffer index is outside the session range");
#if defined(DJL_USE_ROCM_KERNELS)
  TORCH_CHECK(!session->poisoned,
      "fusion session is poisoned by an incomplete failed submission");
  const auto& plan = session->executable->plan;
  TORCH_CHECK(input_count == static_cast<std::size_t>(plan->input_count),
      "fusion input handle count does not match the prepared plan");
  TORCH_CHECK(dimension_count == static_cast<std::size_t>(plan->dimension_count),
      "fusion dimension count does not match the prepared plan");
  TORCH_CHECK(input_count == 0 || input_handles != nullptr,
      "fusion input handle table must not be null");
  TORCH_CHECK(dimension_count == 0 || dimensions != nullptr,
      "fusion dimension table must not be null");
  c10::DeviceGuard device_guard(plan->device);
  c10::impl::VirtualGuardImpl guard_impl(plan->device.type());
  const c10::Stream submission_stream = guard_impl.getStream(plan->device);

  ValidateSubmission(*session, buffer_index, input_handles, dimensions);

  session->WaitForAllocation(buffer_index, submission_stream);
  session->WaitForBinding(buffer_index, submission_stream);
  c10::InferenceMode inference_mode;
  bool work_submitted = false;
  try {
    for (std::size_t command_index = 0;
         command_index < plan->commands.size(); ++command_index) {
      const auto& fusion_command = plan->commands[command_index];
      if (const auto* command = std::get_if<OutputPackCommandSpec>(&fusion_command)) {
        ExecuteCommand(*session, buffer_index, input_handles, dimensions,
            *command, work_submitted);
      } else if (const auto* command =
                     std::get_if<AffineSumCommandSpec>(&fusion_command)) {
        ExecuteCommand(*session, buffer_index, input_handles, dimensions,
            *command, command_index, work_submitted);
      } else if (const auto* command =
                     std::get_if<IndexedAffineCommandSpec>(&fusion_command)) {
        ExecuteCommand(*session, buffer_index, input_handles, dimensions,
            *command, command_index, work_submitted);
      } else {
        ExecuteCommand(*session, buffer_index, input_handles, dimensions,
            std::get<TransformerEncoderStackCommandSpec>(fusion_command),
            command_index, work_submitted);
      }
    }
  } catch (...) {
    const std::exception_ptr submission_failure = std::current_exception();
    if (work_submitted) {
      session->RetainIncompleteSubmission(
          submission_stream, input_handles, input_count);
      try {
        session->DrainIncompleteSubmission();
      } catch (...) {
        TORCH_CHECK(false,
            "fusion session is poisoned because completion of a failed submission "
            "could not be established");
      }
    }
    std::rethrow_exception(submission_failure);
  }
  session->SetSubmissionStream(buffer_index, submission_stream);
#else
  TORCH_CHECK(false, "PyTorch fusion requires a ROCm build");
#endif
}

void SynchronizeFusionOutput(FusionSession* session, int32_t buffer_index) {
  TORCH_CHECK(session != nullptr, "fusion session must not be null");
  TORCH_CHECK(buffer_index >= 0 &&
          static_cast<std::size_t>(buffer_index) < session->storages.size(),
      "fusion output buffer index is outside the session range");
  c10::DeviceGuard device_guard(session->executable->plan->device);
  session->Synchronize(buffer_index);
}

void DeleteFusionPlan(FusionPlan* plan) {
  delete plan;
}

void DeleteFusionExecutable(FusionExecutable* executable) {
  delete executable;
}

void DeleteFusionSession(FusionSession* session) {
  if (session != nullptr && session->poisoned) {
    session->DrainIncompleteSubmission();
  }
  delete session;
}

}  // namespace djl::pytorch::fusion
