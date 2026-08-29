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

#include <c10/core/DeviceGuard.h>

#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
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
constexpr int64_t kDimensionPrefixExtent = 1;
constexpr int64_t kLayoutContiguous = 1;

constexpr int64_t kAttributeInt64 = 1;
constexpr int64_t kAttributeFloat64Bits = 2;
constexpr int64_t kAttributeBoolean = 3;
constexpr std::size_t kAttributeRecordHeaderWords = 4;

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

using FusionCommand = std::variant<OutputPackCommandSpec>;

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
};

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

bool IsOutputPackDataType(torch::ScalarType data_type) {
  return data_type == torch::kFloat16 || data_type == torch::kBFloat16 ||
      data_type == torch::kFloat32;
}

void RecordCurrentStream(const torch::Tensor& tensor) {
  c10::DeviceGuard device_guard(tensor.device());
  c10::impl::VirtualGuardImpl guard_impl(tensor.device().type());
  guard_impl.recordDataPtrOnStream(
      tensor.storage().data_ptr(), guard_impl.getStream(tensor.device()));
}

void ValidateAttributeRecords(DescriptorReader& reader, int32_t attribute_count) {
  std::unordered_set<int64_t> keys;
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
    TORCH_CHECK(keys.insert(key).second, "duplicate fusion command attribute key");
    TORCH_CHECK(type == kAttributeInt64 || type == kAttributeFloat64Bits ||
            type == kAttributeBoolean,
        "unsupported fusion command attribute type");
    TORCH_CHECK(element_count <= reader.Remaining(),
        "truncated fusion command attribute payload");
    for (std::size_t element = 0; element < element_count; ++element) {
      const int64_t payload = reader.Read();
      TORCH_CHECK(type != kAttributeBoolean || payload == 0 || payload == 1,
          "fusion boolean command attribute must contain zero or one");
    }
  }
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

OutputPackCommandSpec BuildOutputPackCommand(FusionPlanData& plan,
    int32_t command_index, const std::vector<int32_t>& results,
    const std::vector<int32_t>& operands, int64_t flags, int32_t attribute_count) {
  TORCH_CHECK(flags == 0, "OUTPUT_PACK_V1 does not support command flags");
  TORCH_CHECK(results.size() == 1,
      "OUTPUT_PACK_V1 requires exactly one result");
  TORCH_CHECK(!operands.empty() && operands.size() <= kMaximumOutputPackSources,
      "OUTPUT_PACK_V1 operand count exceeds the native limit");
  TORCH_CHECK(attribute_count == 0, "OUTPUT_PACK_V1 does not support attributes");

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
  command.result_storage_index = static_cast<int32_t>(plan.storages.size());
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
    TORCH_CHECK(IsOutputPackDataType(operand.data_type),
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
    const auto& command = std::get<OutputPackCommandSpec>(plan.commands[producer_index]);
    for (int32_t operand : command.operand_value_indices) {
      work.push_back(operand);
    }
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
    ValidateAttributeRecords(record_reader, attribute_count);
    TORCH_CHECK(record_reader.IsComplete(),
        "fusion command record size does not match its payload");
    command_reader = DescriptorReader(
        record_reader.Current(), descriptor + output_offset);

    switch (opcode) {
      case kOutputPackV1:
        plan->commands.emplace_back(BuildOutputPackCommand(
            *plan, command_index, results, operands, flags, attribute_count));
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
  }

  void WaitForAllocation(int32_t buffer_index, const c10::Stream& stream) {
    if (!allocation_waited[buffer_index]) {
      allocation_ready->block(stream);
      allocation_waited[buffer_index] = true;
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

  std::shared_ptr<const FusionExecutableData> executable;
  std::vector<std::vector<torch::Tensor>> storages;
  std::unique_ptr<c10::Event> allocation_ready;
  std::vector<bool> allocation_waited;
  std::vector<std::optional<c10::Stream>> submission_streams;
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
    const auto& command = std::get<OutputPackCommandSpec>(fusion_command);
    for (const auto& source : command.sources) {
      const torch::Tensor& tensor = ResolveValue(
          session, buffer_index, input_handles, source.value_index);
      ValidateTensorMetadata(tensor, plan, plan.values[source.value_index],
          dimensions, source.value_index == command.result_value_index, "operand");
    }
    const torch::Tensor& result = session.storages[buffer_index]
        [command.result_storage_index];
    ValidateTensorMetadata(result, plan, plan.values[command.result_value_index],
        dimensions, true, "result storage");
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
  bool work_submitted = false;
  try {
    std::array<OutputPackSource, kMaximumOutputPackSources> launch_sources;
    for (const auto& fusion_command : plan->commands) {
      const auto& command = std::get<OutputPackCommandSpec>(fusion_command);
      const int64_t row_count = dimensions[command.extent_index];
      torch::Tensor& result = session->storages[buffer_index]
          [command.result_storage_index];
      for (std::size_t source_index = 0; source_index < command.sources.size(); ++source_index) {
        const auto& source = command.sources[source_index];
        const torch::Tensor& tensor = ResolveValue(
            *session, buffer_index, input_handles, source.value_index);
        launch_sources[source_index] = OutputPackSource{tensor.data_ptr(),
            source.data_type, source.width, source.destination_offset};
        RecordCurrentStream(tensor);
      }
      RecordCurrentStream(result);
      work_submitted = true;
      LaunchOutputPack(launch_sources.data(), static_cast<int32_t>(command.sources.size()),
          result, row_count, command.output_width);
    }
  } catch (...) {
    const std::exception_ptr submission_failure = std::current_exception();
    if (work_submitted) {
      try {
        c10::Event completion(plan->device.type());
        completion.record(submission_stream);
        completion.synchronize();
      } catch (...) {
        session->poisoned = true;
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
  delete session;
}

}  // namespace djl::pytorch::fusion
