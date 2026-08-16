#include "djl_pytorch_rocm_launch_config.h"

#include <c10/util/Exception.h>

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <system_error>

namespace djl::pytorch::rocm {
namespace {

int read_positive_integer(const char* name, int default_value) {
  const char* text = std::getenv(name);
  if (text == nullptr) {
    return default_value;
  }
  int value = 0;
  const char* end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, value);
  TORCH_CHECK(parsed.ec == std::errc() && parsed.ptr == end && value > 0,
      name, " must be a positive integer, but was '", text, "'");
  return value;
}

bool has_compiled_work_size(int value) {
  switch (value) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
      return true;
    default:
      return false;
  }
}

int read_compiled_work_size(const char* name, int default_value) {
  const int value = read_positive_integer(name, default_value);
  TORCH_CHECK(has_compiled_work_size(value), name,
      " must be one of 1, 2, 4, 8, 16, or 32, but was ", value);
  return value;
}

RocmKernelLaunchConfig load_launch_config() {
  return {
      read_positive_integer(launch_environment::kIndexedRelationBiasForwardWavesPerBlock, 4),
      read_positive_integer(launch_environment::kIndexedRelationBiasBackwardWavesPerBlock, 4),
      read_compiled_work_size(launch_environment::kIndexedRelationBiasForwardQueriesPerWave, 4),
      read_compiled_work_size(launch_environment::kIndexedRelationBiasBackwardQueriesPerWave, 16),
      read_positive_integer(launch_environment::kGroupedIndexedAttentionForwardMaxThreadsPerBlock, 256),
      read_positive_integer(launch_environment::kGroupedIndexedAttentionBackwardMaxThreadsPerBlock, 256),
      read_positive_integer(
          launch_environment::kGroupedIndexedAttentionSharedKeyValueGradientThreadsPerBlock, 256),
      read_compiled_work_size(
          launch_environment::kGroupedIndexedAttentionSharedKeyValueGradientFeaturesPerBlock, 8),
      read_positive_integer(launch_environment::kResidualAddLayerNormThreadsPerBlock, 256),
      read_positive_integer(launch_environment::kFusedAdamUpdateThreadsPerBlock, 256),
      read_positive_integer(launch_environment::kMaskedCategoricalThreadsPerBlock, 256),
      read_positive_integer(launch_environment::kScatterRowsThreadsPerBlock, 256),
  };
}

}  // namespace

const RocmKernelLaunchConfig& rocm_kernel_launch_config() {
  static const RocmKernelLaunchConfig config = load_launch_config();
  return config;
}

}  // namespace djl::pytorch::rocm
