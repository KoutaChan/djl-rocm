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

bool is_specialized_tile(int value) {
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

int read_specialized_tile(const char* name, int default_value) {
  const int value = read_positive_integer(name, default_value);
  TORCH_CHECK(is_specialized_tile(value), name, " must be one of 1, 2, 4, 8, 16, or 32, but was ", value);
  return value;
}

RocmKernelLaunchConfig load_launch_config() {
  return {
      read_positive_integer("DJL_ROCM_RELATION_FORWARD_WAVES_PER_BLOCK", 4),
      read_positive_integer("DJL_ROCM_RELATION_BACKWARD_WAVES_PER_BLOCK", 4),
      read_specialized_tile("DJL_ROCM_RELATION_FORWARD_QUERIES_PER_WAVE", 4),
      read_specialized_tile("DJL_ROCM_RELATION_BACKWARD_QUERIES_PER_WAVE", 16),
      read_positive_integer("DJL_ROCM_GROUPED_ATTENTION_THREADS_PER_BLOCK", 256),
      read_positive_integer("DJL_ROCM_GROUPED_ATTENTION_BACKWARD_THREADS_PER_BLOCK", 256),
      read_positive_integer("DJL_ROCM_SHARED_GRADIENT_THREADS_PER_BLOCK", 256),
      read_specialized_tile("DJL_ROCM_SHARED_GRADIENT_FEATURE_TILE", 8),
      read_positive_integer("DJL_ROCM_RESIDUAL_NORM_THREADS_PER_BLOCK", 256),
      read_positive_integer("DJL_ROCM_OPTIMIZER_THREADS_PER_BLOCK", 256),
      read_positive_integer("DJL_ROCM_MASKED_CATEGORICAL_THREADS_PER_BLOCK", 256),
      read_positive_integer("DJL_ROCM_ROW_OPERATION_THREADS_PER_BLOCK", 256),
  };
}

}  // namespace

const RocmKernelLaunchConfig& rocm_kernel_launch_config() {
  static const RocmKernelLaunchConfig config = load_launch_config();
  return config;
}

}  // namespace djl::pytorch::rocm
