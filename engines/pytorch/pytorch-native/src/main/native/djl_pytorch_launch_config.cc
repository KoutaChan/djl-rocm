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
#include "djl_pytorch_launch_config.h"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <cstring>

#include "djl_pytorch_launch_config.h"

namespace djl::pytorch::launch_environment {

bool IsPlannerEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "TRUE") == 0) {
    return true;
  }
  TORCH_CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0,
      name, " must be 0, 1, false, or true, but was '", value, "'");
  return false;
}

}  // namespace djl::pytorch::launch_environment
