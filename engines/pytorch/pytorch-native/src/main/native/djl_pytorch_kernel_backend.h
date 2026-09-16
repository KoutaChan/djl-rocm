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

#ifndef DJL_PYTORCH_KERNEL_BACKEND_H
#define DJL_PYTORCH_KERNEL_BACKEND_H

#if defined(DJL_USE_ROCM_KERNELS) && defined(DJL_USE_CUDA_KERNELS)
#error "Select one native kernel backend per library"
#endif

#if defined(DJL_USE_ROCM_KERNELS)
#include "djl_pytorch_rocm_kernels.h"
#define DJL_USE_ACCELERATOR_KERNELS
namespace djl::pytorch {
namespace kernel_backend = rocm;
}
#elif defined(DJL_USE_CUDA_KERNELS)
#include "djl_pytorch_layer_norm_cuda.h"
#include "djl_pytorch_masked_categorical_cuda.h"
#include "djl_pytorch_routing_masks_cuda.h"
#include "djl_pytorch_row_ops_cuda.h"
#include "djl_pytorch_structured_attention_cuda.h"
#define DJL_USE_ACCELERATOR_KERNELS
namespace djl::pytorch {
namespace kernel_backend = cuda;
}
#endif

#endif  // DJL_PYTORCH_KERNEL_BACKEND_H
