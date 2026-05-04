cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(djl_torch)

option(BUILD_ANDROID "Build the static djl_torch jni linked with libtorch statically" OFF)
option(REQUIRE_DISTRIBUTED_NCCL "Fail if native distributed NCCL/RCCL support cannot be built" OFF)

if(BUILD_ANDROID AND NOT DEFINED ANDROID_ABI)
    message(FATAL_ERROR "Please specify ANDROID_ABI!")
endif()

if(NOT BUILD_ANDROID)
    message("Building djl_torch with the host...")
    find_package(Torch REQUIRED)

    set(JAVA_AWT_LIBRARY NotNeeded)
    set(JAVA_AWT_INCLUDE_PATH NotNeeded)
    find_package(JNI REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message("Building djl_torch with ${ANDROID_ABI}...")
    # The dependency of the cpuinfo clong.c
    find_library(log-lib log)
    # It is required to set those when using library elsewhere than Android SDK dir during the cross-compiling
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

# find utils header
find_path(UTILS_INCLUDE_DIR NAMES djl/utils.h PATHS ${PROJECT_SOURCE_DIR}/../../../api/src/main/native REQUIRED)

set(SOURCE_FILES
    "src/main/native/djl_pytorch_utils.h"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_system.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_autocast.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_nn_functional.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_inference.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_ivalue.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_optim.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_distributed.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_comparison.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_tensor.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_creation.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_isjm.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_pointwise.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_reduction.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_random_sampling.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_other.cc"
    "src/main/native/djl_pytorch_jni_log.cc"
    "src/main/native/djl_pytorch_jni_log.h"
    "src/main/native/djl_pytorch_jni_exception.h"
    "src/main/native/ai_djl_pytorch_jni_cache.h"
    "src/main/native/ai_djl_pytorch_jni_cache.cc")

if(PT_VERSION)
    add_compile_definitions(${PT_VERSION})
endif()

if(USE_CUDA)
    add_compile_definitions(USE_CUDA)
endif()

set(DISTRIBUTED_NCCL_INCLUDE_DIR "")
set(DISTRIBUTED_RCCL_TARGET "")
if(USE_CUDA AND NOT BUILD_ANDROID)
    if(USE_ROCM)
        find_package(rccl CONFIG QUIET)
        if(rccl_FOUND)
            if(TARGET roc::rccl)
                set(DISTRIBUTED_RCCL_TARGET roc::rccl)
            elseif(TARGET rccl)
                set(DISTRIBUTED_RCCL_TARGET rccl)
            elseif(TARGET rccl::rccl)
                set(DISTRIBUTED_RCCL_TARGET rccl::rccl)
            endif()
        endif()
        if(DISTRIBUTED_RCCL_TARGET)
            add_compile_definitions(USE_DISTRIBUTED_NCCL)
            message(STATUS "Native distributed RCCL support enabled via ${DISTRIBUTED_RCCL_TARGET}")
        else()
            if(REQUIRE_DISTRIBUTED_NCCL)
                message(FATAL_ERROR "Native distributed RCCL support was requested, but a supported rccl CMake target was not found.")
            else()
                message(WARNING "Supported rccl CMake target not found; native distributed training will use JNI fallback stubs")
            endif()
        endif()
    else()
        find_path(DISTRIBUTED_NCCL_INCLUDE_DIR NAMES nccl.h HINTS $ENV{NCCL_INCLUDE_DIR})
        if(NOT DISTRIBUTED_NCCL_INCLUDE_DIR)
            find_path(DISTRIBUTED_NCCL_INCLUDE_DIR
                NAMES nccl.h
                HINTS
                    $ENV{NCCL_ROOT}
                    $ENV{CUDA_HOME}
                    $ENV{CUDA_PATH}
                    /usr/local/cuda
                PATH_SUFFIXES include)
        endif()
        if(DISTRIBUTED_NCCL_INCLUDE_DIR)
            add_compile_definitions(USE_DISTRIBUTED_NCCL)
            message(STATUS "Native distributed NCCL support enabled: ${DISTRIBUTED_NCCL_INCLUDE_DIR}")
        else()
            if(REQUIRE_DISTRIBUTED_NCCL)
                message(FATAL_ERROR "Native distributed NCCL support was requested, but nccl.h was not found.")
            else()
                message(WARNING "nccl.h not found; native distributed training will use JNI fallback stubs")
            endif()
        endif()
    endif()
endif()

# ROCm libtorch keeps the c10/cuda/*.h headers but those transitively pull
# in <cuda_runtime.h>, which is a CUDA-SDK-only file absent on ROCm build
# hosts. USE_ROCM lets JNI sources gate the CUDA-specific code paths
# (CUDACachingAllocator, CUDAStream pool) so the ROCm compile succeeds;
# torch::cuda::is_available() still works on ROCm libtorch via the
# hipified dispatch under the same symbol names.
if(USE_ROCM)
    add_compile_definitions(USE_ROCM)
endif()

add_library(djl_torch SHARED ${SOURCE_FILES})
set_property(TARGET djl_torch PROPERTY CXX_STANDARD 17)

# build host
if(NOT BUILD_ANDROID)
    target_link_libraries(djl_torch "${TORCH_LIBRARIES}")
    if(DISTRIBUTED_RCCL_TARGET)
        target_link_libraries(djl_torch PUBLIC ${DISTRIBUTED_RCCL_TARGET})
    endif()
    target_include_directories(djl_torch PUBLIC build/include ${JNI_INCLUDE_DIRS} ${UTILS_INCLUDE_DIR})
    if(DISTRIBUTED_NCCL_INCLUDE_DIR)
        target_include_directories(djl_torch PUBLIC ${DISTRIBUTED_NCCL_INCLUDE_DIR})
    endif()
    # We have to kill the default rpath and use current dir
    set(CMAKE_SKIP_RPATH TRUE)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set_target_properties(djl_torch PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
    endif()
    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET djl_torch
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:djl_torch>)
    endif(MSVC)
else()
    # build android
    # function to static link the libtorch to djl_pytorch
    function(import_static_lib name)
        add_library(${name} STATIC IMPORTED)
        set_property(
            TARGET ${name}
            PROPERTY IMPORTED_LOCATION
            ${CMAKE_CURRENT_SOURCE_DIR}/libtorch_android/${ANDROID_ABI}/lib/${name}.a)
    endfunction(import_static_lib)
    list(APPEND LIBS libtorch libtorch_cpu libc10 libnnpack libXNNPACK libpytorch_qnnpack libpthreadpool libeigen_blas libcpuinfo libclog libVulkanWrapper)
    foreach(lib IN LISTS LIBS)
        import_static_lib(${lib})
    endforeach()

    target_include_directories(djl_torch PUBLIC build/include libtorch_android/${ANDROID_ABI}/include libtorch_android/${ANDROID_ABI}/include/torch/csrc/api/include ${UTILS_INCLUDE_DIR})
    target_link_libraries(djl_torch
        -Wl,--gc-sections
        -Wl,--whole-archive
        libtorch
        libtorch_cpu
        -Wl,--no-whole-archive
        libc10
        libnnpack
        libXNNPACK
        libpytorch_qnnpack
        libpthreadpool
        libeigen_blas
        libcpuinfo
        libclog
        libVulkanWrapper
        ${log-lib}
    )
endif()
