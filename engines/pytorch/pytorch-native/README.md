# DJL - PyTorch native Library

## Introduction
This project builds the JNI layer for Java to call PyTorch C++ APIs.
You can find more information in the `src`.

## Prerequisite
You need to install `cmake` and C++ compiler on your machine in order to build

### Linux

```sh
apt-get install -y locales cmake curl unzip software-properties-common
```

## CPU Build

Use the following task to build PyTorch JNI library:

### Mac/Linux

```sh
./gradlew compileJNI
```

### Windows

```cmd
gradlew compileJNI
```

This task will send a Jni library copy to `pytorch-engine` model to test locally.

## GPU build

GPU builds require a PyTorch/libtorch distribution and compiler toolchain for the same accelerator
flavor. Fusion Plan is enabled automatically when the JNI library is built for CUDA or ROCm. Its
shared command implementation is in `djl_pytorch_fusion_kernels.hip`; CUDA compiles the thin
`djl_pytorch_fusion_kernels.cu` entry point, and `djl_pytorch_fusion_backend.h` isolates the runtime
and launch differences. Keep changes to common command behavior in the shared source so CUDA and
ROCm do not drift. Fusion storage planner settings are documented in the
[engine guide](../../../docs/engine.md#fusion-storage-planning).

### Fusion execution-lane lifetime

Fusion planner arenas are shared by sessions that submit on the same accelerator device and
stream. Each data-type arena grows to the largest capacity requested on that lane and does not
shrink while any session retains the lane. A smaller storage-capacity profile avoids a larger arena
allocation only when the lane has not already observed a larger plan. The lane and its high-water
allocations are released after the last retaining session is closed; the global registry keeps only
weak references. This grow-only lifetime avoids allocator traffic and synchronization in the steady
state. On ROCm, the lane also owns one lazy, grow-only hipBLASLt workspace sized to the largest
selected fused linear algorithm. This avoids PyTorch's process-lifetime handle/stream workspace
registry while preserving the selected algorithm and adding no submission synchronization.

### NVIDIA CUDA

Install a CUDA toolkit compatible with the selected libtorch flavor and make `nvcc` available to
CMake. Set `TORCH_CUDA_ARCH_LIST` to the compute capabilities that the binary must support, then pass
the libtorch flavor through the `cuda` Gradle property. For example, from this directory:

```sh
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
./gradlew compileJNI -Pcuda=cu128 -Ppt_version=2.11.0
```

On Windows, run the equivalent command from a Visual Studio developer environment with the CUDA
toolkit on `PATH`:

```cmd
set "TORCH_CUDA_ARCH_LIST=8.0 8.6 8.9 9.0"
gradlew compileJNI -Pcuda=cu128 -Ppt_version=2.11.0
```

The build scripts configure CMake with `USE_CUDA=1` and `USE_ROCM=0`. The resulting native library
must be loaded with a CUDA PyTorch runtime of the same version and flavor.

### AMD ROCm

Use a ROCm development environment compatible with the selected PyTorch ROCm distribution. CMake
must be able to find HIP, hipBLASLt, and the ROCm libraries through the standard installation prefix
or `ROCM_PATH`. Set `PYTORCH_ROCM_ARCH` to a semicolon-separated set of target architectures when the
default build matrix is broader than needed:

```sh
export PYTORCH_ROCM_ARCH="gfx942;gfx1100"
./gradlew compileJNI -Pcuda=rocm7.1 -Ppt_version=2.11.0
```

The ROCm flavor configures both `USE_CUDA=1` and `USE_ROCM=1` because hipified libtorch retains the
`torch::cuda` API namespace. The resulting native library must be loaded with a matching ROCm
PyTorch runtime. The provided ROCm Gradle flavor uses `build.sh` on Linux. A direct Windows build
also requires a CMake generator and host compiler combination supported by the installed HIP
toolchain.

ROCm 10 uses AMD's split ROCm and PyTorch wheels instead of the monolithic libtorch archive used by
earlier releases. Install the matching development SDK and PyTorch package, then build against the
installed `torch` directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1100]==2.11.0+rocm10.0.0" \
    "rocm[libraries,devel,device-gfx1100]==10.0.0"
rocm-sdk init
export ROCM_PATH="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export PYTORCH_ROCM_ARCH=gfx1100
./gradlew compileJNI -Pcuda=rocm10.0 -Ppt_version=2.11.0
```

For a native ROCm 10 SDK installation, set `ROCM_PATH` to its development root (for example
`/opt/rocm/core-10.0`) and `LIBTORCH_ROOT` to an extracted AMD `torch` wheel. At runtime, keep the
official wheel packages installed and set `PYTORCH_LIBRARY_PATH` to the wheel's `torch/lib`
directory. This preserves AMD's version-locked ROCm libraries and per-device `.kpack` packages.

### ROCm native JAR packaging

After building JNI, run the packaging task from the repository root:

```sh
ROCM_PATH=/path/to/matching/sdk ./gradlew :engines:pytorch:pytorch-jni-rocm:jar \
    -Ppt_version=2.11.0 -Pflavor=rocm10.0 -Pclassifier=linux-x86_64
```

The JAR task prepares the patched hipBLASLt host library and native loader. It checks the exact
SDK version, source revision, and kernel metadata against
[`profiles.json`](rocm/hipblaslt/profiles.json). The matching SDK and its GPU kernels remain runtime
requirements; the build does not modify the SDK or copy its GPU kernels into the JAR.

`HIPBLASLT_CACHE_DIR` selects the private build cache (default: `pytorch-native/build/hipblaslt-cache`).
`GPU_TARGET` optionally selects a semicolon-separated subset of the SDK's GPU architectures, and
`BUILD_JOBS` controls host-library build parallelism. Library contents, SDK inputs, and build settings
are checked before reusing cached output.

For packaging prebuilt inputs, use `-Pjni_library=/path/to/libdjl_torch.so`,
`-Phipblaslt_bundle_dir=/path/to/rocm-runtime`, and
`-Procm_loader_library=/path/to/libdjl_rocm_loader.so`. Supplied bundles are validated before packaging.
CPU, CUDA, macOS, and Windows artifacts retain JNI-only packaging.

### Format C++ code
It uses clang-format to format the code.

```sh
./gradlew formatCpp
```

## PyTorch native package release

### Step 1: Build new JNI on top of new libtorch on osx, linux-cpu, linux-gpu, windows

1. Spin up a EC2 instance for linux, linux-gpu, windows, windows-gpu and cd pytorch/pytorch-native.
2. Run ./gradlew compileJNI and resolve all the issues you are facing.
3. Raise a PR for the JNI code change and don’t merge it until we have the rest things ready.

### Step 2: Check dependencies of each JNI

1. check the dependencies of each JNI library by `otool -L libdjl_torch.dylib` for osx, `ldd libdjl_torch.so` for linux and `dumpbin /dependents libdjl_torch.dll` for windows.
2. Compare all dependency libraries to those in downloadPyTorchNativeLib, if we miss copying new dependencies, correct the script.
3. modify the version to desired release version in pytorch/pytorch-native/build.gradle and make sure the URL in the task downloadPyTorchNativeLib point to the right, available URL. Usually the URL that is not for the latest version will have %2Bcpu/cuXXX in the end.
4. Make corresponding change on build.sh  and build.cmd
5. Raise PR for script change and get them merge

### Step 3: Upload the new version of libtorch dependencies to S3

1. Spin up a EC2 instance and `cd pytorch/pytorch-native && ./gradlew dPTNL`
2. `cd build/native/lib` and gzip all dependencies (`gzip -r download`)
3. Create a new pytorch-X.X.X in ai.djl/publish bucket with djl-prod account.
4. `aws s3 sync build/native/lib s3://djl-ai/publish/pytorch-X.X.X`

### Step 4: Build new JNI and upload to S3: 

1. Merge the JNI code change.
2. Now every script should point to new PyTorch version except integration and example are still using old pytorch-native version
3. Trigger Native JNI S3 PyTorch and resolve issues if any

### Step 5: Build pytorch-native snapshot

1. Trigger Native Snapshot PyTorch
2. Raise a PR to bump up all PyTorch version to new version and add -SNAPSHOT
3. Test integration test , example and pytorch-engine unit test with snapshot pytorch-native

### Step 6: Publish pytorch-native to staging

1. Trigger Native Release PyTorch
2. Test integration test, example and pytorch-engine unit test with staging pytorch-native
3. Publish to sonatype 
4. Raise a PR to remove all the -SNAPSHOT

## A quick start on implementing pytorch feature with JNI
This is adjusted from the comments in [PR#2349](https://github.com/deepjavalibrary/djl/issues/2349#issuecomment-1409003379).

To implement a simple pytorch feature, generally you can do the following steps.

1. Find the c-api in torch library for the feature to add. This can be done by searching in the document like [this](https://pytorch.org/cppdocs/api/function_namespaceat_1a854b1b19549a17f87a69b5f6b1134e22.html?highlight=bmm) or searching in the torch cpp source code.
2. Implement the JNI and api's in Java. The JNI can then be compiled with gradle commands. Here is the commands you can use on cpu machine, to compile JNI and run it with java api.

    ```sh
    cd engines/pytorch/pytorch-native
    ./gradlew cleanJNI compileJNI
   
    ./gradlew :engines:pytorch:pytorch-jni:clean
    ./gradlew :engines:pytorch:pytorch-engine:test
    ```
    
    **Note**:
    To test with a GPU build, select the matching libtorch flavor. For example:
    
    ```sh
    ./gradlew cleanJNI
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" \
        ./gradlew compileJNI -Pcuda=cu128 -Ppt_version=2.11.0
    ```

3. Document the api and add the unit tests.
4. Format the code and pass the PR tests

    Run the following tasks
    
    ```sh
    ./gradlew fJ fC checkstyleMain checkstyleTest pmdMain pmdTest
    ./gradlew test
    ```

