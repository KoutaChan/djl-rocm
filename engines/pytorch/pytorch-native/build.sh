#!/usr/bin/env bash
#
# Build libdjl_torch.<so|dll|dylib>, the JNI bridge between DJL's Java
# PtNDArrayEx layer and libtorch.
#
# Usage: build.sh <VERSION> <FLAVOR> [precxx11] <ARCH>
#   VERSION  e.g. 2.11.0
#   FLAVOR   cpu / cu128 / cu129 / cu130 / rocm6.4 / rocm7.0 / rocm7.1 / rocm7.2 / rocm10.0
#   3rd arg  "precxx11" to strip the -cxx11-abi- infix (older PyTorch)
#   ARCH     amd64 / aarch64 (linux+darwin)

set -ex

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
VERSION=$1
FLAVOR=$2
CXX11ABI_ARG=$3
ARCH=$4

# PT_VERSION_MACRO is the value passed to cmake via -DPT_VERSION=<x> so it
# becomes an #ifdef symbol for version-gated JNI code. The workflow also
# exports PT_VERSION=<matrix.pt> (e.g. "2.10.0") to Docker; we pick a
# distinct shell name so that env inheritance cannot pollute the cmake flag
# (a numeric value like "2.10.0" is not a valid C identifier).
PT_VERSION_MACRO=""

NUM_PROC=1
if command -v nproc >/dev/null 2>&1; then
  NUM_PROC=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
  NUM_PROC=$(sysctl -n hw.ncpu)
fi

# Derive cxx11-abi suffix used in older libtorch filenames.
AARCH64_CXX11ABI="-cxx11"
CXX11ABI="-cxx11-abi"
if [[ "$CXX11ABI_ARG" == "precxx11" ]]; then
  CXX11ABI=""
  AARCH64_CXX11ABI=""
fi
# PyTorch 2.8+ Linux binaries are CXX11_ABI=1 only and the filename dropped
# the "-cxx11-abi-" infix (last cxx11-abi zip was 2.7.1, first no-cxx11-abi
# was 2.8.0 per download.pytorch.org). Strip the suffix so we request the
# real URL.
if [[ "$VERSION" =~ ^2\.([0-9]+)\. ]] && (( ${BASH_REMATCH[1]} >= 8 )); then
  CXX11ABI=""
fi

#
# libtorch download
#

download_and_extract_zip() (
  local url=$1
  local archive

  archive=$(mktemp "${WORK_DIR}/libtorch.XXXXXX.zip")
  trap 'rm -f "$archive"' EXIT

  curl -fsSL --retry 3 --output "$archive" "$url"
  jar xf "$archive"
)

download_libtorch_linux() {
  if [[ ! "$FLAVOR" =~ ^(cpu|cu117|cu121|cu124|cu128|cu129|cu130|rocm(6|7|10)\.[0-9]+)$ ]]; then
    echo "$FLAVOR is not supported." >&2
    exit 1
  fi
  if [[ $ARCH == 'aarch64' ]]; then
    if [[ "$VERSION" =~ ^2\.([0-9]+)\. ]] && (( ${BASH_REMATCH[1]} >= 7 )); then
      download_and_extract_zip "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-linux-aarch64-${VERSION}.zip"
    else
      download_and_extract_zip "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch${AARCH64_CXX11ABI}-shared-with-deps-${VERSION}-aarch64.zip"
    fi
  else
    download_and_extract_zip "https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip"
  fi
}

find_python_torch() {
  python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("torch"); print(pathlib.Path(spec.origin).parent if spec else "")'
}

find_rocm_root() {
  local requested=${FLAVOR#rocm}
  local candidate
  for candidate in \
      "${ROCM_PATH:-}" \
      "${ROCM_HOME:-}" \
      "/opt/rocm/core-${requested}" \
      "/opt/rocm-${requested}" \
      /opt/rocm; do
    if [[ -n "$candidate" && -x "$candidate/bin/hipcc" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "A ROCm ${requested} development SDK was not found. Set ROCM_PATH." >&2
  exit 1
}

TORCH_ROOT="${LIBTORCH_ROOT:-${WORK_DIR}/libtorch}"
if [[ "$FLAVOR" == rocm10.* && ! -f "$TORCH_ROOT/lib/libtorch.so" ]]; then
  if [[ -n "${PYTORCH_LIBRARY_PATH:-}" ]]; then
    if [[ -f "$PYTORCH_LIBRARY_PATH/libtorch.so" ]]; then
      TORCH_ROOT=$(dirname "$PYTORCH_LIBRARY_PATH")
    elif [[ -f "$PYTORCH_LIBRARY_PATH/lib/libtorch.so" ]]; then
      TORCH_ROOT=$PYTORCH_LIBRARY_PATH
    fi
  fi
  if [[ ! -f "$TORCH_ROOT/lib/libtorch.so" ]]; then
    TORCH_ROOT=$(find_python_torch)
  fi
  if [[ ! -f "$TORCH_ROOT/lib/libtorch.so" ]]; then
    echo "ROCm 10 libtorch is distributed in AMD's torch wheel." >&2
    echo "Install torch[device-<gfx>] from https://stable.repo.amd.com/rocm/whl-next/" >&2
    echo "or set LIBTORCH_ROOT to an extracted torch package." >&2
    exit 1
  fi
fi

download_libtorch_darwin() {
  # PyTorch 2.2+ ships macOS libtorch directly on pytorch.org (arm64 +
  # x86_64 separately). Older 2.0.x/2.1.x only lived on the DJL mirror.
  local pytorch_has_macos=false
  if [[ "$VERSION" =~ ^2\.([0-9]+)\. ]] && (( ${BASH_REMATCH[1]} >= 2 )); then
    pytorch_has_macos=true
  fi
  if [[ "$pytorch_has_macos" == "true" ]]; then
    if [[ $ARCH == 'aarch64' ]]; then
      download_and_extract_zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${VERSION}.zip"
    else
      download_and_extract_zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${VERSION}.zip"
    fi
  else
    if [[ $ARCH == 'aarch64' ]]; then
      download_and_extract_zip "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-macos-${VERSION}-aarch64.zip"
    else
      download_and_extract_zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip"
    fi
  fi
}

# Check for the actual libtorch library, not just the directory. CI bind
# mounts an empty /mnt/libtorch onto this path so the pure directory check
# would silently skip the download and leave us with an empty libtorch/.
if [[ ! -f "$TORCH_ROOT/lib/libtorch.so" && ! -f "$TORCH_ROOT/lib/libtorch.dylib" && ! -f "$TORCH_ROOT/lib/torch.dll" ]]; then
  case "$PLATFORM" in
    linux)  download_libtorch_linux ;;
    darwin) download_libtorch_darwin ;;
    *)      echo "$PLATFORM is not supported." >&2; exit 1 ;;
  esac
fi

if [[ ! -d "$TORCH_ROOT" ]]; then
  echo "ERROR: libtorch directory is missing after download: $TORCH_ROOT" >&2
  exit 1
fi

#
# ROCm-specific libtorch patching
#

stub_cuda_cmake_macros() {
  # ROCm libtorch zips ship c10/cuda/CUDAMacros.h which transitively includes
  # <c10/cuda/impl/cuda_cmake_macros.h>, but that header is only generated by
  # the upstream CUDA CMake build and is absent from the ROCm distribution.
  # Drop a stub so consumers that pull in CUDACachingAllocator & friends can
  # compile. The file only defines build-time #defines so an empty one is
  # equivalent to "default everything".
  local stub="$TORCH_ROOT/include/c10/cuda/impl/cuda_cmake_macros.h"
  if [[ ! -f "$stub" ]]; then
    echo "note: synthesising missing $stub"
    mkdir -p "$(dirname "$stub")"
    printf '#pragma once\n' > "$stub"
  fi
}

set_rocm_arch() {
  # Build DJL's ROCm kernels for the same hardware surface as the matching
  # libtorch package. Callers can narrow the list through PYTORCH_ROCM_ARCH
  # when building a machine-specific binary.
  if [[ -n "${PYTORCH_ROCM_ARCH:-}" ]]; then
    return
  fi
  case "$FLAVOR" in
    rocm6.*)
      export PYTORCH_ROCM_ARCH="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102"
      ;;
    rocm7.*)
      export PYTORCH_ROCM_ARCH="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
      ;;
    rocm10.*)
      export PYTORCH_ROCM_ARCH="gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
      ;;
    *)
      export PYTORCH_ROCM_ARCH="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
      ;;
  esac
}

set_cuda_arch() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    return
  fi
  case "$FLAVOR" in
    cu13*)
      # CUDA 13 supports Turing and newer architectures, while PyTorch's
      # headless fallback list still includes compute_50.
      export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
      ;;
  esac
}

USE_CUDA=0
USE_ROCM=0
case "$FLAVOR" in
  cu*)
    USE_CUDA=1
    set_cuda_arch
    ;;
  rocm*)
    # ROCm libtorch is hipified and still exposes torch::cuda::* symbols.
    # USE_CUDA selects accelerator support and USE_ROCM selects the ROCm
    # stream/allocator source file.
    USE_CUDA=1
    USE_ROCM=1
    stub_cuda_cmake_macros
    set_rocm_arch
    ROCM_PATH=$(find_rocm_root)
    export ROCM_PATH ROCM_HOME="$ROCM_PATH"
    export PATH="$ROCM_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac

# PT_VERSION macro is only defined for 1.13.x / 2.0.x / 2.1.x branches,
# which is where JNI code paths gate on #ifdef V1_13_X. Newer versions
# leave the macro undefined.
if [[ "$VERSION" == "1.13.1" || "$VERSION" == "2.0.1" || "$VERSION" =~ ^2\.1\.[0-9]+$ ]]; then
  PT_VERSION_MACRO=V1_13_X
fi

#
# Build
#

BUILD_TYPE=${DJL_NATIVE_BUILD_TYPE:-Release}

pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes

# Some rocm/dev-ubuntu images (e.g. :6.4-complete) install the JDK under a
# non-standard path that CMake's FindJNI cannot discover. Derive JAVA_HOME
# from `javac` so find_package(JNI) can locate the headers + libjvm.
if [[ -z "${JAVA_HOME:-}" ]] && command -v javac >/dev/null 2>&1; then
  javac_path=$(readlink -f "$(command -v javac)")
  export JAVA_HOME=$(dirname "$(dirname "$javac_path")")
  echo "note: auto-detected JAVA_HOME=${JAVA_HOME}"
fi

cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}${ROCM_PATH:+;${ROCM_PATH}}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DPT_VERSION="${PT_VERSION_MACRO}" \
      -DUSE_CUDA="$USE_CUDA" \
      -DUSE_ROCM="$USE_ROCM" \
      -DREQUIRE_DISTRIBUTED_NCCL="${REQUIRE_DISTRIBUTED_NCCL:-OFF}" ..
cmake --build . --config "${BUILD_TYPE}" -- -j "${NUM_PROC}"

if [[ $PLATFORM == 'darwin' ]]; then
  install_name_tool -add_rpath @loader_path libdjl_torch.dylib
fi

popd
