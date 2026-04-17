#!/usr/bin/env bash

set -ex
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
  NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
  NUM_PROC=$(sysctl -n hw.ncpu)
fi

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
VERSION=$1
FLAVOR=$2
AARCH64_CXX11ABI="-cxx11"
CXX11ABI="-cxx11-abi"
if [[ $3 == "precxx11" ]]; then
  CXX11ABI=""
  AARCH64_CXX11ABI=""
fi
# PyTorch 2.9+ Linux binaries are CXX11_ABI=1 only and the filename dropped
# the "-cxx11-abi-" infix. Strip the suffix so we request the real URL.
if [[ "$VERSION" =~ ^2\.([0-9]+)\. ]] && (( ${BASH_REMATCH[1]} >= 9 )); then
  CXX11ABI=""
fi
ARCH=$4

if [[ ! -d "libtorch" ]]; then
  if [[ $PLATFORM == 'linux' ]]; then
    if [[ ! "$FLAVOR" =~ ^(cpu|cu117|cu121|cu124|cu128|rocm[67]\.[0-9]+)$ ]]; then
      echo "$FLAVOR is not supported."
      exit 1
    fi

    if [[ $ARCH == 'aarch64' ]]; then
      if [[ "$VERSION" =~ ^(2.[7-9].*)$ ]]; then
        curl -fsSL "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-linux-aarch64-${VERSION}.zip" | jar xv >/dev/null
      else
        curl -fsSL "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch${AARCH64_CXX11ABI}-shared-with-deps-${VERSION}-aarch64.zip" | jar xv >/dev/null
      fi
    else
      curl -fsSL "https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip" | jar xv >/dev/null
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    if [[ "$VERSION" =~ ^(2.[2-9].*)$ ]]; then
      if [[ $ARCH == 'aarch64' ]]; then
        curl -fsSL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${VERSION}.zip" | jar xv >/dev/null
      else
        curl -fsSL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${VERSION}.zip" | jar xv >/dev/null
      fi
    else
      if [[ $ARCH == 'aarch64' ]]; then
        curl -fsSL "https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-macos-${VERSION}-aarch64.zip" | jar xv >/dev/null
      else
        curl -fsSL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip" | jar xv >/dev/null
      fi
    fi
  else
    echo "$PLATFORM is not supported."
    exit 1
  fi
fi

# Verify libtorch was actually extracted and expose the TorchConfig.cmake
# location for diagnostics — find_package(Torch) looks for it under
# share/cmake/Torch/ typically, but older / ROCm builds occasionally stage
# it elsewhere.
if [[ ! -d "libtorch" ]]; then
  echo "ERROR: libtorch directory is missing after download." >&2
  exit 1
fi
echo "libtorch top level:"
ls -1 libtorch | head -20
echo "Torch cmake config candidates:"
find libtorch -maxdepth 6 -type f \( -name "TorchConfig.cmake" -o -name "torch-config.cmake" \) \
    2>/dev/null | head -5 || true

if [[ "$VERSION" == "1.13.1" || "$VERSION" == "2.0.1" || "$VERSION" =~ ^2\.1\.[0-9]+$ ]]; then
  PT_VERSION=V1_13_X
fi

if [[ "$FLAVOR" = cu* ]]; then
  USE_CUDA=1
fi
if [[ "$FLAVOR" = rocm* ]]; then
  # ROCm libtorch is hipified and still exposes the c10/cuda/* headers and
  # torch::cuda::* symbols, so keep the JNI USE_CUDA branches (e.g. the
  # CUDACachingAllocator calls) enabled for ROCm as well.
  USE_CUDA=1
  # libtorch's LoadHIP.cmake requires PYTORCH_ROCM_ARCH at configure time
  # even when the downstream project has no HIP kernels. DJL JNI contains
  # zero HIP device code, so the arch list is really a placeholder — the
  # runtime GPU support is determined by the fat-binary libtorch shipped
  # by pytorch.org. Still, list every arch that the matching libtorch
  # actually targets so a future hipified kernel in the JNI covers the
  # same hardware surface. Rocm 7 drops gfx906 and adds gfx1200/1201.
  if [[ -z "${PYTORCH_ROCM_ARCH:-}" ]]; then
    case "$FLAVOR" in
      rocm6.*)
        export PYTORCH_ROCM_ARCH="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102"
        ;;
      rocm7.*)
        export PYTORCH_ROCM_ARCH="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        ;;
      *)
        # Unknown rocm flavor — pass the union of every arch the
        # upstream ROCm 6 & 7 libtorch builds currently target.
        export PYTORCH_ROCM_ARCH="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        ;;
    esac
  fi
fi

pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH="${WORK_DIR}/libtorch" -DPT_VERSION="${PT_VERSION}" -DUSE_CUDA="$USE_CUDA" ..
cmake --build . --config Release -- -j "${NUM_PROC}"
if [[ "$FLAVOR" = cu* ]]; then
  # avoid link with libcudart.so.11.0
  sed -i -r "s/\/usr\/local\/cuda(.{5})?\/lib64\/lib(cudart|nvrtc).so//g" CMakeFiles/djl_torch.dir/link.txt
  rm libdjl_torch.so
  . CMakeFiles/djl_torch.dir/link.txt
fi
if [[ "$FLAVOR" = rocm* ]]; then
  # avoid absolute link to /opt/rocm*/lib stubs so runtime loader picks up
  # whatever ROCm is present on the target machine
  sed -i -r "s#/opt/rocm[^ ]*/lib/lib(amdhip64|hsa-runtime64|rocblas|rocfft|rocrand|hiprtc|MIOpen)\.so[^ ]*##g" CMakeFiles/djl_torch.dir/link.txt
  rm -f libdjl_torch.so
  . CMakeFiles/djl_torch.dir/link.txt
fi

if [[ $PLATFORM == 'darwin' ]]; then
  install_name_tool -add_rpath @loader_path libdjl_torch.dylib
fi

popd
