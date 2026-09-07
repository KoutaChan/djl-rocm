#!/usr/bin/env bash
# Build the patched hipBLASLt host library against the installed ROCm SDK.
# Existing Tensile kernel libraries are reused; this script does not install them.
# Usage: ROCM_PATH=/path/to/sdk bash build.sh <rocm-libraries-source> <build-dir> [cmake-options...]

set -euo pipefail

if (( $# < 2 )); then
    printf 'Usage: ROCM_PATH=/path/to/sdk %s <rocm-libraries-source> <build-dir> [cmake-options...]\n' "$0" >&2
    exit 2
fi
: "${ROCM_PATH:?Set ROCM_PATH to the installed ROCm SDK}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "$1" && pwd)"
build_dir="$2"
shift 2

revision=8d1ae90eff7d022f26019ec55b2ec6a7674b3112
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
    printf 'Expected ROCm/rocm-libraries revision %s\n' "$revision" >&2
    exit 1
fi
patch="$script_dir/patches/0001-bound-tensile-solution-cache.patch"
if ! git -C "$source_root" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$source_root" apply --check "$patch"
    git -C "$source_root" apply "$patch"
fi

export PATH="$ROCM_PATH/bin:$PATH"
# Origami's standalone header checks clear CMAKE_CXX_FLAGS and compile as C++.
# hipcc then omits HIP's include path, so expose the installed headers explicitly.
export CPLUS_INCLUDE_PATH="$ROCM_PATH/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
gpu_target="${GPU_TARGET:-gfx1100}"

# The ROCm SDK need not ship the header-only MessagePack CMake package.
# Keep its pinned source and installation private to this build.
deps_dir="${HIPBLASLT_DEPS_DIR:-$build_dir/deps}"
msgpack_root="$deps_dir/msgpack-cxx-6.1.0"
msgpack_prefix="$msgpack_root/install"
if [[ ! -f "$msgpack_prefix/lib/cmake/msgpack-cxx/msgpack-cxx-config.cmake" ]]; then
    msgpack_revision=8c602e8579c7e7d65d6f9c6703c9699db3fb0488
    if [[ ! -d "$msgpack_root/source/.git" ]]; then
        git init "$msgpack_root/source"
        git -C "$msgpack_root/source" fetch --depth 1 https://github.com/msgpack/msgpack-c.git "$msgpack_revision"
        git -C "$msgpack_root/source" checkout --detach FETCH_HEAD
    fi
    if [[ "$(git -C "$msgpack_root/source" rev-parse HEAD)" != "$msgpack_revision" ]]; then
        printf 'Expected msgpack-cxx revision %s\n' "$msgpack_revision" >&2
        exit 1
    fi
    cmake -S "$msgpack_root/source" -B "$msgpack_root/build" -G 'Unix Makefiles' \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$msgpack_prefix" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DMSGPACK_CXX17=ON \
        -DMSGPACK_USE_BOOST=OFF \
        -DMSGPACK_BUILD_TESTS=OFF \
        -DMSGPACK_BUILD_DOCS=OFF \
        -DMSGPACK_BUILD_EXAMPLES=OFF
    cmake --install "$msgpack_root/build"
fi

cmake -S "$source_root/projects/hipblaslt" -B "$build_dir" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc" \
    -DCMAKE_CXX_FLAGS="--offload-arch=$gpu_target" \
    -DCMAKE_PREFIX_PATH="$msgpack_prefix;$ROCM_PATH;$ROCM_PATH/lib/host-math;$ROCM_PATH/lib/rocm_sysdeps" \
    -DGPU_TARGETS="$gpu_target" \
    -DHIPBLASLT_ENABLE_HOST=ON \
    -DHIPBLASLT_BUILD_SHARED_LIBS=ON \
    -DHIPBLASLT_ENABLE_DEVICE=OFF \
    -DHIPBLASLT_ENABLE_EXTOPS=OFF \
    -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
    -DHIPBLASLT_ENABLE_CLIENT=OFF \
    -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
    -DHIPBLASLT_ENABLE_MARKER=OFF \
    -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF \
    -DHIPBLASLT_ENABLE_LAZY_LOAD=ON \
    -DTENSILELITE_ENABLE_HOST=ON \
    -DTENSILELITE_BUILD_SHARED_LIBS=OFF \
    -DTENSILELITE_ENABLE_CLIENT=OFF \
    -DTENSILELITE_BUILD_TESTING=OFF \
    -DBUILD_TESTING=OFF \
    "$@"
cmake --build "$build_dir" --target hipblaslt --parallel "${BUILD_JOBS:-8}"

printf 'Host library: %s/library/libhipblaslt.so\n' "$build_dir"
printf 'Existing kernels: HIPBLASLT_TENSILE_LIBPATH=%s/lib/hipblaslt/library/%s\n' "$ROCM_PATH" "$gpu_target"
