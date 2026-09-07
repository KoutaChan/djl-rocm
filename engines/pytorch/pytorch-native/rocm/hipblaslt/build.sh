#!/usr/bin/env bash
# Build the patched hipBLASLt host library against the installed ROCm SDK.
# Existing Tensile kernel libraries are reused; this script does not install them.
# Usage: ROCM_PATH=/path/to/sdk bash build.sh <source> <build-dir> [--flavor rocmX.Y] [cmake-options...]

set -euo pipefail

if (( $# < 2 )); then
    printf 'Usage: ROCM_PATH=/path/to/sdk %s <source> <build-dir> [--flavor rocmX.Y] [cmake-options...]\n' "$0" >&2
    exit 2
fi
: "${ROCM_PATH:?Set ROCM_PATH to the installed ROCm SDK}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "$1" && pwd)"
build_dir="$2"
shift 2

flavor=""
if [[ "${1:-}" == --flavor ]]; then
    flavor="$2"
    shift 2
fi
profile_info=$(python3 - "$script_dir" "$source_root" "$flavor" <<'PY'
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bundle", Path(sys.argv[1]) / "prepare-bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)
sdk = Path(os.environ["ROCM_PATH"])
_, _, profile = bundle.sdk_profile(sdk, sys.argv[3] or None)
source = Path(sys.argv[2])
head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if head != profile["source_revision"]:
    raise RuntimeError(
        f"Expected {profile['source_repository']} revision {profile['source_revision']}"
    )
bundle.apply_patches(source, profile)
dynamic = subprocess.check_output(["readelf", "-d", str(sdk / "lib/libhipblaslt.so")], text=True)
print(profile["source_dir"])
print(profile["build_family"])
print("ON" if "librocroller.so" in dynamic else "OFF")
print("ON" if "libroctx" in dynamic else "OFF")
PY
)
mapfile -t profile_values <<< "$profile_info"
project_root="$source_root/${profile_values[0]}"
build_family="${profile_values[1]}"

export PATH="$ROCM_PATH/bin:$PATH"
# Origami's standalone header checks clear CMAKE_CXX_FLAGS and compile as C++.
# hipcc then omits HIP's include path, so expose the installed headers explicitly.
export CPLUS_INCLUDE_PATH="$ROCM_PATH/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
gpu_target="${GPU_TARGET:-gfx1100}"
IFS=';' read -r -a gpu_targets <<< "$gpu_target"
offload_flags=$(printf -- '--offload-arch=%s ' "${gpu_targets[@]}")

# The ROCm SDK need not ship the header-only MessagePack CMake package.
# Keep its pinned source and installation private to this build.
deps_dir="${HIPBLASLT_DEPS_DIR:-$build_dir/deps}"
fetch_dependency() {
    local directory="$1" repository="$2" revision="$3"
    if [[ ! -d "$directory/.git" ]]; then
        git init "$directory"
    fi
    if [[ "$(git -C "$directory" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
        git -C "$directory" fetch --depth 1 "$repository" "$revision"
        git -C "$directory" checkout --detach FETCH_HEAD
    fi
}
msgpack_root="$deps_dir/msgpack-cxx-6.1.0"
msgpack_prefix="$msgpack_root/install"
if [[ ! -f "$msgpack_prefix/lib/cmake/msgpack-cxx/msgpack-cxx-config.cmake" ]]; then
    msgpack_revision=8c602e8579c7e7d65d6f9c6703c9699db3fb0488
    fetch_dependency "$msgpack_root/source" https://github.com/msgpack/msgpack-c.git "$msgpack_revision"
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

# Link the SDK's existing RocRoller when that SDK enables this backend. Its
# exported CMake interface requires fmt and yaml-cpp development packages even
# though the SDK includes neither; install the upstream-pinned dependencies
# privately and keep yaml-cpp static with position-independent code.
rocroller="${profile_values[2]}"
marker="${profile_values[3]}"
dependency_prefix="$deps_dir/install"
if [[ "$rocroller" == ON ]]; then
    fetch_dependency "$deps_dir/fmt/source" https://github.com/fmtlib/fmt.git 9cf9f38eded63e5e0fb95cd536ba51be601d7fa2
    fetch_dependency "$deps_dir/yaml-cpp/source" https://github.com/jbeder/yaml-cpp.git f7320141120f720aecc4c32be25586e7da9eb978
    if [[ ! -f "$dependency_prefix/lib/cmake/fmt/fmt-config.cmake" ]]; then
        cmake -S "$deps_dir/fmt/source" -B "$deps_dir/fmt/build" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dependency_prefix" \
            -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DFMT_TEST=OFF -DFMT_DOC=OFF -DFMT_INSTALL=ON
        cmake --build "$deps_dir/fmt/build" --parallel "${BUILD_JOBS:-8}"
        cmake --install "$deps_dir/fmt/build"
    fi
    if [[ ! -f "$dependency_prefix/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]]; then
        cmake -S "$deps_dir/yaml-cpp/source" -B "$deps_dir/yaml-cpp/build" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dependency_prefix" \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DCMAKE_CXX_FLAGS="-include cstdint" \
            -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF \
            -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF
        cmake --build "$deps_dir/yaml-cpp/build" --parallel "${BUILD_JOBS:-8}"
        cmake --install "$deps_dir/yaml-cpp/build"
    fi
fi

compiler_launcher=""
if command -v ccache >/dev/null 2>&1; then
    compiler_launcher=ccache
fi

if [[ "$build_family" == legacy ]]; then
    family_options=(-DBUILD_SHARED_LIBS=ON -DBUILD_WITH_TENSILE=ON \
        -DAMDGPU_TARGETS="$gpu_target" -DTensile_SKIP_BUILD=ON \
        -DTensile_LAZY_LIBRARY_LOADING=ON -DTensile_NO_LAZY_LIBRARY_LOADING=OFF \
        -DTENSILE_BUILD_CLIENT=OFF -DTENSILE_USE_LLVM=OFF -DTENSILE_USE_MSGPACK=ON \
        -DHIPBLASLT_USE_ROCROLLER="$rocroller" -DHIPBLASLT_ENABLE_MARKER="$marker" \
        -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF)
else
    family_options=(-DGPU_TARGETS="$gpu_target" \
        -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_BUILD_SHARED_LIBS=ON \
        -DHIPBLASLT_ENABLE_DEVICE=OFF -DHIPBLASLT_ENABLE_EXTOPS=OFF \
        -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER="$rocroller" -DHIPBLASLT_ENABLE_THEROCK=ON \
        -DHIPBLASLT_ENABLE_MARKER="$marker" -DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF \
        -DHIPBLASLT_ENABLE_LAZY_LOAD=ON -DTENSILELITE_ENABLE_HOST=ON \
        -DTENSILELITE_BUILD_SHARED_LIBS=OFF -DTENSILELITE_ENABLE_CLIENT=OFF \
        -DTENSILELITE_BUILD_TESTING=OFF)
fi

cmake -S "$project_root" -B "$build_dir" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/amdclang++" \
    -DCMAKE_SKIP_RPATH=ON \
    -DCMAKE_CXX_FLAGS="$offload_flags" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-undefined -Wl,--exclude-libs,ALL" \
    -DCMAKE_PREFIX_PATH="$msgpack_prefix;$dependency_prefix;$ROCM_PATH;$ROCM_PATH/lib/host-math;$ROCM_PATH/lib/rocm_sysdeps" \
    -DCMAKE_LIBRARY_PATH="$ROCM_PATH/lib" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "${family_options[@]}" \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$compiler_launcher" \
    "$@"
cmake --build "$build_dir" --target hipblaslt --parallel "${BUILD_JOBS:-8}"

printf 'Host library: %s/library/libhipblaslt.so\n' "$build_dir"
printf 'Existing kernels: HIPBLASLT_TENSILE_LIBPATH=%s/lib/hipblaslt/library/%s\n' "$ROCM_PATH" "$gpu_target"
