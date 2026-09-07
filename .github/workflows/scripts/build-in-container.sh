#!/usr/bin/env bash
# Build and publish pytorch-native's libdjl_torch for one Linux
# (flavor, classifier) combo inside a Docker container.
#
# Expected env vars:
#   PT_VERSION         e.g. 2.9.1
#   FLAVOR             e.g. cpu, cu128, rocm7.2, rocm10.0
#   CLASSIFIER         e.g. linux-x86_64
#   GITHUB_ACTOR       passed through for Maven publish auth
#   GITHUB_TOKEN       passed through for Maven publish auth
#   GITHUB_REPOSITORY  target GitHub Packages owner/repo
#
# The caller mounts the repo at /ws and bind-mounts the libtorch extraction
# directory from runner temp space so the ~10GB zip unpack does not crowd the
# runner's tight root filesystem.
set -euo pipefail

timed() {
    local label=$1 start=$SECONDS
    shift
    printf '::group::%s\n' "$label"
    "$@"
    printf 'DJL_BUILD_PHASE %s seconds=%d\n' "$label" "$((SECONDS - start))"
    printf '::endgroup::\n'
}

install_base_packages() {
    local packages=(git curl unzip cmake g++ make ca-certificates python3 python3-pip python3-venv)
    if ! command -v java >/dev/null 2>&1; then
        packages+=(openjdk-21-jdk-headless)
    fi

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
    update-ca-certificates
}

install_cuda_packages() {
    local cu="${FLAVOR#cu}"
    local major=$((cu / 10))
    local minor=$((cu % 10))
    local suffix="${major}-${minor}"
    local distro="ubuntu2204"

    curl -fsSLo cuda-keyring_1.1-1_all.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "cuda-toolkit-${suffix}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnccl-dev \
        || echo "libnccl-dev is unavailable; native distributed training will use fallback stubs"

    export CUDA_HOME="/usr/local/cuda-${major}.${minor}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
}

install_rocm_packages() {
    if [[ "$FLAVOR" == rocm10.* ]]; then
        python3 -m venv /opt/djl-rocm-build
        source /opt/djl-rocm-build/bin/activate
        python -m pip install --no-cache-dir \
            --index-url https://stable.repo.amd.com/rocm/whl-next/ \
            "torch[device-gfx1100]==${PT_VERSION}+rocm10.0.0" \
            "rocm[libraries,devel,device-gfx1100]==10.0.0"
        rocm-sdk init
        export ROCM_PATH
        ROCM_PATH=$(rocm-sdk path --root)
        export CMAKE_PREFIX_PATH
        CMAKE_PREFIX_PATH=$(rocm-sdk path --cmake)
        export PATH="$(rocm-sdk path --bin):${PATH}"
        export PYTORCH_ROCM_ARCH=gfx1100
        return
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rccl-dev
    export REQUIRE_DISTRIBUTED_NCCL=ON
}

ensure_rocm_cmake() {
    local version
    version=$(cmake --version | awk 'NR == 1 { print $3 }')
    # ROCm 6.4's host sources require 3.25.2, while Ubuntu 22.04 supplies 3.22.
    if dpkg --compare-versions "$version" lt 3.25.2; then
        python3 -m venv /opt/djl-cmake
        /opt/djl-cmake/bin/python -m pip install --no-cache-dir cmake==3.30.8
        export PATH="/opt/djl-cmake/bin:${PATH}"
    fi
}

finish_build() {
    if command -v ccache >/dev/null 2>&1; then
        ccache --show-stats || true
    fi
    if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
        # Only these small caches leave the container. Do not walk /ws and its
        # multi-GiB libtorch/SDK trees merely to change file ownership.
        for directory in /djl-cache/ccache /djl-cache/gradle /djl-cache/hipblaslt/artifacts; do
            [[ ! -d "$directory" ]] || chown -R "${HOST_UID}:${HOST_GID}" "$directory" || true
        done
    fi
}
trap finish_build EXIT

timed packages install_base_packages
timed ccache-install bash .github/workflows/scripts/install-ccache.sh
export CCACHE_DIR=/djl-cache/ccache
export CCACHE_BASEDIR=/ws
export CCACHE_COMPILERCHECK=content
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-512Mi}
export GRADLE_USER_HOME=/djl-cache/gradle
export HIPBLASLT_CACHE_DIR=/djl-cache/hipblaslt
ccache --zero-stats
case "$FLAVOR" in
    cu*)
        timed cuda-sdk install_cuda_packages
        ;;
    rocm*)
        timed rocm-sdk install_rocm_packages
        timed rocm-cmake ensure_rocm_cmake
        ;;
esac

cd engines/pytorch/pytorch-native
timed native-build ./build.sh "$PT_VERSION" "$FLAVOR" cxx11 amd64
test -f build/libdjl_torch.so

cd /ws
timed publish ./gradlew :engines:pytorch:pytorch-jni-rocm:publish \
    -Ppt_version="$PT_VERSION" \
    -Pflavor="$FLAVOR" \
    -Pclassifier="$CLASSIFIER" \
    -Pgithub \
    -PgithubRepo="$GITHUB_REPOSITORY" \
    --build-cache \
    -x test
