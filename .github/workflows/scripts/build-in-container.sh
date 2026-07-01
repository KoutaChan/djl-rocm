#!/usr/bin/env bash
# Build and publish pytorch-native's libdjl_torch for one Linux
# (flavor, classifier) combo inside a Docker container.
#
# Expected env vars:
#   PT_VERSION         e.g. 2.9.1
#   FLAVOR             e.g. cpu, cu128, rocm7.2
#   CLASSIFIER         e.g. linux-x86_64
#   GITHUB_ACTOR       passed through for Maven publish auth
#   GITHUB_TOKEN       passed through for Maven publish auth
#   GITHUB_REPOSITORY  target GitHub Packages owner/repo
#
# The caller mounts the repo at /ws and bind-mounts the libtorch extraction
# directory from runner temp space so the ~10GB zip unpack does not crowd the
# runner's tight root filesystem.
set -euxo pipefail

install_base_packages() {
    local packages=(git curl unzip cmake g++ make ca-certificates)
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
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rccl-dev
    export REQUIRE_DISTRIBUTED_NCCL=ON
}

restore_workspace_owner() {
    if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" /ws || true
    fi
}
trap restore_workspace_owner EXIT

install_base_packages
case "$FLAVOR" in
    cu*)
        install_cuda_packages
        ;;
    rocm*)
        install_rocm_packages
        ;;
esac

cd engines/pytorch/pytorch-native
./build.sh "$PT_VERSION" "$FLAVOR" cxx11 amd64
test -f build/libdjl_torch.so

cd /ws
./gradlew :engines:pytorch:pytorch-jni-rocm:publish \
    -Ppt_version="$PT_VERSION" \
    -Pflavor="$FLAVOR" \
    -Pclassifier="$CLASSIFIER" \
    -Pgithub \
    -PgithubRepo="$GITHUB_REPOSITORY" \
    -x test
