#!/usr/bin/env bash
# Build pytorch-native's libdjl_torch for one (flavor, classifier) combo
# inside a Linux docker container (ubuntu-based). Used by the jni-rocm
# matrix in Build JNI to target rocm/dev-ubuntu-*-complete images.
#
# Expected env vars:
#   PT_VERSION         e.g. 2.9.1
#   FLAVOR             e.g. rocm7.2
#   CLASSIFIER         e.g. linux-x86_64
#   GITHUB_ACTOR       passed through for Maven publish auth
#   GITHUB_TOKEN       passed through for Maven publish auth
#   GITHUB_REPOSITORY  target GitHub Packages owner/repo
#
# The caller mounts the repo at /ws and typically bind-mounts the libtorch
# extraction directory onto /mnt-backed space so the ~10GB zip unpack does
# not crowd the runner's tight root filesystem.
set -euxo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git curl unzip cmake g++ ca-certificates openjdk-21-jdk-headless
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rccl-dev \
    || DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "rccl-dev${FLAVOR#rocm}.0" \
    || echo "rccl-dev is unavailable; native distributed training will use fallback stubs"
update-ca-certificates

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
