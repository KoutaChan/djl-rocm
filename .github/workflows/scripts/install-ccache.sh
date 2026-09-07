#!/usr/bin/env bash
# Pin a HIP-aware ccache; Ubuntu 22.04's package predates offload support fixes.
set -euo pipefail
version=4.13.3
archive="ccache-${version}-linux-x86_64-musl-static.tar.gz"
digest=0c023908e0027396b3a4e1435278e1d9ffd79a36b4487255b04bacf8eb2d415f
directory=$(mktemp -d)
trap 'rm -rf "$directory"' EXIT
curl -fsSL --retry 3 "https://github.com/ccache/ccache/releases/download/v${version}/${archive}" -o "$directory/$archive"
printf '%s  %s\n' "$digest" "$directory/$archive" | sha256sum --check
tar -xzf "$directory/$archive" -C "$directory"
install -m 0755 "$directory/${archive%.tar.gz}/ccache" /usr/local/bin/ccache
ccache --version
