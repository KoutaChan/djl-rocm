#!/usr/bin/env bash
# Build the small CPU-only bootstrap used to load the private ROCm overlay.
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$1
java_root=${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}
mkdir -p "$(dirname "$output")"
"${CC:-cc}" -O2 -shared -fPIC -Wall -Wextra -Werror \
    -I"$java_root/include" -I"$java_root/include/linux" \
    "$source_dir/djl_rocm_loader.c" -Wl,-soname,libdjl_rocm_loader.so \
    -o "$output" -ldl
