#!/usr/bin/env bash
# Prepare the private runtime matching the selected ROCm native JAR.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/prepare-bundle.py" "$@"
