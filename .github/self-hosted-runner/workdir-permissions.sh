#!/usr/bin/env bash
set -euo pipefail

workdir="${RUNNER_WORKDIR:-}"
if [[ -z "${workdir}" ]]; then
    exit 0
fi

case "${workdir}" in
    /runner/*/_work)
        ;;
    *)
        echo "Refusing to repair unexpected runner workdir: ${workdir}" >&2
        exit 1
        ;;
esac

if [[ ! -d "${workdir}" ]]; then
    exit 0
fi

sudo find "${workdir}" \( ! -user runner -o ! -group runner \) -exec chown runner:runner {} +
sudo find "${workdir}" -type d ! -perm -u+rwx -exec chmod u+rwx {} +
sudo find "${workdir}" -type f ! -perm -u+rw -exec chmod u+rw {} +
