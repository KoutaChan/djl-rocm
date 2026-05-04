#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_URL:?GITHUB_URL is required, for example https://github.com/OWNER/REPO}"

RUNNER_HOME="${RUNNER_HOME:-/actions-runner}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-/runner/_work}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-djl-linux-docker,docker}"
RUNNER_EPHEMERAL="${RUNNER_EPHEMERAL:-false}"

case "$(uname -m)" in
    x86_64)
        RUNNER_ARCH=x64
        ;;
    aarch64|arm64)
        RUNNER_ARCH=arm64
        ;;
    *)
        echo "Unsupported runner architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

mkdir -p "${RUNNER_HOME}" "${RUNNER_WORKDIR}"
chown -R runner:runner "${RUNNER_HOME}" "$(dirname "${RUNNER_WORKDIR}")"
cd "${RUNNER_HOME}"

download_runner() {
    if [[ -x ./config.sh && -x ./run.sh ]]; then
        return
    fi

    local version="${RUNNER_VERSION:-}"
    if [[ -z "${version}" ]]; then
        version="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
            | jq -r '.tag_name | sub("^v"; "")')"
    fi

    if [[ -z "${version}" || "${version}" == "null" ]]; then
        echo "Unable to resolve actions runner version" >&2
        exit 1
    fi

    local archive="/tmp/actions-runner-linux-${RUNNER_ARCH}-${version}.tar.gz"
    curl -fsSL \
        "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-${RUNNER_ARCH}-${version}.tar.gz" \
        -o "${archive}"
    tar -xzf "${archive}" -C "${RUNNER_HOME}"
    rm -f "${archive}"
    chown -R runner:runner "${RUNNER_HOME}"
}

wait_for_docker() {
    if [[ -z "${DOCKER_HOST:-}" ]]; then
        return
    fi

    for _ in $(seq 1 60); do
        if docker version >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done

    echo "Docker daemon did not become ready at ${DOCKER_HOST}" >&2
    exit 1
}

configure_runner() {
    if [[ -f .runner ]]; then
        return
    fi

    : "${RUNNER_TOKEN:?RUNNER_TOKEN is required when the runner has not been configured yet}"

    local args=(
        --unattended
        --url "${GITHUB_URL}"
        --token "${RUNNER_TOKEN}"
        --name "${RUNNER_NAME}"
        --labels "${RUNNER_LABELS}"
        --work "${RUNNER_WORKDIR}"
        --replace
    )

    if [[ -n "${RUNNER_GROUP:-}" ]]; then
        args+=(--runnergroup "${RUNNER_GROUP}")
    fi

    if [[ "${RUNNER_EPHEMERAL}" == "true" ]]; then
        args+=(--ephemeral)
    fi

    gosu runner ./config.sh "${args[@]}"
}

download_runner
wait_for_docker
configure_runner

exec gosu runner ./run.sh
