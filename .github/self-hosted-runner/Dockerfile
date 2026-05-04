FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNER_HOME=/actions-runner
ENV ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/runner-workdir-permissions
ENV ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/usr/local/bin/runner-workdir-permissions

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        curl \
        docker.io \
        git \
        gosu \
        gzip \
        jq \
        libcurl4 \
        libicu70 \
        libkrb5-3 \
        liblttng-ust1 \
        libssl3 \
        sudo \
        tar \
        unzip \
        xz-utils \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1001 --shell /bin/bash runner \
    && usermod -aG docker runner \
    && echo "runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/runner \
    && chmod 0440 /etc/sudoers.d/runner \
    && mkdir -p "${RUNNER_HOME}" /runner \
    && chown -R runner:runner "${RUNNER_HOME}" /runner

COPY entrypoint.sh /usr/local/bin/runner-entrypoint
COPY workdir-permissions.sh /usr/local/bin/runner-workdir-permissions
RUN chmod +x /usr/local/bin/runner-entrypoint /usr/local/bin/runner-workdir-permissions

WORKDIR /actions-runner
ENTRYPOINT ["runner-entrypoint"]
