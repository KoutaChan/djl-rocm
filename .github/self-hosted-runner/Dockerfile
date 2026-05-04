FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNER_HOME=/actions-runner

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
RUN chmod +x /usr/local/bin/runner-entrypoint

WORKDIR /actions-runner
ENTRYPOINT ["runner-entrypoint"]
