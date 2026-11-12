# The `base` sandbox image: a Debian userland with common toolchains. It is
# exported to a read-only ext4 root; everything a sandbox installs or writes
# lands on its own scratch disk.
FROM debian:bookworm-slim
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential bzip2 ca-certificates curl e2fsprogs file git jq less \
        nodejs npm openssh-client pkg-config procps python3 python3-pip python3-venv \
        tar unzip wget xz-utils zip \
    && rm -rf /var/lib/apt/lists/*
COPY mesh-compute-guest /usr/sbin/mesh-compute-guest
