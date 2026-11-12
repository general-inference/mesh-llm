# Toolchain for the sandbox guest: the agent (built against the same Debian
# release as the image, so its glibc matches), the kernel, and mkfs.ext4.
FROM rust:1-bookworm
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bc bison build-essential cpio e2fsprogs flex kmod libelf-dev libssl-dev xz-utils \
    && rm -rf /var/lib/apt/lists/*
