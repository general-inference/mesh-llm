#!/usr/bin/env bash
# Builds a sandbox guest image for this machine's architecture:
#
#   <out>/<name>/<arch>/kernel        uncompressed kernel (Image or vmlinux)
#   <out>/<name>/<arch>/rootfs.ext4   read-only root with the guest agent
#   <out>/<name>/<arch>/image.json    what went into it
#
# Needs Docker (on Linux, or Docker Desktop on a Mac).
# Usage: build.sh <out-dir>   (IMAGE_NAME, KERNEL_VERSION override defaults)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
out="${1:?usage: build.sh <out-dir>}"
name="${IMAGE_NAME:-base}"
kernel_version="${KERNEL_VERSION:-6.12.10}"
arch="$(uname -m)"
[ "$arch" = arm64 ] && arch=aarch64
dest="$out/$name/$arch"
work="$(mktemp -d)"
# The containers write as root; clean up through one too.
trap 'docker run --rm -v "$work:/work" debian:bookworm-slim sh -c "rm -rf /work/*" >/dev/null 2>&1 || true; rm -rf "$work"' EXIT

echo "==> builder image"
docker build -q -t mesh-sandbox-builder -f "$here/builder.Dockerfile" "$here" >/dev/null

echo "==> guest agent"
docker run --rm -v "$repo:/src:ro" -v "$work:/work" -e RUSTC_WRAPPER= -e CARGO_TARGET_DIR=/work/target \
    mesh-sandbox-builder \
    cargo build --quiet --release --locked --manifest-path /src/Cargo.toml \
        -p mesh-llm-sandbox-guest --bin mesh-sandbox-guest
mkdir -p "$work/rootfs-context"
cp "$work/target/release/mesh-sandbox-guest" "$work/rootfs-context/"

echo "==> root filesystem ($name)"
docker build -q -t "mesh-sandbox-rootfs-$name" -f "$here/rootfs.Dockerfile" "$work/rootfs-context" >/dev/null
container="$(docker create "mesh-sandbox-rootfs-$name")"
docker export "$container" -o "$work/rootfs.tar"
docker rm "$container" >/dev/null
docker run --rm -v "$work:/work" -e OWNER="$(id -u):$(id -g)" mesh-sandbox-builder bash -euo pipefail -c '
    mkdir /work/root
    tar -xpf /work/rootfs.tar -C /work/root
    rm -f /work/root/.dockerenv
    printf "127.0.0.1 localhost sandbox\n::1 localhost\n" > /work/root/etc/hosts
    echo sandbox > /work/root/etc/hostname
    : > /work/root/etc/resolv.conf
    size=$(du -sm /work/root | cut -f1)
    mkfs.ext4 -q -F -L sandbox-root -O ^has_journal -d /work/root /work/rootfs.ext4 "$(( size + size / 5 + 256 ))M"
    chown "$OWNER" /work/rootfs.ext4'

echo "==> kernel $kernel_version"
docker run --rm -v "$work:/work" -v "$here:/image:ro" -e KERNEL_VERSION="$kernel_version" \
    -e OWNER="$(id -u):$(id -g)" mesh-sandbox-builder bash -euo pipefail -c '/image/build-kernel.sh && chown "$OWNER" /work/kernel'

mkdir -p "$dest"
mv "$work/kernel" "$dest/kernel"
mv "$work/rootfs.ext4" "$dest/rootfs.ext4"
cat > "$dest/image.json" <<JSON
{
  "name": "$name",
  "arch": "$arch",
  "kernel_version": "$kernel_version",
  "source_commit": "$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
echo "==> $dest"
ls -l "$dest"
