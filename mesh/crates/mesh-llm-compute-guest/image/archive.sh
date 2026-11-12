#!/usr/bin/env bash
# Packs a built image as the release asset the plugin downloads:
#
#   archive.sh <images-dir> <name> <release-tag> <out-dir>
#
# writes <out-dir>/compute-image-<name>-<release-tag>-<arch>.tar.gz (kernel,
# rootfs.ext4 and image.json at the top level) and prints its sha256, which
# the release compiles into the plugin as a pin.
set -euo pipefail
images="${1:?usage: archive.sh <images-dir> <name> <release-tag> <out-dir>}"
name="${2:?}"
tag="${3:?}"
out="${4:?}"
arch="$(uname -m)"
[ "$arch" = arm64 ] && arch=aarch64
source="$images/$name/$arch"
for file in kernel rootfs.ext4 image.json; do
    [ -f "$source/$file" ] || { echo "archive.sh: $source/$file is missing" >&2; exit 1; }
done
mkdir -p "$out"
archive="$out/compute-image-$name-$tag-$arch.tar.gz"
tar -czf "$archive" -C "$source" kernel rootfs.ext4 image.json
if command -v sha256sum >/dev/null; then sha256sum "$archive"; else shasum -a 256 "$archive"; fi | cut -d' ' -f1
