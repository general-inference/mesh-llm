#!/usr/bin/env bash
# Assembles one `compute` plugin archive for `mesh-llm plugins install`.
#
#   package.sh --target <triple> --tag <release-tag> --binary <path> \
#              --manifest <plugin-manifest.json> --out <dir> \
#              [--linux-binary <path>]        # Windows: the Linux build run inside WSL
#
# Linux archives ship Firecracker beside the plugin; Windows archives ship the
# Linux build and Firecracker under linux/ for WSL. Firecracker is fetched from
# its GitHub release and checked against firecracker.pin. macOS binaries must
# already be signed (sign-macos.sh, which also verifies the entitlement);
# packaging never changes their bytes, so it can run on any host.
set -euo pipefail
# macOS tar would add AppleDouble ._ files for extended attributes.
export COPYFILE_DISABLE=1
here="$(cd "$(dirname "$0")" && pwd)"
crate="$(cd "$here/.." && pwd)"

target="" tag="" binary="" manifest="" out="" linux_binary=""
while [ $# -gt 0 ]; do
    case "$1" in
        --target) target="$2" ;;
        --tag) tag="$2" ;;
        --binary) binary="$2" ;;
        --manifest) manifest="$2" ;;
        --out) out="$2" ;;
        --linux-binary) linux_binary="$2" ;;
        *) echo "package.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift 2
done
for required in target tag binary manifest out; do
    [ -n "${!required}" ] || { echo "package.sh: --${required//_/-} is required" >&2; exit 2; }
done

pin() { sed -n "s/^$1=//p" "$here/firecracker.pin"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# Downloads Firecracker for <arch> into <dir>, with its licence files.
fetch_firecracker() {
    local arch="$1" dir="$2" fc_version digest work tarball
    fc_version="$(pin version)"
    digest="$(pin "$arch")"
    work="$(mktemp -d)"
    tarball="$work/firecracker.tgz"
    curl -fsSL --retry 3 -o "$tarball" \
        "https://github.com/firecracker-microvm/firecracker/releases/download/$fc_version/firecracker-$fc_version-$arch.tgz"
    if [ "$(sha256 "$tarball")" != "$digest" ]; then
        echo "package.sh: Firecracker $fc_version $arch does not match firecracker.pin" >&2
        exit 1
    fi
    tar -xzf "$tarball" -C "$work"
    mkdir -p "$dir" "$stage/compute/licenses/firecracker"
    install -m 0755 "$work/release-$fc_version-$arch/firecracker-$fc_version-$arch" "$dir/firecracker"
    cp "$work/release-$fc_version-$arch/"{LICENSE,NOTICE,THIRD-PARTY} "$stage/compute/licenses/firecracker/"
    rm -rf "$work"
}

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/compute" "$out"
cp "$crate/README.md" "$stage/compute/"
printf 'name = "compute"\nversion = "%s"\n' "${tag#v}" > "$stage/compute/plugin.toml"
cp "$manifest" "$stage/compute/plugin-manifest.json"

case "$target" in
    *-windows-msvc)
        cp "$binary" "$stage/compute/compute.exe"
        if [ "$target" = x86_64-pc-windows-msvc ]; then
            [ -n "$linux_binary" ] || { echo "package.sh: Windows needs --linux-binary" >&2; exit 2; }
            mkdir -p "$stage/compute/linux"
            install -m 0755 "$linux_binary" "$stage/compute/linux/mesh-compute"
            fetch_firecracker x86_64 "$stage/compute/linux"
        fi
        archive="$out/compute-$tag-$target.zip"
        rm -f "$archive"
        (cd "$stage" && zip -qrX "$archive" compute)
        ;;
    *-linux-gnu)
        install -m 0755 "$binary" "$stage/compute/compute"
        fetch_firecracker "${target%%-*}" "$stage/compute"
        archive="$out/compute-$tag-$target.tar.gz"
        tar -czf "$archive" -C "$stage" compute
        ;;
    *-apple-darwin)
        install -m 0755 "$binary" "$stage/compute/compute"
        archive="$out/compute-$tag-$target.tar.gz"
        tar -czf "$archive" -C "$stage" compute
        ;;
    *) echo "package.sh: unsupported target $target" >&2; exit 2 ;;
esac
echo "$archive"
