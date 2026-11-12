#!/usr/bin/env bash
# Runs inside the builder container: builds the guest kernel into /work.
set -euo pipefail
version="${KERNEL_VERSION:?}"
cd /tmp
curl -fsSL "https://cdn.kernel.org/pub/linux/kernel/v${version%%.*}.x/linux-${version}.tar.xz" | tar -xJ
cd "linux-${version}"
make -s defconfig
scripts/kconfig/merge_config.sh -m .config /image/kernel.config >/dev/null
make -s olddefconfig
for option in VIRTIO_BLK VIRTIO_CONSOLE VIRTIO_VSOCKETS VIRTIO_MMIO VIRTIO_PCI HW_RANDOM_VIRTIO EXT4_FS OVERLAY_FS DEVTMPFS; do
    grep -qx "CONFIG_${option}=y" .config || { echo "kernel config lacks CONFIG_${option}=y" >&2; exit 1; }
done
case "$(uname -m)" in
    aarch64) make -s -j"$(nproc)" Image && cp arch/arm64/boot/Image /work/kernel ;;
    x86_64) make -s -j"$(nproc)" vmlinux && cp vmlinux /work/kernel ;;
    *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
