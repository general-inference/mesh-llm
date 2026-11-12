//! Guest images installed on this node.
//!
//! An image is a directory `<images>/<name>/<arch>/` holding `kernel` (an
//! uncompressed kernel: `vmlinux` ELF on x86_64, `Image` on aarch64) and
//! `rootfs.ext4` (the read-only root, with the agent at
//! `/sbin/mesh-sandbox-guest`). `just sandbox-image` builds them.

use std::path::PathBuf;

use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::Serialize;

pub const KERNEL_FILE: &str = "kernel";
pub const ROOTFS_FILE: &str = "rootfs.ext4";

pub struct ImageStore {
    root: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub kernel: PathBuf,
    pub rootfs: PathBuf,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ImageInfo {
    pub name: String,
    pub arch: String,
    pub rootfs_bytes: u64,
}

impl ImageStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn get(&self, name: &str) -> Result<Image> {
        validate_name(name)?;
        let dir = self.root.join(name).join(guest_arch());
        let image = Image {
            kernel: dir.join(KERNEL_FILE),
            rootfs: dir.join(ROOTFS_FILE),
        };
        if !image.kernel.is_file() || !image.rootfs.is_file() {
            bail!(
                "image `{name}` is not installed for {} (looked in {}); build it with `just sandbox-image`",
                guest_arch(),
                dir.display()
            );
        }
        Ok(image)
    }

    /// Names of images installed for this node's architecture.
    pub fn list(&self) -> Vec<ImageInfo> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut images: Vec<ImageInfo> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_str()?.to_string();
                let image = self.get(&name).ok()?;
                Some(ImageInfo {
                    name,
                    arch: guest_arch().to_string(),
                    rootfs_bytes: std::fs::metadata(&image.rootfs).ok()?.len(),
                })
            })
            .collect();
        images.sort_by(|left, right| left.name.cmp(&right.name));
        images
    }
}

/// The guest architecture this node runs: the same as the host's.
pub fn guest_arch() -> &'static str {
    std::env::consts::ARCH
}

fn validate_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !valid {
        bail!("invalid image name `{name}`");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_images_are_found_and_listed() {
        let dir = tempfile::tempdir().unwrap();
        let image_dir = dir.path().join("base").join(guest_arch());
        std::fs::create_dir_all(&image_dir).unwrap();
        std::fs::write(image_dir.join(KERNEL_FILE), b"k").unwrap();
        std::fs::write(image_dir.join(ROOTFS_FILE), b"rootfs").unwrap();
        std::fs::create_dir_all(dir.path().join("half").join(guest_arch())).unwrap();

        let store = ImageStore::new(dir.path().to_path_buf());
        assert!(store.get("base").is_ok());
        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "base");
        assert_eq!(listed[0].rootfs_bytes, 6);
        assert!(store.get("half").is_err());
    }

    #[test]
    fn names_cannot_escape_the_image_directory() {
        let store = ImageStore::new(PathBuf::from("/images"));
        for name in ["..", "../etc", "a/b", "", ".hidden"] {
            assert!(store.get(name).is_err(), "`{name}` should be rejected");
        }
    }
}
