//! Guest images on this node.
//!
//! An image is a directory `<images>/<name>/<arch>/` holding `kernel` (an
//! uncompressed kernel: `vmlinux` ELF on x86_64, `Image` on aarch64) and
//! `rootfs.ext4` (the read-only root, with the agent at
//! `/sbin/mesh-compute-guest`).
//!
//! Released plugin builds carry the sha256 of each image published with the
//! same mesh-llm release ([`pins`]) and download a missing one on first use
//! ([`download`]). Other builds use images built locally with
//! `just compute-image`.

#[cfg(feature = "image-download")]
mod download;
pub mod pins;

use std::path::PathBuf;

use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::Serialize;

use self::pins::ImagePin;

pub const KERNEL_FILE: &str = "kernel";
pub const ROOTFS_FILE: &str = "rootfs.ext4";

/// Where published images are fetched from, and which ones this build trusts.
#[derive(Clone, Debug)]
pub struct Published {
    /// Release download base: `<base>/<tag>/<archive>`.
    #[cfg_attr(not(feature = "image-download"), allow(dead_code))]
    pub base_url: String,
    pub pins: Vec<ImagePin>,
}

pub struct ImageStore {
    root: PathBuf,
    published: Option<Published>,
    /// One download at a time; a second caller waits and then finds the image.
    #[cfg_attr(not(feature = "image-download"), allow(dead_code))]
    downloading: tokio::sync::Mutex<()>,
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
    pub fn new(root: PathBuf, published: Option<Published>) -> Self {
        Self {
            root,
            published,
            downloading: tokio::sync::Mutex::new(()),
        }
    }

    /// An installed image.
    pub fn get(&self, name: &str) -> Result<Image> {
        validate_name(name)?;
        let dir = self.root.join(name).join(guest_arch());
        let image = Image {
            kernel: dir.join(KERNEL_FILE),
            rootfs: dir.join(ROOTFS_FILE),
        };
        if !image.kernel.is_file() || !image.rootfs.is_file() {
            bail!(
                "image `{name}` is not installed for {} (looked in {}); build it with `just compute-image`",
                guest_arch(),
                dir.display()
            );
        }
        Ok(image)
    }

    /// An installed image, downloading it first if this build pins one.
    pub async fn ensure(&self, name: &str) -> Result<Image> {
        if let Ok(image) = self.get(name) {
            return Ok(image);
        }
        let Some(pin) = self.pin(name) else {
            return self.get(name);
        };
        self.fetch(name, &pin).await
    }

    #[cfg(feature = "image-download")]
    async fn fetch(&self, name: &str, pin: &ImagePin) -> Result<Image> {
        let _only_one = self.downloading.lock().await;
        if let Ok(image) = self.get(name) {
            return Ok(image);
        }
        let published = self
            .published
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("image downloads are off"))?;
        download::install(&self.root, &published.base_url, pin).await?;
        self.get(name)
    }

    #[cfg(not(feature = "image-download"))]
    async fn fetch(&self, name: &str, _pin: &ImagePin) -> Result<Image> {
        self.get(name)
    }

    fn pin(&self, name: &str) -> Option<ImagePin> {
        self.published.as_ref().and_then(|published| {
            published
                .pins
                .iter()
                .find(|pin| pin.name == name && pin.arch == guest_arch())
                .cloned()
        })
    }

    /// Images this node can run now or after a download.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.list().into_iter().map(|image| image.name).collect();
        if let Some(published) = &self.published {
            for pin in &published.pins {
                if pin.arch == guest_arch() && !names.contains(&pin.name) {
                    names.push(pin.name.clone());
                }
            }
        }
        names.sort();
        names
    }

    /// Images installed for this node's architecture.
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

        let store = ImageStore::new(dir.path().to_path_buf(), None);
        assert!(store.get("base").is_ok());
        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "base");
        assert_eq!(listed[0].rootfs_bytes, 6);
        assert!(store.get("half").is_err());
    }

    #[test]
    fn names_cannot_escape_the_image_directory() {
        let store = ImageStore::new(PathBuf::from("/images"), None);
        for name in ["..", "../etc", "a/b", "", ".hidden"] {
            assert!(store.get(name).is_err(), "`{name}` should be rejected");
        }
    }

    #[test]
    fn pinned_images_for_this_architecture_count_as_available() {
        let dir = tempfile::tempdir().unwrap();
        let pin = |name: &str, arch: &str| ImagePin {
            name: name.into(),
            arch: arch.into(),
            sha256: "0".repeat(64),
        };
        let store = ImageStore::new(
            dir.path().to_path_buf(),
            Some(Published {
                base_url: "https://example.invalid".into(),
                pins: vec![pin("base", guest_arch()), pin("other", "riscv64")],
            }),
        );
        assert_eq!(store.names(), vec!["base".to_string()]);
        assert!(store.list().is_empty());
    }
}
