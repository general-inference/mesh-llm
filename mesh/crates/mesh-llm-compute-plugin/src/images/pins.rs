//! The images a released plugin build trusts.
//!
//! The release workflow builds the guest images first, then compiles the
//! plugin with `MESH_COMPUTE_IMAGE_PINS` set to their digests:
//! `base/x86_64=<sha256>,base/aarch64=<sha256>`. A plugin archive pinned by
//! the catalog therefore pins the images too. Builds without the variable
//! pin nothing and only use locally built images. Images are fetched from the
//! release named by [`release_tag`].

use anyhow::{Context, Result, bail};

/// One published image archive this build accepts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePin {
    pub name: String,
    pub arch: String,
    pub sha256: String,
}

impl ImagePin {
    /// The release asset holding this image in release `tag`.
    #[cfg_attr(not(feature = "image-download"), allow(dead_code))]
    pub fn asset_name(&self, tag: &str) -> String {
        format!("compute-image-{}-{tag}-{}.tar.gz", self.name, self.arch)
    }
}

/// The release this build belongs to. Release builds set
/// `MESH_COMPUTE_RELEASE_TAG`, because a preview's tag is not the crate version.
#[cfg_attr(not(feature = "image-download"), allow(dead_code))]
pub fn release_tag() -> String {
    option_env!("MESH_COMPUTE_RELEASE_TAG").map_or_else(
        || format!("v{}", env!("CARGO_PKG_VERSION")),
        ToString::to_string,
    )
}

/// The pins compiled into this build.
pub fn compiled() -> Vec<ImagePin> {
    option_env!("MESH_COMPUTE_IMAGE_PINS")
        .map(|pins| parse(pins).expect("MESH_COMPUTE_IMAGE_PINS is malformed"))
        .unwrap_or_default()
}

pub fn parse(text: &str) -> Result<Vec<ImagePin>> {
    text.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let (key, sha256) = entry.split_once('=').context("expected name/arch=sha256")?;
            let (name, arch) = key.split_once('/').context("expected name/arch")?;
            if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                bail!("`{entry}` does not end in a sha256");
            }
            Ok(ImagePin {
                name: name.to_string(),
                arch: arch.to_string(),
                sha256: sha256.to_ascii_lowercase(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_parse_and_name_their_release_assets() {
        let digest = "AB".repeat(32);
        let pins = parse(&format!("base/x86_64={digest}, base/aarch64={digest}")).unwrap();
        assert_eq!(pins.len(), 2);
        assert_eq!(pins[1].arch, "aarch64");
        assert_eq!(pins[0].sha256, "ab".repeat(32));
        assert_eq!(
            pins[0].asset_name("v0.80.0"),
            "compute-image-base-v0.80.0-x86_64.tar.gz"
        );
        assert!(parse("").unwrap().is_empty());
    }

    #[test]
    fn malformed_pins_are_refused() {
        for text in ["base=abc", "base/x86_64", "base/x86_64=nothex"] {
            assert!(parse(text).is_err(), "`{text}` should be refused");
        }
    }
}
