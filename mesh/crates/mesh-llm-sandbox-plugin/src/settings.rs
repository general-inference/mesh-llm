//! Operator settings (`[plugin.settings]` in the node config): where state
//! lives, and the most any one sandbox on this node may ask for.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{
    PluginConfigSchemaBuilder, config_boolean, config_enum, config_integer, config_schema,
    config_setting, config_string,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::PLUGIN_ID;
use crate::egress::EgressMode;
use crate::spec::SandboxSpec;

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(default)]
pub struct Settings {
    /// Scratch disks, sockets, snapshots. Default `~/.mesh-llm/sandbox`.
    pub state_dir: Option<PathBuf>,
    /// Guest images. Default `<state_dir>/images`.
    pub images_dir: Option<PathBuf>,
    /// Sandboxes running on this node at once.
    pub max_sandboxes: u32,
    pub max_cpus: u32,
    pub max_memory_mb: u64,
    pub max_disk_gb: u64,
    /// The most network access a sandbox here may ask for.
    pub max_egress: EgressMode,
    pub max_ttl_secs: u64,
    /// Run sandboxes for other nodes in the mesh, not only this one.
    pub accept_remote: bool,
    /// Download the guest images this release publishes when one is
    /// missing. Off means only locally built images are used.
    pub image_download: bool,
    /// Where published images are fetched from: `<image_url>/<release tag>/...`.
    /// Default: the mesh-llm GitHub releases.
    pub image_url: Option<String>,
    /// Linux: the Firecracker binary. Default: the one shipped beside the
    /// plugin, then `firecracker` on PATH.
    pub firecracker_path: Option<PathBuf>,
    /// Windows: the WSL distribution to run Firecracker in. Default: the
    /// default distribution.
    pub wsl_distro: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            state_dir: None,
            images_dir: None,
            max_sandboxes: 4,
            max_cpus: default_max_cpus(),
            max_memory_mb: 8_192,
            max_disk_gb: 50,
            max_egress: EgressMode::Open,
            max_ttl_secs: 24 * 3_600,
            accept_remote: true,
            image_download: true,
            image_url: None,
            firecracker_path: None,
            wsl_distro: None,
        }
    }
}

impl Settings {
    pub fn parse(settings_json: &str) -> Result<Self> {
        let trimmed = settings_json.trim();
        if trimmed.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(trimmed).context("invalid sandbox plugin settings")
    }

    pub fn state_dir(&self) -> PathBuf {
        self.state_dir
            .clone()
            .unwrap_or_else(|| home_dir().join(".mesh-llm").join("sandbox"))
    }

    pub fn images_dir(&self) -> PathBuf {
        self.images_dir
            .clone()
            .unwrap_or_else(|| self.state_dir().join("images"))
    }

    /// The published images this build may download, unless turned off.
    pub fn published_images(&self) -> Option<crate::images::Published> {
        let pins = crate::images::pins::compiled();
        if !self.image_download || pins.is_empty() {
            return None;
        }
        Some(crate::images::Published {
            base_url: self
                .image_url
                .clone()
                .filter(|url| !url.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_IMAGE_URL.to_string()),
            pins,
        })
    }

    /// Refuses a config that asks for more than this node allows. Nothing is
    /// silently reduced: the caller learns which limit it hit.
    pub fn check(&self, spec: &SandboxSpec) -> Result<()> {
        let resources = &spec.resources;
        if resources.cpus == 0 || resources.memory_mb < 128 || resources.disk_gb == 0 {
            bail!("a sandbox needs at least 1 cpu, 128 MB of memory and 1 GB of disk");
        }
        if resources.cpus > self.max_cpus {
            bail!(
                "this node allows at most {} cpus per sandbox",
                self.max_cpus
            );
        }
        if resources.memory_mb > self.max_memory_mb {
            bail!(
                "this node allows at most {} MB of memory per sandbox",
                self.max_memory_mb
            );
        }
        if resources.disk_gb > self.max_disk_gb {
            bail!(
                "this node allows at most {} GB of disk per sandbox",
                self.max_disk_gb
            );
        }
        if spec.egress.mode > self.max_egress {
            bail!(
                "this node allows egress mode `{}` at most",
                mode_name(self.max_egress)
            );
        }
        if spec.ttl_secs == 0 || spec.ttl_secs > self.max_ttl_secs {
            bail!(
                "ttl_secs must be between 1 and {} on this node",
                self.max_ttl_secs
            );
        }
        Ok(())
    }

    pub fn config_schema() -> PluginConfigSchemaBuilder {
        let defaults = Self::default();
        config_schema(PLUGIN_ID)
            .setting(
                config_setting("state_dir", config_string()).description(
                    "Scratch disks, sockets and snapshots. Default ~/.mesh-llm/sandbox.",
                ),
            )
            .setting(
                config_setting("images_dir", config_string())
                    .description("Guest images. Default <state_dir>/images."),
            )
            .setting(
                config_setting("max_sandboxes", config_integer())
                    .default_value(&defaults.max_sandboxes)
                    .description("Sandboxes running on this node at once."),
            )
            .setting(
                config_setting("max_cpus", config_integer())
                    .description("Most cpus one sandbox may ask for. Default: all of them."),
            )
            .setting(
                config_setting("max_memory_mb", config_integer())
                    .default_value(&defaults.max_memory_mb)
                    .description("Most memory one sandbox may ask for."),
            )
            .setting(
                config_setting("max_disk_gb", config_integer())
                    .default_value(&defaults.max_disk_gb)
                    .description("Most disk one sandbox may ask for."),
            )
            .setting(
                config_setting("max_egress", config_enum(["none", "allowlist", "open"]))
                    .default_value(&defaults.max_egress)
                    .description("The most network access a sandbox here may ask for."),
            )
            .setting(
                config_setting("max_ttl_secs", config_integer())
                    .default_value(&defaults.max_ttl_secs)
                    .description("Longest a sandbox may live."),
            )
            .setting(
                config_setting("accept_remote", config_boolean())
                    .default_value(&defaults.accept_remote)
                    .description("Run sandboxes for other nodes in the mesh."),
            )
            .setting(
                config_setting("image_download", config_boolean())
                    .default_value(&defaults.image_download)
                    .description("Download the guest images this release publishes when missing."),
            )
            .setting(
                config_setting("image_url", config_string())
                    .description("Where published images are fetched from. Default: mesh-llm GitHub releases."),
            )
            .setting(
                config_setting("firecracker_path", config_string()).description(
                    "Linux: the Firecracker binary. Default: the bundled one, then firecracker on PATH.",
                ),
            )
            .setting(
                config_setting("wsl_distro", config_string())
                    .description("Windows: the WSL distribution to run Firecracker in."),
            )
    }
}

const DEFAULT_IMAGE_URL: &str = "https://github.com/Mesh-LLM/mesh-llm/releases/download";

pub fn mode_name(mode: EgressMode) -> &'static str {
    match mode {
        EgressMode::None => "none",
        EgressMode::Allowlist => "allowlist",
        EgressMode::Open => "open",
    }
}

fn default_max_cpus() -> u32 {
    std::thread::available_parallelism()
        .ok()
        .and_then(|count| u32::try_from(count.get()).ok())
        .unwrap_or(2)
}

fn home_dir() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map_or_else(std::env::temp_dir, PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::EgressSpec;

    #[test]
    fn empty_settings_take_the_defaults() {
        assert_eq!(Settings::parse("{}").unwrap(), Settings::default());
        assert_eq!(Settings::parse("").unwrap(), Settings::default());
    }

    #[test]
    fn requests_over_a_limit_are_refused_with_the_limit() {
        let settings = Settings {
            max_cpus: 4,
            max_egress: EgressMode::Allowlist,
            ..Settings::default()
        };
        let mut spec = SandboxSpec::default();
        assert!(settings.check(&spec).is_ok());

        spec.resources.cpus = 8;
        let error = settings.check(&spec).unwrap_err().to_string();
        assert!(error.contains("4 cpus"), "{error}");

        spec.resources.cpus = 2;
        spec.egress = EgressSpec {
            mode: EgressMode::Open,
            ..EgressSpec::default()
        };
        let error = settings.check(&spec).unwrap_err().to_string();
        assert!(error.contains("allowlist"), "{error}");
    }

    #[test]
    fn the_operator_can_turn_egress_off_entirely() {
        let settings = Settings::parse(r#"{"max_egress":"none"}"#).unwrap();
        let mut spec = SandboxSpec::default();
        assert!(settings.check(&spec).is_ok());
        spec.egress.mode = EgressMode::Allowlist;
        assert!(settings.check(&spec).is_err());
    }
}
