//! The sandbox config a caller passes to `sandbox_create`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::egress::EgressSpec;

pub const DEFAULT_IMAGE: &str = "base";

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SandboxSpec {
    /// Image name; `base` is a Debian userland with common toolchains.
    #[serde(default = "default_image")]
    pub image: String,
    #[serde(default)]
    pub resources: Resources,
    /// Network access. Omitted means no network at all.
    #[serde(default)]
    pub egress: EgressSpec,
    /// The sandbox is destroyed this long after creation.
    #[serde(default = "default_ttl_secs")]
    pub ttl_secs: u64,
    /// Environment added to every command run in the sandbox.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Where to run it: `local`, or a peer id (or unique prefix of one).
    /// Omitted lets the mesh choose.
    #[serde(default)]
    pub node: Option<String>,
    /// Prefer a node serving this model, so agent turns and tool calls stay
    /// on one machine.
    #[serde(default)]
    pub near_model: Option<String>,
    /// Restore `/workspace` from a snapshot taken with `sandbox_snapshot`.
    #[serde(default)]
    pub from_snapshot: Option<SnapshotRef>,
}

impl Default for SandboxSpec {
    fn default() -> Self {
        Self {
            image: default_image(),
            resources: Resources::default(),
            egress: EgressSpec::default(),
            ttl_secs: default_ttl_secs(),
            env: BTreeMap::new(),
            node: None,
            near_model: None,
            from_snapshot: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    #[serde(default = "default_cpus")]
    pub cpus: u32,
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,
    /// Size of the writable disk holding `/workspace` and installed packages.
    #[serde(default = "default_disk_gb")]
    pub disk_gb: u64,
}

impl Default for Resources {
    fn default() -> Self {
        Self {
            cpus: default_cpus(),
            memory_mb: default_memory_mb(),
            disk_gb: default_disk_gb(),
        }
    }
}

/// A snapshot and the node that holds it. `sandbox_snapshot`'s result can be
/// passed as is, so its other fields are accepted and ignored.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct SnapshotRef {
    pub snapshot_id: String,
    pub node: String,
}

fn default_image() -> String {
    DEFAULT_IMAGE.to_string()
}

fn default_ttl_secs() -> u64 {
    3_600
}

fn default_cpus() -> u32 {
    2
}

fn default_memory_mb() -> u64 {
    2_048
}

fn default_disk_gb() -> u64 {
    10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::EgressMode;

    #[test]
    fn an_empty_config_is_a_small_offline_sandbox() {
        let spec: SandboxSpec = serde_json::from_str("{}").unwrap();
        assert_eq!(spec, SandboxSpec::default());
        assert_eq!(spec.egress.mode, EgressMode::None);
        assert_eq!(spec.resources.cpus, 2);
    }

    #[test]
    fn the_documented_config_parses() {
        let spec: SandboxSpec = serde_json::from_value(serde_json::json!({
            "image": "base",
            "resources": { "cpus": 4, "memory_mb": 8192, "disk_gb": 20 },
            "egress": {
                "mode": "allowlist",
                "allow": ["@registries", "api.example.com:443", "10.20.0.0/16"],
                "deny": ["*.tracking.example.com"]
            },
            "ttl_secs": 3600
        }))
        .unwrap();
        assert_eq!(spec.resources.memory_mb, 8_192);
        assert_eq!(spec.egress.allow.len(), 3);
    }

    #[test]
    fn a_snapshot_result_can_be_passed_back_as_is() {
        let spec: SandboxSpec = serde_json::from_value(serde_json::json!({
            "from_snapshot": { "snapshot_id": "snap-1", "node": "abc", "bytes": 157 }
        }))
        .unwrap();
        assert_eq!(spec.from_snapshot.unwrap().node, "abc");
    }

    #[test]
    fn misspelled_fields_are_rejected_rather_than_ignored() {
        let result: Result<SandboxSpec, _> =
            serde_json::from_value(serde_json::json!({ "egres": { "mode": "open" } }));
        assert!(result.is_err());
    }
}
