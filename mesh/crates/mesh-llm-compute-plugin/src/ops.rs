//! Operations on a node's sandboxes, in a form that runs the same locally or
//! on a peer: a peer receives an [`Op`] on the `sandbox.v1` channel, runs it
//! with [`run`] as [`Caller::Peer`], and sends the [`OpResult`] back.

use mesh_llm_sandbox_guest::protocol::{DirEntry, ExecRequest, ExitStatus};
use serde::{Deserialize, Serialize};

use crate::egress::proxy::EgressEvent;
use crate::manager::{Caller, Capacity, Manager, SandboxInfo, SnapshotInfo};
use crate::spec::SandboxSpec;

/// The most a single file read or snapshot chunk carries between nodes; a
/// channel frame tops out at 10 MB and these travel base64-encoded.
pub const MAX_TRANSFER_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// A snapshot named in the spec is fetched by the node that runs it.
    Create {
        spec: SandboxSpec,
    },
    Exec {
        sandbox_id: String,
        request: ExecRequest,
    },
    ReadFile {
        sandbox_id: String,
        path: String,
        max_bytes: u64,
    },
    WriteFile {
        sandbox_id: String,
        path: String,
        #[serde(with = "b64")]
        data: Vec<u8>,
        mode: Option<u32>,
        append: bool,
    },
    ListDir {
        sandbox_id: String,
        path: String,
    },
    Snapshot {
        sandbox_id: String,
    },
    SnapshotChunk {
        snapshot_id: String,
        offset: u64,
        len: u64,
    },
    Destroy {
        sandbox_id: String,
    },
    List,
    EgressLog {
        sandbox_id: String,
    },
    Capacity,
}

impl Op {
    /// The sandbox this operation addresses, for routing.
    pub fn sandbox_id(&self) -> Option<&str> {
        match self {
            Self::Exec { sandbox_id, .. }
            | Self::ReadFile { sandbox_id, .. }
            | Self::WriteFile { sandbox_id, .. }
            | Self::ListDir { sandbox_id, .. }
            | Self::Snapshot { sandbox_id }
            | Self::Destroy { sandbox_id }
            | Self::EgressLog { sandbox_id } => Some(sandbox_id),
            Self::Create { .. } | Self::SnapshotChunk { .. } | Self::List | Self::Capacity => None,
        }
    }

    /// How long a caller waits for a peer to answer.
    pub fn timeout(&self) -> std::time::Duration {
        let secs = match self {
            Self::Create { .. } => 300,
            Self::Exec { request, .. } => request.timeout_ms.unwrap_or(600_000) / 1_000 + 30,
            Self::Snapshot { .. } => 300,
            _ => 60,
        };
        std::time::Duration::from_secs(secs)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum OpResult {
    Sandbox(SandboxInfo),
    Exec {
        #[serde(with = "b64")]
        stdout: Vec<u8>,
        #[serde(with = "b64")]
        stderr: Vec<u8>,
        status: ExitStatus,
    },
    File {
        #[serde(with = "b64")]
        data: Vec<u8>,
        size: u64,
        truncated: bool,
    },
    Entries {
        entries: Vec<DirEntry>,
    },
    Snapshot(SnapshotInfo),
    SnapshotChunk {
        #[serde(with = "b64")]
        data: Vec<u8>,
        total: u64,
    },
    List {
        sandboxes: Vec<SandboxInfo>,
    },
    EgressLog {
        events: Vec<EgressEvent>,
    },
    Capacity(Capacity),
    Done,
}

/// Runs `op` against this node's sandboxes. A `Create` with a snapshot must
/// have had the archive fetched already; it is passed as `restore`.
pub async fn run(
    manager: &Manager,
    caller: &Caller,
    op: Op,
    restore: Option<Vec<u8>>,
) -> anyhow::Result<OpResult> {
    Ok(match op {
        Op::Create { spec } => {
            OpResult::Sandbox(manager.create(caller.clone(), spec, restore).await?)
        }
        Op::Exec {
            sandbox_id,
            request,
        } => {
            let output = manager.exec(caller, &sandbox_id, request).await?;
            OpResult::Exec {
                stdout: output.stdout,
                stderr: output.stderr,
                status: output.status,
            }
        }
        Op::ReadFile {
            sandbox_id,
            path,
            max_bytes,
        } => {
            let max_bytes = max_bytes.min(MAX_TRANSFER_BYTES);
            let (data, size, truncated) = manager
                .read_file(caller, &sandbox_id, &path, max_bytes)
                .await?;
            OpResult::File {
                data,
                size,
                truncated,
            }
        }
        Op::WriteFile {
            sandbox_id,
            path,
            data,
            mode,
            append,
        } => {
            manager
                .write_file(caller, &sandbox_id, &path, &data, mode, append)
                .await?;
            OpResult::Done
        }
        Op::ListDir { sandbox_id, path } => OpResult::Entries {
            entries: manager.list_dir(caller, &sandbox_id, &path).await?,
        },
        Op::Snapshot { sandbox_id } => {
            OpResult::Snapshot(manager.snapshot(caller, &sandbox_id).await?)
        }
        Op::SnapshotChunk {
            snapshot_id,
            offset,
            len,
        } => {
            let (data, total) = manager
                .snapshot_chunk(&snapshot_id, offset, len.min(MAX_TRANSFER_BYTES))
                .await?;
            OpResult::SnapshotChunk { data, total }
        }
        Op::Destroy { sandbox_id } => {
            manager.destroy(caller, &sandbox_id).await?;
            OpResult::Done
        }
        Op::List => OpResult::List {
            sandboxes: manager.list(caller),
        },
        Op::EgressLog { sandbox_id } => OpResult::EgressLog {
            events: manager.egress_log(caller, &sandbox_id)?,
        },
        Op::Capacity => OpResult::Capacity(manager.capacity()),
    })
}

/// The reply a peer sends for an [`Op`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ok { result: OpResult },
    Error { message: String },
}

mod b64 {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        STANDARD.decode(text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_and_results_survive_the_channel_encoding() {
        let op = Op::WriteFile {
            sandbox_id: "sbx-1".into(),
            path: "/workspace/a.bin".into(),
            data: vec![0, 159, 255],
            mode: Some(0o755),
            append: false,
        };
        let decoded: Op = serde_json::from_slice(&serde_json::to_vec(&op).unwrap()).unwrap();
        assert!(matches!(decoded, Op::WriteFile { data, .. } if data == vec![0, 159, 255]));

        let reply = Reply::Ok {
            result: OpResult::Exec {
                stdout: b"hi\n".to_vec(),
                stderr: Vec::new(),
                status: ExitStatus {
                    code: Some(0),
                    ..ExitStatus::default()
                },
            },
        };
        let decoded: Reply = serde_json::from_slice(&serde_json::to_vec(&reply).unwrap()).unwrap();
        assert!(matches!(
            decoded,
            Reply::Ok { result: OpResult::Exec { stdout, .. } } if stdout == b"hi\n"
        ));
    }

    #[test]
    fn exec_timeouts_cover_the_command_plus_slack() {
        let op = Op::Exec {
            sandbox_id: "sbx".into(),
            request: ExecRequest {
                timeout_ms: Some(120_000),
                ..ExecRequest::default()
            },
        };
        assert_eq!(op.timeout().as_secs(), 150);
    }
}
