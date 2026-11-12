//! Frames exchanged between the host plugin and the guest agent.
//!
//! Every frame is a 4-byte big-endian length followed by that many bytes of
//! JSON. A connection carries one operation: the host sends a [`Request`],
//! may follow it with [`Request::Chunk`] and [`Request::End`] frames for
//! uploads, and the agent answers with [`Reply`] frames ending in a terminal
//! one ([`Reply::is_terminal`]).

use std::collections::BTreeMap;
use std::io::{Read, Write};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Bumped on any incompatible change to the frames below.
pub const PROTOCOL_VERSION: u32 = 1;

/// vsock port the agent accepts host operations on.
pub const AGENT_PORT: u32 = 1024;

/// vsock port on the host that the guest's egress proxy connects to.
pub const EGRESS_PORT: u32 = 1025;

/// vsock context id of the host, as seen from the guest.
pub const HOST_CID: u32 = 2;

/// Largest frame either side accepts.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Payload size used when streaming archives and command output.
pub const CHUNK_BYTES: usize = 256 * 1024;

/// The HTTP proxy address inside the guest that commands are pointed at.
pub const GUEST_PROXY_ADDR: &str = "127.0.0.1:3128";

/// The directory commands run in unless they ask for another.
pub const WORKSPACE_DIR: &str = "/workspace";

/// A host operation, or a frame that continues one.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Report the agent's version; the host sends it until the guest is up.
    /// The guest has no reliable clock, so the host passes its own and the
    /// agent sets the guest clock from it (TLS needs a sane time).
    Hello {
        #[serde(default)]
        unix_time_ms: Option<u64>,
    },
    Exec(ExecRequest),
    ReadFile {
        path: String,
        max_bytes: u64,
    },
    /// Followed by [`Request::Chunk`] frames and a [`Request::End`].
    WriteFile {
        path: String,
        #[serde(default)]
        mode: Option<u32>,
        #[serde(default)]
        append: bool,
    },
    ListDir {
        path: String,
    },
    /// Stream a gzip tar of `path` back as [`Reply::Chunk`] frames.
    ArchiveOut {
        path: String,
    },
    /// Unpack a gzip tar, sent as [`Request::Chunk`] frames, into `path`.
    ArchiveIn {
        path: String,
    },
    Chunk {
        #[serde(with = "bytes_b64")]
        data: Vec<u8>,
    },
    End,
    /// Sync disks and power the guest off.
    Shutdown,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExecRequest {
    /// Program and arguments. Ignored when `shell` is set.
    #[serde(default)]
    pub argv: Vec<String>,
    /// A command line run with `/bin/sh -c`.
    #[serde(default)]
    pub shell: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Added to the agent's default environment, overriding it.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default, with = "opt_bytes_b64")]
    pub stdin: Option<Vec<u8>>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Per stream; output past it is dropped and the exit reports truncation.
    #[serde(default)]
    pub max_output_bytes: Option<u64>,
}

/// An agent frame.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Reply {
    Hello {
        version: u32,
        arch: String,
        kernel: String,
    },
    Stdout {
        #[serde(with = "bytes_b64")]
        data: Vec<u8>,
    },
    Stderr {
        #[serde(with = "bytes_b64")]
        data: Vec<u8>,
    },
    Exit(ExitStatus),
    File {
        #[serde(with = "bytes_b64")]
        data: Vec<u8>,
        size: u64,
        truncated: bool,
    },
    Entries {
        entries: Vec<DirEntry>,
    },
    Chunk {
        #[serde(with = "bytes_b64")]
        data: Vec<u8>,
    },
    Done,
    Error {
        message: String,
    },
}

impl Reply {
    /// Whether this frame ends the operation.
    pub fn is_terminal(&self) -> bool {
        !matches!(
            self,
            Self::Stdout { .. } | Self::Stderr { .. } | Self::Chunk { .. }
        )
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExitStatus {
    pub code: Option<i32>,
    pub signal: Option<i32>,
    pub timed_out: bool,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub kind: EntryKind,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    File,
    Dir,
    Symlink,
    Other,
}

/// First frame on an egress connection, from the guest's proxy.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EgressRequest {
    pub host: String,
    pub port: u16,
}

/// The host's answer. When `allowed`, the connection becomes a raw byte pipe
/// to `host:port`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EgressReply {
    pub allowed: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

pub fn encode_frame<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(value)?;
    if body.len() > MAX_FRAME_BYTES {
        bail!("frame of {} bytes exceeds {MAX_FRAME_BYTES}", body.len());
    }
    let mut frame = Vec::with_capacity(body.len() + 4);
    frame.extend_from_slice(&u32::try_from(body.len())?.to_be_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

pub fn write_frame<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    writer.write_all(&encode_frame(value)?)?;
    writer.flush()?;
    Ok(())
}

/// Reads one frame; `None` on a clean end of stream before the length.
pub fn read_frame<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> Result<Option<T>> {
    let mut len = [0u8; 4];
    match reader.read_exact(&mut len) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let len = frame_len(len)?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).context("truncated frame")?;
    Ok(Some(serde_json::from_slice(&body)?))
}

fn frame_len(prefix: [u8; 4]) -> Result<usize> {
    let len = usize::try_from(u32::from_be_bytes(prefix))?;
    if len > MAX_FRAME_BYTES {
        bail!("frame of {len} bytes exceeds {MAX_FRAME_BYTES}");
    }
    Ok(len)
}

#[cfg(feature = "tokio")]
pub mod async_io {
    //! The same framing over tokio streams, for the host.

    use anyhow::{Context, Result};
    use serde::{Deserialize, Serialize};
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    pub async fn write_frame<T: Serialize>(
        writer: &mut (impl AsyncWrite + Unpin),
        value: &T,
    ) -> Result<()> {
        writer.write_all(&super::encode_frame(value)?).await?;
        writer.flush().await?;
        Ok(())
    }

    pub async fn read_frame<T: for<'de> Deserialize<'de>>(
        reader: &mut (impl AsyncRead + Unpin),
    ) -> Result<Option<T>> {
        let mut len = [0u8; 4];
        match reader.read_exact(&mut len).await {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let len = super::frame_len(len)?;
        let mut body = vec![0u8; len];
        reader
            .read_exact(&mut body)
            .await
            .context("truncated frame")?;
        Ok(Some(serde_json::from_slice(&body)?))
    }
}

mod bytes_b64 {
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

mod opt_bytes_b64 {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(clippy::ref_option)]
    pub fn serialize<S: Serializer>(
        bytes: &Option<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bytes {
            Some(bytes) => serializer.serialize_some(&STANDARD.encode(bytes)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|text| STANDARD.decode(text).map_err(serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip_through_a_byte_stream() {
        let request = Request::Exec(ExecRequest {
            shell: Some("echo hi".into()),
            stdin: Some(vec![0, 1, 2, 255]),
            ..ExecRequest::default()
        });
        let mut buffer = Vec::new();
        write_frame(&mut buffer, &request).unwrap();
        write_frame(&mut buffer, &Request::End).unwrap();
        let mut reader = buffer.as_slice();
        assert_eq!(read_frame::<Request>(&mut reader).unwrap(), Some(request));
        assert_eq!(
            read_frame::<Request>(&mut reader).unwrap(),
            Some(Request::End)
        );
        assert_eq!(read_frame::<Request>(&mut reader).unwrap(), None);
    }

    #[test]
    fn oversized_frames_are_refused_before_allocation() {
        let mut buffer = u32::try_from(MAX_FRAME_BYTES + 1)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        buffer.extend_from_slice(b"{}");
        assert!(read_frame::<Request>(&mut buffer.as_slice()).is_err());
    }

    #[test]
    fn only_streaming_replies_are_non_terminal() {
        assert!(!Reply::Stdout { data: vec![] }.is_terminal());
        assert!(!Reply::Chunk { data: vec![] }.is_terminal());
        assert!(Reply::Done.is_terminal());
        assert!(Reply::Exit(ExitStatus::default()).is_terminal());
        assert!(Reply::error("x").is_terminal());
    }

    #[test]
    fn requests_use_a_stable_tagged_json_shape() {
        let json = serde_json::to_value(Request::ReadFile {
            path: "/workspace/a".into(),
            max_bytes: 10,
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({"op": "read_file", "path": "/workspace/a", "max_bytes": 10})
        );
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_framing_matches_blocking_framing() {
        let mut buffer = Vec::new();
        async_io::write_frame(&mut buffer, &Reply::Done)
            .await
            .unwrap();
        assert_eq!(buffer, encode_frame(&Reply::Done).unwrap());
        let mut reader = buffer.as_slice();
        assert_eq!(
            async_io::read_frame::<Reply>(&mut reader).await.unwrap(),
            Some(Reply::Done)
        );
    }
}
