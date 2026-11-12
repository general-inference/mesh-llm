//! Host side of the guest agent protocol: one connection per operation.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mesh_llm_sandbox_guest::protocol::async_io::{read_frame, write_frame};
use mesh_llm_sandbox_guest::protocol::{
    CHUNK_BYTES, DirEntry, ExecRequest, ExitStatus, Reply, Request,
};
use schemars::JsonSchema;
use serde::Serialize;

use crate::vm::{BoxStream, Vm};

const BOOT_POLL: Duration = Duration::from_millis(100);
/// Covers the slowest step a single frame waits on (unpacking a chunk).
const FRAME_TIMEOUT: Duration = Duration::from_secs(120);
/// Exec output kept per stream. Both streams must fit one channel frame
/// (10 MB) base64-encoded when the sandbox runs on a peer.
pub const MAX_EXEC_OUTPUT: u64 = 2 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct GuestInfo {
    pub agent_version: u32,
    pub arch: String,
    pub kernel: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExecOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: ExitStatus,
}

pub struct GuestClient<'a> {
    vm: &'a dyn Vm,
}

impl<'a> GuestClient<'a> {
    pub fn new(vm: &'a dyn Vm) -> Self {
        Self { vm }
    }

    /// Polls the agent until it answers or `deadline` passes.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<GuestInfo> {
        let deadline = Instant::now() + timeout;
        let mut last_error = None;
        while Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_secs(2), self.hello()).await {
                Ok(Ok(info)) => return Ok(info),
                Ok(Err(error)) => last_error = Some(error),
                Err(_) => last_error = Some(anyhow::anyhow!("agent did not answer")),
            }
            tokio::time::sleep(BOOT_POLL).await;
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no attempt made")))
            .context("the sandbox did not boot in time")
    }

    async fn hello(&self) -> Result<GuestInfo> {
        let reply = self
            .unary(&Request::Hello {
                unix_time_ms: Some(crate::now_ms()),
            })
            .await?;
        match reply {
            Reply::Hello {
                version,
                arch,
                kernel,
            } => Ok(GuestInfo {
                agent_version: version,
                arch,
                kernel,
            }),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn exec(&self, request: ExecRequest) -> Result<ExecOutput> {
        let timeout = request
            .timeout_ms
            .map_or(Duration::from_secs(600), Duration::from_millis)
            + Duration::from_secs(10);
        let mut conn = self.open(&Request::Exec(request)).await?;
        let mut output = ExecOutput::default();
        let deadline = Instant::now() + timeout;
        loop {
            let reply = next_reply(&mut conn, deadline).await?;
            match reply {
                Reply::Stdout { data } => append_capped(&mut output.stdout, &data),
                Reply::Stderr { data } => append_capped(&mut output.stderr, &data),
                Reply::Exit(status) => {
                    output.status = status;
                    return Ok(output);
                }
                Reply::Error { message } => bail!(message),
                other => return Err(unexpected(&other)),
            }
        }
    }

    pub async fn read_file(&self, path: &str, max_bytes: u64) -> Result<(Vec<u8>, u64, bool)> {
        match self
            .unary(&Request::ReadFile {
                path: path.to_string(),
                max_bytes,
            })
            .await?
        {
            Reply::File {
                data,
                size,
                truncated,
            } => Ok((data, size, truncated)),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn write_file(
        &self,
        path: &str,
        data: &[u8],
        mode: Option<u32>,
        append: bool,
    ) -> Result<()> {
        let request = Request::WriteFile {
            path: path.to_string(),
            mode,
            append,
        };
        self.upload(&request, data).await
    }

    pub async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>> {
        match self
            .unary(&Request::ListDir {
                path: path.to_string(),
            })
            .await?
        {
            Reply::Entries { entries } => Ok(entries),
            other => Err(unexpected(&other)),
        }
    }

    /// A gzip tar of `path` inside the guest.
    pub async fn archive_out(&self, path: &str) -> Result<Vec<u8>> {
        let mut conn = self
            .open(&Request::ArchiveOut {
                path: path.to_string(),
            })
            .await?;
        let mut archive = Vec::new();
        loop {
            match next_reply(&mut conn, Instant::now() + FRAME_TIMEOUT).await? {
                Reply::Chunk { data } => archive.extend_from_slice(&data),
                Reply::Done => return Ok(archive),
                Reply::Error { message } => bail!(message),
                other => return Err(unexpected(&other)),
            }
        }
    }

    pub async fn archive_in(&self, path: &str, archive: &[u8]) -> Result<()> {
        self.upload(
            &Request::ArchiveIn {
                path: path.to_string(),
            },
            archive,
        )
        .await
    }

    /// Asks the guest to power off; errors are ignored because the
    /// connection may drop as it does.
    pub async fn shutdown(&self) {
        if let Ok(mut conn) = self.open(&Request::Shutdown).await {
            let _ =
                tokio::time::timeout(Duration::from_secs(2), read_frame::<Reply>(&mut conn)).await;
        }
    }

    async fn upload(&self, request: &Request, data: &[u8]) -> Result<()> {
        let mut conn = self.open(request).await?;
        for chunk in data.chunks(CHUNK_BYTES) {
            write_frame(
                &mut conn,
                &Request::Chunk {
                    data: chunk.to_vec(),
                },
            )
            .await?;
        }
        write_frame(&mut conn, &Request::End).await?;
        match next_reply(&mut conn, Instant::now() + FRAME_TIMEOUT).await? {
            Reply::Done => Ok(()),
            Reply::Error { message } => bail!(message),
            other => Err(unexpected(&other)),
        }
    }

    async fn unary(&self, request: &Request) -> Result<Reply> {
        let mut conn = self.open(request).await?;
        let reply = next_reply(&mut conn, Instant::now() + FRAME_TIMEOUT).await?;
        if let Reply::Error { message } = reply {
            bail!(message);
        }
        Ok(reply)
    }

    async fn open(&self, request: &Request) -> Result<BoxStream> {
        let mut conn = self.vm.connect_agent().await?;
        write_frame(&mut conn, request).await?;
        Ok(conn)
    }
}

async fn next_reply(conn: &mut BoxStream, deadline: Instant) -> Result<Reply> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    tokio::time::timeout(remaining, read_frame::<Reply>(conn))
        .await
        .context("timed out waiting for the sandbox")??
        .context("the sandbox closed the connection")
}

fn append_capped(buffer: &mut Vec<u8>, data: &[u8]) {
    let room = usize::try_from(MAX_EXEC_OUTPUT)
        .unwrap_or(usize::MAX)
        .saturating_sub(buffer.len());
    buffer.extend_from_slice(&data[..data.len().min(room)]);
}

fn unexpected(reply: &Reply) -> anyhow::Error {
    anyhow::anyhow!("unexpected reply from the sandbox agent: {reply:?}")
}
