//! MCP tool inputs and outputs. The host projects these over MCP and HTTP.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use mesh_llm_sandbox_guest::protocol::{DirEntry, ExecRequest, WORKSPACE_DIR};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::egress::proxy::EgressEvent;
use crate::guest::MAX_EXEC_OUTPUT;
use crate::manager::{SandboxInfo, SnapshotInfo};
use crate::mesh::{Mesh, NodeView};
use crate::ops::{MAX_TRANSFER_BYTES, Op, OpResult};
use crate::spec::SandboxSpec;

const DEFAULT_EXEC_TIMEOUT_SECS: u64 = 300;
const MAX_EXEC_TIMEOUT_SECS: u64 = 3_600;
const DEFAULT_READ_BYTES: u64 = 256 * 1024;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SandboxIdArgs {
    pub sandbox_id: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecArgs {
    pub sandbox_id: String,
    /// A shell command line, run with `/bin/sh -c`.
    #[serde(default)]
    pub command: Option<String>,
    /// Program and arguments, run without a shell. Use instead of `command`.
    #[serde(default)]
    pub argv: Option<Vec<String>>,
    /// Working directory. Default `/workspace`.
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Text written to the command's standard input.
    #[serde(default)]
    pub stdin: Option<String>,
    /// Default 300, at most 3600. The command and everything it started are
    /// killed when it passes.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ExecResult {
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileArgs {
    pub sandbox_id: String,
    pub path: String,
    /// Default 262144, at most 4194304.
    #[serde(default)]
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ReadFileResult {
    /// The contents, when they are UTF-8.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The contents, when they are not UTF-8.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_base64: Option<String>,
    pub size: u64,
    pub truncated: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteFileArgs {
    pub sandbox_id: String,
    pub path: String,
    /// Text to write.
    #[serde(default)]
    pub content: Option<String>,
    /// Bytes to write, base64-encoded. Use instead of `content`.
    #[serde(default)]
    pub content_base64: Option<String>,
    /// Unix permission bits, for example 493 (0o755).
    #[serde(default)]
    pub mode: Option<u32>,
    #[serde(default)]
    pub append: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListDirArgs {
    pub sandbox_id: String,
    /// Default `/workspace`.
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ListDirResult {
    pub entries: Vec<DirEntryView>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct DirEntryView {
    pub name: String,
    pub kind: String,
    pub size: u64,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Done {
    pub ok: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SandboxList {
    pub sandboxes: Vec<SandboxInfo>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct NodeList {
    pub nodes: Vec<NodeView>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct EgressLogResult {
    pub events: Vec<EgressEvent>,
}

pub async fn create(mesh: &Mesh, spec: SandboxSpec) -> Result<SandboxInfo> {
    match mesh.dispatch(Op::Create { spec }).await? {
        OpResult::Sandbox(info) => Ok(info),
        other => Err(unexpected(&other)),
    }
}

pub async fn exec(mesh: &Mesh, args: ExecArgs) -> Result<ExecResult> {
    let argv = args.argv.unwrap_or_default();
    if args.command.is_none() == argv.is_empty() {
        bail!("give exactly one of `command` or `argv`");
    }
    let timeout_secs = args
        .timeout_secs
        .unwrap_or(DEFAULT_EXEC_TIMEOUT_SECS)
        .clamp(1, MAX_EXEC_TIMEOUT_SECS);
    let request = ExecRequest {
        argv,
        shell: args.command,
        cwd: args.cwd,
        env: args.env,
        stdin: args.stdin.map(String::into_bytes),
        timeout_ms: Some(timeout_secs * 1_000),
        max_output_bytes: Some(MAX_EXEC_OUTPUT),
    };
    let started = Instant::now();
    let result = mesh
        .dispatch(Op::Exec {
            sandbox_id: args.sandbox_id,
            request,
        })
        .await?;
    let OpResult::Exec {
        stdout,
        stderr,
        status,
    } = result
    else {
        return Err(unexpected(&result));
    };
    Ok(ExecResult {
        exit_code: status.code,
        signal: status.signal,
        timed_out: status.timed_out,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        stdout_truncated: status.stdout_truncated || stdout.len() as u64 >= MAX_EXEC_OUTPUT,
        stderr_truncated: status.stderr_truncated || stderr.len() as u64 >= MAX_EXEC_OUTPUT,
        duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

pub async fn read_file(mesh: &Mesh, args: ReadFileArgs) -> Result<ReadFileResult> {
    let max_bytes = args
        .max_bytes
        .unwrap_or(DEFAULT_READ_BYTES)
        .min(MAX_TRANSFER_BYTES);
    let result = mesh
        .dispatch(Op::ReadFile {
            sandbox_id: args.sandbox_id,
            path: args.path,
            max_bytes,
        })
        .await?;
    let OpResult::File {
        data,
        size,
        truncated,
    } = result
    else {
        return Err(unexpected(&result));
    };
    let (content, content_base64) = match String::from_utf8(data) {
        Ok(text) => (Some(text), None),
        Err(error) => (None, Some(STANDARD.encode(error.into_bytes()))),
    };
    Ok(ReadFileResult {
        content,
        content_base64,
        size,
        truncated,
    })
}

pub async fn write_file(mesh: &Mesh, args: WriteFileArgs) -> Result<Done> {
    let data = match (args.content, args.content_base64) {
        (Some(text), None) => text.into_bytes(),
        (None, Some(encoded)) => STANDARD
            .decode(encoded)
            .context("content_base64 is not valid base64")?,
        _ => bail!("give exactly one of `content` or `content_base64`"),
    };
    if data.len() as u64 > MAX_TRANSFER_BYTES {
        bail!("files over {MAX_TRANSFER_BYTES} bytes must be written in appended parts");
    }
    expect_done(
        mesh.dispatch(Op::WriteFile {
            sandbox_id: args.sandbox_id,
            path: args.path,
            data,
            mode: args.mode,
            append: args.append,
        })
        .await?,
    )
}

pub async fn list_dir(mesh: &Mesh, args: ListDirArgs) -> Result<ListDirResult> {
    let result = mesh
        .dispatch(Op::ListDir {
            sandbox_id: args.sandbox_id,
            path: args.path.unwrap_or_else(|| WORKSPACE_DIR.to_string()),
        })
        .await?;
    let OpResult::Entries { entries } = result else {
        return Err(unexpected(&result));
    };
    Ok(ListDirResult {
        entries: entries.into_iter().map(entry_view).collect(),
    })
}

pub async fn snapshot(mesh: &Mesh, args: SandboxIdArgs) -> Result<SnapshotInfo> {
    match mesh
        .dispatch(Op::Snapshot {
            sandbox_id: args.sandbox_id,
        })
        .await?
    {
        OpResult::Snapshot(info) => Ok(info),
        other => Err(unexpected(&other)),
    }
}

pub async fn destroy(mesh: &Mesh, args: SandboxIdArgs) -> Result<Done> {
    expect_done(
        mesh.dispatch(Op::Destroy {
            sandbox_id: args.sandbox_id,
        })
        .await?,
    )
}

pub async fn list(mesh: &Mesh) -> Result<SandboxList> {
    match mesh.dispatch(Op::List).await? {
        OpResult::List { sandboxes } => Ok(SandboxList { sandboxes }),
        other => Err(unexpected(&other)),
    }
}

pub async fn egress_log(mesh: &Mesh, args: SandboxIdArgs) -> Result<EgressLogResult> {
    match mesh
        .dispatch(Op::EgressLog {
            sandbox_id: args.sandbox_id,
        })
        .await?
    {
        OpResult::EgressLog { events } => Ok(EgressLogResult { events }),
        other => Err(unexpected(&other)),
    }
}

pub fn nodes(mesh: &Mesh) -> NodeList {
    NodeList {
        nodes: mesh.nodes(),
    }
}

fn entry_view(entry: DirEntry) -> DirEntryView {
    DirEntryView {
        name: entry.name,
        kind: serde_json::to_value(entry.kind)
            .ok()
            .and_then(|value| value.as_str().map(ToString::to_string))
            .unwrap_or_default(),
        size: entry.size,
    }
}

fn expect_done(result: OpResult) -> Result<Done> {
    match result {
        OpResult::Done => Ok(Done { ok: true }),
        other => Err(unexpected(&other)),
    }
}

fn unexpected(result: &OpResult) -> anyhow::Error {
    let kind = serde_json::to_value(result)
        .ok()
        .and_then(|value| value.get("result").cloned())
        .unwrap_or_default();
    anyhow::anyhow!("unexpected reply `{kind}` from the sandbox node")
}
