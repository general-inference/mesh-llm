//! A VM backend for tests: each "VM" is an in-process agent that speaks the
//! guest protocol over in-memory pipes, keeping files in a map.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use mesh_llm_sandbox_guest::protocol::async_io::{read_frame, write_frame};
use mesh_llm_sandbox_guest::protocol::{
    DirEntry, EntryKind, ExitStatus, PROTOCOL_VERSION, Reply, Request,
};

use super::{Backend, BoxStream, LaunchSpec, Vm};

type Files = Arc<Mutex<BTreeMap<String, Vec<u8>>>>;

#[derive(Default)]
pub struct FakeBackend {
    pub launched: Mutex<u32>,
}

#[async_trait]
impl Backend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }

    async fn launch(&self, _spec: LaunchSpec) -> Result<Box<dyn Vm>> {
        if let Ok(mut launched) = self.launched.lock() {
            *launched += 1;
        }
        Ok(Box::new(FakeVm {
            files: Files::default(),
        }))
    }
}

struct FakeVm {
    files: Files,
}

#[async_trait]
impl Vm for FakeVm {
    async fn connect_agent(&self) -> Result<BoxStream> {
        let (host, guest) = tokio::io::duplex(1 << 20);
        tokio::spawn(serve(guest, self.files.clone()));
        Ok(Box::new(host))
    }

    async fn stop(&self) {}

    fn console_log(&self) -> Option<PathBuf> {
        None
    }
}

async fn serve(mut conn: tokio::io::DuplexStream, files: Files) -> Result<()> {
    let Some(request) = read_frame::<Request>(&mut conn).await? else {
        return Ok(());
    };
    let reply = match request {
        Request::Hello { .. } => Reply::Hello {
            version: PROTOCOL_VERSION,
            arch: "fake".into(),
            kernel: "fake".into(),
        },
        Request::Exec(exec) => {
            let line = exec.shell.unwrap_or_else(|| exec.argv.join(" "));
            let output = line.strip_prefix("echo ").unwrap_or_default();
            write_frame(
                &mut conn,
                &Reply::Stdout {
                    data: format!("{output}\n").into_bytes(),
                },
            )
            .await?;
            Reply::Exit(ExitStatus {
                code: Some(if line.starts_with("false") { 1 } else { 0 }),
                ..ExitStatus::default()
            })
        }
        Request::ReadFile { path, max_bytes } => {
            match files
                .lock()
                .ok()
                .and_then(|files| files.get(&path).cloned())
            {
                Some(data) => {
                    let size = data.len() as u64;
                    let keep = usize::try_from(max_bytes)
                        .unwrap_or(usize::MAX)
                        .min(data.len());
                    Reply::File {
                        data: data[..keep].to_vec(),
                        size,
                        truncated: keep < data.len(),
                    }
                }
                None => Reply::error(format!("open {path}: not found")),
            }
        }
        Request::WriteFile { path, append, .. } => {
            let data = collect(&mut conn).await?;
            if let Ok(mut files) = files.lock() {
                let entry = files.entry(path).or_default();
                if !append {
                    entry.clear();
                }
                entry.extend_from_slice(&data);
            }
            Reply::Done
        }
        Request::ListDir { path } => {
            let prefix = format!("{}/", path.trim_end_matches('/'));
            let entries = files
                .lock()
                .map(|files| {
                    files
                        .iter()
                        .filter_map(|(name, data)| {
                            Some(DirEntry {
                                name: name.strip_prefix(&prefix)?.to_string(),
                                kind: EntryKind::File,
                                size: data.len() as u64,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            Reply::Entries { entries }
        }
        Request::ArchiveOut { .. } => {
            let archive = files
                .lock()
                .map(|files| serde_json::to_vec(&*files).unwrap_or_default())
                .unwrap_or_default();
            write_frame(&mut conn, &Reply::Chunk { data: archive }).await?;
            Reply::Done
        }
        Request::ArchiveIn { .. } => {
            let archive = collect(&mut conn).await?;
            let restored: BTreeMap<String, Vec<u8>> = serde_json::from_slice(&archive)?;
            if let Ok(mut files) = files.lock() {
                files.extend(restored);
            }
            Reply::Done
        }
        Request::Shutdown => Reply::Done,
        Request::Chunk { .. } | Request::End => Reply::error("unexpected stream frame"),
    };
    write_frame(&mut conn, &reply).await
}

async fn collect(conn: &mut tokio::io::DuplexStream) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    while let Some(Request::Chunk { data: chunk }) = read_frame::<Request>(conn).await? {
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}
