//! The bridge between the Windows plugin and Firecracker inside WSL2.
//!
//! The Windows build starts this binary's Linux build inside WSL2 as
//! `mesh-sandbox wsl-bridge`, writes a [`BridgeHello`] line to its stdin and
//! reads a [`BridgeReady`] line back naming a loopback TCP port, which WSL2
//! forwards to Windows. Each TCP connection then carries one
//! [`BridgeRequest`], authenticated by the token from the hello:
//!
//! - `launch` boots a VM. The connection stays open as the VM's lifeline: the
//!   bridge announces guest egress connections on it (`egress_ready`), and
//!   closing it stops the VM.
//! - `agent` becomes a byte pipe to the guest agent.
//! - `egress` picks up an announced guest egress connection as a byte pipe,
//!   so the Windows side applies the egress policy and dials out itself.
//!
//! The bridge exits, stopping every VM, when its stdin closes: the Windows
//! plugin is gone.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BridgeHello {
    pub token: String,
    /// Where the bridge keeps image copies, scratch disks and sockets.
    pub state_dir: String,
    #[serde(default)]
    pub firecracker_path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BridgeReady {
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum BridgeRequest {
    Launch {
        token: String,
        id: String,
        /// Paths as WSL sees them (`/mnt/c/...`).
        kernel: String,
        rootfs: String,
        scratch_bytes: u64,
        cpus: u32,
        memory_mib: u64,
    },
    Agent {
        token: String,
        id: String,
    },
    Egress {
        token: String,
        id: String,
        connection: u64,
    },
}

impl BridgeRequest {
    #[cfg(target_os = "linux")]
    fn token(&self) -> &str {
        match self {
            Self::Launch { token, .. } | Self::Agent { token, .. } | Self::Egress { token, .. } => {
                token
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum BridgeReply {
    Ok,
    Error { message: String },
    EgressReady { connection: u64 },
}

#[cfg(target_os = "linux")]
pub use server::serve_stdio;

#[cfg(target_os = "linux")]
mod server {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use anyhow::{Context, Result, bail};
    use mesh_llm_sandbox_guest::protocol::async_io::{read_frame, write_frame};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    use super::{BridgeHello, BridgeReady, BridgeReply, BridgeRequest};
    use crate::vm::firecracker::{FirecrackerVm, launch};
    use crate::vm::{BoxStream, LaunchSpec, Vm};

    struct Bridge {
        token: String,
        firecracker: PathBuf,
        state_dir: PathBuf,
        vms: Mutex<HashMap<String, Arc<BridgedVm>>>,
    }

    struct BridgedVm {
        vm: FirecrackerVm,
        egress: Arc<Mutex<HashMap<u64, BoxStream>>>,
    }

    pub async fn serve_stdio() -> Result<()> {
        let mut stdin = BufReader::new(tokio::io::stdin());
        let mut line = String::new();
        stdin.read_line(&mut line).await?;
        let hello: BridgeHello = serde_json::from_str(&line).context("bad bridge hello")?;
        let firecracker = crate::vm::firecracker::FirecrackerBackend::detect(
            hello.firecracker_path.map(PathBuf::from),
        )?
        .binary()
        .to_path_buf();
        let state_dir = PathBuf::from(&hello.state_dir);
        let _ = std::fs::remove_dir_all(state_dir.join("run"));
        std::fs::create_dir_all(state_dir.join("run"))?;
        let bridge = Arc::new(Bridge {
            token: hello.token,
            firecracker,
            state_dir,
            vms: Mutex::new(HashMap::new()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let ready = BridgeReady {
            port: listener.local_addr()?.port(),
        };
        {
            use std::io::Write as _;
            let mut out = mesh_llm_events::machine_out();
            writeln!(out, "{}", serde_json::to_string(&ready)?)?;
            out.flush()?;
        }
        let accepting = bridge.clone();
        tokio::spawn(async move {
            while let Ok((conn, _)) = listener.accept().await {
                let bridge = accepting.clone();
                tokio::spawn(async move {
                    if let Err(error) = bridge.handle(conn).await {
                        tracing::warn!("mesh-sandbox bridge: {error:#}");
                    }
                });
            }
        });
        // The Windows plugin holds our stdin; when it goes, so do we.
        let mut rest = Vec::new();
        let _ = stdin.read_to_end(&mut rest).await;
        let vms: Vec<Arc<BridgedVm>> = bridge
            .vms
            .lock()
            .map(|mut vms| vms.drain().map(|(_, vm)| vm).collect())
            .unwrap_or_default();
        for vm in vms {
            vm.vm.stop().await;
        }
        Ok(())
    }

    impl Bridge {
        async fn handle(self: &Arc<Self>, mut conn: TcpStream) -> Result<()> {
            let request: BridgeRequest = read_frame(&mut conn).await?.context("empty request")?;
            if !constant_time_eq(request.token().as_bytes(), self.token.as_bytes()) {
                bail!("rejected a connection with the wrong token");
            }
            match request {
                BridgeRequest::Launch {
                    id,
                    kernel,
                    rootfs,
                    scratch_bytes,
                    cpus,
                    memory_mib,
                    ..
                } => {
                    let spec = LaunchRequest {
                        id,
                        kernel,
                        rootfs,
                        scratch_bytes,
                        cpus,
                        memory_mib,
                    };
                    self.launch(spec, conn).await
                }
                BridgeRequest::Agent { id, .. } => {
                    // A failed connect is normal while the guest boots: the
                    // client retries, so it gets the error, not our log.
                    let agent = match self.vm(&id) {
                        Ok(vm) => vm.vm.connect_agent().await,
                        Err(error) => Err(error),
                    };
                    let mut agent = match agent {
                        Ok(agent) => agent,
                        Err(error) => {
                            let message = format!("{error:#}");
                            return write_frame(&mut conn, &BridgeReply::Error { message }).await;
                        }
                    };
                    write_frame(&mut conn, &BridgeReply::Ok).await?;
                    let _ = tokio::io::copy_bidirectional(&mut conn, &mut agent).await;
                    Ok(())
                }
                BridgeRequest::Egress { id, connection, .. } => {
                    let vm = self.vm(&id)?;
                    let guest = vm
                        .egress
                        .lock()
                        .ok()
                        .and_then(|mut pending| pending.remove(&connection))
                        .context("no such pending egress connection")?;
                    write_frame(&mut conn, &BridgeReply::Ok).await?;
                    let mut guest = guest;
                    let _ = tokio::io::copy_bidirectional(&mut conn, &mut guest).await;
                    Ok(())
                }
            }
        }

        async fn launch(
            self: &Arc<Self>,
            request: LaunchRequest,
            mut conn: TcpStream,
        ) -> Result<()> {
            let work_dir = self.state_dir.join("run").join(&request.id);
            std::fs::create_dir_all(&work_dir)?;
            let kernel = self.local_copy(Path::new(&request.kernel))?;
            let rootfs = self.local_copy(Path::new(&request.rootfs))?;
            let egress: Arc<Mutex<HashMap<u64, BoxStream>>> = Arc::default();
            let (announce_tx, mut announce_rx) = mpsc::unbounded_channel();
            let next = Arc::new(AtomicU64::new(1));
            let pending = egress.clone();
            let on_egress = Arc::new(move |stream: BoxStream| {
                let connection = next.fetch_add(1, Ordering::Relaxed);
                if let Ok(mut pending) = pending.lock() {
                    pending.insert(connection, stream);
                }
                let _ = announce_tx.send(connection);
            });
            let spec = LaunchSpec {
                id: request.id.clone(),
                kernel,
                rootfs,
                scratch_bytes: request.scratch_bytes,
                cpus: request.cpus,
                memory_mib: request.memory_mib,
                work_dir: work_dir.clone(),
                on_egress,
            };
            let vm = match launch(&self.firecracker, spec).await {
                Ok(vm) => vm,
                Err(error) => {
                    let message = format!("{error:#}");
                    write_frame(&mut conn, &BridgeReply::Error { message }).await?;
                    let _ = std::fs::remove_dir_all(&work_dir);
                    return Ok(());
                }
            };
            let bridged = Arc::new(BridgedVm { vm, egress });
            if let Ok(mut vms) = self.vms.lock() {
                vms.insert(request.id.clone(), bridged.clone());
            }
            write_frame(&mut conn, &BridgeReply::Ok).await?;
            let (mut reader, mut writer) = conn.split();
            let mut probe = [0u8; 1];
            loop {
                tokio::select! {
                    announced = announce_rx.recv() => {
                        let Some(connection) = announced else { break };
                        if write_frame(&mut writer, &BridgeReply::EgressReady { connection }).await.is_err() {
                            break;
                        }
                    }
                    read = reader.read(&mut probe) => {
                        if matches!(read, Ok(0) | Err(_)) {
                            break;
                        }
                    }
                }
            }
            if let Ok(mut vms) = self.vms.lock() {
                vms.remove(&request.id);
            }
            bridged.vm.stop().await;
            let _ = std::fs::remove_dir_all(&work_dir);
            Ok(())
        }

        fn vm(&self, id: &str) -> Result<Arc<BridgedVm>> {
            self.vms
                .lock()
                .ok()
                .and_then(|vms| vms.get(id).cloned())
                .with_context(|| format!("no VM `{id}` in the bridge"))
        }

        /// Firecracker reads the image through WSL's native filesystem, not
        /// the slow Windows mount: copy it once per version of the file.
        fn local_copy(&self, source: &Path) -> Result<PathBuf> {
            let metadata =
                std::fs::metadata(source).with_context(|| format!("read {}", source.display()))?;
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |elapsed| elapsed.as_secs());
            let key = format!(
                "{:016x}-{}-{modified}",
                fnv1a(source.to_string_lossy().as_bytes()),
                metadata.len()
            );
            let name = source.file_name().context("image path has no file name")?;
            let dir = self.state_dir.join("image-cache").join(key);
            let target = dir.join(name);
            if !target.is_file() {
                std::fs::create_dir_all(&dir)?;
                let partial = dir.join("partial");
                std::fs::copy(source, &partial)?;
                std::fs::rename(&partial, &target)?;
            }
            Ok(target)
        }
    }

    struct LaunchRequest {
        id: String,
        kernel: String,
        rootfs: String,
        scratch_bytes: u64,
        cpus: u32,
        memory_mib: u64,
    }

    fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                == 0
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tokens_compare_whole() {
            assert!(constant_time_eq(b"abc", b"abc"));
            assert!(!constant_time_eq(b"abc", b"abd"));
            assert!(!constant_time_eq(b"abc", b"ab"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_replies_use_tagged_json() {
        let request = BridgeRequest::Egress {
            token: "t".into(),
            id: "sbx".into(),
            connection: 7,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["op"], "egress");
        let reply: BridgeReply =
            serde_json::from_value(serde_json::json!({"reply": "egress_ready", "connection": 7}))
                .unwrap();
        assert!(matches!(reply, BridgeReply::EgressReady { connection: 7 }));
    }
}
