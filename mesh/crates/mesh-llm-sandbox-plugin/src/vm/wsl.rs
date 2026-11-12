//! Firecracker behind the bridge in [`super::bridge`].
//!
//! On Windows this is how sandboxes run: the plugin archive carries the
//! Linux build of this binary at `linux/mesh-sandbox` beside `sandbox.exe`,
//! and [`start_wsl`] runs it inside WSL2 (the default distribution, or the
//! `wsl_distro` setting), which must have Firecracker installed and `/dev/kvm`
//! usable by the WSL user (nested virtualization; on by default on Windows 11).
//!
//! On Linux, [`start_local`] runs the same bridge as a child process, so
//! `mesh-sandbox check --bridge` exercises everything but the WSL launch.
//!
//! Egress still leaves from the plugin's side: the bridge hands each guest
//! egress connection back, and the policy and the outbound connection are
//! applied here like on every other platform.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use mesh_llm_sandbox_guest::protocol::async_io::{read_frame, write_frame};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::process::{Child, ChildStdin, Command};

use super::bridge::{BridgeHello, BridgeReady, BridgeReply, BridgeRequest};
use super::{Backend, BoxStream, EgressHandler, LaunchSpec, Vm};
use crate::settings::Settings;

const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// Maps a path on this side to the path the bridge sees.
type PathMapper = fn(&Path) -> Result<String>;

pub struct WslBackend {
    name: &'static str,
    port: u16,
    token: String,
    map_path: PathMapper,
    _child: Child,
    _stdin: ChildStdin,
}

/// Starts the bridge inside WSL2.
#[cfg(windows)]
pub async fn start_wsl(settings: &Settings) -> Result<WslBackend> {
    const LINUX_BINARY: &str = "linux/mesh-sandbox";
    /// The bridge's state lives in WSL's own filesystem.
    const BRIDGE_STATE_DIR: &str = "/tmp/mesh-sandbox";
    let exe = std::env::current_exe()?;
    let linux_binary = exe
        .parent()
        .context("plugin binary has no directory")?
        .join(LINUX_BINARY);
    if !linux_binary.is_file() {
        bail!(
            "the Linux build of the plugin is missing at {}",
            linux_binary.display()
        );
    }
    let mut command = Command::new("wsl.exe");
    if let Some(distro) = &settings.wsl_distro {
        command.arg("--distribution").arg(distro);
    }
    command.arg("--exec").arg(wsl_path(&linux_binary)?);
    WslBackend::start(
        command,
        BRIDGE_STATE_DIR.to_string(),
        settings,
        "firecracker-wsl",
        wsl_path,
    )
    .await
    .context("start the WSL bridge; is WSL2 installed (`wsl --install`) with Firecracker inside?")
}

/// Starts the bridge as a local child process, for testing it on Linux.
#[cfg(target_os = "linux")]
pub async fn start_local(settings: &Settings) -> Result<WslBackend> {
    let command = Command::new(std::env::current_exe()?);
    let state_dir = settings.state_dir().join("bridge");
    WslBackend::start(
        command,
        state_dir.to_string_lossy().into_owned(),
        settings,
        "firecracker-bridge",
        |path| Ok(path.to_string_lossy().into_owned()),
    )
    .await
}

impl WslBackend {
    async fn start(
        mut command: Command,
        state_dir: String,
        settings: &Settings,
        name: &'static str,
        map_path: PathMapper,
    ) -> Result<Self> {
        command
            .arg("wsl-bridge")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let mut stdin = child.stdin.take().context("bridge stdin")?;
        let stdout = child.stdout.take().context("bridge stdout")?;
        let token = uuid::Uuid::new_v4().simple().to_string();
        let hello = BridgeHello {
            token: token.clone(),
            state_dir,
            firecracker_path: settings
                .firecracker_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
        };
        stdin
            .write_all(format!("{}\n", serde_json::to_string(&hello)?).as_bytes())
            .await?;
        stdin.flush().await?;
        let mut line = String::new();
        tokio::time::timeout(READY_TIMEOUT, BufReader::new(stdout).read_line(&mut line))
            .await
            .context("the bridge did not start in time")??;
        let ready: BridgeReady = serde_json::from_str(&line).context(
            "the bridge did not start; check that Firecracker is installed and /dev/kvm is usable",
        )?;
        Ok(Self {
            name,
            port: ready.port,
            token,
            map_path,
            _child: child,
            _stdin: stdin,
        })
    }
}

#[async_trait]
impl Backend for WslBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn launch(&self, spec: LaunchSpec) -> Result<Box<dyn Vm>> {
        let mut conn = TcpStream::connect(("127.0.0.1", self.port)).await?;
        write_frame(
            &mut conn,
            &BridgeRequest::Launch {
                token: self.token.clone(),
                id: spec.id.clone(),
                kernel: (self.map_path)(&spec.kernel)?,
                rootfs: (self.map_path)(&spec.rootfs)?,
                scratch_bytes: spec.scratch_bytes,
                cpus: spec.cpus,
                memory_mib: spec.memory_mib,
            },
        )
        .await?;
        match read_frame::<BridgeReply>(&mut conn).await? {
            Some(BridgeReply::Ok) => {}
            Some(BridgeReply::Error { message }) => bail!(message),
            _ => bail!("the WSL bridge closed the launch"),
        }
        let (mut events, lifeline) = conn.into_split();
        let dialer = Dialer {
            port: self.port,
            token: self.token.clone(),
            id: spec.id.clone(),
        };
        let egress_dialer = dialer.clone();
        let on_egress: EgressHandler = spec.on_egress.clone();
        tokio::spawn(async move {
            while let Ok(Some(BridgeReply::EgressReady { connection })) =
                read_frame::<BridgeReply>(&mut events).await
            {
                if let Ok(stream) = egress_dialer.egress(connection).await {
                    on_egress(stream);
                }
            }
        });
        Ok(Box::new(WslVm {
            dialer,
            lifeline: Mutex::new(Some(lifeline)),
        }))
    }
}

#[derive(Clone)]
struct Dialer {
    port: u16,
    token: String,
    id: String,
}

impl Dialer {
    async fn open(&self, request: &BridgeRequest) -> Result<BoxStream> {
        let mut conn = TcpStream::connect(("127.0.0.1", self.port)).await?;
        write_frame(&mut conn, request).await?;
        match read_frame::<BridgeReply>(&mut conn).await? {
            Some(BridgeReply::Ok) => Ok(Box::new(conn)),
            Some(BridgeReply::Error { message }) => bail!(message),
            _ => bail!("the WSL bridge closed the connection"),
        }
    }

    async fn egress(&self, connection: u64) -> Result<BoxStream> {
        self.open(&BridgeRequest::Egress {
            token: self.token.clone(),
            id: self.id.clone(),
            connection,
        })
        .await
    }
}

struct WslVm {
    dialer: Dialer,
    /// Closing this tells the bridge to stop the VM.
    lifeline: Mutex<Option<OwnedWriteHalf>>,
}

#[async_trait]
impl Vm for WslVm {
    async fn connect_agent(&self) -> Result<BoxStream> {
        self.dialer
            .open(&BridgeRequest::Agent {
                token: self.dialer.token.clone(),
                id: self.dialer.id.clone(),
            })
            .await
    }

    async fn stop(&self) {
        let lifeline = self.lifeline.lock().ok().and_then(|mut slot| slot.take());
        if let Some(mut lifeline) = lifeline {
            let _ = lifeline.shutdown().await;
        }
    }

    fn console_log(&self) -> Option<PathBuf> {
        None
    }
}

/// `C:\Users\me\x` → `/mnt/c/Users/me/x`.
#[cfg(any(windows, test))]
fn wsl_path(path: &Path) -> Result<String> {
    /// Where WSL mounts Windows drives unless `/etc/wsl.conf` says otherwise.
    const DRIVE_MOUNT_ROOT: &str = "/mnt";
    let text = path.to_string_lossy();
    let text = text.strip_prefix(r"\\?\").unwrap_or(&text);
    let mut chars = text.chars();
    let (Some(drive), Some(':')) = (chars.next(), chars.next()) else {
        bail!("{} is not on a drive WSL can reach", path.display());
    };
    let rest = chars.as_str().replace('\\', "/");
    Ok(format!(
        "{DRIVE_MOUNT_ROOT}/{}{rest}",
        drive.to_ascii_lowercase()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_paths_map_to_the_wsl_drive_mounts() {
        assert_eq!(
            wsl_path(Path::new(
                r"C:\Users\me\.mesh-llm\sandbox\images\base\x86_64\kernel"
            ))
            .unwrap(),
            "/mnt/c/Users/me/.mesh-llm/sandbox/images/base/x86_64/kernel"
        );
        assert_eq!(
            wsl_path(Path::new(r"\\?\D:\images\rootfs.ext4")).unwrap(),
            "/mnt/d/images/rootfs.ext4"
        );
        assert!(wsl_path(Path::new(r"\\server\share\x")).is_err());
    }
}
