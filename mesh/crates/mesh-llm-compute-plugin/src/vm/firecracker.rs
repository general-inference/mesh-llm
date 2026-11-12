//! Firecracker on KVM.
//!
//! Each sandbox is one `firecracker --no-api --config-file` process. Its vsock
//! device is a Unix socket on the host: the host reaches the guest agent by
//! connecting to it and sending `CONNECT <port>`, and guest connections to
//! host port `P` arrive on the listener at `<socket>_P`. Unix socket paths
//! are limited to about 100 bytes, so these sockets live in a short private
//! directory under `/tmp` rather than in the (arbitrarily deep) state
//! directory.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use mesh_llm_compute_guest::protocol::{AGENT_PORT, EGRESS_PORT};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

use super::{Backend, BoxStream, LaunchSpec, Vm, kernel_cmdline};

const VSOCK_SOCKET: &str = "v.sock";
const CONFIG_FILE: &str = "firecracker.json";
const CONSOLE_LOG: &str = "console.log";
const GUEST_CID: u32 = 3;
const STOP_GRACE: Duration = Duration::from_secs(3);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

pub struct FirecrackerBackend {
    binary: PathBuf,
}

impl FirecrackerBackend {
    /// Finds Firecracker and checks this user can open `/dev/kvm`.
    pub fn detect(configured: Option<PathBuf>) -> Result<Self> {
        let binary = match configured {
            Some(path) => path,
            None => bundled().or_else(|| find_on_path("firecracker")).context(
                "no Firecracker beside the plugin or on PATH; reinstall the plugin, \
                     or install Firecracker and set the plugin's `firecracker_path` setting",
            )?,
        };
        if !binary.is_file() {
            bail!("firecracker binary {} does not exist", binary.display());
        }
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm")
            .context(
                "cannot open /dev/kvm; enable virtualization and add this user to the `kvm` group",
            )?;
        Ok(Self { binary })
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

#[async_trait]
impl Backend for FirecrackerBackend {
    fn name(&self) -> &'static str {
        "firecracker"
    }

    async fn launch(&self, spec: LaunchSpec) -> Result<Box<dyn Vm>> {
        Ok(Box::new(launch(&self.binary, spec).await?))
    }
}

/// Boots a VM; shared with the WSL bridge, which runs this inside WSL2.
pub async fn launch(binary: &Path, spec: LaunchSpec) -> Result<FirecrackerVm> {
    let scratch = super::create_scratch(&spec.work_dir, spec.scratch_bytes)?;
    let sockets = SocketDir(Some(socket_dir(&spec.id)?));
    let vsock = sockets.path().join(VSOCK_SOCKET);
    let egress_listener =
        UnixListener::bind(egress_socket(&vsock)).context("bind the egress vsock listener")?;
    let config = spec.work_dir.join(CONFIG_FILE);
    std::fs::write(
        &config,
        serde_json::to_vec_pretty(&config_json(&spec, &scratch, &vsock))?,
    )?;
    let console = spec.work_dir.join(CONSOLE_LOG);
    let console_file = std::fs::File::create(&console)?;
    let mut command = Command::new(binary);
    command
        .arg("--no-api")
        .arg("--config-file")
        .arg(&config)
        .arg("--id")
        .arg(&spec.id)
        .stdin(Stdio::null())
        .stdout(console_file.try_clone()?)
        .stderr(console_file)
        .kill_on_drop(true);
    // SAFETY: `prctl` is async-signal-safe and touches no Rust state. The VM
    // dies with the plugin instead of outliving it unmanaged.
    unsafe {
        command.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
    let child = command
        .spawn()
        .with_context(|| format!("start {}", binary.display()))?;
    let on_egress = spec.on_egress.clone();
    let egress_task = tokio::spawn(async move {
        while let Ok((stream, _)) = egress_listener.accept().await {
            on_egress(Box::new(stream));
        }
    });
    Ok(FirecrackerVm {
        child: tokio::sync::Mutex::new(Some(child)),
        vsock,
        sockets: sockets.keep(),
        console,
        egress_task: Mutex::new(Some(egress_task)),
    })
}

fn config_json(spec: &LaunchSpec, scratch: &Path, vsock: &Path) -> serde_json::Value {
    json!({
        "boot-source": {
            "kernel_image_path": spec.kernel,
            "boot_args": kernel_cmdline("ttyS0"),
        },
        "drives": [
            {
                "drive_id": "image",
                "path_on_host": spec.rootfs,
                "is_root_device": false,
                "is_read_only": true,
            },
            {
                "drive_id": "scratch",
                "path_on_host": scratch,
                "is_root_device": false,
                "is_read_only": false,
            },
        ],
        "machine-config": {
            "vcpu_count": spec.cpus,
            "mem_size_mib": spec.memory_mib,
        },
        "vsock": {
            "guest_cid": GUEST_CID,
            "uds_path": vsock,
        },
    })
}

/// A fresh `/tmp/mesh-compute-<uid>/<id>` for one VM's sockets, in a parent
/// only this user can enter.
fn socket_dir(id: &str) -> Result<PathBuf> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    // SAFETY: `getuid` has no preconditions.
    let uid = unsafe { libc::getuid() };
    let base = PathBuf::from(format!("/tmp/mesh-compute-{uid}"));
    match std::fs::create_dir(&base) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).with_context(|| format!("create {}", base.display())),
    }
    let metadata = std::fs::symlink_metadata(&base)?;
    if !metadata.is_dir() || metadata.uid() != uid {
        bail!("{} is not a directory owned by this user", base.display());
    }
    std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o700))?;
    let dir = base.join(id);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

/// Removes a VM's socket directory if the VM never starts.
struct SocketDir(Option<PathBuf>);

impl SocketDir {
    fn path(&self) -> &Path {
        self.0.as_deref().unwrap_or_else(|| Path::new(""))
    }

    fn keep(mut self) -> PathBuf {
        self.0.take().unwrap_or_default()
    }
}

impl Drop for SocketDir {
    fn drop(&mut self) {
        if let Some(dir) = self.0.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn egress_socket(vsock: &Path) -> PathBuf {
    let mut name = vsock.as_os_str().to_owned();
    name.push(format!("_{EGRESS_PORT}"));
    PathBuf::from(name)
}

pub struct FirecrackerVm {
    child: tokio::sync::Mutex<Option<Child>>,
    vsock: PathBuf,
    sockets: PathBuf,
    console: PathBuf,
    egress_task: Mutex<Option<JoinHandle<()>>>,
}

#[async_trait]
impl Vm for FirecrackerVm {
    async fn connect_agent(&self) -> Result<BoxStream> {
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, connect_port(&self.vsock, AGENT_PORT))
            .await
            .context("timed out connecting to the guest agent")??;
        Ok(Box::new(stream))
    }

    async fn stop(&self) {
        if let Ok(mut task) = self.egress_task.lock()
            && let Some(task) = task.take()
        {
            task.abort();
        }
        if let Some(mut child) = self.child.lock().await.take()
            && tokio::time::timeout(STOP_GRACE, child.wait())
                .await
                .is_err()
        {
            let _ = child.kill().await;
        }
        let _ = std::fs::remove_dir_all(&self.sockets);
    }

    fn console_log(&self) -> Option<PathBuf> {
        Some(self.console.clone())
    }
}

/// Opens a host-initiated vsock connection through Firecracker's socket.
pub async fn connect_port(vsock: &Path, port: u32) -> Result<UnixStream> {
    let mut stream = UnixStream::connect(vsock)
        .await
        .with_context(|| format!("connect {}", vsock.display()))?;
    stream
        .write_all(format!("CONNECT {port}\n").as_bytes())
        .await?;
    // Read the acknowledgement a byte at a time so nothing after it is
    // buffered away from the caller.
    let mut reader = BufReader::with_capacity(1, &mut stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    if !line.starts_with("OK ") {
        bail!("guest refused vsock port {port}: {}", line.trim());
    }
    Ok(stream)
}

/// The Firecracker the plugin archive ships beside the plugin binary.
fn bundled() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let candidate = exe.parent()?.join("firecracker");
    candidate.is_file().then_some(candidate)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn config_boots_the_image_read_only_with_a_writable_scratch_disk() {
        let spec = LaunchSpec {
            id: "sbx".into(),
            kernel: "/images/base/vmlinux".into(),
            rootfs: "/images/base/rootfs.ext4".into(),
            scratch_bytes: 1 << 30,
            cpus: 2,
            memory_mib: 1024,
            work_dir: "/run/sbx".into(),
            on_egress: Arc::new(|_| {}),
        };
        let config = config_json(
            &spec,
            Path::new("/run/sbx/scratch.img"),
            Path::new("/run/sbx/v.sock"),
        );
        assert_eq!(config["drives"][0]["is_read_only"], true);
        assert_eq!(config["drives"][1]["is_read_only"], false);
        assert_eq!(config["machine-config"]["vcpu_count"], 2);
        assert!(
            config["boot-source"]["boot_args"]
                .as_str()
                .unwrap()
                .contains("init=/sbin/mesh-compute-guest")
        );
        assert_eq!(
            egress_socket(Path::new("/run/sbx/v.sock")),
            PathBuf::from(format!("/run/sbx/v.sock_{EGRESS_PORT}"))
        );
    }

    #[tokio::test]
    async fn vsock_handshake_returns_the_stream_after_ok() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("v.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            assert_eq!(line, format!("CONNECT {AGENT_PORT}\n"));
            stream.write_all(b"OK 1073741824\nhello").await.unwrap();
        });
        let mut stream = connect_port(&socket, AGENT_PORT).await.unwrap();
        let mut rest = String::new();
        tokio::io::AsyncReadExt::read_to_string(&mut stream, &mut rest)
            .await
            .unwrap();
        assert_eq!(rest, "hello");
    }
}
