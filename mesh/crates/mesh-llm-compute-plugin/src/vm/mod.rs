//! MicroVM backends. Every platform boots the same guest (kernel, read-only
//! image disk, scratch disk, agent on vsock); only the VM monitor differs:
//!
//! - Linux: Firecracker on KVM ([`firecracker`]).
//! - Windows: Firecracker inside WSL2, driven through a bridge process that
//!   runs this binary's Linux build there ([`wsl`], [`bridge`]).
//! - macOS: Apple's Virtualization.framework ([`vz`]).

#[cfg(any(target_os = "linux", windows))]
pub mod bridge;
#[cfg(test)]
pub mod fake;
#[cfg(target_os = "linux")]
pub mod firecracker;
#[cfg(target_os = "macos")]
pub mod vz;
#[cfg(any(windows, target_os = "linux"))]
pub mod wsl;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncWrite};

/// A byte stream to or from the guest.
pub trait GuestStream: AsyncRead + AsyncWrite + Send + Unpin {}

impl<T: AsyncRead + AsyncWrite + Send + Unpin> GuestStream for T {}

pub type BoxStream = Box<dyn GuestStream>;

/// Called with each connection the guest opens to the host egress port.
pub type EgressHandler = Arc<dyn Fn(BoxStream) + Send + Sync>;

/// What to boot.
pub struct LaunchSpec {
    pub id: String,
    pub kernel: PathBuf,
    /// Read-only ext4 root (`/dev/vda`).
    pub rootfs: PathBuf,
    /// Size of the scratch disk (`/dev/vdb`). The backend creates it as a
    /// sparse file in `work_dir`; the guest formats it on first boot.
    pub scratch_bytes: u64,
    pub cpus: u32,
    pub memory_mib: u64,
    /// Per-sandbox directory for sockets and logs; removed with the sandbox.
    /// The WSL bridge keeps its own inside WSL.
    #[cfg_attr(windows, allow(dead_code))]
    pub work_dir: PathBuf,
    pub on_egress: EgressHandler,
}

/// A running microVM.
#[async_trait]
pub trait Vm: Send + Sync {
    /// A new connection to the guest agent.
    async fn connect_agent(&self) -> Result<BoxStream>;
    /// Stops the VM if it is still running. Idempotent.
    async fn stop(&self);
    /// Where the guest console is written, for diagnosing a failed boot.
    fn console_log(&self) -> Option<PathBuf>;
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    async fn launch(&self, spec: LaunchSpec) -> Result<Box<dyn Vm>>;
}

#[cfg(unix)]
/// The kernel command line every backend boots the guest with, after its
/// own console argument. The image disk is the read-only root.
pub fn kernel_cmdline(console: &str) -> String {
    format!(
        "console={console} reboot=k panic=1 quiet root=/dev/vda ro rootfstype=ext4 init=/sbin/mesh-compute-guest"
    )
}

/// The backend for this platform, or why there is none.
pub async fn detect(settings: &crate::settings::Settings) -> Result<Arc<dyn Backend>> {
    #[cfg(target_os = "linux")]
    {
        Ok(Arc::new(firecracker::FirecrackerBackend::detect(
            settings.firecracker_path.clone(),
        )?))
    }
    #[cfg(target_os = "macos")]
    {
        let _ = settings;
        Ok(Arc::new(vz::VzBackend::detect()?))
    }
    #[cfg(windows)]
    {
        Ok(Arc::new(wsl::start_wsl(settings).await?))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = settings;
        anyhow::bail!("sandboxes are not supported on this platform")
    }
}

#[cfg(unix)]
const SCRATCH_FILE: &str = "scratch.img";

/// Creates the sparse scratch disk in `dir`.
#[cfg(unix)]
pub fn create_scratch(dir: &std::path::Path, bytes: u64) -> Result<PathBuf> {
    let path = dir.join(SCRATCH_FILE);
    let file = std::fs::File::create(&path)?;
    file.set_len(bytes)?;
    Ok(path)
}
