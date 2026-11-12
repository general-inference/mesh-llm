//! PID 1 duties: build a writable root, then reap orphans while the agent runs.
//!
//! The image disk (`/dev/vda`) is the read-only root. The per-sandbox scratch
//! disk (`/dev/vdb`) arrives as a sparse, unformatted file; it is formatted on
//! first boot and becomes the upper layer of an overlay over the image, so
//! anything the sandbox installs or writes lands on scratch. The agent then
//! chroots into the overlay.

use std::ffi::CString;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};

const SCRATCH_DEVICE: &str = "/dev/vdb";
const SCRATCH_MOUNT: &str = "/mnt/scratch";
const NEW_ROOT: &str = "/mnt/root";
const EXT4_MAGIC_OFFSET: u64 = 1080;
const EXT4_MAGIC: [u8; 2] = [0x53, 0xEF];
const HOSTNAME: &str = "sandbox";

pub fn prepare_root() -> Result<()> {
    // The kernel may have mounted /dev already (CONFIG_DEVTMPFS_MOUNT).
    mount_unless_busy("proc", "/proc", "proc")?;
    mount_unless_busy("devtmpfs", "/dev", "devtmpfs")?;
    mount("tmpfs", "/mnt", "tmpfs", "mode=0755")?;
    fs::create_dir_all(SCRATCH_MOUNT)?;
    fs::create_dir_all(NEW_ROOT)?;
    format_scratch_if_blank()?;
    mount(SCRATCH_DEVICE, SCRATCH_MOUNT, "ext4", "")?;
    fs::create_dir_all(format!("{SCRATCH_MOUNT}/upper"))?;
    fs::create_dir_all(format!("{SCRATCH_MOUNT}/work"))?;
    mount(
        "overlay",
        NEW_ROOT,
        "overlay",
        &format!("lowerdir=/,upperdir={SCRATCH_MOUNT}/upper,workdir={SCRATCH_MOUNT}/work"),
    )?;
    enter_root(NEW_ROOT)?;
    mount_runtime_filesystems()?;
    fs::create_dir_all(crate::protocol::WORKSPACE_DIR)?;
    set_hostname(HOSTNAME)?;
    bring_up_loopback()?;
    Ok(())
}

/// Forks the agent and stays behind as a reaper, restarting it if it dies.
pub fn supervise(serve: fn() -> Result<()>) -> Result<()> {
    loop {
        // SAFETY: PID 1 is still single-threaded here; the child only calls
        // `serve` and exits.
        let child = unsafe { libc::fork() };
        if child < 0 {
            bail!("fork: {}", std::io::Error::last_os_error());
        }
        if child == 0 {
            let code = match serve() {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("mesh-compute-guest: {error:#}");
                    1
                }
            };
            std::process::exit(code);
        }
        reap_until(child);
        eprintln!("mesh-compute-guest: agent exited; restarting");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn reap_until(agent: libc::pid_t) {
    loop {
        let mut status = 0;
        // SAFETY: plain syscall with a valid out pointer.
        let pid = unsafe { libc::waitpid(-1, &raw mut status, 0) };
        if pid == agent {
            return;
        }
        if pid < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

pub fn power_off() -> ! {
    // SAFETY: plain syscalls; `reboot` does not return on success.
    unsafe {
        libc::sync();
        // Firecracker on x86_64 has no ACPI power-off; it exits on the reboot
        // the kernel performs through the keyboard controller (`reboot=k`).
        if cfg!(target_arch = "x86_64") {
            libc::reboot(libc::RB_AUTOBOOT);
        }
        libc::reboot(libc::RB_POWER_OFF);
    }
    std::process::exit(1);
}

fn format_scratch_if_blank() -> Result<()> {
    if has_ext4_magic(SCRATCH_DEVICE)? {
        return Ok(());
    }
    let status = Command::new("/sbin/mkfs.ext4")
        .args([
            "-q",
            "-F",
            "-E",
            "lazy_itable_init=1,lazy_journal_init=1",
            SCRATCH_DEVICE,
        ])
        .status()
        .context("run mkfs.ext4")?;
    if !status.success() {
        bail!("mkfs.ext4 {SCRATCH_DEVICE} failed: {status}");
    }
    Ok(())
}

fn has_ext4_magic(device: &str) -> Result<bool> {
    let mut file = fs::File::open(device).with_context(|| format!("open {device}"))?;
    file.seek(SeekFrom::Start(EXT4_MAGIC_OFFSET))?;
    let mut magic = [0u8; 2];
    file.read_exact(&mut magic)?;
    Ok(magic == EXT4_MAGIC)
}

fn enter_root(root: &str) -> Result<()> {
    std::env::set_current_dir(root)?;
    let dot = CString::new(".")?;
    // SAFETY: plain syscall with a valid C string.
    if unsafe { libc::chroot(dot.as_ptr()) } != 0 {
        bail!("chroot {root}: {}", std::io::Error::last_os_error());
    }
    std::env::set_current_dir("/")?;
    Ok(())
}

fn mount_runtime_filesystems() -> Result<()> {
    mount("proc", "/proc", "proc", "")?;
    mount("sysfs", "/sys", "sysfs", "")?;
    mount("devtmpfs", "/dev", "devtmpfs", "")?;
    fs::create_dir_all("/dev/pts")?;
    mount("devpts", "/dev/pts", "devpts", "newinstance,ptmxmode=0666")?;
    fs::create_dir_all("/dev/shm")?;
    mount("tmpfs", "/dev/shm", "tmpfs", "mode=1777")?;
    mount("tmpfs", "/run", "tmpfs", "mode=0755")?;
    mount("tmpfs", "/tmp", "tmpfs", "mode=1777")?;
    Ok(())
}

fn mount_unless_busy(source: &str, target: &str, fstype: &str) -> Result<()> {
    match mount(source, target, fstype, "") {
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .and_then(std::io::Error::raw_os_error)
                == Some(libc::EBUSY) =>
        {
            Ok(())
        }
        other => other,
    }
}

fn mount(source: &str, target: &str, fstype: &str, data: &str) -> Result<()> {
    if !Path::new(target).exists() {
        fs::create_dir_all(target)?;
    }
    let source_c = CString::new(source)?;
    let target_c = CString::new(target)?;
    let fstype_c = CString::new(fstype)?;
    let data_c = CString::new(data)?;
    // SAFETY: all pointers are valid NUL-terminated strings for the call.
    let rc = unsafe {
        libc::mount(
            source_c.as_ptr(),
            target_c.as_ptr(),
            fstype_c.as_ptr(),
            0,
            data_c.as_ptr().cast(),
        )
    };
    if rc != 0 {
        let error = std::io::Error::last_os_error();
        return Err(
            anyhow::Error::new(error).context(format!("mount {fstype} {source} on {target}"))
        );
    }
    Ok(())
}

fn set_hostname(name: &str) -> Result<()> {
    // SAFETY: the pointer and length describe `name`.
    if unsafe { libc::sethostname(name.as_ptr().cast(), name.len()) } != 0 {
        bail!("sethostname: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

fn bring_up_loopback() -> Result<()> {
    // SAFETY: plain syscall; checked below.
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        bail!("socket: {}", std::io::Error::last_os_error());
    }
    // SAFETY: `ifreq` is plain old data; the name is NUL-terminated within it
    // and the union member read is the one SIOCGIFFLAGS fills in.
    let result = unsafe {
        let mut request: libc::ifreq = std::mem::zeroed();
        for (slot, byte) in request.ifr_name.iter_mut().zip(b"lo\0") {
            *slot = *byte as libc::c_char;
        }
        if libc::ioctl(fd, ioctl_request(libc::SIOCGIFFLAGS), &raw mut request) != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            request.ifr_ifru.ifru_flags |= (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
            if libc::ioctl(fd, ioctl_request(libc::SIOCSIFFLAGS), &raw const request) != 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    };
    // SAFETY: closing the descriptor opened above.
    unsafe {
        libc::close(fd);
    }
    result.context("bring up lo")
}

/// `ioctl`'s request parameter is `c_ulong` on glibc and `c_int` on musl.
#[allow(clippy::cast_possible_truncation, clippy::unnecessary_cast)]
fn ioctl_request(request: libc::c_ulong) -> libc::Ioctl {
    request as libc::Ioctl
}

/// Sets the guest clock when it is more than two seconds off the host's.
pub fn sync_clock(unix_time_ms: u64) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis());
    if now.abs_diff(u128::from(unix_time_ms)) < 2_000 {
        return;
    }
    // SAFETY: `timespec` is plain old data; all-zero is a valid value.
    let mut spec: libc::timespec = unsafe { std::mem::zeroed() };
    let Ok(secs) = (unix_time_ms / 1_000).try_into() else {
        return;
    };
    spec.tv_sec = secs;
    spec.tv_nsec = ((unix_time_ms % 1_000) * 1_000_000).try_into().unwrap_or(0);
    // SAFETY: plain syscall with a valid timespec; failure (not root) is harmless.
    unsafe {
        libc::clock_settime(libc::CLOCK_REALTIME, &raw const spec);
    }
}

/// The running kernel release, for the hello reply.
pub fn kernel_release() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|release| release.trim().to_string())
        .unwrap_or_default()
}
