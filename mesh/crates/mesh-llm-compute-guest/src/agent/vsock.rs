//! Minimal blocking `AF_VSOCK` stream sockets.

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

pub struct VsockStream {
    file: File,
}

impl VsockStream {
    pub fn connect(cid: u32, port: u32) -> io::Result<Self> {
        let fd = new_socket()?;
        let addr = sockaddr(cid, port);
        // SAFETY: `addr` is a valid `sockaddr_vm` and the length matches it.
        let rc = unsafe {
            libc::connect(
                fd.as_raw_fd(),
                std::ptr::from_ref(&addr).cast(),
                sockaddr_len(),
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            file: File::from(fd),
        })
    }

    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            file: self.file.try_clone()?,
        })
    }

    pub fn shutdown_write(&self) {
        // SAFETY: the descriptor is owned by `self.file` and still open.
        unsafe {
            libc::shutdown(self.file.as_raw_fd(), libc::SHUT_WR);
        }
    }
}

impl Read for VsockStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Write for VsockStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct VsockListener {
    fd: OwnedFd,
}

impl VsockListener {
    pub fn bind(port: u32) -> io::Result<Self> {
        let fd = new_socket()?;
        let addr = sockaddr(libc::VMADDR_CID_ANY, port);
        // SAFETY: `addr` is a valid `sockaddr_vm` and the length matches it.
        let rc = unsafe {
            libc::bind(
                fd.as_raw_fd(),
                std::ptr::from_ref(&addr).cast(),
                sockaddr_len(),
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: plain syscall on an owned, bound socket.
        if unsafe { libc::listen(fd.as_raw_fd(), 64) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd })
    }

    pub fn accept(&self) -> io::Result<VsockStream> {
        loop {
            // SAFETY: a null address is allowed when the peer address is unused.
            let raw = unsafe {
                libc::accept4(
                    self.fd.as_raw_fd(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    libc::SOCK_CLOEXEC,
                )
            };
            if raw >= 0 {
                // SAFETY: `accept4` returned a new descriptor we now own.
                let fd = unsafe { OwnedFd::from_raw_fd(raw) };
                return Ok(VsockStream {
                    file: File::from(fd),
                });
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

fn new_socket() -> io::Result<OwnedFd> {
    // SAFETY: plain syscall; the result is checked before use.
    let raw = unsafe { libc::socket(libc::AF_VSOCK, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `socket` returned a new descriptor we now own.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn sockaddr(cid: u32, port: u32) -> libc::sockaddr_vm {
    // SAFETY: `sockaddr_vm` is plain old data; all-zero is a valid value.
    let mut addr: libc::sockaddr_vm = unsafe { std::mem::zeroed() };
    addr.svm_family = libc::AF_VSOCK as libc::sa_family_t;
    addr.svm_cid = cid;
    addr.svm_port = port;
    addr
}

fn sockaddr_len() -> libc::socklen_t {
    libc::socklen_t::try_from(std::mem::size_of::<libc::sockaddr_vm>())
        .expect("sockaddr_vm fits socklen_t")
}
