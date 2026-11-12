//! Where the agent listens and how it reaches the host: vsock in a VM, TCP
//! when testing on a Linux host.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};

use anyhow::{Context, Result};

use super::vsock::{VsockListener, VsockStream};
use crate::protocol::{AGENT_PORT, EGRESS_PORT, HOST_CID};

const LISTEN_TCP_ENV: &str = "MESH_COMPUTE_GUEST_TCP";
const EGRESS_TCP_ENV: &str = "MESH_COMPUTE_GUEST_EGRESS_TCP";

/// One host connection.
pub trait Conn: Read + Write + Send + 'static {
    fn try_clone_conn(&self) -> io::Result<Box<dyn Conn>>;
    fn shutdown_write(&self);
}

impl Conn for VsockStream {
    fn try_clone_conn(&self) -> io::Result<Box<dyn Conn>> {
        Ok(Box::new(self.try_clone()?))
    }

    fn shutdown_write(&self) {
        VsockStream::shutdown_write(self);
    }
}

impl Conn for TcpStream {
    fn try_clone_conn(&self) -> io::Result<Box<dyn Conn>> {
        Ok(Box::new(self.try_clone()?))
    }

    fn shutdown_write(&self) {
        let _ = self.shutdown(Shutdown::Write);
    }
}

pub enum Listener {
    Vsock(VsockListener),
    Tcp(TcpListener),
}

impl Listener {
    pub fn from_env() -> Result<Self> {
        match env_value(LISTEN_TCP_ENV) {
            Some(addr) => Ok(Self::Tcp(
                TcpListener::bind(&addr).with_context(|| format!("listen on {addr}"))?,
            )),
            None => Ok(Self::Vsock(
                VsockListener::bind(AGENT_PORT).context("listen on vsock")?,
            )),
        }
    }

    pub fn accept(&self) -> io::Result<Box<dyn Conn>> {
        match self {
            Self::Vsock(listener) => Ok(Box::new(listener.accept()?)),
            Self::Tcp(listener) => Ok(Box::new(listener.accept()?.0)),
        }
    }
}

/// Opens connections to the host's egress port.
#[derive(Clone)]
pub enum Dialer {
    Vsock,
    Tcp(String),
}

impl Dialer {
    pub fn from_env() -> Result<Self> {
        Ok(env_value(EGRESS_TCP_ENV).map_or(Self::Vsock, Self::Tcp))
    }

    pub fn dial(&self) -> io::Result<Box<dyn Conn>> {
        match self {
            Self::Vsock => Ok(Box::new(VsockStream::connect(HOST_CID, EGRESS_PORT)?)),
            Self::Tcp(addr) => Ok(Box::new(TcpStream::connect(addr)?)),
        }
    }
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}
