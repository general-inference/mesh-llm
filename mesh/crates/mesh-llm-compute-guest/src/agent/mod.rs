//! The guest agent: PID 1 setup, the operation server and the egress proxy.
//!
//! Run as PID 1 it prepares the root filesystem ([`init`]), forks, and keeps
//! PID 1 as a reaper while the child serves. Run any other way (for tests on
//! a Linux host) it serves straight away; set `MESH_COMPUTE_GUEST_TCP` to a
//! listen address and `MESH_COMPUTE_GUEST_EGRESS_TCP` to the host egress
//! address to use TCP instead of vsock.

mod exec;
mod files;
mod init;
mod proxy;
mod server;
mod transport;
mod vsock;

use anyhow::Result;

pub use transport::{Conn, Dialer, Listener};

pub fn main() -> Result<()> {
    if std::process::id() == 1 {
        if let Err(error) = init::prepare_root() {
            eprintln!("mesh-compute-guest: root setup failed: {error:#}");
            init::power_off();
        }
        return init::supervise(serve);
    }
    serve()
}

fn serve() -> Result<()> {
    let listener = Listener::from_env()?;
    let dialer = Dialer::from_env()?;
    proxy::spawn(dialer)?;
    server::run(listener)
}
