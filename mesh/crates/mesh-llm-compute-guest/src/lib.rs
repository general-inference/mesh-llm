//! The contract between the compute plugin on the host and the agent inside
//! each sandbox microVM.
//!
//! The agent runs as PID 1 in the guest and listens on vsock. The host opens
//! one vsock connection per operation and speaks length-prefixed JSON frames
//! ([`protocol`]). Network egress goes the other way: the guest has no network
//! interface, and its local HTTP proxy opens a vsock connection to the host for
//! every outbound connection ([`protocol::EgressRequest`]).

pub mod protocol;

#[cfg(all(target_os = "linux", feature = "agent"))]
pub mod agent;
