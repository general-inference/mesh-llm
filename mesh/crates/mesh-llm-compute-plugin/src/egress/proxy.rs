//! The host end of the guest's egress connections.
//!
//! Each guest connection names a host and port. The host checks the name,
//! resolves it, keeps only addresses the policy admits, connects to one of
//! them, answers the guest, and then pipes bytes both ways.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mesh_llm_compute_guest::protocol::async_io::{read_frame, write_frame};
use mesh_llm_compute_guest::protocol::{EgressReply, EgressRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use super::{Denial, EgressPolicy};
use crate::vm::BoxStream;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const RESOLVE_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const LOG_CAPACITY: usize = 500;

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct EgressEvent {
    pub at_ms: u64,
    pub host: String,
    pub port: u16,
    pub allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    pub bytes_out: u64,
    pub bytes_in: u64,
    /// False while the connection is still open.
    pub closed: bool,
}

/// The most recent egress decisions for one sandbox.
#[derive(Default)]
pub struct EgressLog {
    entries: Mutex<VecDeque<(u64, EgressEvent)>>,
    next_id: Mutex<u64>,
}

impl EgressLog {
    pub fn snapshot(&self) -> Vec<EgressEvent> {
        self.entries
            .lock()
            .map(|entries| entries.iter().map(|(_, event)| event.clone()).collect())
            .unwrap_or_default()
    }

    fn push(&self, event: EgressEvent) -> u64 {
        let id = self.next_id.lock().map_or(0, |mut next| {
            *next += 1;
            *next
        });
        if let Ok(mut entries) = self.entries.lock() {
            if entries.len() == LOG_CAPACITY {
                entries.pop_front();
            }
            entries.push_back((id, event));
        }
        id
    }

    fn close(&self, id: u64, bytes_out: u64, bytes_in: u64) {
        if let Ok(mut entries) = self.entries.lock()
            && let Some((_, event)) = entries.iter_mut().find(|(entry, _)| *entry == id)
        {
            event.bytes_out = bytes_out;
            event.bytes_in = bytes_in;
            event.closed = true;
        }
    }
}

/// Serves one guest egress connection to completion.
pub async fn serve(mut guest: BoxStream, policy: Arc<EgressPolicy>, log: Arc<EgressLog>) {
    let request = match tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        read_frame::<EgressRequest>(&mut guest),
    )
    .await
    {
        Ok(Ok(Some(request))) => request,
        _ => return,
    };
    match connect(&policy, &request).await {
        Ok((mut upstream, address)) => {
            let reply = EgressReply {
                allowed: true,
                reason: None,
            };
            if write_frame(&mut guest, &reply).await.is_err() {
                return;
            }
            let id = log.push(event(&request, true, None, Some(address)));
            let (bytes_out, bytes_in) = tokio::io::copy_bidirectional(&mut guest, &mut upstream)
                .await
                .unwrap_or((0, 0));
            log.close(id, bytes_out, bytes_in);
        }
        Err(Denial(reason)) => {
            log.push(EgressEvent {
                closed: true,
                ..event(&request, false, Some(reason.clone()), None)
            });
            let _ = write_frame(
                &mut guest,
                &EgressReply {
                    allowed: false,
                    reason: Some(reason),
                },
            )
            .await;
        }
    }
}

async fn connect(
    policy: &EgressPolicy,
    request: &EgressRequest,
) -> Result<(TcpStream, SocketAddr), Denial> {
    let named = policy.check_name(&request.host, request.port)?;
    let resolved = resolve(&request.host, request.port).await?;
    let admitted = policy.admit(&request.host, named, &resolved)?;
    let mut last_error = None;
    for address in admitted {
        match tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(address)).await {
            Ok(Ok(stream)) => return Ok((stream, address)),
            Ok(Err(error)) => last_error = Some(error.to_string()),
            Err(_) => last_error = Some("timed out".into()),
        }
    }
    Err(Denial(format!(
        "could not connect to {}:{}: {}",
        request.host,
        request.port,
        last_error.unwrap_or_default()
    )))
}

async fn resolve(host: &str, port: u16) -> Result<Vec<SocketAddr>, Denial> {
    match tokio::time::timeout(RESOLVE_TIMEOUT, tokio::net::lookup_host((host, port))).await {
        Ok(Ok(addrs)) => Ok(addrs.collect()),
        Ok(Err(error)) => Err(Denial(format!("could not resolve {host}: {error}"))),
        Err(_) => Err(Denial(format!("resolving {host} timed out"))),
    }
}

fn event(
    request: &EgressRequest,
    allowed: bool,
    reason: Option<String>,
    address: Option<SocketAddr>,
) -> EgressEvent {
    EgressEvent {
        at_ms: crate::now_ms(),
        host: request.host.clone(),
        port: request.port,
        allowed,
        reason,
        address: address.map(|address| address.to_string()),
        bytes_out: 0,
        bytes_in: 0,
        closed: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::{EgressMode, EgressSpec};

    #[tokio::test]
    async fn denied_requests_get_a_reason_and_are_logged() {
        let policy = Arc::new(
            EgressPolicy::compile(&EgressSpec {
                mode: EgressMode::None,
                ..EgressSpec::default()
            })
            .unwrap(),
        );
        let log = Arc::new(EgressLog::default());
        let (mut guest, host) = tokio::io::duplex(4096);
        let served = tokio::spawn(serve(Box::new(host), policy, log.clone()));
        write_frame(
            &mut guest,
            &EgressRequest {
                host: "example.com".into(),
                port: 443,
            },
        )
        .await
        .unwrap();
        let reply: EgressReply = read_frame(&mut guest).await.unwrap().unwrap();
        served.await.unwrap();
        assert!(!reply.allowed);
        assert!(reply.reason.unwrap().contains("off"));
        let events = log.snapshot();
        assert_eq!(events.len(), 1);
        assert!(!events[0].allowed);
    }
}
