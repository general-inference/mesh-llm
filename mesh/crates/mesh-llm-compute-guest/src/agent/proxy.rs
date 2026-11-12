//! The guest's HTTP proxy: the only way out of the sandbox.
//!
//! The guest has no network interface. Commands are pointed at this proxy by
//! `HTTP_PROXY`/`HTTPS_PROXY`; for each request it asks the host, over vsock,
//! for a connection to the named host and port. The host applies the
//! sandbox's egress policy and resolves the name, so the guest needs no DNS.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use anyhow::{Context, Result, bail};

use super::transport::{Conn, Dialer};
use crate::protocol::{EgressReply, EgressRequest, GUEST_PROXY_ADDR, read_frame, write_frame};

const MAX_HEAD_BYTES: usize = 64 * 1024;

pub fn spawn(dialer: Dialer) -> Result<()> {
    let listener = TcpListener::bind(GUEST_PROXY_ADDR)
        .with_context(|| format!("listen on {GUEST_PROXY_ADDR}"))?;
    std::thread::spawn(move || {
        for client in listener.incoming().flatten() {
            let dialer = dialer.clone();
            std::thread::spawn(move || {
                let _ = handle(client, &dialer);
            });
        }
    });
    Ok(())
}

fn handle(mut client: TcpStream, dialer: &Dialer) -> Result<()> {
    let (head, rest) = read_head(&mut client)?;
    let request = match ProxyRequest::parse(&head) {
        Ok(request) => request,
        Err(error) => {
            return respond(&mut client, "400 Bad Request", &format!("{error:#}"));
        }
    };
    let mut upstream = dialer.dial().context("reach the host egress port")?;
    write_frame(
        &mut upstream,
        &EgressRequest {
            host: request.host.clone(),
            port: request.port,
        },
    )?;
    let reply: EgressReply = read_frame(&mut upstream)?.context("host closed egress")?;
    if !reply.allowed {
        let reason = reply.reason.unwrap_or_else(|| "egress denied".into());
        return respond(
            &mut client,
            "403 Forbidden",
            &format!("mesh sandbox egress policy: {reason}\n"),
        );
    }
    if request.tunnel {
        client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
    } else {
        upstream.write_all(&request.forward_head)?;
    }
    upstream.write_all(&rest)?;
    splice(client, upstream)
}

fn respond(client: &mut TcpStream, status: &str, body: &str) -> Result<()> {
    write!(
        client,
        "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}

/// Reads up to the end of the request head; returns it and any bytes after.
fn read_head(client: &mut TcpStream) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        if let Some(end) = find_head_end(&buffer) {
            let rest = buffer.split_off(end);
            return Ok((buffer, rest));
        }
        if buffer.len() > MAX_HEAD_BYTES {
            bail!("request head too large");
        }
        let read = client.read(&mut chunk)?;
        if read == 0 {
            bail!("client closed before the request head ended");
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

fn splice(client: TcpStream, upstream: Box<dyn Conn>) -> Result<()> {
    let mut client_read = client.try_clone()?;
    let mut upstream_write = upstream.try_clone_conn()?;
    let outbound = std::thread::spawn(move || {
        let _ = std::io::copy(&mut client_read, &mut upstream_write);
        upstream_write.shutdown_write();
    });
    let mut upstream_read = upstream;
    let mut client_write = client;
    let _ = std::io::copy(&mut upstream_read, &mut client_write);
    let _ = client_write.shutdown(std::net::Shutdown::Write);
    let _ = outbound.join();
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ProxyRequest {
    host: String,
    port: u16,
    /// `CONNECT`: the client tunnels its own bytes (TLS) after our 200.
    tunnel: bool,
    /// For plain HTTP: the head to send upstream, in origin form.
    forward_head: Vec<u8>,
}

impl ProxyRequest {
    fn parse(head: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(head).context("request head is not UTF-8")?;
        let mut lines = text.split("\r\n");
        let request_line = lines.next().context("empty request")?;
        let mut parts = request_line.split(' ');
        let (Some(method), Some(target), Some(version)) =
            (parts.next(), parts.next(), parts.next())
        else {
            bail!("malformed request line");
        };
        if method.eq_ignore_ascii_case("CONNECT") {
            let (host, port) = split_authority(target, 443)?;
            return Ok(Self {
                host,
                port,
                tunnel: true,
                forward_head: Vec::new(),
            });
        }
        let Some(rest) = target.strip_prefix("http://") else {
            bail!("only absolute http:// targets and CONNECT are proxied");
        };
        let (authority, path) = rest
            .find('/')
            .map_or((rest, "/"), |slash| (&rest[..slash], &rest[slash..]));
        let (host, port) = split_authority(authority, 80)?;
        let mut forward = format!("{method} {path} {version}\r\n");
        for line in lines.filter(|line| !line.is_empty()) {
            let name = line.split(':').next().unwrap_or_default().trim();
            if ["connection", "proxy-connection", "keep-alive"]
                .iter()
                .any(|hop| name.eq_ignore_ascii_case(hop))
            {
                continue;
            }
            forward.push_str(line);
            forward.push_str("\r\n");
        }
        forward.push_str("Connection: close\r\n\r\n");
        Ok(Self {
            host,
            port,
            tunnel: false,
            forward_head: forward.into_bytes(),
        })
    }
}

fn split_authority(authority: &str, default_port: u16) -> Result<(String, u16)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest.split_once(']').context("unterminated IPv6 literal")?;
        let port = match after.strip_prefix(':') {
            Some(port) => port.parse().context("bad port")?,
            None => default_port,
        };
        return Ok((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) => Ok((host.to_string(), port.parse().context("bad port")?)),
        None => Ok((authority.to_string(), default_port)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_requests_name_the_tunnel_target() {
        let request =
            ProxyRequest::parse(b"CONNECT registry.npmjs.org:443 HTTP/1.1\r\nHost: x\r\n\r\n")
                .unwrap();
        assert_eq!(request.host, "registry.npmjs.org");
        assert_eq!(request.port, 443);
        assert!(request.tunnel);
    }

    #[test]
    fn plain_http_is_rewritten_to_origin_form_and_closed() {
        let request = ProxyRequest::parse(
            b"GET http://example.com:8080/a?b HTTP/1.1\r\nHost: example.com\r\nProxy-Connection: keep-alive\r\n\r\n",
        )
        .unwrap();
        assert_eq!((request.host.as_str(), request.port), ("example.com", 8080));
        assert_eq!(
            String::from_utf8(request.forward_head).unwrap(),
            "GET /a?b HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn ipv6_literals_and_default_ports_parse() {
        assert_eq!(
            split_authority("[::1]:8443", 443).unwrap(),
            ("::1".into(), 8443)
        );
        assert_eq!(
            split_authority("example.com", 80).unwrap(),
            ("example.com".into(), 80)
        );
    }

    #[test]
    fn https_absolute_targets_are_refused() {
        assert!(ProxyRequest::parse(b"GET https://example.com/ HTTP/1.1\r\n\r\n").is_err());
    }
}
