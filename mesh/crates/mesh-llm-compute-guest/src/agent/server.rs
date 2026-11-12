//! Accepts host connections and runs one operation on each.

use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use super::transport::{Conn, Listener};
use super::{exec, files, init};
use crate::protocol::{CHUNK_BYTES, PROTOCOL_VERSION, Reply, Request, read_frame, write_frame};

pub fn run(listener: Listener) -> Result<()> {
    loop {
        let conn = match listener.accept() {
            Ok(conn) => conn,
            Err(error) => {
                eprintln!("mesh-compute-guest: accept: {error}");
                continue;
            }
        };
        std::thread::spawn(move || {
            if let Err(error) = handle(conn) {
                eprintln!("mesh-compute-guest: connection: {error:#}");
            }
        });
    }
}

fn handle(mut conn: Box<dyn Conn>) -> Result<()> {
    let Some(request) = read_frame::<Request>(&mut conn)? else {
        return Ok(());
    };
    let reply = match request {
        Request::Hello { unix_time_ms } => Ok(hello(unix_time_ms)),
        Request::Exec(request) => return exec::run(&request, conn),
        Request::ReadFile { path, max_bytes } => files::read(&path, max_bytes),
        Request::WriteFile { path, mode, append } => {
            files::write(&path, mode, append, &mut conn).map(|()| Reply::Done)
        }
        Request::ListDir { path } => files::list(&path),
        Request::ArchiveOut { path } => return archive_out(&path, &mut conn),
        Request::ArchiveIn { path } => archive_in(&path, &mut conn).map(|()| Reply::Done),
        Request::Shutdown => {
            write_frame(&mut conn, &Reply::Done)?;
            init::power_off();
        }
        Request::Chunk { .. } | Request::End => Err(anyhow::anyhow!("unexpected stream frame")),
    };
    let reply = reply.unwrap_or_else(|error| Reply::error(format!("{error:#}")));
    write_frame(&mut conn, &reply)
}

fn hello(unix_time_ms: Option<u64>) -> Reply {
    if let Some(unix_time_ms) = unix_time_ms {
        init::sync_clock(unix_time_ms);
    }
    Reply::Hello {
        version: PROTOCOL_VERSION,
        arch: std::env::consts::ARCH.to_string(),
        kernel: init::kernel_release(),
    }
}

/// Streams `tar -czf - .` of `path` as chunks, then `Done` or the error.
fn archive_out(path: &str, conn: &mut Box<dyn Conn>) -> Result<()> {
    let mut child = Command::new("tar")
        .args(["-C", path, "-czf", "-", "."])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("run tar")?;
    let mut stdout = child.stdout.take().context("tar stdout")?;
    let mut buffer = vec![0u8; CHUNK_BYTES];
    loop {
        let read = std::io::Read::read(&mut stdout, &mut buffer)?;
        if read == 0 {
            break;
        }
        write_frame(
            conn,
            &Reply::Chunk {
                data: buffer[..read].to_vec(),
            },
        )?;
    }
    let output = child.wait_with_output()?;
    let reply = if output.status.success() {
        Reply::Done
    } else {
        Reply::error(format!(
            "tar failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    };
    write_frame(conn, &reply)
}

/// Unpacks chunks into `path` until `End`.
fn archive_in(path: &str, conn: &mut Box<dyn Conn>) -> Result<()> {
    std::fs::create_dir_all(path).with_context(|| format!("create {path}"))?;
    let mut child = Command::new("tar")
        .args(["-C", path, "-xzf", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("run tar")?;
    let mut stdin = child.stdin.take().context("tar stdin")?;
    let copied = files::copy_chunks(conn, &mut stdin);
    drop(stdin);
    let output = child.wait_with_output()?;
    copied?;
    if !output.status.success() {
        bail!(
            "tar failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}
