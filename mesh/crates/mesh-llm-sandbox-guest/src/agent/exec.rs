//! Runs a command and streams its output.
//!
//! The command gets its own session so a timeout kills everything it started.
//! Output is streamed as it arrives. Once the command exits, output from
//! background processes it left holding the pipes is waited on only briefly:
//! `server &` must not keep the call open.

use std::io::{Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use super::transport::Conn;
use crate::protocol::{
    CHUNK_BYTES, ExecRequest, ExitStatus, GUEST_PROXY_ADDR, Reply, WORKSPACE_DIR, write_frame,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
const DEFAULT_MAX_OUTPUT: u64 = 1024 * 1024;
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_millis(500);
const POLL_INTERVAL: Duration = Duration::from_millis(10);

type SharedConn = Arc<Mutex<Box<dyn Conn>>>;

pub fn run(request: &ExecRequest, conn: Box<dyn Conn>) -> Result<()> {
    let conn: SharedConn = Arc::new(Mutex::new(conn));
    let mut child = match spawn(request) {
        Ok(child) => child,
        Err(error) => return send(&conn, &Reply::error(format!("{error:#}"))),
    };
    let stopped = Arc::new(AtomicBool::new(false));
    let max_output = request.max_output_bytes.unwrap_or(DEFAULT_MAX_OUTPUT);
    let stdout = child.stdout.take().map(|pipe| {
        pump(pipe, conn.clone(), stopped.clone(), max_output, |data| {
            Reply::Stdout { data }
        })
    });
    let stderr = child.stderr.take().map(|pipe| {
        pump(pipe, conn.clone(), stopped.clone(), max_output, |data| {
            Reply::Stderr { data }
        })
    });
    if let (Some(input), Some(mut pipe)) = (request.stdin.clone(), child.stdin.take()) {
        std::thread::spawn(move || {
            let _ = pipe.write_all(&input);
        });
    }
    let timeout = request
        .timeout_ms
        .map_or(DEFAULT_TIMEOUT, Duration::from_millis);
    let (status, timed_out) = wait_with_timeout(&mut child, timeout)?;
    let stdout_truncated = finish_pump(stdout);
    let stderr_truncated = finish_pump(stderr);
    stopped.store(true, Ordering::SeqCst);
    send(
        &conn,
        &Reply::Exit(ExitStatus {
            code: status.code(),
            signal: status.signal(),
            timed_out,
            stdout_truncated,
            stderr_truncated,
        }),
    )
}

fn spawn(request: &ExecRequest) -> Result<Child> {
    let mut command = match &request.shell {
        Some(line) => {
            let mut command = Command::new("/bin/sh");
            command.arg("-c").arg(line);
            command
        }
        None => {
            let Some((program, args)) = request.argv.split_first() else {
                bail!("exec needs `argv` or `shell`");
            };
            let mut command = Command::new(program);
            command.args(args);
            command
        }
    };
    command
        .current_dir(request.cwd.as_deref().unwrap_or(WORKSPACE_DIR))
        .env_clear()
        .envs(default_env())
        .envs(&request.env)
        .stdin(if request.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: `setsid` is async-signal-safe and touches no Rust state.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command.spawn().context("spawn command")
}

pub fn default_env() -> Vec<(String, String)> {
    let proxy = format!("http://{GUEST_PROXY_ADDR}");
    let mut env = vec![
        (
            "PATH".to_string(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
        ),
        ("HOME".to_string(), "/root".to_string()),
        ("LANG".to_string(), "C.UTF-8".to_string()),
        ("TERM".to_string(), "dumb".to_string()),
        (
            "NO_PROXY".to_string(),
            "localhost,127.0.0.1,::1".to_string(),
        ),
        (
            "no_proxy".to_string(),
            "localhost,127.0.0.1,::1".to_string(),
        ),
    ];
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"] {
        env.push((name.to_string(), proxy.clone()));
    }
    env
}

fn wait_with_timeout(
    child: &mut Child,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, bool)> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok((status, false));
        }
        if Instant::now() >= deadline {
            kill_session(child);
            return Ok((child.wait()?, true));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn kill_session(child: &Child) {
    let Ok(pid) = libc::pid_t::try_from(child.id()) else {
        return;
    };
    // SAFETY: plain syscall; the child leads its own session and group.
    unsafe {
        libc::kill(-pid, libc::SIGKILL);
    }
}

fn pump(
    mut pipe: impl Read + Send + 'static,
    conn: SharedConn,
    stopped: Arc<AtomicBool>,
    max_output: u64,
    frame: fn(Vec<u8>) -> Reply,
) -> JoinHandle<bool> {
    std::thread::spawn(move || {
        let mut buffer = vec![0u8; CHUNK_BYTES];
        let mut sent = 0u64;
        let mut truncated = false;
        loop {
            let read = match pipe.read(&mut buffer) {
                Ok(0) | Err(_) => return truncated,
                Ok(read) => read,
            };
            let room = usize::try_from(max_output.saturating_sub(sent)).unwrap_or(usize::MAX);
            let keep = read.min(room);
            truncated |= keep < read;
            if keep == 0 || stopped.load(Ordering::SeqCst) {
                continue;
            }
            sent += keep as u64;
            if send(&conn, &frame(buffer[..keep].to_vec())).is_err() {
                return truncated;
            }
        }
    })
}

/// Waits briefly for a pump to drain; a pipe still held open by a background
/// process is abandoned.
fn finish_pump(pump: Option<JoinHandle<bool>>) -> bool {
    let Some(pump) = pump else {
        return false;
    };
    let deadline = Instant::now() + OUTPUT_DRAIN_GRACE;
    while !pump.is_finished() && Instant::now() < deadline {
        std::thread::sleep(POLL_INTERVAL);
    }
    if pump.is_finished() {
        pump.join().unwrap_or(false)
    } else {
        false
    }
}

fn send(conn: &SharedConn, reply: &Reply) -> Result<()> {
    let mut conn = conn
        .lock()
        .map_err(|_| anyhow::anyhow!("connection lock poisoned"))?;
    write_frame(&mut *conn, reply)
}
