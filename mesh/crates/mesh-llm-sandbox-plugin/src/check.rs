//! `mesh-sandbox check`: boots a sandbox on this machine and exercises it,
//! without mesh-llm, so an operator can tell whether this node is ready.
//!
//! Reads the same settings the plugin gets, as JSON in `MESH_SANDBOX_SETTINGS`
//! (default `{}`). Steps that need the network are skipped with `--offline`.
//! On Linux, `--bridge` drives Firecracker through the WSL bridge run as a
//! local process, which covers the Windows path short of WSL itself.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_sandbox_guest::protocol::ExecRequest;

use crate::egress::{EgressMode, EgressSpec};
use crate::manager::{Caller, Manager};
use crate::settings::Settings;
use crate::spec::{Resources, SandboxSpec};

/// Progress for the operator running `check`.
macro_rules! say {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(mesh_llm_events::console_out(), $($arg)*);
    }};
}

pub async fn run(offline: bool, bridge: bool) -> Result<()> {
    let settings_json = std::env::var("MESH_SANDBOX_SETTINGS").unwrap_or_else(|_| "{}".into());
    let settings = Settings::parse(&settings_json)?;
    let backend: Arc<dyn crate::vm::Backend> = if bridge {
        bridged(&settings).await?
    } else {
        crate::vm::detect(&settings).await?
    };
    say!("backend: {}", backend.name());
    let manager = Arc::new(Manager::new(settings, Ok(backend))?);
    let result = exercise(&manager, offline).await;
    for sandbox in manager.list(&Caller::Local) {
        let _ = manager.destroy(&Caller::Local, &sandbox.sandbox_id).await;
    }
    result?;
    say!("check passed");
    Ok(())
}

#[cfg(target_os = "linux")]
async fn bridged(settings: &Settings) -> Result<Arc<dyn crate::vm::Backend>> {
    Ok(Arc::new(crate::vm::wsl::start_local(settings).await?))
}

#[cfg(not(target_os = "linux"))]
async fn bridged(_settings: &Settings) -> Result<Arc<dyn crate::vm::Backend>> {
    bail!("--bridge runs the WSL bridge locally and is only available on Linux")
}

async fn exercise(manager: &Manager, offline: bool) -> Result<()> {
    let spec = SandboxSpec {
        resources: Resources {
            cpus: 2,
            memory_mb: 1_024,
            disk_gb: 2,
        },
        egress: EgressSpec {
            mode: EgressMode::Allowlist,
            allow: vec!["@registries".into()],
            deny: Vec::new(),
        },
        ..SandboxSpec::default()
    };
    let started = Instant::now();
    let info = manager.create(Caller::Local, spec.clone(), None).await?;
    let id = info.sandbox_id.clone();
    say!(
        "booted {id} ({}, {}) in {} ms",
        info.arch,
        info.backend,
        started.elapsed().as_millis()
    );

    let uname = shell(manager, &id, "uname -srm && id -u && pwd").await?;
    say!("exec: {}", uname.replace('\n', " | "));
    ensure!(
        uname.contains("/workspace"),
        "commands should start in /workspace"
    );

    manager
        .write_file(
            &Caller::Local,
            &id,
            "/workspace/hello.txt",
            b"hello sandbox\n",
            None,
            false,
        )
        .await?;
    let (data, _, _) = manager
        .read_file(&Caller::Local, &id, "/workspace/hello.txt", 1_024)
        .await?;
    ensure!(
        data == b"hello sandbox\n",
        "file round trip returned {data:?}"
    );
    say!("files: round trip ok");

    let timed = Instant::now();
    let output = manager
        .exec(
            &Caller::Local,
            &id,
            ExecRequest {
                shell: Some("sleep 30".into()),
                timeout_ms: Some(1_000),
                ..ExecRequest::default()
            },
        )
        .await?;
    ensure!(
        output.status.timed_out,
        "a command past its timeout should be killed"
    );
    say!("timeout: killed after {} ms", timed.elapsed().as_millis());

    let background = Instant::now();
    shell(manager, &id, "sleep 60 >/dev/null 2>&1 & echo started").await?;
    ensure!(
        background.elapsed().as_secs() < 10,
        "a background process must not hold exec open"
    );
    say!(
        "background: returned in {} ms",
        background.elapsed().as_millis()
    );

    if offline {
        say!("egress: skipped (--offline)");
    } else {
        check_egress(manager, &id).await?;
    }

    let snapshot = manager.snapshot(&Caller::Local, &id).await?;
    say!(
        "snapshot: {} ({} bytes)",
        snapshot.snapshot_id,
        snapshot.bytes
    );
    let (archive, _) = manager
        .snapshot_chunk(&snapshot.snapshot_id, 0, u64::MAX)
        .await?;
    let restored = manager
        .create(Caller::Local, SandboxSpec::default(), Some(archive))
        .await?;
    let text = shell(manager, &restored.sandbox_id, "cat /workspace/hello.txt").await?;
    ensure!(
        text == "hello sandbox\n",
        "restored workspace returned {text:?}"
    );
    say!("restore: {} has the snapshot's files", restored.sandbox_id);

    let log = manager.egress_log(&Caller::Local, &id)?;
    say!("egress log: {} entries", log.len());
    manager
        .destroy(&Caller::Local, &restored.sandbox_id)
        .await?;
    manager.destroy(&Caller::Local, &id).await?;
    ensure!(
        manager.list(&Caller::Local).is_empty(),
        "destroy should remove sandboxes"
    );
    Ok(())
}

async fn check_egress(manager: &Manager, id: &str) -> Result<()> {
    let allowed = shell(
        manager,
        id,
        "curl -sS -o /dev/null -w '%{http_code}' https://pypi.org/simple/pip/",
    )
    .await
    .context("an allowlisted registry should be reachable")?;
    ensure!(allowed == "200", "pypi.org returned HTTP {allowed}");
    let denied = exec_raw(
        manager,
        id,
        "curl -sS -o /dev/null -w '%{http_code}' https://example.com/",
    )
    .await?;
    ensure!(
        denied.status.code != Some(0) || String::from_utf8_lossy(&denied.stdout) == "403",
        "example.com is not in the allowlist and should be refused"
    );
    let metadata = exec_raw(
        manager,
        id,
        "curl -sS -m 5 -o /dev/null -w '%{http_code}' http://169.254.169.254/",
    )
    .await?;
    ensure!(
        String::from_utf8_lossy(&metadata.stdout) != "200",
        "the metadata address must never be reachable"
    );
    say!("egress: allowlisted registry reachable, others refused");
    Ok(())
}

async fn shell(manager: &Manager, id: &str, line: &str) -> Result<String> {
    let output = exec_raw(manager, id, line).await?;
    if output.status.code != Some(0) {
        bail!(
            "`{line}` exited with {:?}: {}",
            output.status.code,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn exec_raw(manager: &Manager, id: &str, line: &str) -> Result<crate::guest::ExecOutput> {
    manager
        .exec(
            &Caller::Local,
            id,
            ExecRequest {
                shell: Some(line.to_string()),
                timeout_ms: Some(60_000),
                ..ExecRequest::default()
            },
        )
        .await
}
