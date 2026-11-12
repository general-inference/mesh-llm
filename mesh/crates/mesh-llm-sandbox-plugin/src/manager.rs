//! Sandboxes running on this node.
//!
//! Callers are this node (MCP and HTTP on the local host) or a peer, as the
//! mesh authenticated it. A peer sees and drives only the sandboxes it
//! created; the local node is the operator and sees all of them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mesh_llm_sandbox_guest::protocol::{DirEntry, ExecRequest, WORKSPACE_DIR};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::egress::EgressPolicy;
use crate::egress::proxy::{self, EgressEvent, EgressLog};
use crate::guest::{ExecOutput, GuestClient, GuestInfo};
use crate::images::{ImageStore, guest_arch};
use crate::settings::{Settings, mode_name};
use crate::spec::{Resources, SandboxSpec};
use crate::vm::{Backend, LaunchSpec, Vm};

const BOOT_TIMEOUT: Duration = Duration::from_secs(60);
const CONSOLE_TAIL_BYTES: usize = 4_096;
const SNAPSHOT_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 3_600);
const RUN_DIR: &str = "run";
const SNAPSHOT_DIR: &str = "snapshots";

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Caller {
    Local,
    Peer(String),
}

impl Caller {
    fn label(&self) -> &str {
        match self {
            Self::Local => "local",
            Self::Peer(peer) => peer,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct SandboxInfo {
    pub sandbox_id: String,
    /// `local`, or the peer id of the node running it.
    pub node: String,
    pub image: String,
    pub arch: String,
    pub backend: String,
    pub resources: Resources,
    pub egress_mode: String,
    pub created_ms: u64,
    pub expires_ms: u64,
    pub owner: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct SnapshotInfo {
    pub snapshot_id: String,
    pub node: String,
    pub bytes: u64,
}

/// What this node offers, as advertised to the mesh.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct Capacity {
    /// Empty when this node cannot run sandboxes; see `unavailable`.
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable: Option<String>,
    pub arch: String,
    pub images: Vec<String>,
    pub free_slots: u32,
    pub max_cpus: u32,
    pub max_memory_mb: u64,
    pub max_disk_gb: u64,
    pub max_egress: String,
    pub accept_remote: bool,
}

struct Sandbox {
    info: SandboxInfo,
    owner: Caller,
    env: std::collections::BTreeMap<String, String>,
    vm: Box<dyn Vm>,
    egress_log: Arc<EgressLog>,
    dir: PathBuf,
}

pub struct Manager {
    settings: Settings,
    backend: Result<Arc<dyn Backend>, String>,
    images: ImageStore,
    sandboxes: Mutex<HashMap<String, Arc<Sandbox>>>,
    /// Slots taken by sandboxes still booting.
    booting: Mutex<u32>,
}

impl Manager {
    pub fn new(settings: Settings, backend: Result<Arc<dyn Backend>, String>) -> Result<Self> {
        let state = settings.state_dir();
        let run = state.join(RUN_DIR);
        // Sandboxes do not survive a plugin restart; clear what a previous
        // run left behind.
        let _ = std::fs::remove_dir_all(&run);
        std::fs::create_dir_all(&run).with_context(|| format!("create {}", run.display()))?;
        std::fs::create_dir_all(state.join(SNAPSHOT_DIR))?;
        let images = ImageStore::new(settings.images_dir());
        Ok(Self {
            settings,
            backend,
            images,
            sandboxes: Mutex::new(HashMap::new()),
            booting: Mutex::new(0),
        })
    }

    pub fn capacity(&self) -> Capacity {
        let used = self.sandboxes.lock().map_or(0, |map| map.len()) as u32
            + self.booting.lock().map_or(0, |booting| *booting);
        let (backend, unavailable) = match &self.backend {
            Ok(backend) => (backend.name().to_string(), None),
            Err(reason) => (String::new(), Some(reason.clone())),
        };
        Capacity {
            backend,
            unavailable,
            arch: guest_arch().to_string(),
            images: self
                .images
                .list()
                .into_iter()
                .map(|image| image.name)
                .collect(),
            free_slots: self.settings.max_sandboxes.saturating_sub(used),
            max_cpus: self.settings.max_cpus,
            max_memory_mb: self.settings.max_memory_mb,
            max_disk_gb: self.settings.max_disk_gb,
            max_egress: mode_name(self.settings.max_egress).to_string(),
            accept_remote: self.settings.accept_remote,
        }
    }

    /// Boots a sandbox; `restore` is a `/workspace` archive to unpack first.
    pub async fn create(
        &self,
        caller: Caller,
        spec: SandboxSpec,
        restore: Option<Vec<u8>>,
    ) -> Result<SandboxInfo> {
        let backend = self
            .backend
            .clone()
            .map_err(|reason| anyhow::anyhow!("this node cannot run sandboxes: {reason}"))?;
        if matches!(caller, Caller::Peer(_)) && !self.settings.accept_remote {
            bail!("this node does not run sandboxes for other nodes");
        }
        self.settings.check(&spec)?;
        let policy = Arc::new(EgressPolicy::compile(&spec.egress)?);
        let image = self.images.get(&spec.image)?;
        let _slot = self.reserve_slot()?;

        let id = format!("sbx-{}", uuid::Uuid::new_v4().simple());
        let dir = self.settings.state_dir().join(RUN_DIR).join(&id);
        std::fs::create_dir_all(&dir)?;
        let egress_log = Arc::new(EgressLog::default());
        let on_egress = {
            let log = egress_log.clone();
            Arc::new(move |stream| {
                tokio::spawn(proxy::serve(stream, policy.clone(), log.clone()));
            })
        };
        let launch = LaunchSpec {
            id: id.clone(),
            kernel: image.kernel,
            rootfs: image.rootfs,
            scratch_bytes: spec.resources.disk_gb << 30,
            cpus: spec.resources.cpus,
            memory_mib: spec.resources.memory_mb,
            work_dir: dir.clone(),
            on_egress,
        };
        let vm = match backend.launch(launch).await {
            Ok(vm) => vm,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(error.context("start the sandbox VM"));
            }
        };
        let guest = match self.boot(vm.as_ref(), restore.as_deref()).await {
            Ok(guest) => guest,
            Err(error) => {
                let console = console_tail(vm.as_ref());
                vm.stop().await;
                let _ = std::fs::remove_dir_all(&dir);
                return Err(error.context(console));
            }
        };
        let created_ms = crate::now_ms();
        let info = SandboxInfo {
            sandbox_id: id.clone(),
            node: "local".into(),
            image: spec.image.clone(),
            arch: guest.arch,
            backend: backend.name().to_string(),
            resources: spec.resources,
            egress_mode: mode_name(spec.egress.mode).to_string(),
            created_ms,
            expires_ms: created_ms + spec.ttl_secs * 1_000,
            owner: caller.label().to_string(),
        };
        let sandbox = Arc::new(Sandbox {
            info: info.clone(),
            owner: caller,
            env: spec.env,
            vm,
            egress_log,
            dir,
        });
        self.sandboxes
            .lock()
            .map_err(|_| anyhow::anyhow!("sandbox table poisoned"))?
            .insert(id, sandbox);
        Ok(info)
    }

    async fn boot(&self, vm: &dyn Vm, restore: Option<&[u8]>) -> Result<GuestInfo> {
        let client = GuestClient::new(vm);
        let guest = client.wait_ready(BOOT_TIMEOUT).await?;
        if let Some(archive) = restore {
            client
                .archive_in(WORKSPACE_DIR, archive)
                .await
                .context("restore the snapshot")?;
        }
        Ok(guest)
    }

    fn reserve_slot(&self) -> Result<SlotGuard<'_>> {
        let running = self.sandboxes.lock().map_or(0, |map| map.len()) as u32;
        let mut booting = self
            .booting
            .lock()
            .map_err(|_| anyhow::anyhow!("slot counter poisoned"))?;
        if running + *booting >= self.settings.max_sandboxes {
            bail!(
                "this node is running its limit of {} sandboxes",
                self.settings.max_sandboxes
            );
        }
        *booting += 1;
        Ok(SlotGuard(&self.booting))
    }

    pub async fn exec(
        &self,
        caller: &Caller,
        id: &str,
        mut request: ExecRequest,
    ) -> Result<ExecOutput> {
        let sandbox = self.get(caller, id)?;
        for (name, value) in &sandbox.env {
            request
                .env
                .entry(name.clone())
                .or_insert_with(|| value.clone());
        }
        GuestClient::new(sandbox.vm.as_ref()).exec(request).await
    }

    pub async fn read_file(
        &self,
        caller: &Caller,
        id: &str,
        path: &str,
        max_bytes: u64,
    ) -> Result<(Vec<u8>, u64, bool)> {
        let sandbox = self.get(caller, id)?;
        GuestClient::new(sandbox.vm.as_ref())
            .read_file(path, max_bytes)
            .await
    }

    pub async fn write_file(
        &self,
        caller: &Caller,
        id: &str,
        path: &str,
        data: &[u8],
        mode: Option<u32>,
        append: bool,
    ) -> Result<()> {
        let sandbox = self.get(caller, id)?;
        GuestClient::new(sandbox.vm.as_ref())
            .write_file(path, data, mode, append)
            .await
    }

    pub async fn list_dir(&self, caller: &Caller, id: &str, path: &str) -> Result<Vec<DirEntry>> {
        let sandbox = self.get(caller, id)?;
        GuestClient::new(sandbox.vm.as_ref()).list_dir(path).await
    }

    /// Archives `/workspace` and keeps it on this node.
    pub async fn snapshot(&self, caller: &Caller, id: &str) -> Result<SnapshotInfo> {
        let sandbox = self.get(caller, id)?;
        let archive = GuestClient::new(sandbox.vm.as_ref())
            .archive_out(WORKSPACE_DIR)
            .await?;
        let snapshot_id = format!("snap-{}", uuid::Uuid::new_v4().simple());
        let path = self.snapshot_path(&snapshot_id)?;
        tokio::fs::write(&path, &archive).await?;
        Ok(SnapshotInfo {
            snapshot_id,
            node: "local".into(),
            bytes: archive.len() as u64,
        })
    }

    /// A slice of a stored snapshot. Snapshot ids are unguessable; holding
    /// one is what entitles a node to read it.
    pub async fn snapshot_chunk(
        &self,
        snapshot_id: &str,
        offset: u64,
        len: u64,
    ) -> Result<(Vec<u8>, u64)> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let path = self.snapshot_path(snapshot_id)?;
        let mut file = tokio::fs::File::open(&path)
            .await
            .with_context(|| format!("snapshot {snapshot_id} is not on this node"))?;
        let total = file.metadata().await?.len();
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        let mut data = Vec::new();
        file.take(len).read_to_end(&mut data).await?;
        Ok((data, total))
    }

    fn snapshot_path(&self, snapshot_id: &str) -> Result<PathBuf> {
        let valid = snapshot_id
            .strip_prefix("snap-")
            .is_some_and(|hex| hex.len() == 32 && hex.chars().all(|c| c.is_ascii_hexdigit()));
        if !valid {
            bail!("invalid snapshot id `{snapshot_id}`");
        }
        Ok(self
            .settings
            .state_dir()
            .join(SNAPSHOT_DIR)
            .join(format!("{snapshot_id}.tar.gz")))
    }

    pub fn egress_log(&self, caller: &Caller, id: &str) -> Result<Vec<EgressEvent>> {
        Ok(self.get(caller, id)?.egress_log.snapshot())
    }

    pub fn list(&self, caller: &Caller) -> Vec<SandboxInfo> {
        let mut list: Vec<SandboxInfo> = self
            .sandboxes
            .lock()
            .map(|map| {
                map.values()
                    .filter(|sandbox| visible_to(sandbox, caller))
                    .map(|sandbox| sandbox.info.clone())
                    .collect()
            })
            .unwrap_or_default();
        list.sort_by_key(|info| info.created_ms);
        list
    }

    pub async fn destroy(&self, caller: &Caller, id: &str) -> Result<()> {
        self.get(caller, id)?;
        let sandbox = self
            .sandboxes
            .lock()
            .map_err(|_| anyhow::anyhow!("sandbox table poisoned"))?
            .remove(id);
        if let Some(sandbox) = sandbox {
            teardown(&sandbox).await;
        }
        Ok(())
    }

    /// Destroys sandboxes past their TTL and drops old snapshots.
    pub async fn reap(&self) {
        let now = crate::now_ms();
        let expired: Vec<Arc<Sandbox>> = self
            .sandboxes
            .lock()
            .map(|mut map| {
                let ids: Vec<String> = map
                    .iter()
                    .filter(|(_, sandbox)| sandbox.info.expires_ms <= now)
                    .map(|(id, _)| id.clone())
                    .collect();
                ids.iter().filter_map(|id| map.remove(id)).collect()
            })
            .unwrap_or_default();
        for sandbox in expired {
            teardown(&sandbox).await;
        }
        self.prune_snapshots();
    }

    fn prune_snapshots(&self) {
        let Ok(entries) = std::fs::read_dir(self.settings.state_dir().join(SNAPSHOT_DIR)) else {
            return;
        };
        for entry in entries.flatten() {
            let old = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > SNAPSHOT_MAX_AGE);
            if old {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    fn get(&self, caller: &Caller, id: &str) -> Result<Arc<Sandbox>> {
        let sandbox = self
            .sandboxes
            .lock()
            .map_err(|_| anyhow::anyhow!("sandbox table poisoned"))?
            .get(id)
            .cloned();
        match sandbox {
            Some(sandbox) if visible_to(&sandbox, caller) => Ok(sandbox),
            _ => bail!("no sandbox `{id}` on this node"),
        }
    }
}

fn visible_to(sandbox: &Sandbox, caller: &Caller) -> bool {
    *caller == Caller::Local || sandbox.owner == *caller
}

async fn teardown(sandbox: &Sandbox) {
    GuestClient::new(sandbox.vm.as_ref()).shutdown().await;
    sandbox.vm.stop().await;
    let _ = std::fs::remove_dir_all(&sandbox.dir);
}

fn console_tail(vm: &dyn Vm) -> String {
    let Some(path) = vm.console_log() else {
        return "the sandbox failed to boot".into();
    };
    let Ok(console) = std::fs::read(&path) else {
        return "the sandbox failed to boot".into();
    };
    let start = console.len().saturating_sub(CONSOLE_TAIL_BYTES);
    format!(
        "the sandbox failed to boot; guest console:\n{}",
        String::from_utf8_lossy(&console[start..])
    )
}

struct SlotGuard<'a>(&'a Mutex<u32>);

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut booting) = self.0.lock() {
            *booting = booting.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::EgressMode;
    use crate::images::{KERNEL_FILE, ROOTFS_FILE};
    use crate::ops::{Op, OpResult, run};
    use crate::vm::fake::FakeBackend;

    fn manager(dir: &std::path::Path, settings: Settings) -> Manager {
        let image = dir.join("images").join("base").join(guest_arch());
        std::fs::create_dir_all(&image).unwrap();
        std::fs::write(image.join(KERNEL_FILE), b"k").unwrap();
        std::fs::write(image.join(ROOTFS_FILE), b"r").unwrap();
        let settings = Settings {
            state_dir: Some(dir.to_path_buf()),
            images_dir: Some(dir.join("images")),
            ..settings
        };
        Manager::new(settings, Ok(Arc::new(FakeBackend::default()))).unwrap()
    }

    fn peer(name: &str) -> Caller {
        Caller::Peer(name.into())
    }

    #[tokio::test]
    async fn a_sandbox_runs_commands_and_keeps_files() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(dir.path(), Settings::default());
        let info = manager
            .create(Caller::Local, SandboxSpec::default(), None)
            .await
            .unwrap();
        let output = manager
            .exec(
                &Caller::Local,
                &info.sandbox_id,
                ExecRequest {
                    shell: Some("echo hi".into()),
                    ..ExecRequest::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(output.stdout, b"hi\n");
        assert_eq!(output.status.code, Some(0));

        manager
            .write_file(
                &Caller::Local,
                &info.sandbox_id,
                "/workspace/a",
                b"one",
                None,
                false,
            )
            .await
            .unwrap();
        manager
            .write_file(
                &Caller::Local,
                &info.sandbox_id,
                "/workspace/a",
                b"two",
                None,
                true,
            )
            .await
            .unwrap();
        let (data, size, truncated) = manager
            .read_file(&Caller::Local, &info.sandbox_id, "/workspace/a", 4)
            .await
            .unwrap();
        assert_eq!((data.as_slice(), size, truncated), (&b"onet"[..], 6, true));
    }

    #[tokio::test]
    async fn peers_only_reach_their_own_sandboxes_and_the_operator_sees_all() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(dir.path(), Settings::default());
        let alice = manager
            .create(peer("alice"), SandboxSpec::default(), None)
            .await
            .unwrap();
        assert_eq!(alice.owner, "alice");
        let bob = peer("bob");
        assert!(manager.list(&bob).is_empty());
        assert!(
            manager
                .list_dir(&bob, &alice.sandbox_id, "/workspace")
                .await
                .is_err()
        );
        assert!(manager.destroy(&bob, &alice.sandbox_id).await.is_err());
        assert_eq!(manager.list(&Caller::Local).len(), 1);
        manager
            .destroy(&Caller::Local, &alice.sandbox_id)
            .await
            .unwrap();
        assert!(manager.list(&peer("alice")).is_empty());
    }

    #[tokio::test]
    async fn limits_refuse_rather_than_shrink() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(
            dir.path(),
            Settings {
                max_sandboxes: 1,
                accept_remote: false,
                max_egress: EgressMode::Allowlist,
                ..Settings::default()
            },
        );
        let mut open = SandboxSpec::default();
        open.egress.mode = EgressMode::Open;
        assert!(manager.create(Caller::Local, open, None).await.is_err());
        assert!(
            manager
                .create(peer("p"), SandboxSpec::default(), None)
                .await
                .is_err()
        );
        manager
            .create(Caller::Local, SandboxSpec::default(), None)
            .await
            .unwrap();
        let error = manager
            .create(Caller::Local, SandboxSpec::default(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("limit of 1"), "{error}");
        assert_eq!(manager.capacity().free_slots, 0);
    }

    #[tokio::test]
    async fn snapshots_restore_into_a_new_sandbox_through_the_op_layer() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(dir.path(), Settings::default());
        let caller = peer("alice");
        let OpResult::Sandbox(first) = run(
            &manager,
            &caller,
            Op::Create {
                spec: SandboxSpec::default(),
            },
            None,
        )
        .await
        .unwrap() else {
            panic!("expected a sandbox");
        };
        run(
            &manager,
            &caller,
            Op::WriteFile {
                sandbox_id: first.sandbox_id.clone(),
                path: "/workspace/notes.md".into(),
                data: b"remember".to_vec(),
                mode: None,
                append: false,
            },
            None,
        )
        .await
        .unwrap();
        let snapshot = manager.snapshot(&caller, &first.sandbox_id).await.unwrap();
        let OpResult::SnapshotChunk { data, total } = run(
            &manager,
            &peer("someone-else"),
            Op::SnapshotChunk {
                snapshot_id: snapshot.snapshot_id.clone(),
                offset: 0,
                len: 1 << 20,
            },
            None,
        )
        .await
        .unwrap() else {
            panic!("expected a chunk");
        };
        assert_eq!(total, data.len() as u64);
        let second = manager
            .create(caller.clone(), SandboxSpec::default(), Some(data))
            .await
            .unwrap();
        let (restored, _, _) = manager
            .read_file(&caller, &second.sandbox_id, "/workspace/notes.md", 64)
            .await
            .unwrap();
        assert_eq!(restored, b"remember");
        assert!(
            manager
                .snapshot_chunk("snap-../../etc", 0, 1)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn expired_sandboxes_are_reaped() {
        let dir = tempfile::tempdir().unwrap();
        let manager = manager(dir.path(), Settings::default());
        let info = manager
            .create(Caller::Local, SandboxSpec::default(), None)
            .await
            .unwrap();
        if let Ok(mut map) = manager.sandboxes.lock()
            && let Some(sandbox) = map.get_mut(&info.sandbox_id)
        {
            let mut expired = (**sandbox).info.clone();
            expired.expires_ms = 0;
            *sandbox = Arc::new(Sandbox {
                info: expired,
                owner: sandbox.owner.clone(),
                env: sandbox.env.clone(),
                vm: Box::new(NoVm),
                egress_log: sandbox.egress_log.clone(),
                dir: sandbox.dir.clone(),
            });
        }
        manager.reap().await;
        assert!(manager.list(&Caller::Local).is_empty());
    }

    struct NoVm;

    #[async_trait::async_trait]
    impl Vm for NoVm {
        async fn connect_agent(&self) -> Result<crate::vm::BoxStream> {
            bail!("stopped")
        }

        async fn stop(&self) {}

        fn console_log(&self) -> Option<PathBuf> {
            None
        }
    }
}
