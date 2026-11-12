//! Sandboxes across the mesh: capacity adverts, placement, and running
//! operations on the node that holds a sandbox.
//!
//! Everything travels on the `compute.v1` plugin channel. Each node
//! broadcasts its [`Capacity`] as an `advert`. A caller picks a node, sends it
//! a `request` carrying an [`Op`], and waits for the `response` with the same
//! correlation id. The host authenticates the sending peer, and a node runs a
//! peer's request as that peer, so peers only reach their own sandboxes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::PluginContext;
use mesh_llm_plugin::proto;
use tokio::sync::oneshot;

use crate::manager::{Caller, Capacity, Manager, SandboxInfo};
use crate::ops::{self, MAX_TRANSFER_BYTES, Op, OpResult, Reply};
use crate::spec::{SandboxSpec, SnapshotRef};

pub const CHANNEL: &str = "compute.v1";
const KIND_ADVERT: &str = "advert";
const KIND_REQUEST: &str = "request";
const KIND_RESPONSE: &str = "response";
pub const ADVERT_INTERVAL: Duration = Duration::from_secs(20);
const ADVERT_STALE: Duration = Duration::from_secs(90);
const LOCAL: &str = "local";
/// Largest snapshot a node will pull from another.
const MAX_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Local,
    Peer(String),
}

#[derive(Default)]
struct Peer {
    capacity: Option<(Capacity, Instant)>,
    serving_models: Vec<String>,
    rtt_ms: Option<u32>,
}

impl Peer {
    fn fresh_capacity(&self) -> Option<&Capacity> {
        self.capacity
            .as_ref()
            .filter(|(_, at)| at.elapsed() < ADVERT_STALE)
            .map(|(capacity, _)| capacity)
    }
}

#[derive(Default)]
struct State {
    local_id: Option<String>,
    peers: HashMap<String, Peer>,
    /// Which node holds each sandbox this node created.
    routes: HashMap<String, Node>,
}

/// One node as `sandbox_nodes` reports it.
#[derive(Clone, Debug, serde::Serialize, schemars::JsonSchema)]
pub struct NodeView {
    pub node: String,
    pub local: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtt_ms: Option<u32>,
    pub serving_models: Vec<String>,
    pub capacity: Capacity,
}

pub struct Mesh {
    manager: Arc<Manager>,
    state: Mutex<State>,
    pending: Mutex<HashMap<String, oneshot::Sender<Reply>>>,
    context: Mutex<Option<PluginContext<'static>>>,
}

impl Mesh {
    pub fn new(manager: Arc<Manager>) -> Self {
        Self {
            manager,
            state: Mutex::new(State::default()),
            pending: Mutex::new(HashMap::new()),
            context: Mutex::new(None),
        }
    }

    pub fn manager(&self) -> &Arc<Manager> {
        &self.manager
    }

    /// Keeps a handle to the host connection for sending from background
    /// tasks.
    pub fn attach(&self, context: &PluginContext<'_>) {
        if let Ok(mut slot) = self.context.lock() {
            *slot = Some(context.owned());
        }
    }

    fn sender(&self) -> Result<PluginContext<'static>> {
        self.context
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(PluginContext::owned))
            .context("the plugin is not connected to its host yet")
    }

    /// Runs an operation from this node's own tools, wherever it belongs.
    pub async fn dispatch(&self, op: Op) -> Result<OpResult> {
        match op {
            Op::Create { spec } => self.create(spec).await,
            Op::List => Ok(OpResult::List {
                sandboxes: self.list_everywhere().await,
            }),
            op => {
                let sandbox_id = op
                    .sandbox_id()
                    .context("this operation is not addressed to a sandbox")?
                    .to_string();
                let node = self.route(&sandbox_id)?;
                let destroying = matches!(op, Op::Destroy { .. });
                let result = self.run_on(&node, op).await?;
                if destroying && let Ok(mut state) = self.state.lock() {
                    state.routes.remove(&sandbox_id);
                }
                Ok(result)
            }
        }
    }

    async fn create(&self, mut spec: SandboxSpec) -> Result<OpResult> {
        let node = self.place(&spec)?;
        if let Some(snapshot) = spec.from_snapshot.as_mut()
            && snapshot.node == LOCAL
        {
            snapshot.node = self.local_label();
        }
        let result = self.run_on(&node, Op::Create { spec }).await?;
        let OpResult::Sandbox(mut info) = result else {
            bail!("unexpected reply to create");
        };
        info.node = self.node_label(&node);
        if let Ok(mut state) = self.state.lock() {
            state.routes.insert(info.sandbox_id.clone(), node);
        }
        Ok(OpResult::Sandbox(info))
    }

    async fn run_on(&self, node: &Node, op: Op) -> Result<OpResult> {
        match node {
            Node::Local => self.run_here(&Caller::Local, op).await,
            Node::Peer(peer) => self.call(peer, op).await,
        }
    }

    /// Runs `op` on this node, fetching a create's snapshot first.
    async fn run_here(&self, caller: &Caller, op: Op) -> Result<OpResult> {
        let restore = match &op {
            Op::Create { spec } => match &spec.from_snapshot {
                Some(snapshot) => Some(self.fetch_snapshot(snapshot).await?),
                None => None,
            },
            _ => None,
        };
        let mut result = ops::run(&self.manager, caller, op, restore).await?;
        if let OpResult::Sandbox(info) = &mut result {
            info.node = self.local_label();
        }
        if let OpResult::Snapshot(snapshot) = &mut result {
            snapshot.node = self.local_label();
        }
        Ok(result)
    }

    async fn call(&self, peer: &str, op: Op) -> Result<OpResult> {
        let correlation_id = uuid::Uuid::new_v4().simple().to_string();
        let timeout = op.timeout();
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .map_err(|_| anyhow::anyhow!("pending table poisoned"))?
            .insert(correlation_id.clone(), tx);
        let sent = self.send(peer, KIND_REQUEST, &correlation_id, &op).await;
        if let Err(error) = sent {
            self.forget(&correlation_id);
            return Err(error);
        }
        let reply = tokio::time::timeout(timeout, rx).await;
        self.forget(&correlation_id);
        match reply {
            Ok(Ok(Reply::Ok { result })) => Ok(result),
            Ok(Ok(Reply::Error { message })) => {
                Err(anyhow::anyhow!(message)).with_context(|| format!("on node {}", short(peer)))
            }
            Ok(Err(_)) => bail!("lost the request to node {}", short(peer)),
            Err(_) => bail!(
                "node {} did not answer within {}s",
                short(peer),
                timeout.as_secs()
            ),
        }
    }

    fn forget(&self, correlation_id: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(correlation_id);
        }
    }

    async fn send<T: serde::Serialize>(
        &self,
        target: &str,
        kind: &str,
        correlation_id: &str,
        body: &T,
    ) -> Result<()> {
        let mut context = self.sender()?;
        context
            .send_channel_message(proto::ChannelMessage {
                channel: CHANNEL.to_string(),
                target_peer_id: target.to_string(),
                content_type: "application/json".to_string(),
                body: serde_json::to_vec(body)?,
                message_kind: kind.to_string(),
                correlation_id: correlation_id.to_string(),
                ..proto::ChannelMessage::default()
            })
            .await
    }

    /// Sends this node's capacity to `target`, or to every peer when empty.
    pub async fn advertise(&self, target: &str) {
        let _ = self
            .send(target, KIND_ADVERT, "", &self.manager.capacity())
            .await;
    }

    /// Handles a `compute.v1` message. Requests run in their own task so a
    /// long command never holds up the plugin's ordered message delivery.
    pub fn on_message(self: &Arc<Self>, message: proto::ChannelMessage) {
        let source = message.source_peer_id.clone();
        if source.is_empty() || source.starts_with("plugin:") {
            return;
        }
        match message.message_kind.as_str() {
            KIND_ADVERT => {
                if let Ok(capacity) = serde_json::from_slice::<Capacity>(&message.body)
                    && let Ok(mut state) = self.state.lock()
                {
                    state.peers.entry(source).or_default().capacity =
                        Some((capacity, Instant::now()));
                }
            }
            KIND_RESPONSE => {
                let sender = self
                    .pending
                    .lock()
                    .ok()
                    .and_then(|mut pending| pending.remove(&message.correlation_id));
                if let (Some(sender), Ok(reply)) =
                    (sender, serde_json::from_slice::<Reply>(&message.body))
                {
                    let _ = sender.send(reply);
                }
            }
            KIND_REQUEST => {
                let mesh = self.clone();
                tokio::spawn(async move { mesh.serve_request(message).await });
            }
            _ => {}
        }
    }

    async fn serve_request(&self, message: proto::ChannelMessage) {
        let caller = Caller::Peer(message.source_peer_id.clone());
        let reply = match serde_json::from_slice::<Op>(&message.body) {
            Ok(op) => match self.run_here(&caller, op).await {
                Ok(result) => Reply::Ok { result },
                Err(error) => Reply::Error {
                    message: format!("{error:#}"),
                },
            },
            Err(error) => Reply::Error {
                message: format!("malformed request: {error}"),
            },
        };
        let _ = self
            .send(
                &message.source_peer_id,
                KIND_RESPONSE,
                &message.correlation_id,
                &reply,
            )
            .await;
    }

    pub fn on_event(&self, event: &proto::MeshEvent) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !event.local_peer_id.is_empty() {
            state.local_id = Some(event.local_peer_id.clone());
        }
        let Some(peer) = &event.peer else {
            return;
        };
        if event.kind == proto::mesh_event::Kind::PeerDown as i32 {
            state.peers.remove(&peer.peer_id);
            state
                .routes
                .retain(|_, node| *node != Node::Peer(peer.peer_id.clone()));
            return;
        }
        let entry = state.peers.entry(peer.peer_id.clone()).or_default();
        entry.rtt_ms = peer.rtt_ms;
        entry.serving_models = peer
            .serving_models
            .iter()
            .chain(&peer.hosted_models)
            .cloned()
            .collect();
    }

    /// Picks the node for a new sandbox.
    pub fn place(&self, spec: &SandboxSpec) -> Result<Node> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh state poisoned"))?;
        choose(&state, &self.manager.capacity(), spec)
    }

    fn route(&self, sandbox_id: &str) -> Result<Node> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh state poisoned"))?;
        if let Some(node) = state.routes.get(sandbox_id) {
            return Ok(node.clone());
        }
        drop(state);
        if self
            .manager
            .list(&Caller::Local)
            .iter()
            .any(|info| info.sandbox_id == sandbox_id)
        {
            return Ok(Node::Local);
        }
        bail!("unknown sandbox `{sandbox_id}`; `sandbox_list` finds sandboxes this node created")
    }

    /// This node's sandboxes plus the ones it created on peers.
    async fn list_everywhere(&self) -> Vec<SandboxInfo> {
        let mut sandboxes: Vec<SandboxInfo> = self
            .manager
            .list(&Caller::Local)
            .into_iter()
            .map(|mut info| {
                info.node = self.local_label();
                info
            })
            .collect();
        let peers: Vec<String> = self.state.lock().map_or_else(
            |_| Vec::new(),
            |state| {
                state
                    .peers
                    .iter()
                    .filter(|(_, peer)| peer.fresh_capacity().is_some())
                    .map(|(id, _)| id.clone())
                    .collect()
            },
        );
        for peer in peers {
            if let Ok(OpResult::List { sandboxes: remote }) = self.call(&peer, Op::List).await {
                for mut info in remote {
                    info.node = peer.clone();
                    if let Ok(mut state) = self.state.lock() {
                        state
                            .routes
                            .insert(info.sandbox_id.clone(), Node::Peer(peer.clone()));
                    }
                    sandboxes.push(info);
                }
            }
        }
        sandboxes
    }

    pub fn nodes(&self) -> Vec<NodeView> {
        let mut nodes = vec![NodeView {
            node: self.local_label(),
            local: true,
            rtt_ms: None,
            serving_models: Vec::new(),
            capacity: self.manager.capacity(),
        }];
        if let Ok(state) = self.state.lock() {
            for (id, peer) in &state.peers {
                if let Some(capacity) = peer.fresh_capacity() {
                    nodes.push(NodeView {
                        node: id.clone(),
                        local: false,
                        rtt_ms: peer.rtt_ms,
                        serving_models: peer.serving_models.clone(),
                        capacity: capacity.clone(),
                    });
                }
            }
        }
        nodes
    }

    async fn fetch_snapshot(&self, snapshot: &SnapshotRef) -> Result<Vec<u8>> {
        let node = if snapshot.node == LOCAL || Some(&snapshot.node) == self.local_id().as_ref() {
            Node::Local
        } else {
            Node::Peer(snapshot.node.clone())
        };
        let mut archive = Vec::new();
        loop {
            let op = Op::SnapshotChunk {
                snapshot_id: snapshot.snapshot_id.clone(),
                offset: archive.len() as u64,
                len: MAX_TRANSFER_BYTES,
            };
            // Boxed: fetching runs inside `run_here`, which `run_on` calls.
            let OpResult::SnapshotChunk { data, total } = Box::pin(self.run_on(&node, op)).await?
            else {
                bail!("unexpected reply while fetching a snapshot");
            };
            if total > MAX_SNAPSHOT_BYTES {
                bail!("snapshot is {total} bytes; the limit is {MAX_SNAPSHOT_BYTES}");
            }
            let done = data.is_empty();
            archive.extend_from_slice(&data);
            if done || archive.len() as u64 >= total {
                return Ok(archive);
            }
        }
    }

    fn local_id(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.local_id.clone())
    }

    fn local_label(&self) -> String {
        self.local_id().unwrap_or_else(|| LOCAL.to_string())
    }

    fn node_label(&self, node: &Node) -> String {
        match node {
            Node::Local => self.local_label(),
            Node::Peer(peer) => peer.clone(),
        }
    }
}

/// Placement over a snapshot of what this node knows.
fn choose(state: &State, local: &Capacity, spec: &SandboxSpec) -> Result<Node> {
    if let Some(wanted) = &spec.node {
        return place_explicit(state, local, spec, wanted);
    }
    let mut candidates: Vec<(Node, Score)> = Vec::new();
    if fits(local, spec, false).is_ok() {
        candidates.push((Node::Local, Score::new(spec, local, None, true)));
    }
    for (id, peer) in &state.peers {
        if let Some(capacity) = peer.fresh_capacity()
            && fits(capacity, spec, true).is_ok()
        {
            let score = Score::new(spec, capacity, Some(peer), false).with_snapshot(spec, id);
            candidates.push((Node::Peer(id.clone()), score));
        }
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.1));
    candidates
        .into_iter()
        .next()
        .map(|(node, _)| node)
        .with_context(|| {
            let reason = fits(local, spec, false)
                .err()
                .map_or_else(String::new, |error| format!(" (this node: {error})"));
            format!("no node in the mesh can run this sandbox{reason}")
        })
}

fn place_explicit(
    state: &State,
    local: &Capacity,
    spec: &SandboxSpec,
    wanted: &str,
) -> Result<Node> {
    if wanted == LOCAL || state.local_id.as_deref() == Some(wanted) {
        fits(local, spec, false).context("this node cannot run it")?;
        return Ok(Node::Local);
    }
    let matches: Vec<&String> = state
        .peers
        .keys()
        .filter(|id| id.starts_with(wanted))
        .collect();
    let [id] = matches.as_slice() else {
        bail!(
            "node `{wanted}` matches {} known peers; give more of its id",
            matches.len()
        );
    };
    let capacity = state.peers[*id]
        .fresh_capacity()
        .with_context(|| format!("node {} has not advertised sandbox support", short(id)))?;
    fits(capacity, spec, true).with_context(|| format!("node {} cannot run it", short(id)))?;
    Ok(Node::Peer((*id).clone()))
}

/// Whether a node with `capacity` can take `spec`, and if not, why.
fn fits(capacity: &Capacity, spec: &SandboxSpec, remote: bool) -> Result<()> {
    if let Some(reason) = &capacity.unavailable {
        bail!("{reason}");
    }
    if remote && !capacity.accept_remote {
        bail!("it does not run sandboxes for other nodes");
    }
    if !capacity.images.contains(&spec.image) {
        bail!("image `{}` is not installed", spec.image);
    }
    if capacity.free_slots == 0 {
        bail!("it is running its sandbox limit");
    }
    let resources = &spec.resources;
    if resources.cpus > capacity.max_cpus
        || resources.memory_mb > capacity.max_memory_mb
        || resources.disk_gb > capacity.max_disk_gb
    {
        bail!("the requested resources exceed its limits");
    }
    let max_egress: crate::egress::EgressMode =
        serde_json::from_value(serde_json::Value::String(capacity.max_egress.clone()))
            .unwrap_or_default();
    if spec.egress.mode > max_egress {
        bail!("it allows egress mode `{}` at most", capacity.max_egress);
    }
    Ok(())
}

/// Placement preference, compared field by field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    near_model: bool,
    holds_snapshot: bool,
    local: bool,
    /// Lower round-trip time is better; unknown sorts last.
    closeness: u32,
    free_slots: u32,
}

impl Score {
    fn new(spec: &SandboxSpec, capacity: &Capacity, peer: Option<&Peer>, local: bool) -> Self {
        let near_model = match (&spec.near_model, peer) {
            (Some(model), Some(peer)) => peer
                .serving_models
                .iter()
                .any(|serving| serving.eq_ignore_ascii_case(model)),
            _ => false,
        };
        let rtt = if local {
            Some(0)
        } else {
            peer.and_then(|peer| peer.rtt_ms)
        };
        Self {
            near_model,
            holds_snapshot: local
                && spec
                    .from_snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.node == LOCAL),
            local,
            closeness: u32::MAX - rtt.unwrap_or(u32::MAX),
            free_slots: capacity.free_slots,
        }
    }

    fn with_snapshot(mut self, spec: &SandboxSpec, peer_id: &str) -> Self {
        self.holds_snapshot = spec
            .from_snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.node == peer_id);
        self
    }
}

fn short(peer: &str) -> &str {
    &peer[..peer.len().min(12)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::EgressMode;

    fn capacity(images: &[&str], free_slots: u32) -> Capacity {
        Capacity {
            backend: "firecracker".into(),
            unavailable: None,
            arch: "aarch64".into(),
            images: images.iter().map(ToString::to_string).collect(),
            free_slots,
            max_cpus: 8,
            max_memory_mb: 16_384,
            max_disk_gb: 100,
            max_egress: "open".into(),
            accept_remote: true,
        }
    }

    fn peer(capacity: Capacity, rtt_ms: Option<u32>, models: &[&str]) -> Peer {
        Peer {
            capacity: Some((capacity, Instant::now())),
            serving_models: models.iter().map(ToString::to_string).collect(),
            rtt_ms,
        }
    }

    fn pick(state: &State, local: &Capacity, spec: &SandboxSpec) -> Option<Node> {
        choose(state, local, spec).ok()
    }

    fn mesh_state(peers: Vec<(&str, Peer)>) -> State {
        State {
            local_id: Some("me".into()),
            peers: peers
                .into_iter()
                .map(|(id, peer)| (id.to_string(), peer))
                .collect(),
            routes: HashMap::new(),
        }
    }

    #[test]
    fn local_wins_when_it_fits_and_nothing_is_nearer_the_model() {
        let state = mesh_state(vec![("aaaa", peer(capacity(&["base"], 4), Some(1), &[]))]);
        let spec = SandboxSpec::default();
        assert_eq!(
            pick(&state, &capacity(&["base"], 1), &spec),
            Some(Node::Local)
        );
    }

    #[test]
    fn a_peer_serving_the_model_beats_local() {
        let state = mesh_state(vec![
            (
                "aaaa",
                peer(capacity(&["base"], 4), Some(5), &["Qwen3-32B"]),
            ),
            ("bbbb", peer(capacity(&["base"], 4), Some(1), &[])),
        ]);
        let spec = SandboxSpec {
            near_model: Some("qwen3-32b".into()),
            ..SandboxSpec::default()
        };
        assert_eq!(
            pick(&state, &capacity(&["base"], 4), &spec),
            Some(Node::Peer("aaaa".into()))
        );
    }

    #[test]
    fn full_or_imageless_nodes_are_skipped_and_closer_peers_preferred() {
        let state = mesh_state(vec![
            ("full", peer(capacity(&["base"], 0), Some(1), &[])),
            ("far", peer(capacity(&["base"], 4), Some(80), &[])),
            ("near", peer(capacity(&["base"], 1), Some(3), &[])),
            ("noimage", peer(capacity(&["other"], 4), Some(1), &[])),
        ]);
        let spec = SandboxSpec::default();
        assert_eq!(
            pick(&state, &capacity(&[], 4), &spec),
            Some(Node::Peer("near".into()))
        );
    }

    #[test]
    fn egress_beyond_a_nodes_ceiling_rules_it_out() {
        let mut strict = capacity(&["base"], 4);
        strict.max_egress = "allowlist".into();
        let state = mesh_state(vec![("strict", peer(strict, Some(1), &[]))]);
        let mut spec = SandboxSpec::default();
        spec.egress.mode = EgressMode::Open;
        assert_eq!(pick(&state, &capacity(&[], 4), &spec), None);
    }

    #[test]
    fn an_explicit_node_is_honoured_or_explained() {
        let state = mesh_state(vec![
            ("abcd1234", peer(capacity(&["base"], 4), None, &[])),
            ("abce9999", peer(capacity(&["base"], 4), None, &[])),
        ]);
        let local = capacity(&["base"], 4);
        let mut spec = SandboxSpec {
            node: Some("abcd".into()),
            ..SandboxSpec::default()
        };
        assert_eq!(
            pick(&state, &local, &spec),
            Some(Node::Peer("abcd1234".into()))
        );
        spec.node = Some("abc".into());
        assert!(place_explicit(&state, &local, &spec, "abc").is_err());
        spec.node = Some("local".into());
        assert_eq!(pick(&state, &local, &spec), Some(Node::Local));
    }

    #[test]
    fn the_node_holding_a_snapshot_is_preferred() {
        let state = mesh_state(vec![
            ("holder", peer(capacity(&["base"], 1), Some(40), &[])),
            ("other", peer(capacity(&["base"], 4), Some(1), &[])),
        ]);
        let spec = SandboxSpec {
            from_snapshot: Some(SnapshotRef {
                snapshot_id: "snap-x".into(),
                node: "holder".into(),
            }),
            ..SandboxSpec::default()
        };
        assert_eq!(
            pick(&state, &capacity(&[], 4), &spec),
            Some(Node::Peer("holder".into()))
        );
    }
}
