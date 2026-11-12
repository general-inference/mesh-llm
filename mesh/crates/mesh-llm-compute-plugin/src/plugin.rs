//! The plugin's manifest and lifecycle hooks.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use mesh_llm_plugin::mcp::tool;
use mesh_llm_plugin::{
    DeclarativePluginBuilder, PluginContext, PluginError, PluginInitializeRequest, PluginMetadata,
    PluginResult, PluginStartupPolicy, SimplePlugin, capability, mesh_channel,
    mesh_event_local_accepting, mesh_event_peer_down, mesh_event_peer_up, mesh_event_peer_updated,
    plugin_server_info, proto,
};
use serde::Serialize;

use crate::manager::Manager;
use crate::mesh::{ADVERT_INTERVAL, CHANNEL, Mesh};
use crate::settings::Settings;
use crate::spec::SandboxSpec;
use crate::tools::{
    self, ExecArgs, ListDirArgs, NoArgs, ReadFileArgs, SandboxIdArgs, WriteFileArgs,
};
use crate::{PLUGIN_ID, vm};

const REAP_INTERVAL: Duration = Duration::from_secs(30);

/// Plugin state, built once the host hands over the operator's settings.
#[derive(Default)]
pub struct App {
    mesh: OnceLock<Arc<Mesh>>,
}

impl App {
    fn mesh(&self) -> PluginResult<Arc<Mesh>> {
        self.mesh
            .get()
            .cloned()
            .ok_or_else(|| PluginError::internal("the sandbox plugin is still starting"))
    }

    async fn initialize(&self, request: &PluginInitializeRequest) -> anyhow::Result<()> {
        let settings = Settings::parse(&request.settings_json)?;
        let backend = vm::detect(&settings)
            .await
            .map_err(|error| format!("{error:#}"));
        if let Err(reason) = &backend {
            tracing::warn!("{PLUGIN_ID}: sandboxes are unavailable on this node: {reason}");
        }
        let manager = Arc::new(Manager::new(settings, backend)?);
        let prefetch = manager.clone();
        tokio::spawn(async move { prefetch.prefetch_images().await });
        let reaper = manager.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(REAP_INTERVAL).await;
                reaper.reap().await;
            }
        });
        let _ = self.mesh.set(Arc::new(Mesh::new(manager)));
        Ok(())
    }
}

pub fn build(app: Arc<App>) -> SimplePlugin {
    let version = env!("CARGO_PKG_VERSION");
    let metadata = PluginMetadata::new(
        PLUGIN_ID,
        version,
        plugin_server_info(
            PLUGIN_ID,
            version,
            "Sandboxes",
            "Isolated microVM sandboxes for running code, placed across a private mesh",
            Some(INSTRUCTIONS),
        ),
    );
    let builder = DeclarativePluginBuilder::new(metadata)
        .startup_policy(PluginStartupPolicy::PrivateMeshOnly)
        .provide(capability(CHANNEL))
        .config_item(Settings::config_schema())
        .mesh_item(mesh_channel(CHANNEL))
        .event_item(mesh_event_peer_up())
        .event_item(mesh_event_peer_down())
        .event_item(mesh_event_peer_updated())
        .event_item(mesh_event_local_accepting());
    let builder = register_tools(builder, &app);
    let init_app = app.clone();
    let initialized_app = app.clone();
    let channel_app = app.clone();
    let event_app = app.clone();
    let health_app = app;
    builder
        .customize(move |plugin| {
            plugin
                .on_initialize(move |request, _context| {
                    let app = init_app.clone();
                    Box::pin(async move {
                        app.initialize(&request)
                            .await
                            .map_err(|error| PluginError::internal(format!("{error:#}")))
                    })
                })
                .on_initialized(move |context| {
                    let app = initialized_app.clone();
                    Box::pin(async move {
                        if let Ok(mesh) = app.mesh() {
                            mesh.attach(context);
                            tokio::spawn(advertise_forever(mesh));
                        }
                        Ok(())
                    })
                })
                .on_channel_message(move |message, _context| {
                    let app = channel_app.clone();
                    Box::pin(async move {
                        if message.channel == CHANNEL
                            && let Ok(mesh) = app.mesh()
                        {
                            mesh.on_message(message);
                        }
                        Ok(())
                    })
                })
                .on_mesh_event(move |event, _context| {
                    let app = event_app.clone();
                    Box::pin(async move {
                        if let Ok(mesh) = app.mesh() {
                            on_mesh_event(&mesh, &event).await;
                        }
                        Ok(())
                    })
                })
                .with_health(move |_context| {
                    let app = health_app.clone();
                    Box::pin(async move { Ok(health(&app)) })
                })
        })
        .build()
}

const INSTRUCTIONS: &str = "Create a sandbox with sandbox_create, run commands in it with \
sandbox_exec, move files with sandbox_read_file and sandbox_write_file, and remove it with \
sandbox_destroy. Work in /workspace. The sandbox has no network unless its config sets \
egress (mode none, allowlist or open; allow entries such as @registries, api.example.com:443, \
*.example.com, 10.0.0.0/8). sandbox_snapshot saves /workspace; pass the result as \
from_snapshot to sandbox_create to continue from it.";

fn register_tools(builder: DeclarativePluginBuilder, app: &Arc<App>) -> DeclarativePluginBuilder {
    builder
        .mcp_item(
            tool("sandbox_create")
                .description(
                    "Boot a sandbox microVM somewhere in the mesh. Every field is optional: \
                     image, resources {cpus, memory_mb, disk_gb}, egress {mode, allow, deny}, \
                     ttl_secs, env, node, near_model, from_snapshot. Returns its id and node.",
                )
                .input::<SandboxSpec>()
                .handle(with_mesh(app, |mesh, spec| async move {
                    tools::create(&mesh, spec).await
                })),
        )
        .mcp_item(
            tool("sandbox_exec")
                .description(
                    "Run a command in a sandbox and return its exit code and output. Give \
                     `command` (a shell line) or `argv`.",
                )
                .input::<ExecArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::exec(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_read_file")
                .description("Read a file from a sandbox.")
                .input::<ReadFileArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::read_file(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_write_file")
                .description("Write a file in a sandbox, creating its directories.")
                .input::<WriteFileArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::write_file(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_list_dir")
                .description("List a directory in a sandbox (default /workspace).")
                .input::<ListDirArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::list_dir(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_snapshot")
                .description(
                    "Save a sandbox's /workspace. Pass the result as `from_snapshot` to \
                     sandbox_create to start a new sandbox from it.",
                )
                .input::<SandboxIdArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::snapshot(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_destroy")
                .description("Stop a sandbox and delete its disk.")
                .input::<SandboxIdArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::destroy(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_list")
                .description(
                    "Sandboxes this node created, wherever they run, and any running here.",
                )
                .input::<NoArgs>()
                .handle(with_mesh(app, |mesh, _args: NoArgs| async move {
                    tools::list(&mesh).await
                })),
        )
        .mcp_item(
            tool("sandbox_egress_log")
                .description("Recent network connections a sandbox made or was refused, and why.")
                .input::<SandboxIdArgs>()
                .handle(with_mesh(app, |mesh, args| async move {
                    tools::egress_log(&mesh, args).await
                })),
        )
        .mcp_item(
            tool("sandbox_nodes")
                .description("Nodes in the mesh that can run sandboxes, with their limits.")
                .input::<NoArgs>()
                .handle(with_mesh(app, |mesh, _args: NoArgs| async move {
                    Ok(tools::nodes(&mesh))
                })),
        )
}

/// Adapts an async tool body over the shared [`Mesh`] to a tool handler.
fn with_mesh<A, R, F, Fut>(
    app: &Arc<App>,
    body: F,
) -> impl for<'a, 'ctx> Fn(A, &'a mut PluginContext<'ctx>) -> mesh_llm_plugin::JsonOperationFuture<'a, R>
+ Send
+ Sync
+ 'static
where
    A: Send + 'static,
    R: Serialize + Send + 'static,
    F: Fn(Arc<Mesh>, A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = anyhow::Result<R>> + Send + 'static,
{
    let app = app.clone();
    let body = Arc::new(body);
    move |args, context| {
        let app = app.clone();
        let body = body.clone();
        let mesh = app.mesh();
        if let Ok(mesh) = &mesh {
            mesh.attach(context);
        }
        Box::pin(async move {
            body(mesh?, args)
                .await
                .map_err(|error| PluginError::internal(format!("{error:#}")))
        })
    }
}

async fn on_mesh_event(mesh: &Arc<Mesh>, event: &proto::MeshEvent) {
    mesh.on_event(event);
    if event.kind == proto::mesh_event::Kind::PeerUp as i32
        && let Some(peer) = &event.peer
    {
        mesh.advertise(&peer.peer_id).await;
    }
}

async fn advertise_forever(mesh: Arc<Mesh>) {
    loop {
        mesh.advertise("").await;
        tokio::time::sleep(ADVERT_INTERVAL).await;
    }
}

fn health(app: &App) -> String {
    let Ok(mesh) = app.mesh() else {
        return "starting".into();
    };
    let capacity = mesh.manager().capacity();
    match capacity.unavailable {
        Some(reason) => format!("unavailable: {reason}"),
        None => format!(
            "ok: {} backend, {} free sandbox slots",
            capacity.backend, capacity.free_slots
        ),
    }
}
