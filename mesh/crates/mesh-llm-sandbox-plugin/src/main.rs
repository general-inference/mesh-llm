//! `mesh-sandbox`: the mesh-llm sandbox plugin.
//!
//! Launched by mesh-llm with no arguments, it serves the plugin protocol. The
//! Linux build also has a `wsl-bridge` mode, which the Windows build starts
//! inside WSL2 to run Firecracker there. `mesh-sandbox check` boots a sandbox
//! on this machine and exercises it.

mod check;
mod egress;
mod guest;
mod images;
mod manager;
mod mesh;
mod ops;
mod plugin;
mod settings;
mod spec;
mod tools;
mod vm;

use std::sync::Arc;

pub const PLUGIN_ID: &str = "sandbox";

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => {
            let app = Arc::new(plugin::App::default());
            mesh_llm_plugin::PluginRuntime::run(plugin::build(app)).await
        }
        #[cfg(target_os = "linux")]
        Some("wsl-bridge") => vm::bridge::serve_stdio().await,
        Some("check") => check::run(args.iter().any(|arg| arg == "--offline")).await,
        Some("--version") => {
            println!("mesh-sandbox {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown argument `{other}`; mesh-llm launches this plugin"),
    }
}
