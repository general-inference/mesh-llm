//! `mesh-compute`: the mesh-llm compute plugin.
//!
//! Launched by mesh-llm with no arguments, it serves the plugin protocol. The
//! Linux build also has a `wsl-bridge` mode, which the Windows build starts
//! inside WSL2 to run Firecracker there. `mesh-compute check` boots a sandbox
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

use std::io::Write as _;
use std::sync::Arc;

pub const PLUGIN_ID: &str = "compute";

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
        Some("check") => {
            let flag = |name: &str| args.iter().any(|arg| arg == name);
            check::run(flag("--offline"), flag("--bridge")).await
        }
        Some("--print-package-manifest") => {
            use mesh_llm_plugin::Plugin;
            let plugin = plugin::build(Arc::new(plugin::App::default()));
            let manifest = plugin
                .manifest()
                .ok_or_else(|| anyhow::anyhow!("the plugin has no manifest"))?;
            writeln!(
                mesh_llm_events::machine_out(),
                "{}",
                mesh_llm_plugin::package_manifest_json(&manifest)?
            )?;
            Ok(())
        }
        Some("--version") => {
            writeln!(
                mesh_llm_events::machine_out(),
                "mesh-compute {}",
                env!("CARGO_PKG_VERSION")
            )?;
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown argument `{other}`; mesh-llm launches this plugin"),
    }
}
