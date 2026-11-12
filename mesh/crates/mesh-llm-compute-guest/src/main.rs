//! `mesh-sandbox-guest`: the agent that runs as PID 1 inside a sandbox microVM.

fn main() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    {
        mesh_llm_sandbox_guest::agent::main()
    }
    #[cfg(not(target_os = "linux"))]
    {
        anyhow::bail!("mesh-sandbox-guest runs only inside a Linux sandbox guest")
    }
}
