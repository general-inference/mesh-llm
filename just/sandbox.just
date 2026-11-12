# ── Sandbox plugin ─────────────────────────────────────────────

# Build the sandbox plugin (`mesh-sandbox`); on macOS it is signed with the virtualization entitlement.
sandbox-plugin profile="debug":
    cargo build {{ if profile == "release" { "--release" } else { "" } }} --locked -p mesh-llm-sandbox-plugin
    {{ if os() == "macos" { "just sandbox-sign target/" + profile + "/mesh-sandbox" } else { "" } }}

# Build the sandbox guest image (kernel + root filesystem) for this machine's architecture. Needs Docker.
[unix]
sandbox-image out=(home_dir / ".mesh-llm" / "sandbox" / "images"):
    mesh/crates/mesh-llm-sandbox-guest/image/build.sh "{{ out }}"

# Sign a macOS build of the sandbox plugin so it may use Virtualization.framework.
[macos]
sandbox-sign binary="target/debug/mesh-sandbox":
    codesign --force --sign "${MESH_SANDBOX_SIGNING_IDENTITY:--}" --entitlements mesh/crates/mesh-llm-sandbox-plugin/entitlements.plist "{{ binary }}"
