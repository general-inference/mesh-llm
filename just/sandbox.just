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

# Package the sandbox plugin for `mesh-llm plugins install --archive`. On Windows, put the Linux build at linux/mesh-sandbox first (see the plugin README).
[unix]
sandbox-package profile="release":
    #!/usr/bin/env bash
    set -euo pipefail
    just sandbox-plugin {{ profile }}
    version="$(cargo pkgid -p mesh-llm-sandbox-plugin | sed 's/.*[#@]//')"
    target="$(rustc -vV | sed -n 's/^host: //p')"
    stage="$(mktemp -d)"
    trap 'rm -rf "$stage"' EXIT
    mkdir -p "$stage/sandbox" target/sandbox-plugin
    cp "target/{{ profile }}/mesh-sandbox" "$stage/sandbox/sandbox"
    cp mesh/crates/mesh-llm-sandbox-plugin/README.md "$stage/sandbox/"
    printf 'name = "sandbox"\nversion = "%s"\n' "$version" > "$stage/sandbox/plugin.toml"
    archive="target/sandbox-plugin/sandbox-$version-$target.tar.gz"
    tar -czf "$archive" -C "$stage" sandbox
    echo "$archive"
    echo "install: mesh-llm plugins install --archive $archive --name sandbox --version $version"
