# ── Sandbox plugin ─────────────────────────────────────────────

# Build the sandbox plugin (`mesh-sandbox`); on macOS it is signed with the virtualization entitlement.
sandbox-plugin profile="debug":
    cargo build {{ if profile == "release" { "--release" } else { "" } }} --locked -p mesh-llm-sandbox-plugin
    {{ if os() == "macos" { "mesh/crates/mesh-llm-sandbox-plugin/packaging/sign-macos.sh target/" + profile + "/mesh-sandbox" } else { "" } }}

# Build the sandbox guest image (kernel + root filesystem) for this machine's architecture. Needs Docker.
[unix]
sandbox-image out=(home_dir / ".mesh-llm" / "sandbox" / "images"):
    mesh/crates/mesh-llm-sandbox-guest/image/build.sh "{{ out }}"

# Package a local build of the sandbox plugin for `mesh-llm plugins install --archive`. Releases do this in the Release workflow.
[unix]
sandbox-package profile="release":
    #!/usr/bin/env bash
    set -euo pipefail
    just sandbox-plugin {{ profile }}
    version="$(cargo pkgid -p mesh-llm-sandbox-plugin | sed 's/.*[#@]//')"
    target="$(rustc -vV | sed -n 's/^host: //p')"
    manifest="$(mktemp)"
    trap 'rm -f "$manifest"' EXIT
    "target/{{ profile }}/mesh-sandbox" --print-package-manifest > "$manifest"
    archive="$(mesh/crates/mesh-llm-sandbox-plugin/packaging/package.sh --target "$target" \
        --tag "v$version" --binary "target/{{ profile }}/mesh-sandbox" --manifest "$manifest" \
        --out target/sandbox-plugin)"
    echo "$archive"
    echo "install: mesh-llm plugins install --archive $archive --name sandbox --version $version"
