# ── Compute plugin ─────────────────────────────────────────────

# Build the compute plugin (`mesh-compute`); on macOS it is signed with the virtualization entitlement.
compute-plugin profile="debug":
    cargo build {{ if profile == "release" { "--release" } else { "" } }} --locked -p mesh-llm-compute-plugin
    {{ if os() == "macos" { "mesh/crates/mesh-llm-compute-plugin/packaging/sign-macos.sh target/" + profile + "/mesh-compute" } else { "" } }}

# Build the sandbox guest image (kernel + root filesystem) for this machine's architecture. Needs Docker.
[unix]
compute-image out=(home_dir / ".mesh-llm" / "compute" / "images"):
    mesh/crates/mesh-llm-compute-guest/image/build.sh "{{ out }}"

# Package a local build of the compute plugin for `mesh-llm plugins install --archive`. Releases do this in the Release workflow.
[unix]
compute-package profile="release":
    #!/usr/bin/env bash
    set -euo pipefail
    just compute-plugin {{ profile }}
    version="$(cargo pkgid -p mesh-llm-compute-plugin | sed 's/.*[#@]//')"
    target="$(rustc -vV | sed -n 's/^host: //p')"
    manifest="$(mktemp)"
    trap 'rm -f "$manifest"' EXIT
    "target/{{ profile }}/mesh-compute" --print-package-manifest > "$manifest"
    archive="$(mesh/crates/mesh-llm-compute-plugin/packaging/package.sh --target "$target" \
        --tag "v$version" --binary "target/{{ profile }}/mesh-compute" --manifest "$manifest" \
        --out target/compute-plugin)"
    echo "$archive"
    echo "install: mesh-llm plugins install --archive $archive --name sandbox --version $version"
