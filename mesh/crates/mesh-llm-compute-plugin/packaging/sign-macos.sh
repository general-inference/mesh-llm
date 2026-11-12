#!/usr/bin/env bash
# Signs a macOS build of the plugin with the Virtualization.framework
# entitlement it needs. Ad hoc unless MESH_COMPUTE_SIGNING_IDENTITY names a
# Developer ID identity, matching how mesh-llm itself ships.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
binary="${1:?usage: sign-macos.sh <binary>}"
codesign --force --options runtime --sign "${MESH_COMPUTE_SIGNING_IDENTITY:--}" \
    --entitlements "$here/../entitlements.plist" "$binary"
codesign --display --entitlements - "$binary" 2>/dev/null | grep -q com.apple.security.virtualization
