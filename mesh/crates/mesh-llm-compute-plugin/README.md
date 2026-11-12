# mesh-llm-compute-plugin

The `compute` plugin: isolated microVMs for running code, placed across a
private mesh. An agent on any node asks for a sandbox with `sandbox_create`;
the mesh picks a node that can run it (preferring the one serving the agent's
model), boots a microVM there, and routes later calls to it.

It owns sandbox lifecycle, placement, the per-sandbox egress policy and the
host side of the guest protocol. The guest side (the agent that runs inside
each VM, and the image build) is
[`mesh-llm-compute-guest`](../mesh-llm-compute-guest/README.md). The host
(`mesh-llm`) only launches the plugin and projects its tools over MCP and
HTTP; nothing in core knows about sandboxes. Full guide:
[docs/plugins/compute.md](../../docs/plugins/compute.md).

## Platforms

| Host | VM monitor |
| --- | --- |
| Linux | Firecracker on KVM |
| Windows | Firecracker inside WSL2, through `mesh-compute wsl-bridge` |
| macOS (Apple Silicon) | Virtualization.framework |

Every platform boots the same guest: kernel, read-only image disk, a scratch
disk for the sandbox's writes, and the guest agent on vsock.

## Install, build and release

Users install it with `mesh-llm plugins install compute`; releases publish it
(see [RELEASE.md](../../../RELEASE.md#compute-plugin)), and the plugin
downloads the guest images that release pins. To work on it:

```bash
just compute-image                 # guest image for this architecture (needs Docker)
just compute-plugin                # builds mesh-compute; signs it on macOS
target/debug/mesh-compute check    # boots a sandbox here and exercises it
just compute-package               # archive for `mesh-llm plugins install --archive`
```

`check` reads the plugin settings as JSON from `MESH_COMPUTE_SETTINGS`, for
example `{"firecracker_path": "/opt/firecracker"}`.

`packaging/` holds the release packaging: `package.sh` assembles an archive
(bundling Firecracker per `firecracker.pin`), `sign-macos.sh` adds the
entitlement, and `catalog-entry.json` is the catalog entry each stable release
pins.
