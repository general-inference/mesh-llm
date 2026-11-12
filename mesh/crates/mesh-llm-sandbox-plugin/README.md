# mesh-llm-sandbox-plugin

The `sandbox` plugin: isolated microVMs for running code, placed across a
private mesh. An agent on any node asks for a sandbox with `sandbox_create`;
the mesh picks a node that can run it (preferring the one serving the agent's
model), boots a microVM there, and routes later calls to it.

It owns sandbox lifecycle, placement, the per-sandbox egress policy and the
host side of the guest protocol. The guest side (the agent that runs inside
each VM, and the image build) is
[`mesh-llm-sandbox-guest`](../mesh-llm-sandbox-guest/README.md). The host
(`mesh-llm`) only launches the plugin and projects its tools over MCP and
HTTP; nothing in core knows about sandboxes. Full guide:
[docs/plugins/sandbox.md](../../docs/plugins/sandbox.md).

## Platforms

| Host | VM monitor |
| --- | --- |
| Linux | Firecracker on KVM |
| Windows | Firecracker inside WSL2, through `mesh-sandbox wsl-bridge` |
| macOS (Apple Silicon) | Virtualization.framework |

Every platform boots the same guest: kernel, read-only image disk, a scratch
disk for the sandbox's writes, and the guest agent on vsock.

## Build and try it

```bash
just sandbox-image                 # guest image for this architecture (needs Docker)
just sandbox-plugin                # builds mesh-sandbox; signs it on macOS
target/debug/mesh-sandbox check    # boots a sandbox here and exercises it
just sandbox-package               # archive for `mesh-llm plugins install --archive`
```

`check` reads the plugin settings as JSON from `MESH_SANDBOX_SETTINGS`, for
example `{"firecracker_path": "/opt/firecracker"}`.

On Windows, build the Linux binary inside WSL (`just sandbox-plugin release`
there) and place it at `linux/mesh-sandbox` in the plugin's directory, next to
`sandbox.exe`.
