# Compute

The `compute` plugin runs code for agents across a private mesh. It does that
in sandboxes. Each is a microVM on some node in the mesh, with its own kernel,
a Debian userland, a writable `/workspace`, and no network unless its config
asks for one. Placement prefers the node serving the agent's model, so a turn
of inference and the tool call after it stay on one machine.

The plugin refuses to start on a public mesh.

## Install

On each node that should run sandboxes, or only call them:

```bash
mesh-llm plugins install compute
```

The plugin is published with every mesh-llm release, for the platforms
mesh-llm ships on (macOS on Apple Silicon, Linux x86_64 and arm64, Windows
x86_64), and the plugin catalog pins each stable release's archives by
sha256. The Linux and Windows archives include Firecracker.

The first time a node needs the guest image (about 250 MB) it downloads it
from the same release and checks it against the sha256 compiled into the
plugin; the plugin also starts that download when it loads. Images live in
`~/.mesh-llm/compute/images/<name>/<arch>/`. Set `image_download = false` to
use only locally built images, or `image_url` to fetch from a mirror laid out
like the release (`<image_url>/<tag>/compute-image-<name>-<tag>-<arch>.tar.gz`).

A node that cannot run VMs still loads the plugin: it creates sandboxes on
other nodes, and `sandbox_nodes` says why it cannot run its own.

### Per platform

- **Linux**: read/write access to `/dev/kvm` (usually the `kvm` group). The
  bundled Firecracker is used unless `firecracker_path` names another.
- **macOS**: Apple Silicon. Released binaries carry the
  `com.apple.security.virtualization` entitlement they need.
- **Windows**: WSL2 (`wsl --install`), with `/dev/kvm` usable by the WSL user
  (nested virtualization, on by default on Windows 11 x64). The archive ships
  the Linux side and Firecracker for WSL. Windows on ARM is not supported.

Run `mesh-compute check` (the `compute` binary in the plugin's directory) on a
node to boot a sandbox and exercise it without the mesh.

### Building it yourself

```bash
just compute-image      # guest image for this architecture (needs Docker)
just compute-package    # local archive, signed on macOS; prints the install command
```

A local build pins no published images, so it uses the ones `just
compute-image` built.

## Tools

| Tool | Does |
| --- | --- |
| `sandbox_create` | Boot a sandbox from a config (below); returns its id and node |
| `sandbox_exec` | Run a command (`command` shell line or `argv`); returns exit code and output |
| `sandbox_read_file` / `sandbox_write_file` | Move files in and out (4 MiB per call) |
| `sandbox_list_dir` | List a directory (default `/workspace`) |
| `sandbox_snapshot` | Save `/workspace`; pass the result as `from_snapshot` to start from it |
| `sandbox_destroy` | Stop a sandbox and delete its disk |
| `sandbox_list` | Sandboxes this node created, wherever they run |
| `sandbox_egress_log` | Recent connections the sandbox made or was refused, and why |
| `sandbox_nodes` | Nodes that can run sandboxes, and their limits |

The host projects them over MCP and over HTTP under the plugin's namespace.

## Sandbox config

Every field is optional.

```jsonc
{
  "image": "base",
  "resources": { "cpus": 2, "memory_mb": 2048, "disk_gb": 10 },
  "egress": {
    "mode": "allowlist",              // "none" (default) | "allowlist" | "open"
    "allow": ["@registries", "api.example.com:443", "*.internal.example.com", "10.20.0.0/16"],
    "deny": ["*.tracking.example.com"]
  },
  "ttl_secs": 3600,                   // destroyed this long after creation
  "env": { "CI": "1" },               // added to every command
  "node": "local",                    // or a peer id prefix; omit to let the mesh choose
  "near_model": "Qwen3-32B",          // prefer a node serving this model
  "from_snapshot": { "snapshot_id": "snap-…", "node": "…" }
}
```

Unknown fields are rejected, so a misspelled `egress` cannot silently leave a
sandbox offline or open.

## Egress

The guest has no network interface. Commands reach the network through an
HTTP proxy inside the guest (`HTTP_PROXY`/`HTTPS_PROXY` are set), which hands
each connection to the host over vsock. The host applies the sandbox's
policy, resolves the name itself, and connects only to addresses that passed:

1. `none` refuses everything.
2. Loopback, link-local (including cloud metadata at 169.254.169.254),
   unspecified, multicast and broadcast addresses are never reachable, in any
   mode. The node's own services and mesh ports live on loopback.
3. `deny` entries refuse matching names and addresses.
4. Private and shared ranges (RFC 1918, CGNAT such as Tailscale, IPv6 ULA)
   are reachable only when an `allow` entry names them, by host or CIDR, even
   in `open`.
5. `allowlist` needs an `allow` match; `open` allows the rest.

Entry forms: `host`, `host:port`, `*.example.com` (subdomains only), IP
addresses, CIDRs, and presets. `@registries` covers npm, PyPI, crates.io, Go
modules, Docker Hub, GHCR, GitHub, Debian, RubyGems, Maven Central and
Hugging Face.

Programs that ignore proxy settings, and protocols other than HTTP(S) through
`CONNECT`, have no route out.

## Operator settings

`[plugin.settings]` for `compute` in the node config. A request beyond a limit
is refused with the limit named; nothing is silently reduced.

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_sandboxes` | 4 | Sandboxes running on this node at once |
| `max_cpus` | all cores | Per sandbox |
| `max_memory_mb` | 8192 | Per sandbox |
| `max_disk_gb` | 50 | Per sandbox |
| `max_egress` | `open` | The most network access a sandbox here may ask for |
| `max_ttl_secs` | 86400 | Longest a sandbox may live |
| `accept_remote` | true | Run sandboxes for other nodes |
| `state_dir` | `~/.mesh-llm/compute` | Scratch disks, sockets, snapshots |
| `images_dir` | `<state_dir>/images` | Guest images |
| `image_download` | true | Download the release's pinned guest images when missing |
| `image_url` | mesh-llm GitHub releases | Mirror to download images from |
| `firecracker_path` | bundled, then `firecracker` on PATH | Linux (a WSL path on Windows) |
| `wsl_distro` | default distribution | Windows |

## Trust and ownership

Sandboxes are for private meshes, where members already trust each other with
compute. Within one:

- A peer reaches only the sandboxes it created; the mesh authenticates which
  peer sent each request. The local node's own tools see every sandbox on it.
- The VM boundary is the isolation; the VM monitor runs as the plugin's user.
- Snapshot ids are unguessable; holding one is what lets a node read it.
- Sandboxes do not survive a plugin restart, and a sandbox is lost with its
  node. Snapshot what you need to keep. Snapshots are kept on the node that
  took them for seven days.

## How it works

Each node broadcasts its capacity (backend, architecture, images, free slots,
limits) on the `compute.v1` plugin channel. `sandbox_create` places the
sandbox by preference: a node serving `near_model`, the node holding
`from_snapshot`, this node, then the lowest round-trip time, then the most
free slots. Later calls go to the node that holds the sandbox as `request`
messages on the same channel; the receiving node runs them as the sending
peer.

Inside the VM, the agent ([`mesh-llm-compute-guest`](../../crates/mesh-llm-compute-guest/README.md))
runs as PID 1. The image disk is the read-only root; the sandbox's sparse
scratch disk is formatted on first boot and becomes the upper layer of an
overlay, so installs and writes land on scratch and the image stays shared.
