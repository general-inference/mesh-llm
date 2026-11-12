# mesh-llm-sandbox-guest

The inside of a sandbox microVM: the agent that runs as PID 1, the vsock wire
protocol it shares with the host, and the image build.

- [`src/protocol.rs`](src/protocol.rs) — length-prefixed JSON frames for exec,
  files, archives and egress handshakes. The host side uses the `tokio`
  feature for async framing.
- [`src/agent/`](src/agent) — Linux only. Builds a writable root (an overlay
  of the read-only image and the sandbox's scratch disk), serves host
  operations on vsock port 1024, and runs the HTTP proxy that is the guest's
  only route out: each connection is handed to the host on vsock port 1025,
  where the sandbox's egress policy decides.
- [`image/`](image) — `build.sh` produces `kernel` and `rootfs.ext4` for the
  build machine's architecture with Docker; run it as `just sandbox-image`.

It does not own sandbox lifecycle, placement or egress policy; those are the
plugin's ([`mesh-llm-sandbox-plugin`](../mesh-llm-sandbox-plugin/README.md)).
