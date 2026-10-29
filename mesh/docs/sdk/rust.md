# Rust SDK

`mesh-llm-sdk` exposes one embedded `MeshNode`. Choose its role with the builder:

| Builder | Consumes mesh inference | Serves local models |
|---|---:|---:|
| `.client()` | yes | no |
| `.serve_only()` | no | yes |
| `.serve()` | yes | yes |

```toml
[dependencies]
mesh-llm-sdk = { version = "0.78.0" }
```

```rust
use mesh_llm_sdk::MeshNode;

# async fn example() -> anyhow::Result<()> {
let node = MeshNode::builder()
    .client()
    .auto_join_public_mesh()
    .start()
    .await?;
let client = node.openai_client()?;
let models = client.models().await?;
println!("{models}");
node.stop().await?;
# Ok(())
# }
```

For a private mesh, replace `.auto_join_public_mesh()` with `.join_token(token)`.
For serve-only or combined use, select `.serve_only()` or `.serve()` and add
`.model("publisher/model:quant")`. Serving needs a compatible installed native
runtime; see [runtime artifacts](../SDK.md#native-runtime-artifacts).

The raw `OpenAiClient::request` and `OpenAiClient::stream` methods preserve
OpenAI-compatible request bodies, including tools and response-format fields.
A serve-only node rejects `openai_client()` at the SDK boundary.

For a locally loaded model, `node.inference_client()` provides in-process streaming
without an HTTP connection or another model load. It works in serve-only mode as
well as combined mode. Use the exact serving ID and the SDK's
`inference::ChatCompletionRequest`; the client enables streaming automatically.
The returned stream exposes status and `next_event()`. Drop it to cancel a request.
Each request resolves the node's current loaded instance, and shutdown interrupts
pending dispatch and stream reads. This local path uses the model's existing
frontend, guardrails and lifecycle observers; it does not route to remote peers.

**Migration:** The old `MeshClient` re-export and `node` feature are removed from
this facade. Migrate callers to `MeshNode`; the embedded node owns mesh join,
local HTTP routing, and shutdown. This changes the SDK API but not mesh wire
protocols.
