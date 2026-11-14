# Metal decode optimization evidence

## Gated activations

Core patch `0020-metal-fuse-contiguous-float32-gated-activations.patch` adds
Metal fusion for `SiLU(x) * y` and `sigmoid(x) * y`. The loaded Qwen3.6 MoE
graph uses these chains in recurrent output gates, attention gates, and shared
experts. The fused kernel avoids storing and reading the activation intermediate
and submits one kernel for the pair.

This is a new backend fusion capability, separate from the runtime packaging
changes in core patch 0005. It does not change the public Skippy ABI; both Rust
and native retain ABI 0.1.82.

Admission is deliberately limited:

- Inputs and output are float32 and contiguous.
- Only dimension zero may broadcast, and only from width one. Remaining
  dimensions must agree with the output.
- Either multiply operand may contain the unary result.
- Captured intermediates and intermediates with other consumers remain
  materialized through the existing generic subgraph checks.
- Equal-shape in-place output is allowed. Partial overlaps and output aliases
  with a broadcast input are rejected.
- Unsupported layouts, operations, and types retain their separate kernels.

The normal graph optimizer, allocation dependencies, concurrency checks, and
command-buffer fusion boundaries apply. The kernel preserves a float32 unary
result before multiplying; lower-precision types are not admitted.

## Local measurements

Hardware: M5 Max, 128 GiB. Baseline: commit `e1a48798e`. Candidate: the same
source plus patch 0020. The same release CLI selected saved baseline and
candidate native runtime bundles explicitly. Target weights, sampler, context
budgets, and request settings were unchanged.

Target: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, revision
`5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`,
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Native MTP and speculation were disabled for
the plain comparison. KV caches were f16; prefix and disk caches were off.

The pinned llama-benchy revision
`e9be344578cec17745066b220798b80a0d2686d3` ran 512-token prompts and
256-token generations, with ordinary EOS, greedy sampling, thinking disabled,
two warmup batches, and three measured batches per concurrency per process.
The order was baseline, candidate, candidate, baseline, with a fresh server
process each time. Values below are medians of six measured batches.

| Decode workload | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 61.000 | 61.663 | +1.09% |
| Two requests, aggregate | 95.254 | 95.831 | +0.61% |

The two-request distributions overlap; this is a small measured improvement,
not evidence of a large throughput change. Prompt strings differed between
llama-benchy batches, while corpus and requested lengths were held constant.
Two separate fixed 64-token completion checks matched across all four server
processes (eight outputs).

A separate DFlash1 check used shared verification and shared draft, max draft
six, and two concurrent fixed 128-token completions. All twelve measured
outputs matched across three fresh processes per arm. Aggregate rates ranged
157.6–174.8 tok/s for baseline and 162.4–202.0 for candidate. The short-run
variance is too large to attribute a speculative speedup to this patch.

Raw commands, runtime manifests, responses, logs, and summaries are retained
locally under `target/unary-fusion-evidence/`; they are ignored build artifacts.

## Validation

- Full ordered patch replay and Metal runtime packaging passed.
- `test-metal-graph-optimize` passed, including 21 new admission checks for
  operand order, scalar gates, reuse, captured outputs, strided inputs,
  unsupported operations/types, and safe/unsafe aliases.
- `test-backend-ops test -b MTL0 -o 'SILU_MUL,SIGMOID_MUL'` passed 52 numerical
  cases. The existing suite also exercised unfused fallback shapes and f16;
  unsupported bf16 cases were reported as unsupported.
- Opening the full target on Metal compiled the shaders and completed requests.
  Native graph logs confirmed the fused gated-activation dispatches executed.

Build native test targets through the repository workflow, for example:

```sh
LLAMA_STAGE_BACKEND=metal LLAMA_STAGE_LINK_MODE=dynamic \
  LLAMA_STAGE_UPSTREAM_TESTS=ON \
  just with-lld scripts/build-llama.sh --build-only \
    --target test-backend-ops --target test-metal-graph-optimize
just with-lld just release-runtime-build metal
```

## Remaining gap

A separate practical BaseRT 0.5.0 comparison measured about 200 tok/s for one
request and 253 aggregate tok/s for two requests. It used the same original
model with a different Q4 quantization, not identical weights or matched
accuracy. BaseRT speculation was disabled. This patch does not establish
parity with that engine.

The small gain despite confirmed fusion means removing these dispatches alone
does not explain the gap. The next investigation is routed-expert matrix/vector
execution and its memory throughput, using attributable kernel measurements.
No proprietary engine implementation was inspected or copied for this change.
