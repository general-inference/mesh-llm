# Metal decode optimization evidence

The measurements below were collected on the branch based at `c8bdb738f`,
before general-inference/mesh-llm PR #33. They remain historical measurements,
not performance claims for the combined implementation. Integration with
merged commit `74c7d5bd2` is in progress; the fresh comparison below qualifies
only its stated workloads. In the
combined queue, the original core patches 0017–0022 are numbered 0036–0041,
following PR #33's core patches; the combined native ABI is 0.1.85.

## Combined implementation against PR #33

The control is the clean merged PR #33 commit `74c7d5bd2`, built in a
separate worktree with its own ABI 0.1.84 native runtime. The candidate is
`perf/skippy-decode-pr33` with the combined ABI 0.1.85 runtime. Both are full
release product builds. The packaged candidate native source is
`c568f66885340c801cb5ae14989aaff42d8a2c2b`; subsequent queue repairs change
only tests, with no differences in production `src`, `ggml`, or headers.

Both arms use the same target/draft revisions listed below, DFlash with a
maximum of six draft tokens, adaptive sizing, host draft readout, f16 KV,
8192 context cells, batch 2048, microbatch 512, two generation lanes, and no
prefix or disk cache. Shared draft and shared verification are explicitly
enabled on the candidate; the control does not implement those switches.
These remain opt-in capabilities, so the results do not describe every
unconfigured installation.

On the same M5 Max, the pinned llama-benchy revision listed below ran PP512 /
TG256 at concurrency one and two. Each fresh process warmed two batches and
measured three per concurrency, in control/candidate/candidate/control order.
The table gives medians of six measured batches per arm.

| Decode workload | PR #33 tok/s | Combined tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 124.762 | 135.368 | +8.50% |
| Two requests, aggregate | 124.453 | 180.830 | +45.30% |

All 36 measured requests completed without reported errors. Ordinary EOS was
enabled: one control request ended at 159 tokens; the other 35 produced 256.
Prompt strings differ between benchmark batches. The single-request candidate
rates range from 121.4 to 165.8 tok/s, so its median is not a stable prediction
for arbitrary text. Eight separate fixed-prompt outputs (two prompts in each
process) all produced 64 tokens and matched byte for byte across both arms.
This establishes a gain for the tested workload, not parity with BaseRT.

An additional candidate-only eight-lane check used 32768 context cells and
fixed 128-token prompts: one warmup batch plus three measured batches. All
32 requests reached the length limit without errors. The three measured
batch rates were 270.36, 268.90, and 266.13 aggregate tok/s, calculated as
1024 completion tokens divided by the slowest request's HTTP elapsed time.
There is no fresh eight-lane PR #33 control in this check; it is a correctness
smoke and cannot establish an eight-lane speedup or exact output parity.

A separate instrumented run on the combined implementation captured normal
Metal command timings for two single requests. GPU execution occupied
1365.65 of 1584.00 ms for the 256-token prompt and 280.47 of 314.19 ms for
the 64-token prompt (unions of GPU execution intervals inside each decode
span). CPU Metal encoding occupied 104.52 and 17.37 ms respectively; these
intervals overlap GPU work. This small sample points toward GPU work as the
larger remaining cost; removing all encoding cannot recover the whole gap.
A further class-split diagnostic identified matrix multiplication as the
largest GPU class, but splitting produced hundreds of command buffers per
forward and slowed execution. Its timings are invasive diagnostics, not
production fractions or an additional speedup claim. Both profiles and the
interval audit are retained beside the benchmark evidence.

Validation of the combined implementation includes the full Skippy and Mesh
release product build, 517 runtime lib tests, 1352 serving lib tests, 12 FFI
tests, and all-target Clippy with warnings denied. The native roster selected
336 cases: the initial run had three failures and one explicit real-model
skip. The three failures subsequently passed focused reruns after building
the missing activation-validation binary and repairing outdated capability
and empty-command timing assertions. The standard native full-replay command
still stops on inherited references to two nonexistent targets; that command
is not a passing gate. No push or PR was made.

Raw ABBA commands, CLI hashes, runtime manifests, request outputs, per-request
progress, and the audit are retained locally in
`target/pr33-performance-evidence/`. Integration build and test logs are under
`target/pr33-integration-evidence/`. The following sections retain their
original pre-PR #33 measurements and patch numbers for provenance.

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

## Active weight traffic audit

The total file-size difference understates the quantization difference in the
comparison. A tensor inventory of both artifacts gives the following storage
estimate for one target token, using 40 layers, eight of 256 experts, and one
output-head application. Values are decimal GB.

| Weight category | Current GGUF | BaseRT Q4 artifact |
| --- | ---: | ---: |
| Dense weights and output head | 2.015 | 1.098 |
| Active expert weights, estimated as total expert bytes × 8/256 | 0.612 | 0.566 |
| Sum | 2.626 | 1.665 |

The GGUF has approximately **58% more active stored weight bytes per token**,
despite being only about 10% larger overall. Its dense projections include
Q8_0 matrices, its routing gates include f32 matrices, and its output head is
Q6_K. The BaseRT artifact uses its canonical Q4 representation for many of
these weights. Large inactive expert stacks dominate total file size and hide
the relative dense-weight difference.

This is an inventory calculation, not a measurement of DRAM traffic. It omits
embedding-row reads, KV/state traffic, padding, cache reuse, and implementation
details. It neither establishes matched accuracy nor attributes all of the
observed throughput gap to quantization. MTP layers are excluded. The GGUF
inventory comes from the existing `skippy-package-builder inspect` command;
the BaseRT inventory comes from its public SDK inspection surface.

Artifacts and calculation assumptions are retained locally in
`target/moe-kernel-evidence/weight-traffic.json` and the adjacent inventories.
An equal-quantization comparison is needed before treating the practical
same-model-family result as a kernel-efficiency ratio.

## Follow-up probes

The following local experiments were not retained as production changes:

- Forcing the existing few-row matrix kernel for one-row Q8_0 input slowed a
  4096×14336 projection from about 86 to 116 microseconds. The default threshold
  was restored.
- Changing routed Q4_K/Q6_K workgroups from two to four SIMD groups gave small,
  mixed changes at existing MoE benchmark shapes.
- One-SIMD-group Q8_0 variants, including a wider block mapping, helped some
  projection shapes and hurt others. They did not justify a general dispatch
  change. The short microbenchmarks also reuse weights more than a complete
  model forward, so their timings are not end-to-end speedup estimates.
- Making all GPU buffers private with `mmap=false` measured 56.7 tok/s for one
  request, below the roughly 61.7 tok/s shared-buffer control.
- A temporary backend preference for private **weights only**, leaving graph
  buffers shared, measured 63.4 tok/s in one process. This small improvement is
  unqualified; the two-request run also included a 118-token early EOS and
  cannot be compared directly with all-256-token runs. The preference was
  removed rather than changing defaults without stronger evidence.

Launching Skippy under Instruments with the **Metal GPU Counters** instrument
enabled shader-profiler data that was absent from the earlier attached trace.
A warmup-forward PC-range attribution suggests Q8_0 dense kernels are a larger
target than the routed expert kernels. Program-counter ranges can overlap or
be reused, and concurrent shader intervals are not additive wall time; this
ranking needs a carefully scoped steady-decode analysis before it supports
an optimization claim. Only Skippy was launched for profiling, with a minimal
environment; the proprietary engine was not profiled.

## Rejected post-PR #33 matrix tiling probe

The current few-row MMA crossover was originally measured on an M3 Ultra.
A sweep on the M5 Max used the actual target's five Q8 dense projection
shapes at 1, 2, 4, 6, 8, 12, and 16 rows. Smaller K splits were generally
slower. A finer split of 16 SIMD groups improved the 2048-input / 8192-output
projection at 12 and 16 rows by about 8% and 16% in a repeated-weight
microbenchmark. An experimental native patch restricted that choice to this
geometry, 12–16 rows, and M5 Max, retaining all normal resource limits.

The candidate passed 41 GPU-versus-CPU numerical cases, including each
changed row count and the 11/17-row boundaries. A full decoder ABBA comparison
used the same release CLI, target/draft weights, and shared-verification
configuration on both arms. Each fresh process first completed two simultaneous
fixed 128-token prompts, then ran the PP512/TG256 benchmark with two warmups
and three measured batches per concurrency. All eight fixed outputs matched
byte for byte, and all 36 measured requests completed without reported errors.
Ordinary EOS ended one control request at 238 tokens and one candidate request
at 190; the other 34 produced 256. Prompt strings varied between batches.

| Decode workload | Qualified control tok/s | Tiling candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 133.806 | 130.841 | -2.22% |
| Two requests, aggregate | 184.960 | 181.395 | -1.93% |

These medians show no useful end-to-end gain. The single-request difference,
where the changed geometry is largely absent, also illustrates process/text
variation. The experiment does not prove a causal 2% regression, but it does
not justify changing defaults. The patch was removed from the active queue,
the qualified runtime bundle was restored, and the original queue replayed
cleanly. Raw commands, numerical cases, sweep results, runtime bundles,
manifests, and the rejected patch remain under `target/mma-tiling-evidence/`.

## Host greedy placement on M5 Max

Core patch `0021-metal-prefer-host-greedy-sampling-on-qualified-share.patch`
prefers the existing host greedy sampler on M5 Max when graph buffers are
shared with the host. A private GGML backend preference communicates that
choice to Skippy's sampler attachment code. Other devices and private-buffer
configurations retain their previous behavior. Explicit draft GPU readout uses
a separate path and is unchanged. No public Skippy API or ABI changed.

Earlier fixed-length requests disabled backend greedy sampling, while
ordinary-EOS requests enabled it. A temperature-one/top-k-one probe, which
still forces an argmax, suggested that placement caused part of the difference.
The retained change was then tested with the **original temperature-zero
requests**, unchanged weights and sampler settings.

Baseline is `4f787f529`; candidate adds patch 0021. The same CLI explicitly
selected the saved baseline and candidate native bundles. The workload and
ABBA fresh-process method are the same as the plain comparison above: 512-token
prompts, up to 256 generated tokens, two warmups and three measured batches per
concurrency per process, ordinary EOS, greedy sampling, thinking disabled, and
f16 KV. Medians below cover six measured batches per concurrency.

| Greedy decode workload | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 61.524 | 66.512 | +8.11% |
| Two requests, aggregate | 95.770 | 103.136 | +7.69% |
| Eight requests, aggregate | 160.051 | 213.713 | +33.53% |

The eight-request probe used 32,768 total context slots, one warmup and two
measured batches per process, again in ABBA order. Its medians cover four
batches per arm. Both arms used the same rebuilt CLI; its hash is recorded
separately from the earlier one/two-request comparison. All 64 measured
requests succeeded. Three baseline requests stopped at 255, 221, and 135
tokens; the remaining baseline requests and all candidate requests produced
256 tokens. Actual token counts determine throughput. The sole entirely
full-length baseline batch measured 161.035 tok/s, which also shows a large
improvement against the candidate median, but is only one batch.

All 36 measured requests succeeded. Thirty-five produced 256 tokens; one
candidate request in a two-request batch stopped at 193 tokens. The benchmark
uses actual token counts. The five full-length candidate two-request batches
have median throughput 103.263 tok/s, so the early EOS does not create the
reported improvement.

All eight fixed 64-token plain outputs matched. All twelve fixed 128-token
speculative outputs also matched across three fresh processes per arm. The
eight-request configuration additionally produced sixteen fixed 64-token
outputs in eight baseline/candidate comparisons; every pair matched exactly.
Across these checks, all 36 generated outputs matched their counterparts. The
short speculative measurements remain noisy; this patch's speedup claim is
for plain greedy decode. A separate graph log recorded 7,490 Metal node lines
and zero `temp_max_idx`, `greedy_argmax`, or `greedy_sampled_token` node lines,
confirming that an actual temperature-zero completion used the host path.

Validation includes the complete native patch replay and runtime package
build, eight device/storage preference checks in `test-metal-graph-optimize`,
the retained gated-activation checks, and 15 native sampling/history/suppression
tests. The composed Skippy and Mesh release build also passed.

Commands, raw responses, timings, request audits, native manifests, graph logs,
and validation output are retained under `target/host-greedy-evidence/`.
These measurements qualify the tested M5 Max workloads; they do not establish
BaseRT parity or performance on other hardware.

## Rejected draft geometry and sampler probes

The next round kept the target/draft weights fixed and tested three possible
reductions in speculative decode cost. None was retained. The control includes
the committed host-greedy placement change above.

| Probe | Control aggregate tok/s | Probe aggregate tok/s | Change |
| --- | ---: | ---: | ---: |
| Full 16-row trained draft block, return only six proposals | 209.663 | 136.524 | −34.88% |
| GPU greedy readout, retain token output only | 206.476 | 206.296 | −0.09% |
| Batch independent GPU argmax reductions, two requests | 208.542 | 212.223 | +1.77% |
| Batch independent GPU argmax reductions, eight requests | 245.473 | 242.087 | −1.38% |

These are short direct-completion coding probes, not the 512-token prompt
benchmark above. Each process warmed one batch and measured three batches in
ABBA fresh-process order; medians cover six batches per arm. Every request
produced 128 tokens. Shared verification and drafting were enabled, the proposal
limit was six, and adaptive window selection was disabled in both arms. The
first three comparisons used two coding prompts and 8,192 total context slots;
the eight-request comparison used eight prompts and 32,768 slots. The sampler
comparisons enabled draft GPU readout in both arms. The batching comparison
used the same CLI and selected saved/control and candidate native bundles.

The full-block probe kept attention at the checkpoint's trained width but
projected logits only for the requested prefix. It reduced accepted drafts on
the Rust prompt and increased runtime. It is not evidence that the current
shorter-block path is incorrect. The DFlash paper's
[block-size ablation](https://arxiv.org/html/2602.06036v1#S5.SS4.SSS4)
also treats inference width as a performance tradeoff, including smaller widths
than training.

The readout audit found that a plain native greedy sampler retains its input
logits as an output even after selecting a token. A private interface guard
suppressed unused logits/probability/candidate outputs when a token was
available, preserving fallback outputs otherwise. Removing those readbacks
alone did not improve this workload. A second probe shared one matrix argmax
across structurally unmodified raw-logit rows, preserving filtered/remapped
samplers. CPU and Metal numerical checks passed at vocabulary widths 19 and
248,320, including row reordering, repeated rows, and each backend's existing
tie behavior. A serving graph log contained six `greedy_argmax_batched` node
lines and no individual `greedy_argmax` node lines, confirming execution of
the proposed graph. Its mixed end-to-end result did not justify adoption.

All 24 measured outputs matched across arms in each two-request comparison.
All 96 eight-request completions succeeded, but four prompts had two output
variants; every candidate variant also occurred in the control. This is not
an exact-output parity qualification for the eight-request configuration.

Raw responses, timings, output audits, graphs, numerical tests and rejected
source patches are retained under `target/dflash-full-block-evidence/` and
`target/dflash-token-only-evidence/`. The canonical source and patch queue were
restored after these probes. Draft GPU readout remains opt-in, and the
two-request batching result is not claimed as a shipped gain.

## Adaptive window alignment lead

A follow-up configuration comparison used the saved committed CLI and native
bundle in both arms, changing only `SKIPPY_DFLASH_ADAPTIVE`: adaptive sizing
versus fixed six-token proposals. It used the same two 128-token direct coding
completions, shared verification/drafting, host draft readout, and ABBA method
as the short probes above. No sampler or native batching probe was enabled.

Fixed sizing reached median **206.875 aggregate tok/s**, versus **178.843**
for adaptive sizing (**+15.67%**). All 24 measured completions produced 128
tokens and matched exactly by prompt. Including warmups, telemetry recorded
384 of 432 fixed-window lane verifications as shared (**88.9%**), versus
332 of 501 adaptive-window lane verifications (**66.3%**). The adaptive arm
was also more variable: its six measured batches ranged from 164.3 to 227.3
tok/s, while fixed sizing ranged from 206.2 to 212.4.

The scheduler currently coalesces only equal-width verification requests.
Adaptive sizing chooses widths independently per request, so differing choices
can prevent shared target execution. This is a candidate explanation, not an
isolated causal proof: acceptance estimates, serial probes, and cost-model
learning also differ between these arms. The result warrants testing aligned
adaptive windows or compatible ragged verification, including rollback,
sampling history, constraints, request tails, and capacity accounting. It does
not qualify disabling adaptation by default for other prompts or loads.

Raw responses, shared-verification event counts, output checks and commands
are retained in `target/dflash-window-policy-evidence/`.

## Align adaptive widths for shared verification

Concurrent local DFlash requests now use aggregate acceptance and one cached
width for eight observed windows per lane when both shared drafting and shared
verification are enabled. Cost learning stays isolated by draft model and
active lane count. Decisions are cached separately for each proposal limit,
so a request tail cannot shorten its peers. A serial decision is reconsidered
after the next draft probe. Single-lane, unshared and split paths retain their
per-request selection. Each session retains ownership of its sampling,
checkpoints and repair; the verifier receives its selected window without
padding. There is no new public API or ABI.

This changes the **policy**, rather than padding native verification windows.
Padding could advance a stateful sampler beyond a request's original window;
the aligned policy submits only the token rows the request actually selected.
Both shared switches remain opt-in.

Control is `18e4e04e6`; candidate adds this policy. Both arms use the same native
bundle (patch digest `9bdd0ef972dd5488927f604413d927ed880ea611c2b7c5e6a4bca14f417f8a54`),
weights, context/batch limits, host draft readout, six-token proposal limit and
adaptive sizing. Each arm has two fresh processes, in ABBA order.

| Workload | Control aggregate tok/s | Candidate aggregate tok/s | Change |
| --- | ---: | ---: | ---: |
| Two short coding prompts, 128 tokens each | 191.055 | 214.662 | +12.36% |
| Two requests, 512-token prompts, 256 generated tokens | 138.165 | 162.787 | +17.82% |

The short probe warms one pair and measures three pairs per process. All 24
measured outputs match exactly by prompt. Including warmups, shared verification
covers 402 of 555 control lane-windows (72.4%) and 444 of 487 candidate
lane-windows (91.2%).

The longer pinned llama-benchy probe warms twice and measures three batches at
each concurrency per process. All 36 measured requests completed all 256 tokens
without errors. Its single-request medians are 123.387 and 125.606 tok/s; the
wide run-to-run variation does not establish a single-request gain. Eight
additional fixed 64-token outputs also matched exactly by prompt.

An eight-request comparison is **inconclusive**: both candidate processes
completed their six measured batches (48 requests), but the last control
process returned HTTP 502 during its first measured batch. A separate fresh
control process with error bodies retained reproduced
`linear proposal repair failed and the session was retired: RuntimeError:
failed to trim hybrid memory suffix`. Both failures occurred in the control
binary. This demonstrates a control-path rollback defect, not its root cause
or that the candidate fixes it. No eight-request speedup or exact-output parity
is claimed from this interrupted comparison. Higher-concurrency rollback
qualification remains outstanding.

The serving suite passed 1,201 tests with 18 existing opt-in tests ignored.
Focused policy tests cover stable intervals, row-count exploration, limits,
serial-probe reconsideration, lane isolation, and the unchanged single-lane
and unshared choices. Skippy-serving and Mesh checks and warning-denying Clippy,
workspace formatting, and the no-console-print contract passed.

Evidence is retained in `target/dflash-aligned-window-evidence/`, including
saved binaries/hashes, commands, native manifest, raw responses, progress
events, output audits, test output and the interrupted/diagnostic eight-request
runs. These results qualify the tested two-request workloads; BaseRT parity
and performance at other loads remain unproven.

## Preserve recurrent history when unrelated lanes run

The eight-request failure above now has a deterministic native reproducer.
After A and B verify together, B can retain a pending repair while A and C run.
The recurrent allocator gathers active lanes contiguously, relocating B as an
inactive peer. It previously moved B's current-state source and sequence
metadata without preserving its rollback depth and snapshot planes. A later
trim could fail or select a different state. This is a native memory ownership
defect independent of the adaptive-width policy.

Core patch 0022 preserves the logical owner's rollback depth during cell swaps
and materializes live inactive history before a forward can overwrite its
sources. The copy gathers each snapshot plane before writing it, handles
pending rollbacks and fresh zero-state lanes, and covers recurrent R, S and P
banks through backend tensor access. Scratch storage is reused. Planning never
copies tensor data. The materialization is conditional on live inactive
history moving or shifting; it is a correctness fallback whose overhead still
needs throughput qualification.

This patch owns generic recurrent cell relocation, separately from patch
0015's staged verification-window trim interface. It adds no public symbols
or headers and retains Skippy ABI 0.1.82. The complete pinned core, model and
generated queues replay successfully. The packaged Metal source is
`16cb58f537c151171cc64e5f49e46726a32df6fd`, with patch digest
`0182eb8a76d72c37e0b1124e347b38753b7259e36248305b37078295e9527dd9`.

The regression fails before the fix with the same hybrid suffix-trim error.
With the fix, all eight CPU fixture/rollback tests pass. Four delayed-peer
scenarios also pass with recurrent state explicitly asserted to reside on
Metal, including active and inactive pending rollbacks and a new zero-state
lane. The tests compare every retained snapshot byte before and after peer
execution, then compare 15 generated predictions against an independent serial
reference after repair. CPU coverage includes Qwen, GLM and LFM2 fixtures.

Two fresh processes of the old, independently adaptive serving CLI completed
64 of 64 eight-way 128-token coding requests with the repaired native engine,
without errors. These were correctness stress runs with debug logging, not
performance measurements. Some prompts produced more than one output across
the runs; exact serving-output parity is not established. The interrupted
eight-request speed comparison above remains historical evidence, not a
speedup claim. Artifacts and commands are in `target/peer-rollback-evidence/`.

The interrupted comparison was then repeated with the repaired native bundle
in **both** arms. The independent controller (`18e4e04e6`) and aligned
controller (`8a0e553fe`, unchanged by the native fix) each ran two fresh
processes in ABBA order, one warmup batch and three measured batches per
process, using eight concurrent 128-token coding completions. Median batch
throughput increased from **196.960 to 259.046 aggregate tok/s (+31.52%)**.
Control batches ranged from 162.0 to 252.7 tok/s; aligned batches ranged from
252.0 to 262.0. All 96 measured requests completed 128 tokens without errors;
all 128 requests including warmups completed. Across arms and repeats, five
of eight prompts had two output variants, so exact text parity remains
unqualified. Including warmups, shared verification covered 88.1% of control
lane-windows and 95.4% of aligned lane-windows. Eight-lane shared draft batches
increased from 29 to 199. These figures qualify this policy on the tested
eight-request workload, not an independent throughput gain from the memory fix
or parity with BaseRT. Raw responses, timings, logs and audits are retained in
`target/peer-rollback-evidence/aligned-c8-*`.

The complete Skippy and Mesh release build passed after the final native queue
replay. Runtime library tests passed 503 tests with 12 existing opt-in tests
ignored; serving library tests passed 1,201 with 18 ignored.

Jianyang job `job_arc0cvavycafvvb` built exact local commit
`f6dbb9090694e907684191968528968545b9499d` natively on Dan's M4 mini
(16 GB, macOS 26.5.1). The eight fixture/rollback tests and all four explicit
Metal delayed-peer scenarios passed; the job succeeded at $0. Source was
fetched at the pinned commit from a temporary read-only private-LAN Git
listener, which was stopped after checkout. No GitHub push or binary transfer
was needed. The static test recipe first exposed an existing unquoted
prefix-map path problem under JY's `Application Support` workspace; validation
used the normal dynamic Metal build instead. This run establishes additional
hardware correctness coverage, not M4 model throughput.
