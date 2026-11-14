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

## Batch equal-width selector draft blocks

The Rust draft scheduler previously admitted multiple noise blocks only for
the basic DFlash head. The native selector and convolution graphs already
isolate sequences by reshaping rows into equally sized blocks. The scheduler
now admits that layout when every active block has the same actual row count
and the whole batch fits one native microbatch. Basic heads still accept
ragged blocks. Markov heads and unverified lattice heads remain independent.
Admission uses tensor and graph metadata: a selector requires the complete
predecessor, successor, and hidden projection set; a top-k count alone is
insufficient. There is no new model-family allowlist or native ABI change.

These measurements compare independent versus shared DFlash2 submissions on
M5 Max, starting from `69e122f69`, which includes PR #33 and the earlier decode
changes. Both arms use the original target weights, the same Q8 DFlash2 draft,
the same ABI 0.1.85 native bundle, and six maximum draft tokens. Shared verify,
shared draft, and adaptive widths are enabled in both arms; GPU draft readout
and device feature retention are disabled. Shared execution remains opt-in.

Each comparison uses four fresh processes in control/candidate/candidate/control
order and six measured batches per arm. No build or test load ran alongside
the performance comparisons.

| Workload | Independent tok/s | Shared tok/s | Change |
| --- | ---: | ---: | ---: |
| PP512/TG256, one request | 136.378 | 133.086 | -2.41% |
| PP512/TG256, two requests, aggregate | 163.905 | 184.419 | +12.52% |
| Fixed coding prompts, two requests, aggregate | 219.785 | 231.738 | +5.44% |
| Fixed coding prompts, eight requests, aggregate | 272.976 | 302.289 | +10.74% |

The long benchmark uses pinned llama-benchy
`e9be344578cec17745066b220798b80a0d2686d3`, two warmups and three measured
batches per concurrency per process. All 36 measured requests succeeded; two
control requests stopped at 240 and 241 tokens and all others reached 256.
Its single-request results overlap substantially and show no established gain.
Telemetry confirms zero shared draft forwards in the controls and 457/473
two-lane shared forwards in the candidates. Eight separate fixed 64-token
outputs match exactly across arms.

The fixed coding probes use one warmup and three measured batches per process,
128 output tokens per request, greedy sampling, f16 KV, and disabled prefix and
disk caches. Their metric is total completion tokens divided by the slowest
request's HTTP elapsed time, including prefill and request overhead; it differs
from the long benchmark's metric and workload. All 32 two-request completions
and all 128 eight-request completions reached 128 tokens. Each prompt produced
one byte-identical output across both arms and all measured repeats. The
two-request probe uses the final CLI including the startup guards below.

The real-checkpoint Metal regression compares three independent lanes with
shared proposals at widths 1, 3, and 6, reverses lane order, checks scratch
cleanup by proposing independently again, and verifies that ragged widths
fall back without incrementing shared-forward counts. It passes with the
actual selector/convolution draft and target features. Unit tests cover
incomplete selector projections, absent blocks, unequal widths, and serial
head admission.

Qualification also exposed model-free startup calls into an unloaded dynamic
FFI table. Scheduler platform detection now returns its existing generic
policy before native startup, and the media marker uses its existing default
until the native runtime is loaded. All 21 affected serving tests pass in
separate fresh processes without a runtime-bundle environment override.
The final full Skippy/Mesh release build passes, as do 521 runtime library
tests, 1,352 serving library tests, warning-denying all-target Clippy for the
affected runtime/serving/benchmark/Mesh packages, formatting, and the console
output contract. This Rust change does not resolve the previously recorded
inherited native test-target gate failure.

Commands, CLI hashes, native manifest, raw responses, telemetry, audits, and
validation logs remain under `target/dflash2-batch-evidence/`. These gains
qualify batching this particular draft; replacing DFlash1 with DFlash2 alone
was slower in the initial diagnostic. They do not establish BaseRT parity or
justify choosing DFlash2 for every workload.

## Compatible speculative sharing by default

Serving coalesces compatible drafts and verification windows by default.
No model weights, sampling parameters,
native ABI or graph admission rules change. Complete causal targets, distinct
sessions, equal verification widths and actual native capacity are still
required; unsupported target layouts retain independent verification. Draft
head contracts continue to decide which noise blocks may share. Single-lane
waves do not wait to collect peers. Explicit `SKIPPY_SHARED_DRAFT=0` and
`SKIPPY_SHARED_VERIFY=0` retain the independent paths for diagnostics.

The control is the saved release CLI at `5770b20c8`; candidate changes only
the sharing defaults and their tests/documentation. Both select the same
frozen ABI 0.1.85 native bundle, original target weights, original DFlash1
draft, six maximum draft tokens, f16 KV, and disabled prefix/disk caches.
The sharing, adaptive, GPU-readout and device-feature environment variables
are explicitly removed in both arms. Thus this comparison measures ordinary
sharing defaults rather than an explicitly enabled benchmark configuration.

The pinned PP512/TG256 benchmark uses four fresh processes in ABBA order,
two warmup batches and three measured batches per concurrency per process.
Medians cover six measured batches per arm.

| Workload | Previous defaults tok/s | New defaults tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 132.169 | 138.602 | +4.87% |
| Two requests, aggregate | 107.839 | 175.682 | +62.91% |

Single-request values overlap substantially, so their median is not an
established speedup. All six two-request candidate batches exceed all six
controls. All 36 measured requests succeed; one control request stops at 222
tokens and one candidate at 190, with the other 34 reaching 256. Excluding
those batches leaves five full-length batches per arm, with medians 106.662
and 176.805 aggregate tok/s, so early EOS does not create the gain. All eight
separate fixed 64-token outputs match exactly across arms.

A separate fixed coding comparison at eight requests uses one warmup and
three measured batches per fresh process, 128 completion tokens per request,
and 32,768 context slots. End-to-end aggregate throughput (total completion
tokens divided by the slowest HTTP request's elapsed time, including prefill)
improves from 222.803 to 261.038 tok/s (+17.16%). Control batches vary widely
(145–241 tok/s), but every candidate batch (245–265 tok/s) exceeds every
control. All 128 completions reach 128 tokens. Seven prompts match exactly
throughout; one prompt exhibits the target numerical variation below.

The subsequent eight-request check exposes a target numerical difference on
one of eight prompts, including variation within a candidate process. The
same difference reproduces at prediction 74 in teacher-forced target
verification without draft proposals or rollback. Serial evaluation and
independent seven-row windows agree through all 128 predictions on every
prompt. Shared batches of 16 and 24 packed rows agree through 80 predictions
per lane; 32 and 56 packed rows flip the same top token. At 32 rows the Metal
routed-expert operation switches from matrix/vector to matrix/matrix dispatch.

A rejected dispatch experiment requires eight expected rows per expert
before choosing routed matrix tiles. That restores all 1,024 teacher-forced
predictions at 56 packed rows, but slows a fresh eight-request ABBA comparison
from 266.878 to 208.829 end-to-end aggregate tok/s (-21.75%). Both arms use
the same candidate CLI with default sharing and unchanged original weights;
only the native dispatch differs. All 128 completions reach 128 tokens.
The dispatch experiment is removed from the active queue. Its patch, native
bundle, commands and raw results remain under `target/routed-density-evidence/`.
A second experiment retains float operands in Q4_K/Q5_K routed matrix tiles.
It still flips the same prediction with both the tensor and SIMD paths, and
is also removed from the active queue. Its patch, bundle and reports remain
under `target/routed-matrix-precision-evidence/`. No speed claim is made for
that experiment. The restored original kernels pass 24 targeted CPU-reference
checks across Q4_K, Q5_K and Q8_0, broadcast and per-expert inputs, and 32–256
rows. Eight existing sparse-routing cases with 512/1,024 experts also pass.

The retained configuration uses the original native kernels. A separate
public-API quality diagnostic teacher-forces the same 32,640 corpus targets
in four execution shapes. All modes use the same eight sequence slots,
identical independent 256-token prefills, 128 reset contexts, unchanged
original weights and f16 KV. It reads raw causal logits without draft capture,
proposals, rollback or sampling RNG; it measures target arithmetic rather
than complete speculative serving.

| Execution shape | Perplexity | Top-token agreement with serial |
| --- | ---: | ---: |
| Independent one-row steps | 6.651468 | 100% |
| Eight lanes, one row each | 6.651067 | 99.942% |
| Eight lanes, four rows each | 6.648762 | 98.447% |
| Eight lanes, seven rows each | 6.649120 | 98.456% |

Mean NLL differences versus serial are -0.000407 and -0.000353 for the two
large shapes, with standard errors across context means of 0.000390 and
0.000397. There is no measurable perplexity loss on this corpus, and no
established improvement either. Contexts are not guaranteed statistically
independent. This supports retaining normal native batching, while explicitly
not promising identical greedy text across batch geometries or general task
quality equivalence. The standalone scorer, exact position/token audit and
per-position scores are retained with the default-sharing evidence.

Real HTTP cancellation checks close four streams after first content for
each of DFlash1 and DFlash2. In every round the peer completes unchanged and
both reused lanes match their pre-cancellation outputs. Explicit zero switches
produce no shared forwards and reproduce all eight previous-default outputs.
A valid 16-row microbatch test with eight lanes completes all 16 control and
candidate requests at 128 tokens, with exact paired outputs; native capacity
limits its shared verification waves to two lanes in this workload.

Probes with one- and two-row microbatches expose an inherited recurrent-tail
assertion during control startup, before either shared path runs. Those failed
logs are preserved separately; this change does not repair tiny-microbatch
startup validation.

The full Skippy/Mesh release build passes, as do all 1,354 serving library
tests (including the complete crate/doc-test command), Cargo checks and warning-denying all-target Clippy for serving and
Mesh, formatting and the console-output contract. The two new policy tests
exercise unset/empty/enabled values and explicit zero without mutating the
process environment. Historical sections above describe the opt-in policy
at the time of those measurements. The inherited native test-target gate
failure remains separately recorded; this change does not repair that gate.

Commands, saved CLI hashes, raw responses, throughput audits and validation
logs remain under `target/default-batching-evidence/`. These numbers qualify
the tested original-weight workload, not parity with BaseRT or a universal
speedup for every model and backend.

## Post-PR #33 dense-weight precision diagnostic

This is a model-format experiment, not an equivalent-output engine change.
The same `193d8a83b` release CLI and qualified ABI 0.1.85 runtime served both
arms. The original target/draft revisions are unchanged. The trial requantizes
250 main-target, two-dimensional Q8 tensors to Q4_K, without an importance
matrix, using the existing native quantizer. It starts from the current GGUF,
not FP16. Exact per-tensor type overrides preserve the other 503 tensor types;
a SHA256 range audit additionally proved their payloads are byte-identical.
Experts, routing gates, embeddings, output head and MTP sidecar are preserved.
The original file is untouched. The trial SHA256 is
`086f79da132cf18f91440bebd34c8c50e9a242ba5b2e0d540e7d655e5a752cfe`.

Total file bytes fall from 22,663,387,424 to 21,960,841,504 (3.10%). The
changed tensors are active dense projections, so the earlier stored-active-
weight estimate falls from about 2.626 to 1.924 GB per decoded token, a much
larger reduction than total file size suggests. This remains an inventory
estimate, not measured DRAM traffic.

Both plain and DFlash experiments used fresh control/candidate/candidate/
control processes, PP512/TG256, two warmups and three measured batches at each
concurrency. Each table value is the median of six measured batches. Sampling,
context, cache and batch settings match the combined comparison above;
plain decoding disables speculation on both arms. The speculative comparison
uses the same six-token DFlash draft and enables shared draft/verification on
both arms. Only the target precision changes.

| Workload | Original GGUF tok/s | Dense-Q4 trial tok/s | Change |
| --- | ---: | ---: | ---: |
| Plain, one request | 113.612 | 132.829 | +16.91% |
| Plain, two requests aggregate | 160.502 | 178.739 | +11.36% |
| DFlash, one request | 125.400 | 133.353 | +6.34% |
| DFlash, two requests aggregate | 175.272 | 200.572 | +14.43% |

Each experiment completed all 36 measured requests without reported errors.
Ordinary EOS remained enabled: one plain trial request produced 195 tokens;
the other 35 produced 256. The speculative experiment had one control output
of 181 tokens and trial outputs of 236, 172 and 231 tokens; the other 32
produced 256. Prompt strings differ between benchmark batches. The speculative
single-request distributions overlap substantially. These numbers do not
establish an equal-quality comparison with BaseRT.

Two additional fixed 64-token prompts ran in every process. For each weight
variant separately, plain and speculative outputs matched byte for byte
across both fresh processes. Outputs differ between the original and trial
weights, so this is not exact-output parity across quantizations.

The quality diagnostic follows the native
[perplexity/KL method](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity)
on [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) test text,
from the ggml-org CI archive at revision
`927b3642933080f1b0e811e2f916e14c292992f9`. With 128 independent 512-token
chunks it scores the final 255 predictions per chunk: 32,640 positions.
The initial 16-chunk pilot scores 4,080 positions. All weights are on Metal;
context, batches and microbatches are 512, with eight host threads.

The larger trial reports PPL 6.698454 versus direct original PPL 6.6468.
The paired comparison to stored original logits reports PPL ratio
1.008072 +/- 0.001831, mean KL 0.038167 +/- 0.000567, and matching most-likely
tokens at 91.624 +/- 0.153 percent. Thus the trial has a measurable roughly
0.8% perplexity penalty and changes the most likely token at roughly 8.4%
of positions. The stored-reference format compresses logits: an original-
versus-itself 16-chunk check reports KL approximately zero and 100% most-
likely-token agreement, with a small PPL roundoff offset. The reference model
is the existing mixed GGUF, not an FP16 accuracy reference or BaseRT model.
This test does not cover instruction following, coding accuracy or long context.

The original-weight build additionally passed an eight-lane teacher-forced
verification check: all 512 predictions matched the serial target, over ten
shared forwards at width seven, with DFlash feature capture attached. This
isolates shared-forward prediction correctness; it deliberately does not
exercise draft proposals or rollback, and is not a full serving qualification.
The command and report are retained as `shared-reference-commands.sh` and
`shared-reference-8.json` in the same evidence directory.

The trial remains a local experimental artifact. It is not selected by default,
and no production quantization or serving setting changes in this experiment.
The results show that precision accounts for part of the practical gap, but
reducing dense precision alone does not reach the prior BaseRT rates. Raw
recipes, commands, source/candidate tensor audits, hashes, logits, quality
reports and both ABBA comparisons are in `target/mixed-precision-evidence/`.

## Public-API quality comparison with BaseRT

A subsequent diagnostic scores the installed BaseRT 0.5.0 engine through its
published C API against the same WikiText-2 inputs as the precision study.
It loads the existing `basecompute/Qwen3.6-35B-A3B` default-Q4 artifact, SHA256
`7171b37ffccdf16dfd920f55be766e6faffa31f64e04ca9cd0064943bf72881b`.
The engine reports ABI 1, host-logit and sequence capabilities, and effective
f16 KV. Its tokenizer matches all 65,536 input token IDs in the existing
reference file exactly, and its add-BOS policy is false.

Each of 128 fresh sequences prefills the first 256 tokens of a 512-token
context. Serial teacher forcing then feeds positions 256 through 510 and
scores the supplied next tokens at positions 257 through 511: 255 predictions
per context, 32,640 total. Logit readback uses the engine-reported row stride;
the public selected-token log-probability helper supplies the score. No engine
implementation or private weight format is examined. BaseRT's default
acceleration policy is retained, including its reported Neural Engine prefill.

| Artifact and execution | Perplexity, lower is better |
| --- | ---: |
| Skippy original mixed GGUF, direct native evaluation | 6.6468 |
| Skippy dense-Q4 diagnostic, direct native evaluation | 6.698454 |
| BaseRT default Q4, serial teacher forcing | 6.916309 |

The paired comparison against the compressed original reference gives mean
NLL difference 0.0400453, or a 1.040858 perplexity ratio for BaseRT/original.
The standard error across 128 context means is 0.003081 NLL; this treats
contexts as independent, which is not guaranteed for adjacent corpus text.
The compressed reference has perplexity 6.644816 rather than the direct
6.6468 because its uint16 affine encoding clips low tails; eight selected
tokens hit its minimum quantized value. Both numbers are retained so the
compression effect is visible.

On the first context, repeating serial scoring produces exactly the same
selected-token log probabilities. A separate 31-row public-API submission,
with the documented plain-step commits between feeds, differs from serial
by up to 0.46579 in selected-token log probability and changes two of 255
top predictions. Consequently the table uses serial scoring throughout.
It does not silently substitute multi-row verification math for decode.

This result supports investigating lower-traffic Skippy formats: the existing
dense-Q4 diagnostic still has lower perplexity than this BaseRT artifact on
this corpus. It does not establish general quality equivalence, identify the
cause of the difference, or establish performance parity. The original
precision-loss measurements remain valid relative to Skippy's original
weights. No format is selected as the default by this experiment.

The standalone C++ public-API client, its Just build recipe, exact token and
position audit, engine/header/model hashes, per-position scores and complete
logs are retained under `target/basert-quality-evidence/`. This diagnostic
makes no speed claim.

## Output-head precision follow-up

After the public-API quality comparison, a separate model diagnostic changes
the output head from Q6_K to Q4_K in addition to the previous 250 dense-Q4
projections. The native quantizer refuses routing-matrix quantization even
when explicitly requested; the tensor audit caught this, and no router-Q8
result is claimed. The actual artifact retains every routing matrix in F32.
All 502 tensors outside the 251 planned changes are byte-identical to the
original source. Experts, embeddings and the unused MTP sidecar remain
unchanged. The new file is 21,829,728,544 bytes, with SHA256
`f3563b1a68af6efab240ce31d3cc213d80f5ea8dd596cf8d06862853ab72fab1`.
The original model files are unchanged.

On the same 32,640 WikiText-2 predictions, the new artifact has perplexity
6.732749, versus 6.698454 for dense-Q4 alone and 6.6468 for the original direct
evaluation. Against the compressed original reference, its perplexity ratio
is 1.013233 ± 0.001954, mean KL is 0.044716 ± 0.000581, and top-token
agreement is 90.098 ± 0.165%. It remains below the BaseRT artifact's 6.916309
perplexity on this corpus, while worsening quality relative to the original
Skippy weights. This is an optional format experiment, not a default change.

A fresh plain-decode ABBA comparison uses the same final CLI and frozen
ABI 0.1.85 native bundle for both arms. Control is dense-Q4 alone; candidate
adds the Q4 output head. The pinned PP512/TG256 benchmark uses two warmups,
three measured batches per concurrency per process, greedy sampling, f16 KV,
and disabled prefix/disk caches. Across six measured single-request batches
per arm, medians are 133.518 versus 136.780 tok/s (+2.44%); all six candidate
values exceed all six controls. The two-request medians are 170.126 versus
180.398 aggregate tok/s, but control process rates differ substantially
(183–186 versus 154–157 tok/s), so their pooled +6.04% is not a qualified
two-request gain. All 36 measured requests succeed; two candidate requests
stop at 212 and 253 tokens and the rest reach 256. Output text differs across
weight formats, as expected from the quality measurements.

A separate fixed two-prompt ABBA comparison removes the early-EOS and varying
text confounders. Each process warms one batch and measures three, with 128
tokens per request. All 32 requests reach 128 tokens, and each arm reproduces
its own output text exactly across measured repeats. The text differs between
weight formats. Median end-to-end aggregate throughput, including prefill
and HTTP overhead, is 191.781 versus 195.596 tok/s (+1.99%); all six candidate
batch rates exceed all six controls. This qualifies a small gain on these
fixed prompts, rather than the pooled long benchmark's apparent 6% gain.

Recipes, the rejected routing request, actual tensor audit, model hash,
quality evaluation, raw requests and benchmark audit are retained under
`target/head-router-precision-evidence/`. Reducing output precision supplies
only a small additional single-request gain and does not close the BaseRT
speed gap. Further work must measure execution cost as well as stored bytes.

## Multi-row routed vector activation reuse

A separate native capability reuses activation registers across four Q4_K
output rows or two Q5_K output rows inside the existing routed vector kernels.
The baseline handles two and one rows respectively. The arithmetic for each
output row and stored weights are unchanged. This is a new core patch because
it owns a distinct routed-vector specialization, rather than extending the
shared-verification or graph-fusion capabilities.

Dispatch is restricted to M5 Max with shared buffers, 3–31 routed token rows,
inner dimensions 512–4096, and at least 512 output rows. Output rows must divide
exactly into complete unrolled SIMDgroup tiles: the kernel bodies load every
unrolled row, so a write-boundary check alone cannot make partial tiles safe.
One- and two-row operations retain the baseline dispatch, as do other devices,
private buffers, types, and unqualified shapes. Selection uses device and tensor
metadata, without model-family checks or a new operator setting. There is no
public native ABI change.

The initial policy also selected the specializations for one and two rows.
Repeated-ID operator microbenchmarks improved, but the full-model plain
benchmark slowed by 0.79% at concurrency one and 0.57% at concurrency two.
That policy was rejected. These microbenchmarks repeatedly reuse a small set
of expert IDs; their printed byte-throughput estimates do not represent real
model bandwidth and are not used as a speedup claim.

The restricted policy was compared with original native source `9403301c7`
using the same CLI, original target weights, original DFlash draft, and default
shared execution. Four fresh processes ran in ABBA order. Each process ran
one warmup and three measured fixed-prompt batches per concurrency, with
128 output tokens per request. Rates below divide aggregate output tokens by
the slowest HTTP request time and include prefill.

| Concurrency | Baseline median tok/s | Restricted reuse median tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 171.409 | 175.803 | +2.56% |
| 2 | 216.403 | 223.957 | +3.49% |

All 48 requests completed with 128 tokens and length termination. Outputs
matched exactly across processes and arms for both prompts. Every measured
two-request candidate batch exceeded every control batch. Single-request
batches overlap, with adaptation still visible after the first warmup; this
is evidence for this workload, not a universal steady-state gain or BaseRT
parity. Raw operator tests, commands, responses and audits are retained under
`target/routed-vector-evidence/`, with the restricted full-model comparison
in `fixed-spec-multirow-summary.json`.

The restricted policy's variable-prompt plain benchmark was +0.19% at
concurrency one and −2.07% at concurrency two. One candidate request ended
at 210 tokens; the other 35 measured requests produced 256. The two candidate
processes also differed noticeably at concurrency two. This result is not
counted as a plain-decode improvement or dismissed as a passing regression
check. A subsequent fixed-prompt, fixed-128-token ABBA comparison on the final
release build measured:

| Plain concurrency | Baseline median tok/s | Restricted reuse median tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 110.841 | 111.194 | +0.32% |
| 2 | 166.751 | 167.054 | +0.18% |

All 48 requests completed at the expected length, and measured outputs
matched exactly across both arms. Batches overlap: these figures support
preserving ordinary decode performance on this fixed workload, not a plain
speedup. Raw evidence is in `plain-multirow-summary.json` and
`fixed-plain-multirow-summary.json`.

Validation includes 18 routed GPU-versus-CPU operator cases: Q4_K 2048×512,
Q5_K 512×2048, and Q4_K 2048×768, each at 1, 2, 3, 8, 24, and 32 token rows.
The graph optimizer test also passed with 14 routed selection/fallback
assertions, including complete-tile alignment, device and buffer exclusions,
and dimension/token boundaries. A fresh queue replay reproduced the intended
source hashes. The complete `just with-lld just release-build` passed for
Skippy and Mesh. The final native source differs from the measured speculative
candidate only in stronger test assertions; production source is identical.
This selected validation does not make the previously failing broad native
build gate green: its inherited missing ASR/diffusion targets remain unresolved.

## Rejected flat-Q4 dense/head format

The next format diagnostic uses Q4_0 for the same 251 dense/head matrices as
the Q4_K dense/head trial. Both formats store 4.5 bits per weight at these
aligned shapes, and both complete files are exactly 21,829,728,544 bytes.
The flat format removes packed scale/minimum decoding at the cost of a less
expressive quantizer. All 502 other tensors remain byte-identical to the
original source, including experts, routers, embeddings and the MTP sidecar.
Original model files and serving defaults are unchanged.

On 32,640 scored WikiText-2 positions, flat Q4 has perplexity 6.873154 versus
6.732749 for the preceding Q4_K dense/head trial. Against the compressed
original reference, its perplexity ratio is 1.034363 ± 0.002194, mean KL is
0.056068 ± 0.000625, and top-token agreement is 88.824 ± 0.174%. Its perplexity
is slightly below the previously tested BaseRT artifact's 6.916309 on this
corpus; this does not establish general or statistically demonstrated quality
superiority.

A fresh plain-decode ABBA comparison uses the same CLI and frozen native
runtime for both formats, the pinned PP512/TG256 benchmark, two warmups and
three measured batches per concurrency per process, greedy sampling, f16 KV,
and disabled prefix/disk caches. Medians cover six measured batches per arm.

| Workload | Q4_K dense/head tok/s | Q4_0 dense/head tok/s | Change |
| --- | ---: | ---: | ---: |
| One request | 136.418 | 137.347 | +0.68% |
| Two requests, aggregate | 188.036 | 178.992 | -4.81% |

All 36 measured requests succeed; one flat-Q4 request stops at 253 tokens
and the other 35 reach 256. Every two-request control batch exceeds every
candidate batch. The small single-request difference does not compensate for
the two-request slowdown and worse quality. The format is rejected rather
than supplied as a performance profile. Recipes, tensor audits, quality logs,
raw responses and throughput audit remain under `target/flat-q4-evidence/`.

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
