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
was slower in the initial diagnostic. The later target-format-qualified
comparison below supersedes that diagnostic for the original target. These
measurements do not establish BaseRT parity or
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

Dispatch is restricted to M5 Max in shared-buffer backend mode, 3–31 routed token rows,
inner dimensions 512–4096, and at least 512 output rows. Output rows must divide
exactly into complete unrolled SIMDgroup tiles: the kernel bodies load every
unrolled row, so a write-boundary check alone cannot make partial tiles safe.
One- and two-row operations retain the baseline dispatch, as do other devices,
private-buffer backend mode, types, and unqualified shapes. Selection uses device and tensor
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

## Convolution channel dispatch

A new GPU counter trace of the retained native runtime and routed-down recipe
covers a warmed request loop at trace-relative 13.120687–25.056605 seconds.
Shader compilation ended at 10.777350 seconds. Submission IDs map uniquely to
processes; 1,404,052 scoped shader intervals are attributed to Skippy, with
1,536 Skippy intervals unmapped. Latest preceding compiled PC ranges are a
lifetime heuristic, not proof of shader allocation lifetime. Per-kernel union
intervals overlap and are not additive wall-time fractions. The loop includes
brief prefill for each 128-token request, and instrumentation is invasive; this
trace is a bottleneck lead, not a throughput result. Raw exports and attribution
scripts remain in `target/steady-gpu-evidence/`. A separate CPU stack sample
in `target/steady-cpu-evidence/` frequently finds the scheduler waiting for
Metal completion; it does not by itself locate the GPU cost.

The trace highlights the existing one-token SSM convolution path. Its shader
computes one channel per threadgroup with a single thread. The new geometry
places 32 independent channels in each threadgroup, retaining each channel's
arithmetic and guarding partial channel groups. Both scalar and float4
convolution kernels use the mapping. Batched-token convolution and SiLU fusion
keep their existing behavior. Selection uses the actual one-token operation
shape, without a model-family or chip allowlist. There is no public ABI change.
This is a separate core capability: earlier patches do not own this upstream
convolution dispatch implementation.

The CPU-reference Metal suite passed 396 convolution cases, including 216 new
channel-grid cases. Those cover 1/7/31/32/33/127/129/8192/12288 channels,
3/4/5/8-wide filters, one or three sequences, plain convolution, SiLU fusion,
and bias-plus-SiLU graphs. New cases live in the narrowly scoped `test-backend-ssm-conv-channels.h`;
the oversized backend operator test file calls that generator and retains its
existing common test framework.

Fresh-process ABBA comparisons used identical CLI, model weights, settings,
and fixed 128-token requests within each pair. Each process ran a warmup plus
three measured batches per concurrency. Rates include prefill and divide total
output tokens by the slowest HTTP request duration:

| Weights | Concurrency | Baseline median tok/s | Packed channels median tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Original | 1 | 110.284 | 112.388 | +1.91% |
| Original | 2 | 166.161 | 170.678 | +2.72% |
| Routed-down Q4_K recipe | 1 | 139.934 | 143.960 | +2.88% |
| Routed-down Q4_K recipe | 2 | 211.510 | 217.138 | +2.66% |

All 48 requests in each comparison completed at 128 tokens. Every measured
candidate batch exceeded every control at each concurrency and weight format.
Measured outputs matched exactly across arms. A separate serial teacher-forcing
check used eight fixed 512-token corpus contexts with identical 256-token
prefills and 255 one-row decode steps per context. For each weight format,
all 2,040 scored log probabilities and top-token IDs matched byte-for-byte
between control and candidate. This checks actual recurrent decode, whereas
a single 512-row perplexity forward would use the unchanged batched path.
It does not establish universal bitwise equivalence or general accuracy.

A final-runtime speculative comparison on the original weights completed all
48 requests at 128 tokens with identical measured outputs. Medians were
174.724→175.877 tok/s at concurrency one (+0.66%) and
223.072→224.569 aggregate tok/s at concurrency two (+0.67%). Batches overlap
and adaptation is visible, so no additional speculative speedup is established.
The retained performance claim is the plain-decode improvement above. Raw
evidence is in `fixed-spec-original-summary.json`.

A genuine fresh M4 checkout exposed stale context in core patch 18's CMake
hunk: it still named `resident_checkpoint_gpu.cpp` after the owning build
patch had moved that test source to `resident_attention.cpp`. Reconstructing
that hunk against its actual predecessor fixes the owning patch. All 140
patches then replayed on a fresh upstream-only checkout, producing final tree
`6c19b4bdb441a8776f07b5daae718187e5e05db7`, exactly matching the measured
candidate. The repair changes queue reproducibility, not runtime semantics.
The local Skippy and Mesh release build passed for that tree. The inherited
broad native gate's missing ASR/diffusion targets are a separate unresolved
issue; this selected validation does not make that gate green.

Jianyang job `job_4s332ewhhs3wdj9` checked exact local commit
`fb775b9dc5fa1462c4d2783170218a9d1d34b8be` on Dan's M4 mini
(`macmini-764jw9`, macOS 26.5.1). Its fresh checkout replayed the repaired
queue, built the backend operator test natively, and passed all 396 convolution
cases on Metal against the CPU reference. The job succeeded at $0. This is
additional chip correctness coverage, not an M4 throughput measurement. Source
was fetched through a temporary read-only private-LAN Git listener, which was
stopped after checkout; no GitHub push or binary transfer was involved. The
final local packaged native source is `b60068a0e39b2d1a3fad15fe8c2a4c702009ff2e`,
with the same measured source tree and ABI 0.1.85.

Raw tests, request timings, scores, replay proofs and build logs remain in
`target/ssm-conv-channels-evidence/`. These measurements establish a local
convolution-dispatch gain, not parity with BaseRT.

## Rejected dense Q4_K vector row-width probes

After retaining the routed-down Q4_K recipe, two dense-vector variants tested
whether activation register reuse or reduced register pressure improved the
remaining dense projections and output head. Both use the existing Q4_K
arithmetic with a different compile-time output-row count. The production
baseline handles two rows per SIMDgroup. A private policy limited the probes
to M5 Max in shared-buffer mode, inner dimensions 512–4096 and at least 512
output rows, with complete row tiles. Neither probe is retained in the queue.

Both variants passed 11 GPU-versus-CPU operator cases: 512×2048, 2048×4096,
and 4096×2048 at 1/2/8 token rows, plus the 2048×248320 output head at 1/2
rows. Dispatch selection/fallback tests passed. Four fresh processes per
variant ran the same fixed prompts in ABBA order, with one warmup and three
measured batches per concurrency. All 48 requests per comparison completed
at 128 tokens; every warm and measured output matched across both arms.

| Dense rows per SIMDgroup | C1 baseline → probe tok/s | Change | C2 baseline → probe aggregate tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 | 139.746 → 136.749 | −2.14% | 212.513 → 208.881 | −1.71% |
| 1 | 139.358 → 139.815 | +0.33% | 211.913 → 205.875 | −2.85% |

The four-row candidate is consistently slower. The one-row single-request
result overlaps the controls, while every measured two-request candidate
batch is slower than every control. The baseline two-row specialization
remains. No gain is attributed to either probe. Rejected patches, bundles,
raw requests, numerical tests, timing audits and the restoration log remain
under `target/dense-vector-evidence/`.

Removing the temporary patch and replaying the retained queue restored native
source `04f7cff4e6a9600d31a78f484e53929f92b06b8f`, primary-library SHA-256
`9b3b6350ecea9ec5c0626e8e47f9bc9839c4a1d5efdc288cc0c8b4e41bc201e4`, and
Metal-library SHA-256
`39e00dc70cb8ec810bd58ba07f9163293f3cd614f7d18964cd8b06417c23be71`, matching
the frozen control. The retained precision recipe and routed-vector policy
are unchanged.

## Routed-down Q4_K performance recipe

A further format experiment retains the earlier dense/head Q4_K changes and
changes the 37 decoder expert-down Q5_K tensors to Q4_K. The three source
Q6_K expert-down layers (34, 38, 39) and unused block 40 MTP sidecar retain their
source representations. The audit confirms exactly 288 changed tensors and
465 byte-identical preserved tensors. The resulting artifact is 20,588,214,560
bytes, versus 21,829,728,544 for the dense/head-only variant. SHA-256 is
`2becfbe82ceb5862f90145c69b8545dcf30982aec2b81ce52ddeaec32648ecf9`.

At the same 128×512-token WikiText-2 contexts and 32,640 scored positions,
perplexity is 6.733014, versus dense/head-only 6.732749 and original compressed
reference 6.644816. The ratio against the original is 1.013273 ± 0.002024;
mean KL is 0.047777 ± 0.000636 and top-token agreement 89.939 ± 0.167%.
The small perplexity point-estimate change does not establish general quality
equivalence. KL is slightly higher and top-token agreement slightly lower than
the dense/head-only diagnostic.

An initial variable-prompt plain ABBA comparison had substantial process
variation: single-request controls ranged from about 117 to 137 tok/s, and five
of 36 measured requests ended before 256 tokens. Its nominal +12.71%/+21.48%
medians are not retained as qualified speed claims. A subsequent fixed-prompt,
128-output-token plain comparison used the same final CLI and native runtime
in both arms, changing only those 37 expert-down tensor representations:

| Concurrency | Dense/head Q4_K median tok/s | With routed-down Q4_K median tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 132.742 | 140.391 | +5.76% |
| 2 | 196.339 | 212.506 | +8.23% |

Four fresh processes ran ABBA, with one warmup and three measured batches per
concurrency. Rates divide total output tokens by the slowest HTTP duration and
include prefill. All 48 requests produced 128 tokens. Every candidate measured
batch exceeded every control at both concurrencies. Outputs repeat within each
format but differ between formats. These results do not measure speculative
decode or establish parity with BaseRT.

A subsequent fixed-prompt DFlash comparison completed all 48 requests at 128
tokens. Nominal medians were +8.01% at concurrency one and +5.23% at concurrency
two, but controls varied substantially and one candidate single-request prompt
had two generated variants as batch shapes changed. These are unqualified
speculative results, not an additional established gain or an exact-output
guarantee. Raw evidence is in `fixed-spec-summary.json`.

The audited, source-specific recipe is retained at
`skippy/scripts/recipes/quantization/qwen3.6-35b-a3b-ud-q4-k-m-decode.tensor-types.txt`.
It is explicit and opt-in; it does not change loading defaults. Source pin,
reproduction commands and quality limitations are in the adjacent recipe
README. The product validator accepted all 753 entries, and a complete product
quantization run produced the same full-file SHA-256 as the measured artifact.
Raw plans, tensor audits, quality logs, commands, responses and timing
audits remain under `target/routed-down-precision-evidence/`.

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

## Target-qualified DFlash 2 draft choice

With retained native `b60068a0e39b2d1a3fad15fe8c2a4c702009ff2e`, the same
release CLI and original target weights in both arms, replacing the original
Q8 DFlash draft with Q8 DFlash 2 at a six-proposal ceiling improved fixed
128-token HTTP completion throughput:

| Target | Concurrency | DFlash 1 | DFlash 2 | Change |
| --- | --- | --- | --- | --- |
| Original UD-Q4_K_M | 1 | 176.194 | 192.354 | +9.17% |
| Original UD-Q4_K_M | 2 | 220.023 | 236.669 | +7.57% |
| Dense/head/down Q4 recipe | 1 | 185.402 | 184.474 | -0.50% |
| Dense/head/down Q4 recipe | 2 | 245.003 | 243.657 | -0.55% |

Each target format used fresh-process ABBA order, three warmups and five
measured batches per concurrency per process. All 96 requests per format
completed 128 tokens; measured output text matched between drafts. Every
original-target DFlash 2 measured batch exceeded every corresponding control.
The broader Q4 recipe has overlapping results and no established draft-choice
benefit. All arms used default adaptive DFlash sizing and shared execution,
f16 KV, two lanes, ctx8192, batch2048, microbatch512, disabled prefix/disk
caches, greedy seed42, and the same two short coding prompts. Rates include
prefill and HTTP overhead; concurrency two is aggregate tok/s.

For the original target at concurrency one, measured acceptance increased
from 0.505 to 0.607 and verification windows fell from 33 to 28 per request.
Drafting increased from about 87 to 102 ms, while verification fell from about
588 to 512 ms. These are per-request telemetry intervals, not additive GPU
execution totals. The generic `spec.adaptive_enabled` telemetry field does
not describe DFlash's separate adaptive controller.

A separate original-target, DFlash-2-only ABBA comparison raised the proposal
ceiling from six to its trained default seven: 192.924 to 195.167 tok/s
(+1.16%) at concurrency one, and 238.626 to 243.563 (+2.07%) at concurrency
two. All 96 requests completed 128 tokens and measured text matched. The
single-request candidate range clears every control; concurrency-two ranges
overlap. The draft's block size is eight with one anchor position. These
separate draft-choice and ceiling gains must not be multiplied into a claimed
combined measurement.

The opt-in serving preset is
`skippy/scripts/recipes/serving/qwen3.6-35b-a3b-dflash2.toml`; its README pins
the measured draft and target artifacts. Commands, raw responses, statistics,
and the pinned upstream draft card are in `target/draft-choice-evidence/`.
This establishes a configuration improvement for the tested target and
workload, not BaseRT parity or a global draft default.

## Rejected small-batch convolution extension

Extending the retained one-token channel packing to batched convolution
packed channels into threadgroup y while retaining token positions in x.
The CPU-reference Metal suite passed 2,340 cases across token counts
1/2/3/4/7/8/15/16/17/33, including channel tails and fused bias/SiLU.
However, fresh-process ABBA measurements did not establish a serving benefit.
Original-target DFlash 1 changed by +0.73% / +0.49% at concurrency one/two;
plain broad-recipe decode changed by -0.41% / -0.31%, with overlapping
process variation. A short concurrency-eight comparison gave +1.24%, but a
longer three-warmup/five-measured repeat gave 271.321 to 270.753 aggregate
tok/s (-0.21%). All measured outputs matched the respective controls.

The extension was rejected. Owning patch 0043 and the packaged runtime were
restored to retained native B600; libllama and libggml-metal hashes matched
the frozen control byte-for-byte. The rejected owner patch, logs, responses,
and audits remain in `target/ssm-conv-small-batch-evidence/`. The retained
one-token optimization remains unchanged.

## Expert-down-only precision probe

A narrower candidate converts only the original target's 37 Q5_K routed
expert-down tensors to Q4_K. All 716 other tensor payloads, names, and shapes
are preserved byte-for-byte, including dense weights, output head, routers,
three Q6_K expert-down tensors, and MTP. The product quantizer validated the
753 literal tensor overrides and produced a 21,421,873,440-byte GGUF, SHA-256
`4ad842f4ed3ef1070d7818d95163e47402e7de17c79ecb85a0e272aab3ae52f5`.
This saves 1,241,513,984 bytes versus the original compressed target.

On 128 WikiText-2 raw test chunks at context512, with 32,640 scored positions,
perplexity was 6.657436 versus the original compressed target's 6.644816.
The paired PPL ratio was 1.001899 ± 0.000820, mean KL divergence
0.008046 ± 0.000157, and top-token agreement 96.317 ± 0.104%. Uncertainties
are the tool's standard errors. The reference is the original compressed
GGUF, not BF16. This is a small measured corpus-specific quality change,
not proof of equivalent model quality. Source audit, complete recipe,
quantization commands, artifact digest, and quality logs are retained in
`target/expert-down-only-evidence/`. Performance qualification follows below;
no default precision setting is changed by this probe.

### Complete-configuration comparison

A fresh-process ABCCBA comparison held CLI/native runtime constant and tested
A: original target + DFlash 1 cap6; B: original target + DFlash 2 cap7; and
C: expert-down-only Q4 target + DFlash 2 cap7. Each process ran three warmups
and five measured batches at each of concurrency one/two, totaling 144
requests, 90 measured. All completed 128 tokens with length finish reasons.

| Configuration | Concurrency 1 tok/s | Concurrency 2 aggregate tok/s |
| --- | --- | --- |
| A: original + DFlash 1 | 175.838 | 219.546 |
| B: original + DFlash 2 | 194.153 | 230.376 |
| C: expert-down Q4 + DFlash 2 | 225.408 | 239.088 |

This directly measured B/A at +10.42% and C/A at +28.19% for one request;
C/B was +16.10%. Every C single-request batch exceeded every B batch, and
every B batch exceeded every A batch. The two-request medians changed by
+4.93% B/A, +8.90% C/A, and +3.78% C/B, but their ranges overlap materially
and include slow batches. Those two-request point estimates do not establish
an isolated precision gain. A and B produced identical measured text; C was
stable within each prompt but differed from A/B, as expected for changed
weights. These are complete configuration measurements, not identical-output
kernel comparisons.

At concurrency one, C reduced DFlash 2 verification time from approximately
509–514 to 426–429 ms per request, with 25 versus 26 verification windows
and acceptance 0.611 versus 0.574. Changed token paths and acceptance can
contribute to the gain alongside faster expert-down arithmetic. A wider
prompt qualification is required before treating the single-prompt gain as
representative. Raw evidence is `target/draft-choice-evidence/combined*`.

The new serving TOML passed the product's `--print-effective-config` parser
with file-attributed DFlash strategy, native MTP false, and ceiling seven.
A normal product server using that TOML loaded the pinned original target and
draft and returned a successful 128-token length-terminated completion.
Evidence is `target/draft-choice-evidence/preset-effective-config*` and
`preset-smoke/`.

### Wider expert-down qualification

The same original + DFlash 2 cap7 versus expert-down Q4 + DFlash 2 cap7
comparison was repeated across eight short coding prompts, concurrency one,
fresh-process ABBA, one full warmup cycle plus three measured cycles/process.
All 128 requests completed 128 tokens; 96 were measured. Dividing measured
tokens by the sum of HTTP request durations gave 186.864 versus 197.225 tok/s
(+5.54%). Six prompt medians improved and two regressed, with changes spanning
-15.31% to +17.45%. Text was stable across measured repeats within each
prompt/target but differs between targets. The first merge-slices prompt
repeated its approximately +16% precision benefit; binary search regressed
15.31%, demonstrating why that first-prompt result is not representative.

The complete recipe is retained as an opt-in source-specific artifact at
`skippy/scripts/recipes/quantization/qwen3.6-35b-a3b-ud-q4-k-m-expert-down.tensor-types.txt`.
No model-loading or precision default changes. Commands, requests, responses,
timings, and audit are retained in `target/draft-choice-evidence/wide*`.

After restoring the retained native source, the static backend tests were
rebuilt through Just and all 396 retained convolution CPU-reference cases
passed again on Metal. This removes the stale expanded test binary left by
the rejected extension. The full native gate's inherited missing ASR and
diffusion targets remain unresolved; this focused result does not replace it.

## Fresh public-API engine configuration comparison

A balanced ABCDDCBA comparison used the same eight raw coding prompts,
128-token requests, greedy sampling, f16 KV, context8192, two resident lanes,
and concurrency one. Each fresh process ran one eight-prompt warmup cycle and
three measured cycles. All 256 requests completed 128 tokens with length
finish reasons; all reported cached prompt tokens were zero. Prompt token
counts matched across all four configurations. Measured rates divide total
completion tokens by summed HTTP elapsed time, including prefill.

| Configuration | HTTP tok/s | First / second process tok/s |
| --- | --- | --- |
| BaseRT 0.5.0, canonical Q4, speculation off | 197.402 | 199.747 / 195.111 |
| BaseRT 0.5.0, canonical Q4, DFlash 2 cap7 | 320.994 | 322.182 / 319.814 |
| Skippy original target, DFlash 2 cap7 | 186.912 | 186.721 / 187.102 |
| Skippy expert-down Q4, DFlash 2 cap7 | 197.362 | 197.153 / 197.572 |

The expert-down improvement reproduces across the wider prompt set, but
matching BaseRT plain performance does not establish parity with BaseRT's
speculative configuration. That configuration is approximately 62.6% faster
than the tested Skippy expert-down configuration on this workload. Different
target quantizations and token paths prevent an engine-only efficiency or
equal-quality conclusion. The earlier plain-BaseRT comparison remains
historical; this comparison includes an enabled DFlash 2 configuration.

Only public CLI/API behavior was used. Superfluid's raw completions endpoint
accepted the same prompt bytes. Its public metrics snapshot showed 4,465
speculative tokens proposed and 3,010 accepted, with zero warm-prefix tokens,
confirming speculation was active during the first DFlash process. Defaults
for BaseRT's adaptive sizing and throughput gate were retained. Both engines
used a seven-proposal ceiling; their controllers and kernels differ.

The BaseRT drafter was converted with public `basert convert` 0.5.0 from
`incoai/Qwen3.6-35B-A3B-DFlash2`, full revision
`51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, using `--target base-q8
--quantize-drafter`. The converter retained 103 tensors and wrote a 636 MiB
`.base` sidecar. Direct GGUF loading was rejected by the public loader; the
public converter also rejected DFlash GGUF input, so conversion used the
pinned upstream safetensors source. Skippy used the previously pinned Q8_0
GGUF of that draft. These converter-specific Q8 representations are not
byte-identical. The BaseRT target remains its canonical Q4 artifact from the
public-API quality comparison above; no proprietary implementation was
inspected or copied.

Commands, conversion logs, pinned source metadata, model listings, responses,
timings, public metrics, and audit are retained in
`target/draft-choice-evidence/engine-comparison*` and adjacent
`basert-draft*` files. The next performance investigation must target the
cost of speculative verification rather than claim closure from a plain
BaseRT baseline.

## Warmed DFlash 2 GPU attribution

A new Instruments launch profiles the expert-down Q4 target with DFlash 2
cap7, the retained B600 native bundle, two lanes, and the same eight raw
prompts at concurrency one. Two complete warmup cycles precede one measured
profile cycle. All 24 requests complete 128 tokens. The 5.450454-second
steady interval begins after the last recorded Skippy shader compilation
(20.803932 seconds versus scope start 23.578249 seconds).

Submission ownership maps 24,942 submissions with zero PID collisions;
23,456 belong to Skippy. Within the steady scope, 339,780 shader intervals
are attributed to Skippy kernels and 1,303 Skippy intervals remain unmapped.
The attribution uses the latest preceding compiled PC range for that PID,
which remains a code-lifetime heuristic. Per-kernel interval unions overlap
and are not additive wall-time fractions or exclusive GPU durations.

| Shader category | Union active ms within profile scope |
| --- | --- |
| Routed Q4_K vector, four output rows | 1,588.552 |
| Dense Q8_0 few-row MMA, nt4 | 593.185 |
| Q6_K few-row MMA, nt4 | 380.395 |
| Dense Q8_0 few-row MMA, nt2 | 375.672 |
| Gated delta net | 308.990 |
| Dense Q8_0 few-row MMA, nt1 | 236.353 |
| RMS norm/multiply | 88.713 |
| State/data copy F32 | 45.631 |
| Batched convolution | 46.310 |

This changes the next investigation from the earlier plain-decode ranking:
short-batch routed expert work is now the largest individual category. The
existing grouped expert matrix path starts at 32 token rows; the profiled
verification windows are generally shorter. A short-batch crossover probe
can test existing kernel reuse before introducing a new routed kernel.
These invasive profiling timings do not establish a throughput gain.
Trace, scope timestamps, submission ownership, exports, and attribution are
retained in `target/speculative-gpu-evidence/`.

### Rejected blanket short-batch matrix crossover

Lowering the existing grouped Q4_K expert-matrix crossover from 32 to eight
rows (inner/output at least 512) passed 72 CPU-reference Metal cases, covering
Q4/Q5/Q6/Q8, broadcast/per-expert activations, rows 1/7/8/9/16/31/32,
expert-down geometry, and dimension boundaries. A fresh-process ABBA serving
test used the expert-down target, DFlash 2 cap7, eight prompts, one warmup
cycle and three measured cycles/process. All 128 requests completed 128
tokens. Aggregate measured throughput fell from 197.192 to 188.276 tok/s
(-4.52%). Seven of eight prompt medians regressed. The candidate also showed
two measured text variants on three prompts, whereas each control prompt
had one. The blanket crossover is rejected; its patch and runtime are
archived in `target/routed-short-matrix-evidence/`.

A separate repeated-weight ABBA microbenchmark explains why a blanket
crossover is unsuitable: at eight rows, K2048/M512 grouped matrix latency
increased by 19–23%, whereas K512/M2048 fell about 60%. At sixteen rows,
latencies fell by 32–35% and about 78%, respectively. Broadcast and per-expert
activation cases agree in direction. These synthetic repeated-weight numbers
are not DRAM bandwidth or end-to-end speedup claims. They motivate a narrower
geometry-sensitive probe that preserves the tall-projection vector path
below sixteen rows and admits the short/wide projection from eight.

### Rejected geometry-sensitive matrix crossover

A follow-up admits Q4_K grouped matrix execution from eight rows for short,
wide weights (inner 512–1024, output at least1024), and from sixteen rows
for the other tested projection regime (inner/output at least 512). The
candidate passed 48 CPU-reference Metal cases at rows 7/8/15/16, with
broadcast/per-expert inputs, both actual expert geometries, and boundaries
around inner 512 and output 1024.

The serving comparison used fresh-process ABBA, the same expert-down target
and DFlash 2 cap7 in both arms, all eight raw prompts at concurrency one,
then paired prompts at concurrency two. Each process ran one full warmup
cycle and three measured cycles per concurrency. All 256 requests completed
128 tokens without cache hits; 192 were measured.

| Concurrency | Retained tok/s | Probe tok/s | Pooled change |
| --- | --- | --- | --- |
| 1 | 193.499 | 198.194 | +2.43% |
| 2 aggregate | 226.304 | 223.197 | -1.37% |

The single-request control processes differ materially (199.694 versus
187.677 tok/s), while candidates are 199.368 and 197.034. Thus the pooled
single-request increase is not an established causal gain. Two-request
controls are 232.186 and 220.713, versus candidates 219.793 and 226.709.
Each single-request prompt has stable text within an arm, but text differs
between arms. The two-request candidate has multiple measured output
variants on four prompts, versus one variant per control prompt.

This probe is also rejected. The original owning 0042 patch is restored;
neither rejected crossover changes the retained policy. Patch, candidate
runtime, numerical cases, logs, raw responses and audit are archived under
`target/routed-geometry-matrix-evidence/`. The existing grouped matrix path's
mapping, tile padding and numerical behavior need a more specialized
short-batch solution; the repeated-weight microbenchmark gains alone do not
justify a crossover change.

After both crossover probes, native B600 was rebuilt through Just. The
restored `libllama.dylib` and `libggml-metal.dylib` SHA-256 digests match the
frozen control byte-for-byte, and the rebuilt backend-test executable is also
byte-identical to its pre-probe control. Restoration audit and build logs are
under `target/routed-geometry-matrix-evidence/`. No rejected dispatch change
remains in the native patch queue.

## Rejected 32x8 routed tensor tile

A separate short-batch grouped Q4_K kernel used a 32-output/8-entry tensor
tile, two SIMD groups, and a 32-value inner step. The existing compact routing
map was specialized to eight-entry descriptors. Scratch sizing reserved the
corresponding tile-count bound, including dense/duplicate expert reuse. The
path was restricted to M5 Max shared buffers with tensor API support,
contiguous F32 activations, supported routing widths, and 8–31 token rows;
other paths retained their original kernels.

The new capability was prototyped in its own 0044 patch, separately from the
0042 vector activation-reuse capability. The GPU smoke test first caught a
signed/unsigned function-constant mismatch, which was corrected before timing.
Hardware/shape dispatch checks passed. Then 32 actual CPU-reference Metal
routing cases passed, covering both target expert geometries, broadcast and
per-expert inputs, rows 7/8/9/16/31/32, dense expert reuse, duplicate/negative
IDs, odd output tails, strided weights, and offset ID views. The test selection
was corrected to the backend's actual `MTL` registry name; zero-selected-case
runs are excluded from qualification.

Repeated-weight ABBA timings improved expert-down latency by about 61% at
eight rows and 72% at sixteen; tall-projection latency increased by 4.5–5.2%
at eight rows and fell about 36% at sixteen. These microbenchmarks did not
predict a serving gain. A full fresh-process ABBA comparison used the same
expert-down target, DFlash 2 cap7, eight prompts at concurrency one and paired
at concurrency two, one warmup cycle and three measured cycles/concurrency.
All 256 requests completed 128 tokens; 192 were measured, with no cache hits.

| Concurrency | Retained tok/s | Short tile tok/s | Change |
| --- | --- | --- | --- |
| 1 | 199.402 | 191.594 | -3.92% |
| 2 aggregate | 231.829 | 228.344 | -1.50% |

Single-request processes were consistent: 199.189/199.615 for controls versus
191.494/191.694 for the probe. One candidate single-request prompt had two
measured text variants. Every prompt was stable within each two-request arm,
but output text differed between arms. The probe is rejected. Its runtime
source pin was `c3ae4773a85f2a04db9331bb0519d8c799563d74`; the subsequent
`MTL` test-selection correction changes only the test executable.

The measured patch, corrected test-reproduction patch, native runtime,
commands, numerical and timing logs, raw responses, and audits are archived in
`target/routed-short-tile-evidence/`. The 0044 prototype is removed from the
active queue and native B600 restored. The repeated-weight gains alone do not
justify replacing routed vector work in the serving path.

## Device feature capture and broader precision qualification

With restored native B600, the existing opt-in device feature path was
checked directly using the runtime benchmark. With the expert-down target,
DFlash 2 cap7 and one 128-token coding prompt, the counter reported 203 target
rows consumed from device capture and the generated token IDs matched the
serial target. These checks establish active device ingestion on that smoke;
its isolated runtime timing is not an HTTP throughput comparison.

A fresh ABCDDCBA comparison then tested four combinations: expert-down-only
versus broad dense/head/down Q4 target, each with host feature ingestion
(`SKIPPY_DFLASH_DEVICE_FEATURES=0`) versus device ingestion (`=1`). All other
settings stayed fixed: native B600, same release CLI and Q8 DFlash 2 cap7,
default adaptive/shared execution, f16 KV, disabled caches, eight raw coding
prompts at concurrency one then paired at concurrency two. Each process ran
one warmup cycle and three measured cycles/concurrency. All 512 requests
completed 128 tokens without cache hits; 384 were measured.

| Target / feature path | Concurrency 1 HTTP tok/s | Concurrency 2 aggregate tok/s |
| --- | --- | --- |
| Expert-down Q4 / host | 198.917 | 231.622 |
| Expert-down Q4 / device | 201.091 | 230.568 |
| Dense/head/down Q4 / host | 177.386 | 223.712 |
| Dense/head/down Q4 / device | 196.697 | 220.348 |

For the expert-down target, device features changed rates by +1.09% and
-0.46%; measured text matched between host/device at both concurrencies.
For the broader target they changed rates by +10.89% and -1.50%. The broader
single-request results have target-text variation: host repeats have two
variants on two prompts and device repeats have two on one prompt; text
is not identical across the two arms. At concurrency two, each broader-target
prompt is stable and text matches across feature paths. That broader
single-request point estimate is not an identical-output speedup.

Even with device features, the broader target does not beat the expert-down
host baseline (-1.12% at one request, -4.87% at two). The precision recipes
remain workload-qualified; neither broader precision reduction nor device
capture closes the BaseRT speculative gap. Device features remain opt-in;
this study does not change the default. Commands, primitive offload counters,
responses, timings and audit are in `target/dflash-device-feature-evidence/`.

## Opt-in code-preserving affine expert layout

Core patch 0044 adds a separate quantizer capability: Q4_K to Q4_1 conversion
that preserves each 32-value group's integer codes and rounds its decoded
scale and offset to f16. This is not a lossless weight transformation. The
ordinary quantizer remains in use when calibration is supplied or an affine
coefficient cannot be represented finitely. No serving kernels or public ABI
change. This capability is independent of patch 0003's model-loading changes.

The experimental artifact starts from the qualified expert-down Q4 target
(SHA-256 `4ad842f4ed3ef1070d7818d95163e47402e7de17c79ecb85a0e272aab3ae52f5`).
It converts 117 routed expert tensors and preserves the remaining 636 tensor
payloads byte-for-byte. Its size is 23,384,807,712 bytes, SHA-256
`e7b415ddddfa7bd8348d643534cd559313a9831f332466cfc68ba2cd40dec7d3`.
Each converted tensor grows 11.11%; the complete model grows 9.16%.
Independent full-payload auditing verified all 31,406,948,352 integer codes.
Public GGML dequantization of 122,234,112 sampled weights measured relative
squared error 4.12888849149e-07 and maximum absolute difference
0.000137329101562. Native unit tests cover 1,280 randomized blocks, batch
boundaries, output sentinels, invalid lengths, and coefficient overflow.

The same 128-chunk quality corpus gives perplexity 6.658400 versus 6.657436
for the parent and 6.644816 for the original compressed target. Mean KL
against the original is 0.008213 and top-token agreement 96.403% over 32,640
scored positions. This single corpus does not establish general quality
equivalence. Repeated-weight microbenchmarks show lower Q4_1 latency, but
they do not establish a serving gain.

A first eight-prompt ABBA serving run measured 179.123 versus 197.539 HTTP
tok/s at concurrency one, and 202.138 versus 242.107 aggregate at two.
Both arms used frozen native B600, the same CLI, DFlash 2 cap7 and f16 KV.
The controls were slower than earlier runs, so these results are provisional.
A subsequent comparison is explicitly invalid for speed qualification:
single-request parent controls collapsed from 169.501 to 90.673 tok/s while
concurrent unrelated local `maxd` activity was observed (about 30 GB RSS and
substantial CPU use). These observations do not establish GPU utilization or
prove the entire cause. All 78 power snapshots showed battery power. Its
pooled percentages must not be used as causal gains or evidence of BaseRT
parity. Raw responses and invalidation metadata are retained for diagnosis.

Jianyang job `job_ryjqgw7d0ta8gtk` succeeded on a Dan-lab Apple M4 mini at
local commit `86474d10d`. A cold queue replay produced native commit
`748825b4eab791f3dc768608f37c9ffee96d02d8` and tree
`d625e60be8c3daed38929f3104dc6cf27ff27747`, matching this workstation's
prepared tree. All 1,280 converter blocks passed with the same measured error.
This is a correctness check, not a performance result; its budget was $0.

The full Skippy and Mesh release build (`just with-lld just release-build`)
also passes with this converter and the packaged Metal runtime. This does not
resolve the inherited native gate's missing ASR/diffusion targets or substitute
for a whole-workspace test run. The cooled confirmation below qualifies an
opt-in affine recipe for the tested workload; it does not change the default.
Evidence is in `target/affine-code-layout-evidence/`.

An isolated M1 Ultra ABBA run subsequently succeeded through Jianyang
(`job_np7hvq415syy8sv`, run `run_psk6a3jnyt46smn`). It resumed the exact
verified source snapshot after an initial job stopped on an incorrect Just
recipe name, before any conversion or timing. The release build and corrected
`just skippy-quantize-build` both passed. Host-native regeneration reproduced
the parent and affine GGUF SHA-256 pins above exactly.

Each fresh process ran three warmup and three measured cycles per concurrency,
using the same eight coding prompts, DFlash 2 cap7, f16 KV and disabled caches.
Independent auditing of all 384 raw responses verified 128-token length
completions, no cache hits, matching input token counts, and 192 measured
requests. All four power snapshots report AC power; `pmset` reported no
thermal/performance warning. Late Jianyang telemetry reported moderate
thermal status, so temperature-independent performance is not established.

| Concurrency | Parent HTTP tok/s | Affine HTTP tok/s | Pooled change |
| --- | --- | --- | --- |
| 1 | 98.172 | 125.204 | +27.53% |
| 2 aggregate | 117.515 | 146.901 | +25.01% |

Candidate processes were consistent (125.459/124.950 at one request and
146.566/147.238 at two), and each beat both corresponding controls.
Controls drifted: 106.230/91.251 and 121.564/113.728 respectively. The
single-request spread is 16.42%, so the pooled percentages require a reversed
order confirmation. Each prompt was stable within each process except one
two-request candidate prompt with two variants. Text differs between formats;
this is not an identical-weight speedup or a BaseRT comparison.

Single-request candidate telemetry had approximately the same acceptance and
window counts as the first control, with lower measured verification time
(736/739 versus 905 ms per request; the later control was 1,070 ms).
These host telemetry intervals are not additive GPU time. Raw artifacts were
retrieved with SHA-256 verification, including the archived response directory;
the independent audit is under `m1-artifacts/`. The cooled reverse-order
confirmation below supersedes these provisional percentages.

The first reverse attempt was cancelled when an unrelated GitHub Actions job
appeared on the same host. Cancellation reached the process approximately
50 seconds after the control-plane state changed; final log upload and the
return to idle GPU memory verified cleanup. That CI job began at 22:10:27Z,
after the original completed run ended at 22:08:03Z, so it does not overlap
the first run above. No other participant's job was interrupted.

A guarded BAAB repeat (`job_yd1aj0vfcphvv6m`, run
`run_jj79pb3jk5q1x1g`) completed all requests but is invalid for pooled speed
qualification. Its later parent process fell from 106.101 to 48.931 tok/s at
one request. Time-aligned telemetry, using sample timestamps rather than
observation timestamps, found heavy thermal pressure and explicit throttling
during measured requests: three of six samples at parent concurrency one and
both samples at concurrency two. The later candidate also had throttled
samples. Its pooled +84.69% / +17.66% figures must not be claimed as causal
gains. `pmset`'s lack of a recorded warning is insufficient to exclude this.

The initial unthrottled candidate/control pair again measured approximately
125 versus 106 tok/s at one request, with the same ordering advantage seen
in the earlier run. That subset is diagnostic, not a replacement for the
planned full comparison. The earlier +27.53% / +25.01% figures also remain
provisional because of control drift and incomplete thermal coverage.
Confirmation required cooling between fresh processes, no overlapping
jobs, and rejection of any active-process thermal-throttling observation.
The rejected repeat's raw artifacts and thermal invalidation metadata are
retained under `m1-reverse-artifacts/`.

A fresh cooled BAAB comparison (`job_s68vr3cqryksg8k`, run
`run_hfr9ag2mkka1apz`) passed on the M1 Ultra. Each fresh process was preceded
by 180 seconds idle and ran three eight-prompt warmup cycles followed by three
measured cycles at each concurrency. Both arms used the same CLI, native
`748825b4eab791f3dc768608f37c9ffee96d02d8`, DFlash 2 cap7, f16 KV,
8192-token context, two lanes and disabled caches. This native version includes
the converter but neither experimental routed NR8 nor routed GLU fusion.

| Concurrency | Parent HTTP tok/s | Affine HTTP tok/s | Pooled change |
| --- | --- | --- | --- |
| 1 | 106.229 | 125.198 | +17.86% |
| 2 aggregate | 120.383 | 146.719 | +21.88% |

Every candidate process beat both controls; process spreads were below 0.22%.
All sampled Jianyang thermal states were nominal, with no throttling or
overlapping jobs. Independent raw-response auditing checked all 384 requests,
including 192 measured requests: matching input token counts, 128 output
tokens, zero cache hits, and stable per-prompt text within each format across
processes and concurrency levels. Text differs across formats. Source and
regenerated model checksums match the pins above. Checksummed artifacts and
the independent audit are retained under `m1-cooled-artifacts/`; the separate
watcher's telemetry uses sample timestamps to establish measured-window
coverage.

The explicit 753-tensor recipe is
`skippy/scripts/recipes/quantization/qwen3.6-35b-a3b-ud-q4-k-m-affine-experts.tensor-types.txt`.
Its README specifies the required expert-down parent, converter and output
pins. These are additional target-format gains over our existing improved
baseline, including PR #33; they are not an engine-only comparison against
PR #33, a general quality guarantee, an M5 serving qualification, or evidence
of BaseRT parity. Component percentages must not be added or multiplied.

## Dense affine format screening on M4

Jianyang job `job_nbtr6z2fftgpd91` succeeded on the Dan-lab M4 mini with
source snapshot `bb636ca76a4739adcd2c6116b2e3ff95981d389d`. Its tree
`fb26abdbe203b6f8e15007fbff2e150ab6015336` equals branch commit `86474d10d`;
the snapshot gives the private lab transfer a valid Git commit without
publishing the branch. All 45 actual Metal CPU-reference cases passed.

Five loaded-model dense geometries were screened at 1/8/16 activation rows
using Q8_0, Q5_1 and Q4_1. Four fresh performance processes used forward,
reverse, reverse, forward case order. These are repeated-weight operation
latencies, not serving throughput, equal-weight quality comparisons or DRAM
bandwidth measurements. The host was on AC power, with no reported thermal
warning at completion. Downloads and artifact retrieval preserve SHA-256
checksums. No new native dispatch or format default was introduced.

| Inner / output dimensions | Q5_1 vs Q8_0 latency, 8 / 16 rows | Q4_1 vs Q8_0 latency, 8 / 16 rows |
| --- | --- | --- |
| 2048 / 4096 | +8.53% / +17.99% | -34.31% / -1.67% |
| 2048 / 8192 | -6.65% / +15.40% | -41.52% / -6.87% |
| 512 / 2048 | +51.92% / +20.65% | -11.27% / +1.05% |
| 2048 / 512 | +67.06% / +11.10% | +6.31% / +3.75% |
| 4096 / 2048 | +14.37% / +17.56% | -29.24% / +2.26% |

Q5_1 often improves one-row latency but does not justify a blanket change for
speculative verification widths. Q4_1 is a more promising eight-row candidate
for the larger projections; sixteen-row gains are small or absent. A next
experimental plan targets only 110 main-decoder Q8 large dense tensors and
preserves 643 tensors, including smaller shared-expert matrices, routers, MTP
and the output head. This plan is not promoted to a product recipe. Raw correctness/performance
logs, the exact cases and source pins, checksummed artifacts, and analysis
are in `target/dense-affine-format-evidence/`.

The follow-up M1 Ultra quality job `job_pt9s6jz177e53r5` (run
`run_6f4y4z247xpem9n`) succeeded at $0. A full payload audit verified exactly
110 changed tensors and 643 byte-identical preserved tensors. The candidate
is 22,834,305,312 bytes versus the affine parent's 23,384,807,712. On the
same SHA-pinned WikiText corpus and 128 chunks, the quantized-logit comparison
reported PPL 6.781923 versus reference-logit PPL 6.652538, ratio 1.019449
plus/minus 0.001739, mean KL 0.035864 plus/minus 0.000502 and top-token
agreement 92.053%. The parent's direct PPL run was 6.6546; its packed reference
logits introduce a small separate measurement difference. This is a measurable
quality tradeoff, not an equivalent-weight improvement. No serving throughput
qualification has been run. Raw checksummed logs and payload audit are in
`target/dense-affine-format-evidence/m1-quality-artifacts/`.

## DFlash 2 selector precision diagnostic

The public BaseRT 0.5 inspector reports that its converted Q8 DFlash 2
sidecar retains `candidate_selector.hidden_projection.weight` and both
selector codebooks as F16. The Skippy Q8_0 GGUF stores the corresponding
three tensors in Q8_0. This public metadata difference suggested checking
whether selector quantization materially reduced proposal acceptance; it
does not reveal or establish an engine implementation difference.

An experimental draft restored only these three tensors from the original
open BF16 checkpoint, revision `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`.
A temporary C++ diagnostic used public GGML/GGUF APIs to write the mixed
artifact. Independent full-payload hashing verified that the other 93 tensor
payloads and shapes remain unchanged. BF16-to-F16 conversion's maximum
absolute rounding difference was 2.98023223877e-08. The draft grew from
570,468,544 to 690,153,664 bytes; its SHA-256 is
`326649243b7277d4522c8cf86eeb2445bec6d080fe53052252e8f64e1c951812`.
No proprietary format reader, engine code or kernels were copied.

With frozen native B600, the same expert-down target, a seven-proposal limit,
eight coding prompts and 128 output tokens per prompt, both drafts matched
their serial target on all eight prompts. Token IDs also matched across
arms. The Q8 selector accepted 792 of 1,564 proposals in 230 windows; the F16
selector accepted 793 of 1,562 in 229 windows. Thus this probe gains only one
accepted token over 1,024 generated tokens and does not explain the large
acceptance or throughput gap. The attached-draft teacher-forcing diagnostic
also matched 128 serial target positions, but that separate diagnostic does
not run draft proposals or rollback.

These two unbalanced runtime processes do not qualify a causal speedup.
Selector precision is not promoted to a product recipe or default. Public
inspector output, checkpoint/artifact hashes, the diagnostic writer, payload
audit, exact prompts, correctness reports and acceptance counts are retained
in `target/draft-selector-precision-evidence/` and the public draft metadata
files under `target/draft-choice-evidence/`.

A follow-up width screen kept the original Q8 selector and compared proposal
caps three, five and seven. All eight 128-token prompts matched their serial
targets in each process. Caps 3/5/7 required 319/259/230 speculative windows,
accepted 704/763/792 tokens, and proposed 948/1277/1564. Mean draft time per
window was 4.407/4.538/4.624 ms, whereas target time was
15.180/17.752/19.622 ms. The smaller blocks needed enough extra windows to
lose their lower per-window target cost. Isolated runtime rates were
163.825/177.268/183.492 tok/s, with serial baselines 97.739/99.930/98.198.
These unbalanced processes are a screening result, not a causal serving
speedup qualification. They give no reason to lower the cap or prioritize a
row-dependent draft-time fit. They also illustrate why proposal acceptance
percentages cannot be compared without accounting for proposal widths:
the shortest, slower arm has the highest percentage. The controller remains
unchanged. Reports are in `width-screening.json` beside the selector evidence.
