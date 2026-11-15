# Metal decode optimization evidence

The measurements below were collected on the branch based at `c8bdb738f`,
before general-inference/mesh-llm PR #33. They remain historical measurements,
not performance claims for the combined implementation. Integration with
merged commit `74c7d5bd2` is in progress; the fresh comparison below qualifies
only its stated workloads. In the
combined queue, the original core patches 0017–0022 are numbered 0036–0041,
following PR #33's core patches; the combined native ABI is 0.1.85.

The latest retained addition is the [portable down-only NR8 policy](#retained-portable-down-only-nr8-policy):
fresh-start M1 C2 comparisons confirm about 1.1% additional throughput. This is
a separate hardware-specific measurement, not an addition to the M5 headline.

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

## Routed affine reuse on M5 Max

The local extension of core patch 0042 reuses activations across eight output
rows for eligible shared-memory M5 Max Q4_1 routed-vector projections at token
widths 3 through 31. Other widths, incomplete row tiles and unsupported inner
dimensions retain their earlier paths. Forty actual CPU-reference Metal cases
and the dispatch-boundary tests passed. Repeated-weight operation tests at
widths 8 and 16 measured roughly 8–11% lower gate/up latency and 18–20% lower
down-projection latency. A serving comparison was stopped when unrelated
`maxd` activity appeared, so there is no qualified serving gain for NR8 yet.
Evidence is under `target/routed-affine-reuse-evidence/`.

A completed two-resident-server comparison isolated NR8 against frozen B600,
using the same affine-expert model, CLI, DFlash 2 cap7, f16 KV, two lanes and
disabled caches. The candidate also contains quantizer-only patch 0044, which
does not alter serving arithmetic. Three warmup and three measured eight-prompt
cycles ran per concurrency, with each measured prompt/batch ordered
control/candidate/candidate/control. Predefined load/thermal checks rejected
five whole blocks for both arms and retried after three quiet samples ten
seconds apart. An independent audit verified 288 accepted requests, including
192 measured; 305 raw response files include rejected attempts. Every accepted
batch had nominal thermal state and no observed competing CPU load before and
after it. Input token counts matched, outputs reached 128 tokens, and caches
were unused. Batch elapsed time includes client validation and artifact writes.

| Concurrency | NR4 control tok/s | NR8 tok/s | Change |
| --- | --- | --- | --- |
| 1 | 216.829 | 218.499 | +0.77% |
| 2 aggregate | 253.060 | 259.430 | +2.52% |

All measured single-request outputs match byte-for-byte. Each measured cycle
is faster with NR8 (218.361/218.968/218.171 versus
216.537/217.714/216.242). The small single-request improvement supports keeping
the narrow dispatch change. Two-request output variation prevents treating
that point estimate as an identical-output speedup. These are within-prompt
comparisons with two resident servers, not a fresh-process qualification or
evidence of BaseRT parity. The real-model diagnostic below also loaded the
`kernel_mul_mv_id_q4_1_f32_r8_nsg=2_split=0` pipeline. The restored packaged
runtime exactly reproduces the pre-fusion NR8 primary-library SHA-256
`32bd947b9520a78188989647810f69845f6443b49ede639d12e9c95b0e160efd`.
Raw guard observations, rejected blocks, response files and independent audit
are under `target/routed-affine-reuse-evidence/paired-resilient/`.

## Current affine/NR8 configuration against BaseRT

A fresh paired comparison used BaseRT 0.5's public canonical Q4 target and
converted DFlash 2 Q8 sidecar against Skippy's affine-expert target and GGUF
DFlash 2 Q8 sidecar. Both used a seven-proposal ceiling, greedy sampling,
f16 KV, 8192 context cells, two lanes and disabled prefix reuse. Target and
draft representations differ; this is a complete-configuration comparison,
not equal-quality or engine-only efficiency. Full SHA-256 pins were refreshed
before either server started, including BaseRT target
`7171b37ffccdf16dfd920f55be766e6faffa31f64e04ca9cd0064943bf72881b`
and its draft `ee4c3d0902c7e9bc9e8f0ed9798caae8a5584ac80362ec32ebc0651946141eb1`.

The two resident servers ran three warmup and three measured eight-prompt
cycles at concurrency one and two. Every measured prompt/batch alternated
BaseRT/Skippy/Skippy/BaseRT. Three whole blocks affected by unrelated `maxd`
activity were rejected for both arms and retried after three quiet samples
ten seconds apart. Independent auditing verified 288 accepted requests,
including 192 measured, with matching input token counts, 128-token length
completions and zero cache hits. All accepted batches had nominal thermal
state and no observed competing CPU load before and after them. Batch elapsed
time includes client validation and artifact writes. There are 300 raw response
files, including excluded attempts.

| Concurrency | BaseRT tok/s | Skippy affine/NR8 tok/s | Skippy relative change |
| --- | --- | --- | --- |
| 1 | 309.412 | 215.712 | -30.28% |
| 2 aggregate | 367.784 | 272.092 | -26.02% |

The candidate's single-request cycles are stable at
215.383/216.328/215.428 tok/s. Text differs between engines as expected from
their different target quantizations. This comparison confirms that BaseRT
parity has not been achieved: matching these rates would require approximately
43.4% and 35.2% more Skippy throughput respectively on this workload. It is
not valid to combine these percentages with component gains or results from
other machines and workloads. Commands, guard/rejection records, all raw
responses, refreshed source hashes and independent audit are under
`target/basert-current-gap-evidence/`.

## Affine/NR8 GPU attribution

A fresh Metal System Trace launch profiles native `e795659936a8` with the
affine-expert target and DFlash 2 cap7. Two complete eight-prompt warmup cycles
precede one measured profile cycle. All 24 requests complete 128 tokens without
cache hits. The 5.018145-second steady interval starts at trace time
24.087373 seconds, after the last Skippy shader compilation at 20.787230333.

The mapping table contains 18 submission-ID ownership reuses across processes.
The analysis therefore uses each interval's latest preceding mapping timestamp
and rejects conflicting PID ownership within that interval, rather than taking
the final owner of an ID. The corrected temporal join leaves the steady-scope
ranking unchanged: 325,093 shader intervals attributed to Skippy, 1,286 Skippy
intervals without a matching shader PC range, and three unowned intervals.
Shader attribution still uses the latest preceding compiled PC range for the
owned PID, a code-lifetime heuristic. Per-kernel interval unions can overlap;
they are not additive or exclusive wall-time fractions.

| Shader category | Union active ms within profile scope |
| --- | --- |
| Routed Q4_1 vector, eight output rows | 1,218.782 |
| Dense Q8_0 few-row MMA, nt4 | 576.559 |
| Dense Q8_0 few-row MMA, nt2 | 367.582 |
| Q6_K few-row MMA, nt4 | 366.003 |
| Gated delta net | 294.553 |
| Dense Q8_0 few-row MMA, nt1 | 230.036 |

Matrix work remains the largest target after the affine layout and NR8 change.
The Q8 category includes both target and draft operations; it does not by itself
justify quantizing every dense tensor. This profile is invasive diagnostic
evidence, not a throughput result or a causal comparison against the older
Q4_K trace. The full trace, raw exports, timestamp-aware ownership analysis,
steady-window markers and request validation are retained under
`target/affine-nr8-gpu-evidence/`.

## Public-API BaseRT GPU attribution

A matching 90-second Metal System Trace launch profiles the public Superfluid
CLI and BaseRT 0.5 with its canonical Q4 target, converted DFlash 2 Q8 sidecar,
cap7, f16 KV, 8192 context cells and two lanes. It runs the same eight raw
prompts, two warmup cycles and one profile cycle. All 24 requests produce
128-token length completions without cache hits. No implementation code is
reconstructed or copied.

The launched HTTP process is PID 27898. The recorded process inventory lists
two workers with the expected executable path; only PID 27910 has GPU shader
records. Attribution uses that GPU-active worker, not the HTTP parent. Its
last shader compilation is at trace time 10.811935625 seconds, before the
3.315475-second steady window starting at 13.254581 seconds. A timestamp-aware
submission-owner join handles four reused submission IDs. It attributes
184,107 steady shader intervals to the worker, out of 187,271 total intervals;
no worker intervals remain without a shader PC mapping. The PC-range
code-lifetime heuristic and non-additive union-time limitations still apply.

| Observed BaseRT shader label | Union active ms within profile scope |
| --- | --- |
| `moe_silu_gate_up_shexp_gated_q4_s1r2` | 735.152 |
| `moe_expert_gemm_shexp_scatter_gated_q4` | 494.726 |
| `tc_gemm_q4_tiled_ns_n128` | 250.575 |
| `tc_gemm_q8_u8_w128_sk` | 216.464 |
| `tc_gemm_q4_tiled_ns_n128_seg_ab` | 215.265 |
| `gdn_verify_fused_defer_hd128_f16` | 173.002 |

Public startup logs also report 131 target projections tiled at load time
(959 MiB replacing packed rows; 948 MiB of mapped pages released, with 111
tensors never wrapped). These logs and labels provide research leads for
weight packing, shared/routed expert fusion and recurrent verification; names
alone do not establish their algorithms or full numerical precision.

During the respective eight-prompt steady windows, Skippy records 3,695 unique
Metal command-buffer IDs and BaseRT's worker 527, selected by the mapping
timestamps and owning PID. The counts motivate investigating submission and
synchronization, but different target/draft representations and token paths
prevent treating them as an engine-only overhead ratio. Attributed shader
interval unions cover 3,735.643 ms of Skippy's 5,018.145 ms window and
2,894.416 ms of BaseRT's 3,315.475 ms window. These are profiler coverage
measurements, not hardware utilization or exclusive runtime fractions.

The public metrics delta for BaseRT's steady cycle reports 1,024 emitted
tokens, 1,179 proposals and 777 accepted proposals. Skippy's eight corresponding
profile requests report 1,506 proposals, 791 accepted and 222 windows.
Acceptance and proposal counts depend on the controller, quantization and
text; they do not establish a single causal explanation. They support
rechecking proposal width after the affine format change. Trace, logs, metric
snapshots, raw responses and corrected worker attribution are retained under
`target/basert-gpu-evidence/`.

## Affine/NR8 proposal width confirmation

The lower proposal count seen in BaseRT's profile prompted a new width test
after the format change. Both Skippy arms use the exact retained affine/NR8
runtime and model/draft bytes; only the proposal ceiling changes from seven
to five. The same two-server, within-prompt ABBA method runs three warmup and
three measured eight-prompt cycles per concurrency. Five whole blocks affected
by unrelated `maxd` CPU load are rejected for both arms and retried after three
quiet samples ten seconds apart.

Independent auditing verifies 288 accepted requests, including 192 measured,
with matched input token counts, 128 output tokens, zero cache hits, nominal
thermal state and no observed competing CPU load before and after each
accepted batch. All measured output text matches byte-for-byte at both
concurrencies. The 310 raw response files include excluded attempts.

| Concurrency | Cap7 tok/s | Cap5 tok/s | Cap5 change |
| --- | --- | --- | --- |
| 1 | 217.040 | 206.888 | -4.68% |
| 2 aggregate | 272.475 | 268.450 | -1.48% |

Cap5 is slower in every measured cycle at both concurrencies. The cap7 serving
preset remains unchanged; fewer draft proposals alone do not establish a
speed improvement for this runtime. This confirms the width choice on the
new format without assuming that the older Q4_K result transfers. Raw
responses, guards, rejected blocks and independent audit are retained under
`target/affine-width-evidence/`.

## Rejected routed gate/up/SwiGLU fusion

The rejected core patch 0045 prototype was a separate capability: it combined paired
Q4_1 routed gate/up projections and split SwiGLU in one M5 Max kernel, reusing
F32 activations and keeping both accumulators in F32. It does not change
weights or the public ABI. Its graph matcher admits the parallel projection
branches explicitly, preserves captured intermediate outputs, keeps structural
packing independent of token width, and rejects output overlap with every
input, including exact activation aliases and leaf weights or IDs. The full
encoder requires all inputs and output on one shared M5 Max device and uses
the existing fallback outside token widths 3–31.

The actual M5 Max kernel compiled and executed. All 58 CPU-reference numerical
cases passed across both geometries, broadcast/per-expert activations, negative
routing IDs, widths 1/2/3/8/16/31/32, captured intermediates and an exact input
alias. The short-row cases now use a 1e-4 relative squared error threshold,
50 times tighter than the inherited 5e-3. A trial 1e-6 threshold failed near
3e-5 because the CPU Q4_1 reference quantizes activations to Q8_1. An independent
oracle dequantizes weights, retains F32 activations and accumulates in double.
It checks eight sampled output rows per token/expert, including four/eight-row
tile boundaries and the last row. All 16 eligible cases pass with maximum relative squared error
2.77041678939e-14. Together with 20 allocation-backed policy checks, that test
reports 36 checks and zero failures. The existing Metal graph-policy test also
passed. Queue replay produces the exact intended native tree
`c48d7438de526c9cf48c8c498eec66b96cc742e2`. The packaged runtime build passed
at `bba02246f508fa828cad6d7ebebb6800ec8c6669`; subsequent changes are test-only.

The generic operation performance harness repeats only its final node and is
unsuitable for timing this whole graph. An explicit `--benchmark` mode on the
policy test instead times the complete three-node graph. Four ABBA processes
used the same binary with fusion disabled only for the control. At 2048 inner
dimensions and 512 output rows, width-eight latency fell 4.44% with broadcast
activations and 6.36% with per-expert activations. Width-sixteen reductions
were 1.17% and 3.10%. The reverse geometry improved 8.05–11.58% at widths
eight/sixteen. Unchanged width-one cases varied from -1.73% to +4.41%, so
these modest component measurements retain a noise limitation. The fixture
uses 16 experts with eight selected and repeated weights; it does not measure
serving throughput or DRAM bandwidth. Source, raw logs, load guard and analysis
are under `target/routed-glu-fusion-evidence/`.

A full-serving comparison uses the frozen NR8 runtime as control, rather
than disabling every existing fusion. Its first attempt was invalidated during
the first candidate process: the load snapshot recorded unrelated `maxd`
PID 68931 at 45.2% CPU and about 26.4 GB RSS. Only the owned server was stopped;
the unrelated daemon was untouched. No complete comparison or causal serving
gain may be claimed from those partial responses.

A separate real-model diagnostic, with debug logging and fusion statistics
enabled, completed two 128-token requests. The new pipeline loaded, and the
device-wide counter reported 2,118 `MUL_MAT_ID+MUL_MAT_ID+GLU` fusions. The
same shared counter is logged by two backend contexts and must not be summed.
These instrumentation settings alter encoding concurrency; the diagnostic is
path-activation evidence, not a performance result. Logs and responses are in
`target/routed-glu-fusion-evidence/serving-diagnostic/`.

A shorter two-resident-server comparison then completed, with predefined
thermal/load exclusions applied to whole within-prompt ABBA blocks. Three
warmup and three measured eight-prompt cycles ran at each concurrency. Every
measured prompt or paired batch ran control/candidate/candidate/control.
The guard rejected two blocks (fair thermal state, then unrelated `maxd`
activity) and retried them after three quiet samples ten seconds apart.
An independent audit verified 288 valid requests, including 192 measured,
matching input token counts, 128 output tokens, zero cached tokens, and
nominal thermal state with no observed competing CPU load before and after
each accepted batch. There are 291 raw response files, including rejected
attempts. Batch wall time includes client validation and artifact writes.

| Concurrency | NR8 control tok/s | Fused tok/s | Change |
| --- | --- | --- | --- |
| 1 | 214.796 | 210.255 | -2.11% |
| 2 aggregate | 272.404 | 266.254 | -2.26% |

The single-request outputs match byte-for-byte, and the candidate is slower
in every measured cycle. Two-request output variation prevents interpreting
that percentage as an identical-output regression; it also provides no reason
to promote the prototype. The small component latency gain does not survive
full serving. Patch 0045 was removed from the active queue, with its complete
source, frozen runtime, oracle binary, raw responses and independent audit
retained under `target/routed-glu-fusion-evidence/`. NR8 is qualified separately
above; the rejected-fusion comparison includes it in both arms. Neither
kernel is included in the clean M1 Ultra affine-format percentages above.

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

A cooled M1 Ultra BAAB serving run (`job_nm5x3wvnsbkab5j`, run
`run_075fthbct662tyb`) then completed at $0. Both arms used native
`748825b4eab791f3dc768608f37c9ffee96d02d8`, the same CLI and DFlash 2 cap7,
f16 KV, two lanes, disabled caches and eight coding prompts. Each process
had 180 seconds idle before loading, then three warmup and three measured
cycles per concurrency. Independent auditing checked all 384 raw responses,
including 192 measured: matching input token counts, 128-token length
completions and zero cache hits. All 126 guard observations through completion
reported nominal thermal state, without throttling or overlapping jobs.
Each process pair's spread is below 0.37%.

| Concurrency | Affine parent HTTP tok/s | Selective dense HTTP tok/s | Change |
| --- | --- | --- | --- |
| 1 | 125.135 | 127.342 | +1.76% |
| 2 aggregate | 146.391 | 147.338 | +0.65% |

Text differs between formats. These small complete-configuration changes do
not justify replacing the preferred affine-expert recipe given the measured
1.94% perplexity increase, and do not establish BaseRT parity. The selective
dense plan remains an experimental quality/performance tradeoff rather than
a promoted recipe. Checksummed returned artifacts, independent raw-response
audit and telemetry are under
`target/dense-affine-format-evidence/m1-serving-artifacts/` and
`m1-serving-guard.jsonl`.

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

## Rejected post-fence shared-output readback experiments

Native candidate `9c3bf3dbebc37b5c2b4b47e2d06bdeb3b7e11047` queues
host-readable decode outputs until the existing recurrent completion fence,
then reads them through the owning buffer API. Private buffers and callers
without that fence retain their original asynchronous transfers. Metal shared
buffer readability is exposed through a private backend capability; `is_host`
remains false. This is additional to PR #33's KV placement and gather-free
attention work, which is already present in both arms.

The diagnostic retained 1,376 graph command buffers but removed all 2,097
target output-copy command buffers. The 222 draft lattice copies remained.
All eight measured texts and input token counts matched the control, and
telemetry reported no dropped events. A focused native unit also checks
producer completion, destination lifetimes, views, fallback buffers and the
unchanged fence.

A separate uninstrumented comparison used identical affine target and Q8
draft files, NR8 in both arms, cap seven, two resident servers and within-prompt
ABBA ordering. The independent audit validated 288 accepted requests, including
192 measured requests; 300 raw responses include rejected load-affected blocks.
Accepted blocks passed the nominal thermal and competing-load guards.

| Concurrency | Original tok/s | Post-fence reads tok/s | Change |
|---|---:|---:|---:|
| 1 | 215.790 | 216.797 | +0.47% |
| 2 | 265.910 | 266.577 | +0.25% |

All measured output texts matched exactly. C1 improved in each of three cycles;
C2 had one negative cycle and throughput drift in both arms. These small changes
are not a qualified serving gain or evidence of BaseRT parity. Removing copy
submissions alone did not deliver the expected large improvement. The candidate
was not promoted. A follow-up restored asynchronous bulk logits transfers
while retaining post-fence feature reads, to isolate whether CPU logits
copying offset the submission savings. Evidence, frozen candidate and
control packages, and independent audit remain in
`target/host-readback-evidence/`.

The feature-only follow-up used native
`62f0a924d95486284189783b66e68dfa26a2c449`, primary-library SHA-256
`09434e1704a809141eb9e34ed96afa624ccc2c37350e455a8ff0bac326e1ba6f`.
Its diagnostic matched all eight texts and input counts exactly, removed
1,864 feature-copy commands, and retained 233 logits copies and 222 draft
lattice copies alongside the original 1,376 graph command buffers.

The same paired serving protocol independently validated 288 accepted requests,
192 measured, with 297 raw response files. Six whole blocks were rejected for
competing CPU load or non-nominal thermal state; both arms were retried after
three quiet, nominal samples. Every measured output text matched across arms.

| Concurrency | Original tok/s | Feature-only reads tok/s | Change |
|---|---:|---:|---:|
| 1 | 218.642 | 219.283 | +0.29% |
| 2 | 267.503 | 267.750 | +0.09% |

Both concurrency levels include a negative comparison cycle, and C2 drifts
materially across cycles in both arms. Neither version establishes a useful
serving gain. Both prototypes are rejected, and experimental core 0045 is
removed from the active patch queue. Frozen runtimes, durable prototype
patches, complete diagnostics and separate independent audits remain under
`target/host-readback-evidence/`. These results narrow the next work toward
matrix execution and weight layout: substantially fewer copy submissions
alone did not materially close the measured engine-configuration gap.

## Cumulative retained configuration against merged PR #33

A fresh same-machine M5 Max comparison directly measures the original merged
PR #33 configuration against the retained current configuration. The control
uses clean root `74c7d5bd2`, native `7166b39c6add8b266b5d1d80b36d4c31520256f3`,
the original Unsloth UD Q4_K_M target and original Q8 DFlash draft, with a
six-proposal ceiling and the earlier adaptive settings. The candidate uses
native `e795659936a8bb5fd0554d4ee7b788fd6f78b6be`, affine-expert target,
Q8 DFlash 2 and cap seven. Both primary libraries were freshly hashed against
their pinned manifests; CLI checksums and manifests are retained with the run.
Startup telemetry confirms DFlash versus DFlash 2 and the six/seven limits.
Neither rejected readback prototype nor the rejected short-tile prototype
is part of the candidate runtime.

This is a complete-configuration result with different target representations,
drafts and controllers. It is not an engine-only or equal-quality claim. Both
arms use the same eight fixed prompts, greedy generation, 128-token length
completions, f16 KV, 8192 context cells, two generation lanes and disabled prefix
and disk caches. Two resident servers perform three warmup and three measured
cycles per concurrency, with control/candidate/candidate/control ordering
within each measured prompt or paired batch.

| Concurrency | Original PR #33 tok/s | Retained current tok/s | Change |
|---|---:|---:|---:|
| 1 | 151.001 | 215.265 | +42.56% |
| 2 aggregate | 147.924 | 267.924 | +81.12% |

The independent audit checks 288 accepted requests, including 192 measured;
296 raw responses include excluded attempts. Five whole blocks were rejected
for competing CPU load or non-nominal thermal state and both arms retried after
three quiet nominal samples ten seconds apart. All accepted measurements pass
the pre/post batch guards, matching input-token counts, length and zero-cache
checks. Each candidate cycle exceeds every corresponding control cycle. Text
differs across configurations, so this is not exact-output parity. Timing
includes HTTP client validation and artifact writes. The benchmark uses resident
processes, not a fresh-process hardware qualification, and cannot be combined
with earlier component percentages or treated as BaseRT parity. Full scripts,
response files, guard/rejection observations, startup logs, pins and independent
audit are under `target/cumulative-pr33-evidence/`.

## Rejected affine routed 32x8x128 tensor tile

A new M5 Max experiment revisited routed tensor execution after the affine
format change. It uses Q4_1 weights, a 32-output/8-entry tile, two SIMD groups,
and a 128-value inner step instead of the older rejected Q4_K tile's 32-value
step. The staging allocation is 10,240 bytes. Dispatch requires shared buffers,
tensor API support, 8–31 token rows, supported routing widths and aligned
512–4096 inner dimensions; other cases retain their existing paths. This
experimental core 0045 is unrelated to the rejected readback core 0045.

Native `7c4c3d31b6041dbf5fe76c8379869bec9a3cfca6` cold-replays to the intended
tree `c64c41406e1107ffae44ca59b4737c0d05fcf817`. It builds through Just and
passes the dispatch policy suite and 36 independently selected actual Metal
versus CPU routing cases. Cases cover both expert geometries, rows around the
dispatch boundaries, broadcast and per-expert activations, duplicate/negative
IDs, offset ID views, odd output tails, strided weights and non-128-aligned
fallback dimensions. The kernel stages weights and activations in F16, so
passing those operation tolerances does not establish unchanged model quality.

A full serving comparison uses identical affine target and Q8 DFlash 2 files,
cap seven, f16 KV8192, two lanes and disabled caches. Both runtimes include
NR8; the candidate replaces eligible routed vector operations with the new
tile. Independent auditing checks 288 accepted requests, including 192 measured,
and 311 raw responses. Seven complete load/thermal-affected blocks were rejected
for both arms and retried after three quiet nominal samples ten seconds apart.
Measured ordering remains within-prompt control/candidate/candidate/control.

| Concurrency | Retained NR8 tok/s | Short tile tok/s | Change |
|---|---:|---:|---:|
| 1 | 191.623 | 170.651 | -10.94% |
| 2 aggregate | 256.101 | 196.061 | -23.44% |

Every candidate cycle is slower than its corresponding control cycle. Output
text differs between arms. The single-request control drifts from 195.301 to
187.922 tok/s across cycles, but remains above every candidate cycle. Thus the
experiment provides no basis for retention or an additional quality campaign.
The prototype is rejected and removed from the active queue. Frozen runtime,
patch, build/test logs, request files, guard observations and independent audit
are retained under `target/affine-short-tile-evidence/`. The measured cumulative
headline above continues to refer to the retained NR8 configuration.

## Routed Q4_0 format quality screen

A separate $0 experiment on the explicitly authorized idle M1 Ultra changes
only the 117 main routed expert tensors from retained Q4_1 to ordinary Q4_0.
It uses existing product quantization tooling and unchanged native
`748825b4eab791f3dc768608f37c9ffee96d02d8` in pinned source
`bb636ca76a4739adcd2c6116b2e3ff95981d389d`. The original affine source SHA-256
was verified before conversion. A full independent tensor-payload audit checks
all 117 planned changes and exact preservation of the other 636 payloads.

The new file is 21,421,873,440 bytes, versus 23,384,807,712 for affine, with
SHA-256 `97dfb68b5d4cb4e46fea9611a1f7cf0d6a7701a3eb63ab4d1b3a177e5d33dd86`.
A fresh matched 128-chunk WikiText quality run completes in
`job_a8txbmm8fyb87h1` / `run_kvvgg509x5cvm06`. Candidate PPL is 6.700972 versus
6.652538 for the packed affine reference: ratio 1.007281 ± 0.001220,
KL divergence 0.017774 ± 0.000359 and top-token agreement 94.620 ± 0.125%.
The direct affine baseline PPL is 6.6546; packing the reference slightly changes
its reconstructed score. These measurements describe one corpus and a real
quality tradeoff, not equivalent quality or a general quality guarantee.

A fresh-process cooled BAAB serving comparison was launched separately as
`job_7afzt1brwz4g9jc` / `run_bjqv10zh619mt7j`, with the same CLI/runtime,
DFlash 2 cap seven and disabled caches in both arms. It includes 180-second
cooldowns, repeated warmup and measured cycles, and a separate thermal/overlap
watcher. No serving gain or product recipe/default promotion was established.
Quality artifacts,
checksums, payload audit, recipe, immutable job specs and serving observations
are under `target/routed-q4-0-evidence/`.

The user explicitly parked quantization/model tuning to focus on inference
engine speed. The serving job above was cancelled at that request before a
complete BAAB result; no serving gain or format promotion is claimed. Future
engine comparisons keep target and draft artifacts fixed.

## Rejected grouped F32 routed vector execution

A subsequent engine-only prototype groups up to four routed token entries,
reusing each reconstructed Q4_1 weight across those entries without F16 staging
or threadgroup barriers. It retains F32 reconstruction and accumulation, uses
two SIMD groups and sixteen output rows, and admits only eligible M5 Max
shared-buffer geometries. Native
`a1ccedbeea9a51c1a3f7d58ae09c12a81662f894` has intended tree
`e07ac5e82c265d9170bfe4b0e1d906ee66a0bf86`. The build, dispatch policy checks
and 36 actual Metal versus CPU cases passed.

The serving screen reached its predefined thirty-minute deadline after
repeated competing-load interruptions. It is incomplete: 264 accepted requests
include 168 measured requests, with eighteen rejected blocks and 321 raw
responses. The accepted results show 196.291 versus 174.037 tok/s at concurrency
one (-11.34%) and 235.877 versus 182.519 aggregate tok/s at concurrency two
(-22.62%). Every corresponding measured cycle is slower; the final concurrency
two cycle is partial. Target, draft and serving settings are fixed across arms,
but generated text differs. These are rejection-screen observations, not a
completed performance qualification. The prototype was removed from the active
queue and retained NR8 restored. Patch, frozen candidate runtime, raw responses,
guard observations and partial audit are under
`target/affine-grouped-vector-evidence/`.

## Rejected dense cooperative Q8 input kernels

Two dense projection prototypes consume the existing Q8_0 integer codes through
Metal cooperative register inputs, retain F32 activations, and apply each
stored F16 block scale in F32. Both use one SIMD group, sixteen output rows,
eight activation rows and a 32-value inner step. The first uses an int8
cooperative right operand; the second represents the same integers exactly as
half values. Neither changes model files or quantizes activations.

Both pass dispatch policy tests, 47 actual Metal versus CPU cases, and ten
independent FP64 stored-weight reference cases. The latter exercise both plain
and fused-residual pipelines, rather than relying on intermediate-output
backend comparisons that prevent fusion. Explicit opt-out smoke tests also
pass. The half-code runtime cold-replays to tree
`fb434268e79aebccf2c9f4d172a2a39adea4a3cb`; its primary library SHA-256 is
`9579d3b7b1282060ade270779050fde84498a9b6dedb5e65624ea690c5ed3fa7`.

Ten-shape hot-weight latency screens compare the same frozen backend executable
with only a library-initialization opt-out toggled in control/candidate/candidate/
control order. Each process must pass nominal thermal and quiet competing-CPU
checks before and after execution; an affected sequence is rejected in full.
The integer-input screen completes on attempt one and is 4.4–22.7 times slower
than the existing kernels. The half-code screen completes on attempt eighteen,
after load and thermal exclusions, and is 4.8–24.4 times slower. Every tested
shape regresses in both screens. These are operation-level rejection screens,
not DRAM-bandwidth or serving-throughput qualifications. Neither warrants a
full serving run or retention. Frozen half runtime, patches, raw logs, guards
and screen summaries are under `target/dense-cooperative-evidence/half-screen/`
and `target/dense-cooperative-evidence/int8-screen/`.

A third variant reconstructs existing Q8_0 values into F32 cooperative registers
and accumulates 128-value inner tiles, with four SIMD groups splitting K and
2,048 bytes of shared memory used only for final reduction. K must be divisible
by 128; the policy test explicitly rejects the former 576-value admission.
The replayed tree is `fbb7442ffa58f98fb00af4aa02987ed7702fa089`.
Policy, the same 47 Metal cases and the ten direct backend reference cases
pass; maximum independent NMSE is `2.97487734709e-12`. Explicit opt-out also
passes without loading the cooperative kernel.

Its guarded ten-shape screen completes on attempt one. Every shape remains
slower than the existing implementation, by factors of 2.7–6.5. For example,
K2048/M4096/N8 takes 96.670 versus 17.510 microseconds, and K2048/M512/N8
takes 34.630 versus 5.320 microseconds. An independent audit verifies the
frozen executable hash, all eight accepted guards, kernel activation in the
correct arms, forty raw timing observations and all ten computed comparisons.
Artifacts and frozen runtime are under
`target/dense-cooperative-evidence/f32-split-screen/`. This variant is also
rejected without a serving run; the experimental core 0045 is removed rather
than left on an eligible production dispatch path. These experiments establish
no additional engine speedup.

After removal, `just release-runtime-build metal` succeeds for retained NR8.
Its build metadata and all 24 native-library hashes match the frozen known-good
control; the primary library remains
`32bd947b9520a78188989647810f69845f6443b49ede639d12e9c95b0e160efd`.
The full retained Skippy and Mesh release build had also passed before these
native-only prototypes. No rejected cooperative dispatch remains in the active
patch queue.

## Exploratory byte-preserving weight layouts

Following the rejected cooperative kernels, standalone probes test changes to
weight-byte addresses while leaving the target and draft files untouched.
These probes are not integrated runtime caches and do not establish serving
speedups. Packing is checked by reconstructing every original quantization
block and comparing its complete bytes, including scales.

The first Q4_1 probe separates sixteen blocks' quant bytes from their scales,
without increasing size, and replaces scattered scalar quant reads with aligned
vector reads. Its arithmetic and lane assignment follow the retained NR8 body.
Twelve projection/routing cases cover the two expert geometries, 3/8/16 tokens,
broadcast and per-selection inputs, 128 experts and negative routing IDs.
Forty-eight FP64 stored-weight reference comparisons and 24 tight paired
numerical checks pass. Outputs have small floating-point differences despite
the exact byte permutation; bit equality is explicitly not claimed.

The initial paired screen covers 24 cases with fixed and changing expert
selections. It completes a guarded control/packed/packed/control sequence on
attempt one. GPU latency changes range from -9.71% to +8.06%, with regressions
among the measured cases and inconsistent host-wall changes. An independent
audit checks executable/source hashes, all eight accepted guards, 96 raw timing
pairs and all 24 comparisons. This short-duration screen does not justify a
production expert-weight cache. Source, correctness evidence and timing files
are under `target/affine-byte-layout-evidence/`.

Dense Q8_0 probes instead start from the retained native MMA shader, including
its original arithmetic, tiling policy, residual fusion and K splitting. Checked
source substitutions add a layout constant and change only quant/scale loads.
A row-separated variant groups sixteen blocks within each output row; a second
variant also groups the eight output rows consumed by a SIMD group. Both
preserve every stored value, with the same byte count on complete row tiles.
The eight-row variant pads an incomplete final tile in odd-row correctness
cases.

Each dense variant passes 36 projection cases, including plain/fused residual,
token tails, an odd output tail and K8192. All 72 old/packed full-output pairs
are bit-identical. The 144 independent FP64 reference comparisons check every
output row for M at most 2048 and every seventeenth row for the larger matrices;
they do not substitute for the full paired-output checks. Four weight banks
are inverse-byte checked for each geometry.

The row-separated variant's initial 60-case screen is mixed and is not
promoted. Its raw files and independent audit are under
`target/dense-byte-layout-evidence/`. Very short per-case measurement durations
make GPU clock transitions a concern, so the eight-row variant uses at least
20 ms of GPU warmup and at least 30 ms of measured GPU time per case, across at
least four 256-dispatch command buffers. GPU timestamps and host-wall latency
are reported separately; fixed and rotating weight banks remain separate cases.
The stronger screen completes its guarded ABBA sequence on attempt one. It is
mixed overall: 21 of 60 cases improve GPU latency and eighteen improve host-wall
latency. Thus it does not support replacing the existing path generally. A
consistent signal is the K2048/M8192/N16 projection, including both residual
variants and both weight-bank modes:

| Residual | Weight banks | Original GPU us | Tiled GPU us | GPU latency change |
| --- | --- | ---: | ---: | ---: |
| No | Fixed | 52.040 | 47.563 | -8.60% |
| No | Rotating | 64.310 | 49.930 | -22.36% |
| Yes | Fixed | 53.130 | 47.773 | -10.08% |
| Yes | Rotating | 64.907 | 50.601 | -22.04% |

Both candidate repeats beat both control repeats in each of these four cases.
Host-wall latency improves in all four, by 8.47–22.08%. The independent audit
checks the frozen executable/source hashes, eight accepted guards, 240 raw
timing pairs, all sixty comparisons and the minimum measured GPU duration and
dispatch count. This remains a standalone projection result, not an integrated
decode speedup. Smaller activation batches include regressions and are not
admitted merely because they share a weight shape. The result motivates
selective runtime integration followed by a fixed-model serving comparison;
cache allocation, invalidation, residency and lifetime have not yet been
implemented or qualified. The retained NR8 runtime and target/draft artifacts
remain unchanged. All source and observations are under
`target/dense-tiled-byte-layout-evidence/`.

## Integrated selective dense Q8 cache prototype

An additional standalone width sweep holds K2048/M8192 fixed and tests every
activation width from nine through sixteen, both residual variants and both
weight-bank modes. It passes 22 numerical cases with 44 bit-identical paired
outputs and 88 FP64 reference checks, including the retained tail/long-K cases.
The sustained 32-case ABBA latency screen completes on attempt one. Every
rotating-bank case improves GPU latency, by 16.19–23.87%; fixed-bank cases are
mixed, including small regressions. These observations motivate a serving
experiment covering widths nine through sixteen, rather than a general kernel
replacement. Source, raw timings and the independent 128-pair audit are under
`target/dense-tiled-row-width-evidence/`.

Experimental core 0045 adds a buffer-owned cache for contiguous, original
Q8_0 K2048/M8192 tensors on shared-buffer M5 Max devices. It retains original
weight storage for all other operations and admits F32 activation widths 9–16
only. The packed path uses the existing geometry's NT4/RT2/NSG8 tiling and
original F32 arithmetic; its shader has a distinct entry point, checked against
the actual compiled library before use. Old precompiled/override libraries
without that entry retain original storage and dispatch.

The cache owns and tracks Metal resources in the source buffer's residency
set, bounds additional resources to 1 GiB per process, and declines allocation
when it would exceed the device's working-set limit or allocation fails. Repeated reads
reuse the packed resource. Any supported host write after cache creation
revokes caching for the source buffer, retaining old resources through buffer
destruction for pending GPU work. Async and graph writes revoke it even before
the first entry, so queued GPU results are never packed by the CPU. Graph
preflight ignores non-Metal buffers. Weight views and unsupported layouts use
the original path; no model-family or tensor-name selection is introduced.

Final native `2cca840a370ee22198db2384b373f014dc4076dd` cold-replays to intended
tree `65c5f9f653917cba8abbeaf8d92fefbc01fc5002` and builds through Just. Its
primary-library SHA-256 is
`32b5bf64d11589f2b891af8420d76b06c202bb582e605a6804d1ad662c0e8b6f`.
The private backend change does not alter the public Skippy ABI. The actual
native test covers cache byte preservation, reuse, allocation failure, global
quota/reclamation, source writes and fallback, plus 34 real Metal cases. The
same 34 cases pass with the internal opt-out; each arm includes sixteen full
bit-identical comparisons against original dispatch. Both packed residual
variants are proven loaded only in the enabled arm. Ten CPU-reference fallback
projections, the existing graph policy suite and 240 FA-vector cases also pass.
Static-test and packaged-runtime MMA shader sources hash identically.

The fixed-file serving comparison completes and independently audits 288
accepted requests, including 192 measured. Twenty-three complete blocks are
excluded for competing load or thermal state; 328 raw responses include those
attempts. The audit checks the exact block/prompt roster, ABBA ordering, every
accepted pre/post guard, response lengths, cache status and text hashes. Both
frozen native bundles, the shared CLI and all target/draft hashes are verified
before startup and after the performance processes exit.

| Concurrency | Retained NR8 HTTP tok/s | Packed cache HTTP tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 193.385 | 193.422 | +0.019% |
| 2 aggregate | 243.622 | 243.883 | +0.107% |

Cycle signs are mixed at both concurrencies; these point estimates do not
establish a useful serving gain. All measured text matches at concurrency one,
but not at concurrency two. Timing includes the full HTTP batch and artifact
validation, using resident servers rather than fresh-process qualification.

A separate debug-enabled diagnostic runs only after performance processes exit.
It validates eight additional 128-token completions with no cache hits and
observes the packed plain-projection pipeline loading after startup. Thus the
new path is exercised in real serving, not solely in the standalone probe;
the diagnostic is not used as a latency measurement or evidence of its decode
frequency. The prototype is rejected and experimental core 0045 removed from
the active queue. No additional decode gain is claimed. Builds, failed-fixture
diagnostics, corrected fixture, test logs, pre/post source pins, frozen candidate
bundle, archived patch, raw responses and independent audits are retained under
`target/dense-packed-runtime-evidence/`.

After removing the prototype, `just release-runtime-build metal` succeeds for
retained NR8. Its build metadata and all 24 native-library hashes again match
the frozen control; no packed-weight cache remains in the active native queue.

### Stream-bounded C2 decode diagnostic after cache rejection

A new retained-NR8 capture validates 32 fixed-file completions: sixteen warmup
requests followed by sixteen streamed requests. Eight disjoint windows begin
after both concurrent requests emit their first nonempty text and end at the
earlier last text event. All windows follow the final observed target shader
compilation. They total 7,690.977 ms; these client observation boundaries are
not exact GPU phase markers.

The saved trace exports successfully, but the recorder's own status file is
**54**, despite its log reporting a completed, saved trace. The TOC records
time-limit termination and a SIGKILL of the launched target. The outer recording
script's zero exit status does not prove recorder success. Consequently this
is a diagnostic ranking with a recording warning, not a clean capture,
complete-sample guarantee, throughput qualification or new performance gain.

The temporal submission-owner/PC join attributes 349,396 shader rows to the
target, leaving 1,413 target rows unmapped (99.597% of these target rows map).
One conflicting ownership interval is excluded. The leading per-kernel unions
inside the streamed windows are:

| Kernel | Union of observed intervals, ms |
| --- | ---: |
| Routed Q4_1, eight output rows | 2,005.927 |
| Dense Q8_0, NT2/RT2 | 692.077 |
| Dense Q6_K, NT4/RT2 | 651.820 |
| Gated delta net | 468.200 |
| Dense Q8_0, NT4/RT2 | 406.487 |
| Dense Q8_0, NT1/RT2 | 236.865 |

These unions overlap and must not be added or treated as exclusive wall-time
fractions or hardware utilization. The cache experiment targeted a subset of
the fifth entry, whereas routed Q4_1 remains the largest observed kernel union.
The diagnostic supports prioritizing routed arithmetic over another narrow
dense weight-layout experiment; it does not establish a speedup ceiling.

The next screen should evaluate direct affine-code MMA for routed Q4_1:
multiply its existing integer codes by F32 activations, then apply the stored
scale and offset in F32. This retains target/draft files and avoids the F16
dequantized-weight staging used by the rejected short-tile prototype. It still
needs routed gather/scatter, masked IDs, tail and numerical checks, realistic
expert occupancy, and a guarded comparison against retained NR8 before any
production integration. No gain or working implementation is claimed yet.

Raw streams, client timestamp records, saved trace, streamed export analyzers,
and an audit of response hashes, roster, scope geometry and attribution
consistency are under `target/retained-c2-decode-profile/`. The audit explicitly
retains the recording warning and does not independently re-export every GPU
sample. The active native runtime and patch queue remain retained NR8.

### Direct affine-code routed MMA feasibility screen

A standalone prototype now multiplies the stored Q4_1 integer codes, represented
exactly in half, by F32 activations with SIMDgroup MMA. It applies the original
block scale and offset in F32; it does not stage dequantized weights in half or
modify weight bytes. Four SIMDgroups split K. CPU-precomputed route tiles gather
up to eight entries per expert and scatter results to their original positions.
Routing construction and upload are excluded from timing, so this screen cannot
establish an engine or serving gain.

The original NR8 arithmetic body supplies the control. Both arms pass 24 routed
cases, 96 stored-weight FP64 references and 48 paired numerical comparisons.
F32 reduction order differs and output is not bit-identical. The cases cover
both expert projection geometries, 3/8/16 tokens, broadcast/per-selection inputs,
16/128 active experts, two ID frames and a masked entry.

A sustained 48-case ABBA screen completes on attempt two; attempt one is
rejected for competing MaxDaemon load. Each case has at least 20 ms of GPU
warmup, 30 ms of GPU timing and 1,024 dispatches per arm. The independent audit
checks source/binary hashes, all eight accepted pre/post guards, the exact
geometry roster, 192 raw timing pairs and reported ratios.

The candidate wins 16 of 24 concentrated-routing cases. With sixteen tokens
and sixteen active experts (roughly eight entries per expert), observed GPU
latency falls 47.09–59.27%; with eight tokens (roughly four entries per expert),
it falls 12.28–18.85%. With three tokens it regresses. All 24 cases with 128
active experts regress, by 33.59–147.64%. These are synthetic projection latency
changes, not serving results or a new headline metric.

This rules out an unconditional replacement. A separate selective screen keeps
NR8 for sparse routes and dispatches MMA only for tiles with at least four
entries. It includes the extra projection dispatch, but still excludes routing
construction/upload. It must pass numerical checks and mixed occupancy cases
before the cost of production GPU routing can be evaluated. Initial evidence,
driver, source and independent audit are retained under
`target/routed-affine-code-mma-evidence/`; the selective variant is under
`target/routed-affine-selective-evidence/`. Neither changes the active runtime.

The CPU-mapped selective variant passes its 24 numerical cases, then its timing
campaign is superseded before an accepted complete sweep. Only that owned
driver and its verified child are stopped. No latency result is accepted from
its interrupted campaign.

A further variant under `target/routed-affine-gpu-map-evidence/` builds routing
tiles and sparse IDs entirely on the GPU, writes indirect dispatch arguments,
then runs sparse NR8 and rich affine-code MMA projections. Every timed
projection includes the route scan, compaction, two explicit buffer barriers
and indirect MMA dispatch; there is no CPU route upload or readback in the
timed loop. It passes 24 numerical cases, 96 stored-weight FP64 references,
48 paired comparisons and 48 checks of GPU route ownership and indirect grids.

Its latency campaign uses 24 short, independently guarded ABBA blocks, each
covering fixed and rotating IDs for one geometry. Completed blocks survive
interference in later blocks. CPU/thermal samples before, after and every
100 ms during each process reject affected blocks. The campaign completes and
passes an independent audit of 24 ABBA blocks, 48 projection cases, 192 raw
timing pairs and 228 mid-run guard samples, as well as the correctness checks
above. With sixteen tokens and sixteen active experts, GPU projection latency
falls 21.37–28.27% across all eight cases, including routing and dispatch costs.
All other cases regress: 3-token cases by 14.39–32.15%, 8-token cases by
10.77–23.30%, and 16-token cases with 128 active experts by 5.43–10.68%.
These results restrict further work to sufficiently occupied routes; they do
not establish real-model route occupancy or a serving gain.

The initial implementation still dispatches sparse NR8 groups for rich entries,
which return after writing zero before MMA overwrites those outputs. A prepared
variant under `target/routed-affine-compact-evidence/` instead compacts sparse
entry indices and generates indirect grids for both projections. It is not yet
built into the native runtime. Its standalone Just build and numerical run
pass the same 24 cases, 96 FP64 references, 48 paired comparisons and 48 GPU
ownership/grid checks. Those ownership checks require every original entry to
appear exactly once across rich tiles and the compact sparse list, including
masked entries. Its latency campaign begins after the prior owned campaign
exits and passes its audit, so the two never measure concurrently. Its work is
isolated from the active retained runtime; no serving gain is claimed.

The compact campaign completes and passes its independent audit: 24 guarded
ABBA blocks, 48 projection cases, 192 raw timing pairs and 228 mid-run guard
samples. Its sixteen-token/sixteen-active-expert cases improve GPU projection
latency by 24.51–39.39%, including GPU mapping and both indirect dispatches.
All other cases regress: three-token cases by 6.62–32.08%, eight-token cases by
3.95–31.02%, and sixteen-token/128-active-expert cases by 4.88–11.14%. The
synthetic screen supports a narrow serving experiment, not general admission
or a new headline metric.

### Experimental native routed affine implementation

Experimental core 0045 introduces a separate private GPU route-compaction and
affine-code MMA capability; it does not replace the activation-register reuse
policy owned by core 0042. New host and Metal modules own the implementation.
Scratch belongs to the output allocation and is rebuilt from current graph IDs
on every operation; no route state is cached across graphs. Explicit barriers
preserve producer and temporary-buffer ordering on concurrent encoders.

Admission is limited to shared-buffer M5 Max devices, F32 activations, Q4_1
weights, sixteen token rows, eight selections, and the two screened projection
geometries. Weight/input/output tensors must be contiguous; expert IDs may be
strided and offset top-k views. The source can contain 128 or 256 experts.
Sparse entries retain NR8 arithmetic; sufficiently occupied tiles use direct
affine-code MMA. All unsupported cases retain existing dispatch. An internal
presence opt-out supports validation. Missing functions or unavailable pipeline
geometry decline before encoding; combined and override-library lookup uses
the actual library's function index. Public Skippy ABI remains 0.1.85.

The first native build exposes a scratch-size template type mismatch, corrected
before qualification. The initial 128-expert-only candidate passes its native
tests but does not activate on the serving target. Its debug diagnostic confirms
the target has 256 experts; no latency run is started for that candidate.
The mapper is then generalized to the admitted expert count, with a sufficiently
large threadgroup for the masked-entry owner. The 128-only bundle, diagnostic
failure and pins remain archived rather than being overwritten.

The corrected native source passes 48 actual Metal CPU-reference cases with the
feature enabled and the same 48 disabled. Coverage includes both expert counts,
both projection geometries, broadcast/per-selection inputs, strided/offset IDs,
masked and repeated expert IDs, adjacent token counts and strided-weight
fallbacks. All three new pipeline names load only in the enabled arm. Private
shape/device/scratch policy tests pass. A separate explicit hardware target
checks per-kind function discovery and two old/partial combined-library
fallbacks without issuing any encoder commands. That target is absent from
default CTest runs, which therefore gain no new GPU requirement.

Final cold replay matches intended tree
`a51242b233a9b2a4ae77ff0c3048d0ea5ada6d2f`; native SHA is
`d69322d0ceff0a2d9ce15d6a315e7027aeaeff8a`. The packaged primary-library SHA-256
is `9f0d2ef29e7fffd0b4d69b2c421b75ae5643983c0d581f9fd1cf56fcf9eb6da4`.
The Just native runtime build succeeds, and tested static/packaged dynamic
shader sources hash identically. A frozen versioned bundle preserves those
files for serving. The corrected real-model diagnostic validates eight further
128-token, zero-cache completions and observes all three affine pipelines
loading after startup. This diagnostic confirms activation, not latency or
decode frequency.

The fixed-file resident-server A/B is now running against frozen retained NR8,
with three warmup and three measured cycles per concurrency and predefined
load/thermal exclusions. Both servers must prove they loaded their respective
frozen library paths. Target, draft, shared CLI and all native library hashes
pass pre-run verification. No serving gain is claimed yet. The local queue and
dist package contain the experimental candidate; there is no deployment, push
or PR. Source, builds, failed and corrected diagnostics, pins, raw requests and
audits live under `target/routed-affine-native-evidence/`.

At the user's request, portable kernel work moves to an idle Dan-lab M4 mini
through Jianyang. Both available M4 minis have 16 GiB memory; the fixed target
file is about 23 GB, so a single mini cannot replace the full-model serving
comparison. No smaller model is substituted for that headline comparison.

M4 job `job_ypyds4xkda98vd4` succeeds at $0: the standalone 256-expert kernel
passes 24 cases, 96 FP64 references, 48 paired comparisons and routing checks.
It builds immutable source on the host through Just and transfers no model or
prebuilt binary. This research harness directly invokes the kernels; it does
not change the native M5-only admission policy. A subsequent timing job catches
a Ruby command-path issue caused by spaces in the job directory, before any
accepted timing. The corrected job `job_wzw9x77qteskk6d` is running on
`macmini-764jw9`, with both fixed/rotating IDs and GPU routing/indirect dispatch
cost included. Its source payloads, job records and logs are under
`target/routed-affine-m4-evidence/`. M4 projection timings cannot establish the
M5 full-model serving gain.

The corrected M4 timing job succeeds at $0. Its audit checks 24 guarded ABBA
blocks, 48 projection cases, 192 raw timing pairs, 776 mid-run guard samples,
96 FP64 references and 48 paired numerical comparisons. With sixteen active
experts, eight-token GPU projection latency falls 11.95–17.81%, and
sixteen-token latency falls 54.01–54.87%. Sparse sixteen-token/256-active-expert
cases regress by 0.65–2.12%. The control is the retained NR8 arithmetic body,
not a claim about the default shipped M4 dispatcher. These observations do not
establish real-model routing distribution or serving throughput.

The local M5 resident-serving campaign reaches its fixed 30-minute deadline
under competing workload. It accepts 124 requests, including 76 measured, all
at concurrency one; it never reaches the required concurrency-two dataset.
This is incomplete evidence, not a measured absence of gain. No complete
comparison, additional serving headline or BaseRT parity claim is made.
Post-run target/draft/shared-CLI and frozen-library pins still pass, and the
owned servers exit before verification. The inherited descriptive method label
in the executed driver is documented; its final summary is never produced and
no incomplete result is finalized as a serving gain.

The candidate remains unpromoted. Its complete patch is archived as
`target/routed-affine-native-evidence/unpromoted-0045.patch`, and core 0045 is
removed from the active queue. `just release-runtime-build metal` succeeds for
retained NR8; all 24 native-library hashes and build metadata match the frozen
control exactly. No owned serving process remains. The M4 projection result is
promising research evidence, while full-model qualification remains outstanding.
Nothing is pushed and no PR is raised.

The dedicated M1 Ultra comparison subsequently completes in Jianyang job
`job_6q2p4afd6294hpz`, run `run_2s2s03ngfkfsaqs`, at $0. Its private native
candidate admits M1 Ultra in addition to M5 and otherwise retains the same
sixteen-token geometry and original weight bytes. Native correctness,
real-model activation, 288 completion validations (192 measured), zero rejected
blocks, independent serving audit, and pre/post frozen-file pins all pass.
The measured C1 result is 124.8793 versus 124.8087 tokens/s (−0.0565%); C2 is
146.2881 versus 144.7085 aggregate tokens/s (−1.0798%). C1 text is identical;
C2 has one prompt with differing output. This candidate supplies no useful
serving improvement and remains unpromoted. These are resident M1 measurements,
not fresh-process qualification or a conclusion about M5 throughput.

Digest-verified server artifacts permit a post-hoc telemetry audit without
another timing run. Each measured arm/concurrency contains 48 requests and
6,144 output tokens; all 144 request-decode records per server match the full
warmup/measured roster, and telemetry reports no dropped events or export
errors. In C2, shared verification covers 89.87% of control request-windows
and 90.79% of candidate request-windows. Aggregate draft acceptance is 51.98%
versus 52.60%. C1 proposal, acceptance, and window counts match exactly. Thus
neither absent shared batching nor lower aggregate draft acceptance explains
this result. Summed per-request C2 verification elapsed time rises from
61,173.785 to 62,728.966 ms; these overlapping request totals are not GPU time
and do not establish which kernel or synchronization cost caused the regression.

Existing shared-verification events report request count, not token rows or
expert occupancy. They cannot establish how often the sixteen-token native
path runs or whether real routes have the synthetic screen's reuse. A temporary
opt-in graph callback and eight-request streamed occupancy diagnostic are
prepared under `target/routed-affine-density-evidence/`. It reads graph-produced
expert IDs after scheduler synchronization, validates tile-count conservation,
and groups projection observations inside intervals after both requests emit
their first text. Synchronization perturbs timing and may change scheduler
cohorts, so its results cannot qualify throughput or natural shape frequency.
The diagnostic is not submitted while the approved M1 is occupied by unrelated
CI. The active local native source remains retained NR8 at `e795659936a8bb5fd0554d4ee7b788fd6f78b6be`;
no diagnostic or candidate is promoted, pushed, or deployed.

Reviewing the standalone screen also exposes an execution-context limitation:
its encoder uses default serial dispatch, whereas the digest-verified M1
activation log reports `use concurrency = true` for both target and draft.
This does not invalidate the isolated serial projection measurements, but they
do not represent the native concurrent graph's synchronization or overlap.
A separate $0 M4 job, `job_rg4j2vv5axbkpe5`, tests one specific dependency:
sparse and rich dispatches write disjoint output rows, so it compares the same
affine kernels with and without their intervening buffer barrier. Both arms
use concurrent encoders and an explicit end-of-projection barrier to protect
output/scratch reuse between timed iterations. The mapper dependency remains
protected in both arms. Numerical qualification compares the barrierless arm
with NR8 and validates complete, unique route ownership before timing. This is
a synchronization screen, not a full-graph or serving throughput result.

That M4 job completes successfully at $0, run `run_zmg2kvc22ec8gev`. Its audit
validates 24 ABBA blocks, 48 projection cases, 192 raw timing pairs, 747
mid-run guard samples, 48 route checks, 96 FP64 references and 48 numerical
paired checks. For sixteen-token/sixteen-active-expert cases, removing the
sparse-to-MMA barrier reduces projection latency by only 0.0667–0.9799%.
Sparse sixteen-token cases range from a 1.0143% reduction to a 0.0963%
regression. Eight-token/concentrated cases improve 2.47–4.72%, but these are
outside the currently qualified native admission. This does not justify
promoting the failed full-serving candidate or forecasting additional TPS.
The result and digest-verified artifacts are retained under
`target/routed-affine-barrier-evidence/`.

The occupancy diagnostic is submitted as `job_z40g0v6adxc8a5c` with a $0 cap
after the approved M1 reports idle. Admission succeeds, but a subsequent agent
snapshot shows another CI job running and the diagnostic still queued with no
attempt. It has not built or executed yet. No competing job is interrupted and
the same diagnostic handle is retained rather than resubmitted.

The occupancy job later starts but fails during the native build: its temporary
module includes `ggml-impl.h`, which the llama library's include path does not
provide. No diagnostic requests execute. The corrected module uses the existing
`llama-impl.h` / `LLAMA_LOG_INFO` interface, as other Skippy modules do. A repair
spec requires the exact failed native tree and patch digest before replacement;
it cannot silently overwrite a different checkout. The original failed spec,
patch and log are preserved. Corrected intended tree is
`2597d4c1c6d3f71aca066489cb5994c14c2faa42`; the repair remains unsubmitted while
CI owns the M1.

A separate concurrent two-projection screen completes successfully on the M4
in $0 job `job_m1ctj1gr42cp0ea`, run `run_nz30fgkfref2nhg`. Both projections
share input/ID buffers and have private weights, outputs and route scratch.
NR8 can dispatch the siblings concurrently; the candidate retains both internal
global barriers. Both arms protect output/scratch reuse with an end-of-pair
barrier. The audit validates 24 guarded ABBA blocks, 48 timing cases, 192 raw
timing pairs, 1,566 mid-run guard samples, 96 route checks, 192 FP64 references
and 96 paired numerical comparisons. Submitted and retrieved probe/harness/audit
source hashes match. The timing unit is a projection pair, not an individual
kernel. Sixteen-token/sixteen-active-expert pair latency falls 54.17–55.26%;
sparse sixteen-token cases regress 1.55–2.24%. This screen preserves the strong
concentrated-route gain even with independent sibling work, but cannot rule out
other whole-graph concurrency costs or establish real-model expert occupancy.
The K2048/M512 geometry represents gate/up sibling work; K512/M2048 pairs are
synthetic independent work. Artifacts remain under
`target/routed-affine-pair-evidence/`. Full serving still supplies no additional
retained gain, and neither experimental native variant is promoted.

The corrected occupancy module passes a local Just-driven C++17 syntax compile
against the actual native-stage headers with `-Wall -Wextra -Wpedantic -Werror`.
This checks the logging/include repair without a full local runtime build or
model execution. After confirming there is no Jianyang lease reserving the M1,
the repair is queued at priority 1 as `job_9xnyqyextbjayg9`, with a $0 cap.
External CI currently occupies the agent; it is not interrupted. This is a new
immutable repair spec for the terminal build failure, not a duplicate of a
live diagnostic attempt. The latest authoritative job read still reports
queued with no attempt, so no occupancy result is claimed.

The corrected occupancy run subsequently succeeds at $0: job
`job_9xnyqyextbjayg9`, run `run_t9x378gwjt09jd7`. Its native source cold replay,
loaded-library path check, eight zero-cache 128-token streamed completions,
and post-run target/draft/CLI/frozen-library pins pass. Digest-verified artifacts
are retrieved, and the occupancy audit is rerun locally against the full saved
server log. All 16,380 diagnostic sequences are present and uncapped; ID bounds
and tile/entry conservation pass. Four intervals after both requests' first
text contain 13,923 projection records. Their sixteen-token observations are:

| Projection | Records | Mean active experts | Entries in tiles of at least four |
| --- | ---: | ---: | ---: |
| K2048/M512, broadcast inputs | 7,360 | 54.12 | 45.63% |
| K512/M2048, per-selection inputs | 3,404 | 53.78 | 46.11% |

Those records have contiguous weights/inputs/outputs and no output view, so the
observed shapes satisfy those candidate admission requirements. All scoped
sixteen-token records together number 10,764. This is a synchronized,
projection-weighted observation, not an estimate of uninstrumented shape
frequency or independent router samples. It reveals a substantial mismatch
with the favorable sixteen-active-expert synthetic screen: more than half of
entries still take the sparse path. It does not alone prove why serving fails
to improve.

`prepare-profiles.rb` validates anonymous route reconstructions for all 10,764
observations and selects 32 rich-entry quantiles per projection. Each selected
profile preserves active-expert count, aggregate eight-entry tile histogram,
rich-entry count, and unique top-k IDs within each token. Expert identities,
token correlations, and associations between full tiles and their tails are
synthetic; histograms do not uniquely determine those associations. Selected
profiles span 24–93 active experts and 0–104 rich entries out of 128. A $0 M4
screen is submitted as `job_6tw3y38jsqv7r2r` to compare unchanged NR8 and
affine kernels under these profiles, with a concurrent encoder and protected
output/scratch reuse. Fixed routing uses the lowest-rich quantile; rotating
routing cycles all 32 profiles. This is not exact route replay or full-model
throughput qualification. Artifacts are under
`target/routed-affine-density-evidence/` and
`target/routed-affine-observed-evidence/`.

The observed-occupancy synthetic M4 job succeeds at $0, run
`run_y0bv91whjpxs4r7`. Its audit passes eight ABBA blocks, sixteen projection
timing cases, 64 raw timing pairs, 309 mid-run guard samples, 48 route checks,
96 FP64 references and 48 paired numerical comparisons. Submitted and retrieved
probe, harness, audit and profile-header hashes match. Concentrated synthetic
control cases still improve 53.93–54.71%, but rotating reconstructed profiles
produce much smaller changes:

| Projection | Reconstructed rotating-profile GPU latency change |
| --- | ---: |
| K2048/M512, broadcast input (gate/up geometry) | +0.47% |
| K2048/M512, per-selection input (synthetic cross-case) | −1.75% |
| K512/M2048, broadcast input (synthetic cross-case) | −3.92% |
| K512/M2048, per-selection input (down geometry) | −2.77% |

The fixed lowest-rich quantile regresses 6.47–12.04%. These synthetic
observations are consistent with the failed serving transfer, but are not
causal proof for the M1 full model or exact route replay. They substantially
weaken the case for further integration of the mapper/compact-sparse/MMA
candidate as designed. A next kernel approach should reduce overhead on
sparse expert work without assuming the sixteen-active-expert reuse pattern;
no further serving gain is claimed and the retained engine remains unchanged.

### Rejected direct diagonal affine matrix screen

A standalone alternative removes route mapping entirely. Each 32-thread SIMD
group forms a matrix product whose rows belong to independent selected experts,
then consumes only diagonal outputs. Quantization codes remain exact half
integers; scales, offsets and activations retain F32 arithmetic. The kernel has
no route scratch, indirect dispatch or internal barrier. Job
`job_9c5cd2x3g9n8jv4`, run `run_9pmfkgg951w6n3b`, succeeds on the M4 at $0.
Poisoning outputs before candidate execution verifies complete finite writes
and masked zeros; 48 coverage checks, 96 FP64 references and 48 full paired
numerical checks pass. Eight guarded ABBA blocks cover sixteen timing cases,
64 raw timing pairs and 587 mid-run guard samples. Despite removing routing
overhead, concentrated cases regress 188.47–226.25% in GPU latency and
reconstructed-profile cases regress 73.68–185.28%. This approach is rejected;
unused matrix arithmetic does not pay off in this screen. Evidence is under
`target/affine-diagonal-evidence/`.

### Wider vector reuse and portable NR8 qualification

The next M4 screen changes only accumulated output rows per SIMD group from
eight to sixteen or thirty-two, retaining original bytes, lane assignment and
per-output arithmetic. Separate numerical qualification precedes each guarded
timing phase. Job `job_t3f820wf47402j9`, run `run_68mfspk3vx7ht9r`, completes
at $0. With rotating reconstructed profiles, NR16 gate/up latency regresses
7.50%, while down changes −0.39%. NR32 regresses 27.74% and 7.04%, respectively.
Synthetic cross-cases also offer no consistent gain. Neither widening is
integrated. Both phases' audits and raw artifacts are under
`target/affine-vector-width-evidence/`.

Source review also clarifies a hardware distinction in earlier comparisons:
the retained native source contains NR8, but `ggml-metal-routed-tuning.h`
admits that Q4_1 width only on M5 Max. Native M1/M4 fall back to four rows
(`N_R0_Q4_1 = 4`); the captured M1 debug log loads the unsuffixed routed Q4_1
pipeline. Thus descriptions of the retained engine as an NR8 source baseline
do not mean that its M1 full-serving control executes NR8. The standalone
screens explicitly compare NR8 arithmetic and cannot establish a speedup over
the shipped M1/M4 kernel. A direct NR4/NR8 comparison is submitted on M4 as
`job_yrzgn1g1m9x6sdm` and on approved M1 Ultra as `job_p47j2py4496d4t0`.
Both use identical source, host-native builds, numerical/write coverage checks,
and concentrated/reconstructed-occupancy profiles. The old arm is NR8 and the
packed label denotes NR4, so a positive reported packed-arm latency means
NR8 is faster. These remain synthetic operation screens, not serving metrics.

Both portable NR4/NR8 jobs complete successfully at $0: M4 run
`run_fh6sqq4q115f4zs` and M1 Ultra run `run_y9tgs33t5m4adjd`. Each audit
validates eight ABBA blocks, sixteen timing cases, 64 raw timing pairs, 48
poisoned-output/masked-zero coverage checks, 96 FP64 references and 48 paired
numerical comparisons. Retrieved shader hashes and probe/harness/audit sources
match the submitted source. Numerical differences remain nonzero, so these
checks are not bit-equality or full-model quality qualification.

| Device | NR8 versus NR4 gate/up latency, rotating reconstructed profiles | NR8 versus NR4 down latency, rotating reconstructed profiles |
| --- | ---: | ---: |
| M1 Ultra | +3.60% | −6.30% |
| M4 | +1.47% | −3.86% |

Gate/up uses K2048/M512 with broadcast input; down uses K512/M2048 with
per-selection input. The M1 down improvement also holds for the fixed sparse
profile (−6.24%); the M4 fixed sparse down case changes only −0.32%. This
supports investigating a narrow down-only policy, not generally enabling NR8
on either device. No serving TPS gain is established.

An isolated native worktree over retained `e795659936a8bb5fd0554d4ee7b788fd6f78b6be`
contains an unpromoted policy prototype. It selects the existing NR8 shader
only on M1 Ultra/base M4 shared buffers, Q4_1 K512/M2048, exactly sixteen token
rows, eight expert selections and two SIMD groups. Existing M5 policy and all
other cases retain their previous choices. Dispatch-boundary tests are added
in the existing policy suite; they have not executed yet. The prototype's
intended tree is `986e05c36526af9b6cf4a73d0f47db173f8d2a3f`. It introduces no
shader or public ABI change. Before any promotion, actual native backend
correctness/activation, fixed-file serving comparison and cold replay remain
required, and the change must be folded into its owning core 0042 patch. The
active local source and queue remain retained; nothing is pushed or deployed.

The portable prototype gains an opt-in native backend fixture module with 35
cases, using the suite's actual CPU references. Cases include broadcast and
per-selection inputs, 16/256 experts, masked and repeated IDs, offset/strided
ID views, strided expert weights and adjacent-width/geometry/type/selection
fallbacks. Default backend test rosters are unchanged unless
`SKIPPY_TEST_PORTABLE_R8` is set. Both updated native test translation units
pass a local Just-driven C++17 syntax compile. This does not execute their
policy assertions or GPU comparisons. Updated intended prototype tree is
`68ac922e99396abf34ab969a26c64a7fda003fb5`.

Jianyang job `job_7x957p7v5ghfw77` is submitted at priority 1 with a $0 cap for
native build/correctness, real-model activation and fixed-file serving ABBA on
the approved M1. It requires the exact prior diagnostic source tree, committed
root source, old private-patch digests and frozen-file pins before archiving
and replacing its own previous research patches. Retained control libraries,
target, draft and shared host-built CLI are preserved in a new evidence
directory. The method labels name native NR4 control and the down-only NR8
candidate, rather than the earlier mapper/MMA experiment. Both static-tested
and packaged vector shader sources must match before serving. Three warmup and
three measured eight-prompt cycles at C1/C2 retain within-prompt ABBA ordering,
predefined whole-block interference exclusions, loaded-library checks and
pre/post file pins. Latest authoritative job state is queued with no attempt;
no new native correctness, activation or serving result is claimed yet.

The first job starts but its clean-checkout guard rejects four untracked
native preparation markers before mutating any patches or running tests.
Read-only job `job_zjr426rvyw28p3q` confirms there are no tracked native changes;
the native tree still matches the prior diagnostic. The repaired guard checks
the exact four marker names and their contents (upstream SHA, patched SHA,
patch digest and preparation schema 5), while continuing to reject all other
changes. It does not delete or ignore an arbitrary dirty checkout.

Corrected job `job_y2gqn1kzwdraw5p`, run `run_9x5myy3r8r8k2za`, succeeds at $0.
Cold replay matches prototype tree `68ac922e99396abf34ab969a26c64a7fda003fb5`.
The native policy suite and all 35 actual Metal CPU-reference cases pass,
including observed NR8 and fallback pipelines. Tested static and packaged
vector shader sources match. Eight real-model diagnostic completions validate
activation of `kernel_mul_mv_id_q4_1_f32_r8_nsg=2_split=0`. All pre/post
target/draft/shared-CLI and frozen-library pins pass. The resident comparison
completes 288 valid requests, 192 measured, with no rejected blocks:

| Concurrency | Frozen native NR4 control | Down-only NR8 candidate | TPS change |
| --- | ---: | ---: | ---: |
| 1 | 125.0321 | 125.0424 | +0.0082% |
| 2 aggregate | 146.9441 | 148.4653 | +1.0352% |

All measured output text matches exactly across arms at both concurrencies.
All three C2 candidate cycles beat their corresponding control cycles
(148.5936/147.8473/148.9595 versus 147.0617/147.0450/146.7262). All 288 raw
responses, batch timing records and load/thermal guards are retrieved with
artifact SHA-256 verification. Running the executed audit locally against
those raw files reproduces the complete remote audit exactly. This is a small
M1 resident-serving improvement, not an addition to the M5 cumulative headline,
fresh-process confirmation or BaseRT parity.

Job `job_zyq446m48tse421` is submitted at $0 to repeat C2 from two newly started
pairs of the same frozen control/candidate servers. Each restart retains three
warmup and three measured eight-prompt cycles and within-prompt ABBA; this adds
two fresh starts per arm and 192 measured C2 requests across both repeats.
Independent raw-response/guard audits and pre/post frozen-file pins remain
required. No build or GPU correctness work runs during measurement. The
confirmation measures warmed serving across process starts, not cold-start
latency. The prototype remains outside the active local queue until that
confirmation and ownership folding are complete.

## Retained portable down-only NR8 policy

Fresh-process confirmation job `job_zyq446m48tse421`, run
`run_7xk88szwax20bxm`, completes successfully at $0. The two new resident
process pairs independently improve M1 C2 throughput by 0.8610% and 1.3366%.
Their combined control/candidate rates are 147.0620 and 148.6776 aggregate
tokens/s (+1.0986%). All 288 requests validate, including 192 measured;
no blocks are rejected. Measured output text matches across arms and process
starts. Pre/post frozen-file pins pass. All raw responses and guards are
retrieved with artifact SHA-256 verification, and both independent audits and
the pooled result are recomputed locally from those files. This confirms a
small warmed-serving gain across process starts, not cold-start latency.

The actual M4 native qualification also passes: job `job_ee9qgsnp7528rhs`, run
`run_kw5wzd3cbsp0mj7`, uses exact committed root source plus the pinned native
prototype, builds through Just on the mini, and passes the native policy suite
and all 35 CPU-reference Metal cases. Its native tree matches the M1 tested
tree `68ac922e99396abf34ab969a26c64a7fda003fb5`. No model or prebuilt binary
is transferred; the M4 result is operation qualification, not serving TPS.

The change is now folded into owning core 0042. Core 0043's test-list insertion
is rebased to retain its convolution fixtures alongside the new opt-in routed
fixtures. An isolated full queue replay, then the active local Just native
build, both reproduce the exact benchmarked prototype tree. The folded native
commit is `57450b1e3fc0ad06c2060dfd994a4f3c9812e59a`, with queue digest
`7095c93fb5def828bd69cb28ed999a42a5be8e755322917b502cbd7a951de518`.
The local M5 policy suite and the same 35 backend reference cases pass. Tested
static and packaged vector shader sources match, and the vector shader file
is byte-identical to the retained pre-change source. All 24 packaged library
hashes verify. Public ABI remains 0.1.85 and runtime label 0.76.1; the packaged
primary-library SHA-256 is
`4d0ff11e0aba5b5a988b5ce389d9f49dbf90f71ca73a3560fc8c76139c440039`.

This retains existing NR8 arithmetic only for the qualified M1 Ultra/base M4
Q4_1 down shape: K512/M2048, sixteen token rows, eight expert selections and
two SIMD groups on shared buffers. No additional M5 throughput or BaseRT parity
is claimed. The integrated package and audit are frozen under
`target/affine-portable-r8-evidence/native-integrated/` and
`target/affine-portable-r8-evidence/integration-audit.json`. Nothing is pushed
and no PR is raised.
