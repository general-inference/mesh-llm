# Drafter swap: per-workload speculative drafters

Would swapping the DFlash draft per workload (HTML, code review, code edit, Rust, prose,
agent turns) make Max faster than one general draft? Measured on dan-lab DGX Sparks with
Qwen3.6-35B-A3B UD-Q4_K_M and its shipped DFlash2 draft: greedy, thinking off, 512-token
replies, 40 held-out prompts per workload, one request at a time.

## Round 2: overfitting, complete replies, breadth (2026-11-14)

Round 1's drafts were trained for 3 epochs on replies cut at 1024 tokens, with no held-out
data, and the HTML draft saw one prompt style. Round 2 fixes each of those.

**Complete replies.** HTML pages run a median ~4.2K tokens (max ~10K); at 4096 tokens half
still stopped early, so capped replies were continued to completion through
`/v1/completions` (`continue_replies.py`, checked against finished replies first): 998 of
999 HTML v2 replies and 599 of 600 code reviews end on their own.

**Overfitting.** A 10% held-out split now reports validation accuracy and loss:

| Data | 1 epoch val acc | 2 epochs | 3 epochs | Verdict |
|---|---:|---:|---:|---|
| rust_gen (470 rows) | 0.849 | 0.857 (train 0.925) | – | memorises from 2 epochs |
| code_review v2 (600) | 0.536 | 0.544 | 0.543, loss up | 3 epochs overfits |
| html_v2 (999 long pages) | 0.833 | 0.851 | 0.860 | still improving |

Round 1's 3-epoch drafts were partly memorised; with enough data, 2 epochs is the served
optimum (3 is no faster).

**Breadth.** v1 (Tailwind pages only) against v2 (four formats, complete pages, 2 epochs),
served decode change against the shipped draft (`results/round2/html-v2-by-format.md`):

| Format | v1 | v2 |
|---|---:|---:|
| Tailwind page | +57% | +48% |
| Plain-CSS page | +2% | +11% |
| Page edit | +1% | +14% |
| React + Tailwind component | −2% | +7% |
| Real GitHub HTML/JSX/Vue edits (`html_ood`, no training source) | 0% | +6% |
| Prose, code review | ~0 | ~0 |

Token traces (`trace.sh`, `trace_table.py`) show v1's gain is almost all Tailwind class
strings (class/style tokens from the draft 57% → 83%; other kinds +0–7 points).

**Q4 capture.** Features from the served Q4_K_M target (`llama-spec-bench --dump-features`)
match bf16 at cosine 0.987–0.9997 by layer; a draft trained on them performs the same as the
bf16-trained one, so bf16 capture stays.

**Code review does not specialise.** Acceptance stayed 0.39–0.41 for every variant; 2–3
epochs cost 6–9% elsewhere.

**One coding draft beats the specialists.** `coding-v1` is the shipped draft fine-tuned for
2 epochs on all nine coding sets at once (5,860 complete replies: Rust/TS/Python/Go edits,
TS/Go/Rust generation, HTML v2, code review, agent turns cut at their first tool call;
`build_coding.py`). Validation improves on every source from 1 to 2 epochs (0.820 → 0.829).
Served, against the shipped draft on one Spark (10 prompts per workload, 1024 tokens;
`results/round2/coding-draft-evals.txt`):

| Workload | Shipped | coding-v1 (2 epochs) | Acceptance | HTML-only draft |
|---|---:|---:|---|---:|
| html_v2 | 107.5 | 129.6 (+21%) | 0.54 → 0.67 | 114.1 (0.67) |
| agent turns | 30.4 | 33.9 (+12%) | 0.40 → 0.57 | 31.2 |
| rust_gen | 134.9 | 146.6 (+9%) | 0.73 → 0.81 | 121.8 |
| gen_go / gen_ts | 139.5 / 134.2 | +7% / +4% | 0.84 → 0.90 / 0.76 → 0.77 | below shipped |
| edits (Py, Go, real HTML) | | −2…+1% | unchanged | ~0 |
| prose | 74.6 | 76.1 | 0.37 → 0.32 | 72.7 |

Over the ten coding workloads at equal token shares, coding-v1 decodes **+8.1%** faster than
the shipped draft (+6.7% without agent turns); the HTML-only draft nets −0.1%. One coding draft
needs no routing or swapping among coding workloads; the swap API is only useful for switching
to the shipped draft for chat, if prose acceptance (0.32 against 0.37 here, 10 prompts) holds
up as a real loss. Agent turns still decode slower with speculation than without (33.9 against
39.9 off): long-context verification, not the draft, limits them.

### Bugs found (not filed: the hand-off server was unreachable)

- Served `ngram-suffix` never drafts on local `mesh-llm serve` for this target (`draft_n` 0,
  decode equal to off), with the documented config and with `SKIPPY_SPECULATION_GATE=0`.
  Offline, the same replies would accept ~2.9 tokens per step on code edits.
- Four lanes with DFlash and ~5K-token prompts killed `mesh-llm serve` without a log line, as
  decoding started (`continue_replies.py` at parallel 4); one lane works.

## Round 1 results

**A specialist draft pays where the general draft is weak and the output is regular.** The
HTML draft lifts HTML decode by 54%; code review, Rust and code edit gain 3–7%. Swapping costs
about 1 µs with drafts kept open, and replaying the last 2048 feature rows into the new draft
(~130 ms) removes the cost of starting it cold. Routed by task kind over a mixed workload,
decode is 5.6–8.7% faster than one general draft. (Round 2 shows the HTML number came from one
prompt style and the 3-epoch drafts were partly memorised.)

### 1. Headroom with the shipped draft

| Workload | Off tok/s | DFlash2 tok/s | × | Accepted / drafted |
|---|---:|---:|---:|---:|
| html | 62.7 | 82.8 | 1.32 | 0.49 |
| code_review | 61.9 | 69.7 | 1.13 | 0.49 |
| prose | 63.7 | 81.8 | 1.28 | 0.45 |
| code_edit | 59.7 | 131.4 | 2.20 | 0.89 |
| rust_gen | 63.6 | 130.2 | 2.05 | 0.71 |
| agent 4–8K | 26.7 | 28.1 | 1.05 | 0.61 |
| agent 8–16K | 22.4 | 31.7 | 1.42 | 0.82 |
| agent 16–32K | 23.2 | 28.2 | 1.22 | 0.80 |
| agent 32–48K | 24.1 | 20.7 | 0.86 | 0.69 |

Agent turns accept well but gain little, and lose at 32–48K: verifying at long context, not
draft quality, limits them.

Copy-from-context (suffix n-gram, `ngram_sim.py` on the same replies, min 5, 7 tokens per
window) accepts 2.92 tokens per verify step on code_edit, 0.7–1.1 on long agent turns, 0.56
on rust_gen and under 0.1 on html, code_review and prose. The served `ngram-suffix` strategy
never drafted (`draft_n` 0, decode equal to `off`), even with the documented config and the
speculation gate off, so it has no measured speed here.

### 2. Specialist drafts

DFlash2 fine-tuned from `incoai/Qwen3.6-35B-A3B-DFlash2` on the target's own replies to
held-out training prompts (`data/`), 3 epochs at 1e-4, with the models repo `drafter-swap`
pipeline. Each row compares arms from one job on one machine.

| Draft (training replies) | On its workload | Accepted / drafted | Other workloads |
|---|---|---:|---|
| html (600, 614K tokens) | 95.6 → **147.2** tok/s (+54%) | 0.47 → 0.81 | prose −0.5%, code_review −0.5% |
| code_review (600, 605K) | 79.0 → 81.2 (+2.8%) | 0.46 → 0.40 | html −2.2%, prose −1.3% |
| code_edit (1195, 309K) | 127.7 → 136.5 (+6.9%) | 0.89 → 0.89 | rust −2.5%, prose −4.0% |
| rust_gen (470, 87K) | 134.5 → 138.1 (+2.7%) | 0.71 → 0.75 | code_edit +1.4%, prose −2.1% |

The general draft's speed on one workload varies up to ~15% between runs (html 82.8 and 95.6
on the same Spark), so only within-job ratios are comparable.

### 3. Swap cost (`swap_probe.sh`, `probe_table.py`)

| Swap | Time | Next 256 tokens vs no swap |
|---|---:|---:|
| load the draft from its file | ~620 ms | as resident |
| resident (`ParkedDFlashDraft`) | ~1 µs | |
| … at a new request (draft sees the whole prompt) | | 1.00 |
| … mid-conversation, draft starts cold | | 0.76 |
| … after 256 / 1024 / 2048 new rows | | 0.87 / 0.95 / 0.96 |
| resident + replay of the last 2048 feature rows | 17 ms (short) – 130 ms (2048 rows) | **1.00** |

The draft's KV is built from target layer inputs injected after every decode, so a swapped-in
draft knows nothing until it is fed rows. `StageSession::dflash_keep_history` keeps the rows
(~64 KB each, 128 MB per lane at 2048 in f32) and `dflash_restart_replaying` feeds them in.

### 4. Routing over a mixed workload (`compose.py`, `results/`)

400-token replies, the workload changing between requests with probability 0.3, specialist
speeds from step 2 (unmeasured draft/workload pairs at the worst measured off-workload loss,
−4%).

| Mix | Swap | General draft | Route by task kind | Classifier, 10% wrong |
|---|---|---:|---:|---:|
| equal shares | at request start | 97.5 tok/s | 106.9 (+9.6%) | 106.4 |
| equal shares | resident + replay | 97.5 | 106.0 (+8.7%) | 105.1 |
| equal shares | load from file | 97.5 | 102.7 (+5.3%) | 100.3 |
| coding (edit 30, review 20, rust 20, prose 20, html 10) | resident + replay | 99.7 | 105.3 (+5.6%) | 104.7 |

The HTML draft alone gives +6.8% of the +8.7%. Routing by what Max already knows (the
routine, the file type, the tool) matches the oracle; a classifier wrong 30% of the time still
gains 4.4%.

Not done: the serving path (`mesh-llm serve`) does not swap drafts yet; the swap and replay
live in `skippy-runtime` and are exercised by `llama-spec-bench --swap-probe`. No Metal
cross-check: the dan-lab Macs cannot hold the 22 GB target.

## Running it

```bash
python3 -I build_corpora.py --src ~/.cache/drafter-swap-src   # workloads.jsonl
python3 -I build_corpora.py --src ~/.cache/drafter-swap-src --train 1200   # train-prompts.jsonl
./submit.py                                   # step 1: off / dflash / ngram per workload
./submit.py --phase datagen [--limit 600]     # target replies to the training prompts
./fetch_data.py <datagen job> ...             # copy them into data/
./submit.py --phase headroom --only <spark> --domains html,prose --arms "off dflash spec_html" \
  --specialists "html=/home/gi/.cache/gi-drafters/specialists/html-v1.gguf" --tag spec-html
./submit.py --phase swap [--probes "resident replay"]
./table.py; ./probe_table.py <swap job>; ./compose.py --matrix results/compose-matrix.json --swap-cost-s 0.13
```

Fine-tunes run from general-inference/models, branch `drafter-swap`:
`./submit.py html-v1 '<raw data URL>#sha256=<hex>' --node <spark>`.
