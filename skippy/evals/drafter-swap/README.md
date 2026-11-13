# Drafter swap: per-workload speculative drafters

Experiment: would swapping the DFlash draft per workload (HTML, code review, code edit,
Rust, agent turns, prose) make Max faster than one general draft?

## Step 1: headroom

```bash
python3 -I build_corpora.py --src ~/.cache/drafter-swap-src   # writes workloads.jsonl
./submit.py                                                    # one jy job per dan-lab Spark
```

Each job measures Qwen3.6-35B-A3B (UD-Q4_K_M) on its workloads with three arms on one
machine: `off`, `dflash` (the shipped DFlash2 draft, adaptive blocks up to 7) and
`ngram` (n-gram suffix speculation). Greedy, thinking off, 512 tokens per reply, 40
prompts per workload. The agent workload replays `../agent-transcripts` at 64K context.

`arms.jsonl` in each job's artifacts keeps every reply. The `off` replies are the target
model's own greedy output, so they are drafter training data and the reference the other arms
are checked against.

If one draft accepts about as well on every workload, specialist drafts have little room to
win. A low-acceptance workload where n-gram does well suggests trying copy-from-context drafting
before training anything.
