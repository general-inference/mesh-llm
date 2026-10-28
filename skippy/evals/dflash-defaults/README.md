# DFlash default speculation benchmark

Reproduces the table in general-inference/mesh-llm#1: decode speed with speculation off,
with the best fixed DFlash block, and with the shipped default (no speculation config), for
each model DFlash is on by default for, on a DGX Spark, an M3 Ultra and an M6.

```bash
cd skippy/evals/dflash-defaults
./bench.py run --rev origin/main          # submit, wait, print the table (several hours)
./bench.py submit --only qwen3.5-9b       # or one model, then later:
./bench.py status
./bench.py table                          # writes dflash-bench-out/table.md and results.json
./bench.py record --label main            # store this commit's runs in MLflow
./bench.py compare <commit-a> <commit-b>  # per-model change between two recorded commits
```

`--machines Spark,M6` limits machines. `--rev` must be pushed to
github.com/general-inference/mesh-llm and contain this directory, because each job builds
that commit and runs `run.sh` from it. You need a signed-in `jy` CLI.

## What each job does

One jy job per model and machine (`matrix.json`). Jobs for the same machine kind spread over
its nodes; the M3 Ultra runs its four jobs one after another.

1. `build.sh` builds the release host and the machine's native runtime.
2. `run.sh` downloads the target and the fixed-arm draft from pinned Hugging Face revisions
   and checks their SHA-256. It doesn't use the jy model store, which has returned damaged
   copies of large files.
3. It starts the server once so the default path fetches its own draft in the background,
   and waits for that file.
4. Arms, each a fresh server: `off`, `default`, every fixed block in `fixed_blocks`
   (adaptive sizing off), then `off_t07` and `default_t07` at temperature 0.7. Each arm
   records the GPU clock and temperature first.
5. `measure.py` streams the 77 prompts of the `report` split of
   `skippy/evals/spec-prompts` (seeded, 256 tokens, one at a time). Decode tok/s is total
   tokens after the first divided by total time from first to last token. It prints a
   `DFLASH_BENCH` JSON line, which `bench.py table` reads from the job log.

## Agent workload

When `agent_transcripts` is true in `matrix.json`, each job also runs `agent_off` and
`agent_default` over `skippy/evals/agent-transcripts`. That set holds 32 real OpenHands
coding-agent histories from nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0): system
prompt, reasoning, tool calls and tool output, cut before an assistant turn. They come in
four bands of history length (4–8K, 8–16K, 16–32K and 32–48K tokens), and the model writes
the next turn.

These arms serve with a 64K context, so they show whether DFlash still pays off with long
histories, and whether time to first token changes. The table gets a second section by
history band, and MLflow records these runs with `workload=agent`. James's AgentX traces
carry only token counts, not text, so they can't measure speculation; this set
complements them.

## Reading the table

- **Before** is the `off` arm, **All together** the `default` arm, and **Gain** their ratio
  from the same job and machine.
- **+DFlash** is the fastest fixed block by all-prompt tok/s, with its per-workload numbers.
- Sampled rows have no fixed arms.
- A model the default serves without speculation (Qwen3-Coder-Next on CUDA) is marked
  "off by default"; its default arm should match `off`.

Runs on different nodes of the same kind aren't interchangeable: one Spark ran a baseline
25% slow under thermal throttling. Compare rows only within a job, and check the `gpu:`
lines in the log when a baseline looks off.

## MLflow

`run` records automatically when `MLFLOW_TRACKING_URI` is set, using
`MLFLOW_TRACKING_USERNAME`/`PASSWORD` or `MLFLOW_TRACKING_TOKEN`. Everything goes into one
experiment, "mesh-llm DFlash defaults — decode speed by commit":

- **A parent run per measured commit**, named `mesh-llm @ <sha> · <subject>`. It holds
  `table.md`, `matrix.json` and the jy job IDs.
- **A child run per model, machine and arm**, for example
  `Qwen3.5-9B · Spark · shipped default · greedy`. Its metrics are `decode_tok_s_*`,
  `acceptance_*`, `speedup_vs_off_*` (against the same job's off arm), `gpu_clock_mhz`,
  `gpu_temp_c`, `ttft_ms_p50_*` and `commit_time_unix`, and its params include `workload`
  (`spec-prompts` or `agent`).

The experiment page is the dashboard. Its description is rewritten after each recording
with three sections:

- the latest commit's table, as in the PR;
- the shipped default's change from the previous recorded commit;
- an index of recorded commits linking to their runs.

Each commit's parent run shows its own table the same way.

Every run carries `mesh_llm.commit`, `mesh_llm.commit_short`, `mesh_llm.commit_time`,
`mesh_llm.label` and MLflow's own `mlflow.source.git.commit`. To compare commits in the UI,
filter on these:

```text
tags.mesh_llm.commit_short IN ('f19b54a88', '<other>') AND params.arm = 'default'
```

To see a trend, chart `decode_tok_s_all` against `commit_time_unix` as a scatter plot. Runs
are keyed by `gi.key` (commit, model, machine, arm), so recording again only adds what is
missing. `bench.py compare` prints the same comparison as a table.

