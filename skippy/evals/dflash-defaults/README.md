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
