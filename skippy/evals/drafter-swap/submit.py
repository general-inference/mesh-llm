#!/usr/bin/env python3
"""Submits drafter-swap jobs: one jy job per Spark, each owning some workloads.

    submit.py [--phase headroom|datagen|swap] [--rev REV] [--dry-run] [--only NODE,...]

headroom  measures off / shipped DFlash2 / n-gram per workload (run.sh)
datagen   has the target answer train-prompts.jsonl, the drafters' training data (datagen.sh)
swap      acceptance and cost of swapping the draft mid-session (swap_probe.sh)

Every arm of a workload runs on the same machine, so speedups compare within a job.
Acceptance does not depend on the machine, so workloads can be spread freely.
REV must be pushed to github.com/general-inference/mesh-llm.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/general-inference/mesh-llm.git"
TARGET = {
    "url": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "sha256": "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    "bytes": 22134528992,
}
# The CUDA default draft for this target (dflash_catalog.rs).
DRAFT = {
    "url": "https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/resolve/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf",
    "sha256": "69bc81f5344dfeb67bcfc09277dd9617500fca1823f72e6e08d5df92e73a8b44",
    "bytes": 570468544,
}
# The original DFlash draft for the same target, swapped against DFlash2 in step 3.
ALT_DRAFT = {
    "url": "https://huggingface.co/Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp/resolve/2aa57b642f211d2639083954c5b66d9c1ee503fc/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf",
    "sha256": "724ac8a102763e0d896cd49bf558fdf5bfdb146ec9f7097d18b035e4a95eac3f",
    "bytes": 421060800,
}
# dan-lab Sparks.
ASSIGNMENT = {
    "spark-000280": "html",
    "spark-000265": "code_review",
    "spark-000468": "code_edit",
    "spark-000499": "rust_gen,prose",
    "spark-000907": "agent",
}
DATAGEN_ASSIGNMENT = {
    "spark-000280": "html",
    "spark-000265": "code_review",
    "spark-000468": "code_edit",
    "spark-000499": "rust_gen",
}
PHASES = {
    "swap": {"assignment": {"spark-000907": "agent,code_edit,code_review,html"}, "command": "swap_probe.sh",
             "version": "drafter-swap-probe-1", "trace": "workloads.jsonl", "requests": 40,
             "title": "Drafter swap probe",
             "description": "Acceptance right after a DFlash draft swap, by rows the new draft has seen, and "
                            "the time a swap takes."},
    "headroom": {"assignment": ASSIGNMENT, "command": "run.sh", "version": "drafter-swap-headroom-1",
                 "trace": "workloads.jsonl", "requests": 40,
                 "title": "Drafter swap headroom",
                 "description": "Qwen3.6-35B-A3B decode tok/s and draft acceptance per workload: off, shipped "
                                "DFlash2, n-gram suffix."},
    "datagen": {"assignment": DATAGEN_ASSIGNMENT, "command": "datagen.sh", "version": "drafter-swap-datagen-1",
                "trace": "train-prompts.jsonl", "requests": 1200,
                "title": "Drafter swap datagen",
                "description": "Qwen3.6-35B-A3B greedy replies to a workload's training prompts, as DFlash "
                               "draft training data."},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, default="headroom")
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arms", help="headroom arms to run, e.g. 'ngram' (default: all)")
    parser.add_argument("--limit", type=int, help="datagen: prompts per domain (default: all)")
    args = parser.parse_args()
    rev = subprocess.run(["git", "rev-parse", args.rev], cwd=HERE, check=True, capture_output=True,
                         text=True).stdout.strip()
    out = HERE / "out"
    (out / "specs").mkdir(parents=True, exist_ok=True)
    phase = PHASES[args.phase]
    state = {"rev": rev, "phase": args.phase, "jobs": []}
    for node, domains in phase["assignment"].items():
        if args.only and node not in args.only.split(","):
            continue
        spec = {"spec": {
            "name": f"{phase['title']}{f' ({args.arms})' if args.arms else ''} · {domains} · {node} @ {rev[:9]}",
            "description": phase["description"],
            "priority": 20,
            "target": {"executor": "lab", "nodes": [node]},
            "harness": {"name": "custom", "version": phase["version"],
                        "trace": {"kind": "file", "ref": f"skippy/evals/drafter-swap/{phase['trace']}@{rev[:9]}"},
                        "seed": 0, "warmup_requests": 1, "concurrency": [1], "request_count": phase["requests"]},
            "budget": {"max_usd": 0, "max_duration_s": 21600},
            "labels": {"suite": "drafter-swap", "phase": args.phase, "domains": domains, "rev": rev[:9]},
            "runtime": {
                "model_package": {"ref": "hf:unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5#*Q4_K_M*",
                                  "egress": "lab_only"},
                "models": [],
                "workspace": {"repos": [{"url": REPO_URL, "rev": rev, "path": "mesh-llm"}],
                              "build": "bash mesh-llm/skippy/evals/dflash-defaults/build.sh",
                              "retain": "on_failure"},
                "env": {
                    "LLAMA_STAGE_USE_SCCACHE": "0",
                    "BENCH_TARGET_URL": TARGET["url"], "BENCH_TARGET_SHA256": TARGET["sha256"],
                    "BENCH_TARGET_BYTES": str(TARGET["bytes"]),
                    "BENCH_DRAFT_URL": DRAFT["url"], "BENCH_DRAFT_SHA256": DRAFT["sha256"],
                    "BENCH_DRAFT_BYTES": str(DRAFT["bytes"]),
                    "BENCH_DFLASH_CAP": "7",
                    "BENCH_DOMAINS": domains,
                    **({"BENCH_ARMS": args.arms} if args.arms else {}),
                    **({"BENCH_LIMIT": str(args.limit)} if args.limit else {}),
                    "BENCH_ALT_DRAFT_URL": ALT_DRAFT["url"], "BENCH_ALT_DRAFT_SHA256": ALT_DRAFT["sha256"],
                },
                "command": f"bash mesh-llm/skippy/evals/drafter-swap/{phase['command']}",
            },
        }}
        path = out / "specs" / f"{args.phase}{'-' + args.arms if args.arms else ''}-{node}.json"
        path.write_text(json.dumps(spec, indent=1))
        if args.dry_run:
            print(f"{node}: {domains} -> {path}")
            continue
        result = subprocess.run(["jy", "submit", str(path)], capture_output=True, text=True)
        match = re.search(r"\bjob_[a-z0-9]+", result.stdout + result.stderr)
        if not match:
            sys.exit(f"submit failed for {node}:\n{result.stdout}{result.stderr}")
        print(f"{node:14s} {domains:16s} {match.group(0)}")
        state["jobs"].append({"node": node, "domains": domains, "job_id": match.group(0)})
        (out / f"{args.phase}{'-' + args.arms if args.arms else ''}-state.json").write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
