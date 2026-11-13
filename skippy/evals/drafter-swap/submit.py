#!/usr/bin/env python3
"""Submits the drafter-swap headroom jobs: one jy job per Spark, each owning some workloads.

    submit.py [--rev REV] [--dry-run] [--only NODE,...]

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
# dan-lab Sparks.
ASSIGNMENT = {
    "spark-000280": "html",
    "spark-000265": "code_review",
    "spark-000468": "code_edit",
    "spark-000499": "rust_gen,prose",
    "spark-000907": "agent",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rev = subprocess.run(["git", "rev-parse", args.rev], cwd=HERE, check=True, capture_output=True,
                         text=True).stdout.strip()
    out = HERE / "out"
    (out / "specs").mkdir(parents=True, exist_ok=True)
    state = {"rev": rev, "jobs": []}
    for node, domains in ASSIGNMENT.items():
        if args.only and node not in args.only.split(","):
            continue
        spec = {"spec": {
            "name": f"Drafter swap headroom · {domains} · {node} @ {rev[:9]}",
            "description": "Qwen3.6-35B-A3B decode tok/s and draft acceptance per workload: off, shipped DFlash2, "
                           "n-gram suffix.",
            "priority": 20,
            "target": {"executor": "lab", "nodes": [node]},
            "harness": {"name": "custom", "version": "drafter-swap-headroom-1",
                        "trace": {"kind": "file", "ref": f"skippy/evals/drafter-swap/workloads.jsonl@{rev[:9]}"},
                        "seed": 0, "warmup_requests": 1, "concurrency": [1], "request_count": 40},
            "budget": {"max_usd": 0, "max_duration_s": 21600},
            "labels": {"suite": "drafter-swap", "phase": "headroom", "domains": domains, "rev": rev[:9]},
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
                },
                "command": "bash mesh-llm/skippy/evals/drafter-swap/run.sh",
            },
        }}
        path = out / "specs" / f"headroom-{node}.json"
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
        (out / "headroom-state.json").write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
