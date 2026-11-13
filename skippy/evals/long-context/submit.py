#!/usr/bin/env python3
"""Submits one long-context ladder job per model, pinned to one Spark.

    submit.py MODEL[,MODEL...] [--node spark-000464] [--rev REV] [--depths ...] [--dry-run]

REV must be pushed to github.com/general-inference/mesh-llm.
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/general-inference/mesh-llm.git"
HF = "https://huggingface.co"
MODELS = {
    # Max's model: hybrid (10 of 40 layers full attention), so its KV cache stays small.
    "qwen36-35b": {
        "target": (f"{HF}/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                   "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61", 22134528992),
        "draft": (f"{HF}/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/resolve/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf",
                  "69bc81f5344dfeb67bcfc09277dd9617500fca1823f72e6e08d5df92e73a8b44", 570468544),
        "cache": "f16",
    },
    "qwen35-35b": {
        "target": (f"{HF}/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf",
                   "3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375", 22016023168),
        "draft": (f"{HF}/Anbeeld/Qwen3.5-35B-A3B-DFlash-GGUF/resolve/31335d2996d3fcaf269225f17693c5d9205822a4/Qwen3.5-35B-A3B-DFlash-Q8_0.gguf",
                  "137520fa4fcd6964d0683189638dc1389bd9f26ddeaac66e642ea439c4bb2029", 421060928),
        "cache": "f16",
    },
    # Every layer is full attention: 96 KiB of f16 KV per token, so 1M needs q8_0.
    "qwen3-coder-30b": {
        "target": (f"{HF}/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                   "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad", 18556689568),
        "draft": (f"{HF}/AtomicChat/Qwen3-Coder-30B-A3B-DFlash-GGUF/resolve/22a0c8f320ad784a89dac14f03b8afedcdb14d45/Qwen3-Coder-30B-A3B-DFlash.Q8_0.gguf",
                  "abd0e078f1e5683ba83c4ced92836b136d9e2e6b2626715b699088fd5c3a51fc", 509674848),
        "cache": "q8_0",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models")
    parser.add_argument("--node", default="spark-000464")
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--depths")
    parser.add_argument("--arms")
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--yarn", help="YaRN factor; 0 serves the published GGUF")
    parser.add_argument("--priority", type=int, default=20)
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rev = subprocess.run(["git", "rev-parse", args.rev], cwd=HERE, check=True, capture_output=True,
                         text=True).stdout.strip()
    (HERE / "out").mkdir(exist_ok=True)
    for name in args.models.split(","):
        model = MODELS[name]
        cache = args.cache or model["cache"]
        env = {
            "LLAMA_STAGE_USE_SCCACHE": "0",
            "BENCH_TARGET_URL": model["target"][0], "BENCH_TARGET_SHA256": model["target"][1],
            "BENCH_TARGET_BYTES": str(model["target"][2]),
            "BENCH_DRAFT_URL": model["draft"][0], "BENCH_DRAFT_SHA256": model["draft"][1],
            "BENCH_DRAFT_BYTES": str(model["draft"][2]),
            "BENCH_CACHE_TYPE": cache,
            **({"BENCH_DEPTHS": args.depths} if args.depths else {}),
            **({"BENCH_ARMS": args.arms} if args.arms else {}),
            **({"BENCH_CTX": str(args.ctx)} if args.ctx else {}),
            **({"BENCH_YARN_FACTOR": args.yarn} if args.yarn else {}),
        }
        spec = {"spec": {
            "name": f"Long context 1M{' ' + args.tag if args.tag else ''} · {name} · {args.node} @ {rev[:9]}",
            "description": "Prefill and decode (off vs DFlash) at 32K..1M tokens, YaRN x4 in the GGUF, "
                           "with a passphrase-retrieval check at every depth.",
            "priority": args.priority,
            "target": {"executor": "lab", "nodes": [args.node]},
            "harness": {"name": "custom", "version": "long-context-1",
                        "trace": {"kind": "file", "ref": f"skippy/evals/long-context/ladder.py@{rev[:9]}"},
                        "seed": 0, "warmup_requests": 1, "concurrency": [1], "request_count": 5},
            "budget": {"max_usd": 0, "max_duration_s": 8 * 3600},
            "labels": {"suite": "long-context", "model": name, "rev": rev[:9]},
            "runtime": {
                "model_package": {"ref": "hf:unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5#*Q4_K_M*",
                                  "egress": "lab_only"},
                "models": [],
                "workspace": {"repos": [{"url": REPO_URL, "rev": rev, "path": "mesh-llm"}],
                              "build": "bash mesh-llm/skippy/evals/dflash-defaults/build.sh",
                              "retain": "on_failure"},
                "env": env,
                "command": "bash mesh-llm/skippy/evals/long-context/run.sh",
            },
        }}
        path = HERE / "out" / f"{name}{'-' + args.tag if args.tag else ''}-{args.node}.json"
        path.write_text(json.dumps(spec, indent=1))
        if args.dry_run:
            print(path)
            continue
        result = subprocess.run(["jy", "submit", str(path)], capture_output=True, text=True)
        print(name, result.stdout.strip() or result.stderr.strip())


if __name__ == "__main__":
    main()
