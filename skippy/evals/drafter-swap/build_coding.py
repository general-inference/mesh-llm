#!/usr/bin/env python3
"""Adds the coding-draft sets: more languages and agent turns, beside the existing ones.

    uv run --with pyarrow python build_coding.py --src DIR [--eval 20]

A single coding draft is trained on the target's replies to every coding set: the existing
code_edit (Rust/TypeScript edits), rust_gen, html_v2 and code_review training data, plus
the sets built here. Eval rows go to workloads.jsonl, training rows to train-prompts.jsonl,
never sharing a commit, problem or repository with an eval row.

  edit_py   bigcode/commitpackft Python commits: apply the subject, reply with the file (MIT)
  edit_go   the same for Go
  gen_ts    nuprl/MultiPL-E humaneval-ts and mbpp-ts: complete the function (MIT)
  gen_go    nuprl/MultiPL-E humaneval-go and mbpp-go
  agent_v2  nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0): an OpenHands history
            cut before an assistant turn, 2-12K estimated tokens, the next message asked
            for. Repositories in ../agent-transcripts are excluded; eval and training
            draw from disjoint repositories.

`--src` holds commitpackft's python.jsonl and go.jsonl, and trajectories.parquet at the
revision ../agent-transcripts pins.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from build_corpora import MAX_FILE_CHARS, rows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent-transcripts"))
from build import CHARS_PER_TOKEN, INSTRUCTION, render  # noqa: E402

TRAIN = {"edit_py": 700, "edit_go": 400, "gen_ts": 500, "gen_go": 400, "agent_v2": 600}
AGENT_BAND = (2_000, 12_000)


def edits(src, lang, fence):
    out = []
    for line in open(Path(src) / f"{lang}.jsonl"):
        row = json.loads(line)
        old, new = row["old_contents"], row["new_contents"]
        if not old.strip() or old == new or len(old) > MAX_FILE_CHARS or len(new) > MAX_FILE_CHARS:
            continue
        out.append({"id": f"{lang}edit-{row['commit'][:12]}", "source": f"bigcode/commitpackft:{lang}",
                    "prompt": (f"Apply this change to `{row['old_file']}`: {row['subject']}\n\n"
                               "Reply with the complete updated file in one code block and nothing else."
                               f"\n\n```{fence}\n{old}```")})
    return out


def generation(lang, fence):
    out = []
    for config in (f"humaneval-{lang}", f"mbpp-{lang}"):
        offset = 0
        while True:
            page = rows("nuprl/MultiPL-E", config, "test", offset, 100)
            for row in page:
                out.append({"id": f"gen{lang}-{row['name']}", "source": f"nuprl/MultiPL-E@{config}",
                            "prompt": f"Complete this {fence} function. Reply with the whole function in one "
                                      f"code block.\n\n```{fence}\n{row['prompt'].rstrip()}\n```"})
            offset += len(page)
            if len(page) < 100:
                break
    return out


def agent_turns(path, rng, held_repos):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    ids = parquet.read(columns=["trajectory_id"]).column("trajectory_id").to_pylist()
    wanted = set(sorted(ids, key=lambda i: hashlib.sha256(f"coding:{i}".encode()).hexdigest())[:6000])
    by_repo = {}
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=["trajectory_id", "repo", "trajectory"])
        for row in table.to_pylist():
            if row["trajectory_id"] in wanted and row["repo"] not in held_repos:
                by_repo.setdefault(row["repo"], row)
    out = []
    for repo, row in sorted(by_repo.items()):
        rendered = [render(message) for message in row["trajectory"]]
        prefix = [0]
        for text in rendered:
            prefix.append(prefix[-1] + len(text) + 2)
        cuts = [i for i, message in enumerate(row["trajectory"])
                if i >= 3 and message["role"] == "assistant"
                and AGENT_BAND[0] <= prefix[i] / CHARS_PER_TOKEN <= AGENT_BAND[1]]
        if not cuts:
            continue
        cut = rng.choice(cuts)
        out.append({"id": f"agentv2-{row['trajectory_id']}@{cut}", "repo": repo,
                    "source": "nebius/SWE-rebench-openhands-trajectories",
                    "prompt": INSTRUCTION + "\n\n".join(rendered[:cut])})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--eval", type=int, default=20)
    args = parser.parse_args()
    rng = random.Random(20261115)
    workloads = [json.loads(line) for line in open(HERE / "workloads.jsonl") if line.strip()]
    if any(row["domain"] in TRAIN for row in workloads):
        raise SystemExit("workloads.jsonl already has the coding sets")
    held_repos = {json.loads(line)["repo"] for line in open(HERE.parent / "agent-transcripts" / "agent-transcripts.jsonl")}
    pools = {"edit_py": edits(args.src, "python", "python"), "edit_go": edits(args.src, "go", "go"),
             "gen_ts": generation("ts", "typescript"), "gen_go": generation("go", "go"),
             "agent_v2": agent_turns(Path(args.src) / "trajectories.parquet", rng, held_repos)}
    eval_rows, train_rows = [], []
    for domain, pool in pools.items():
        rng.shuffle(pool)
        held, kept = pool[:args.eval], pool[args.eval:args.eval + TRAIN[domain]]
        if domain == "agent_v2":  # one cut per repository, so eval and training repositories differ
            assert not {r["repo"] for r in held} & {r["repo"] for r in kept}
        eval_rows += [{"domain": domain, "split": "eval"} | row for row in held]
        train_rows += [{"domain": domain, "split": "train"} | row for row in kept]
        print(f"{domain:9s} eval {len(held):3d}  train {len(kept):4d}  pool {len(pool)}")
    with open(HERE / "workloads.jsonl", "a") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")
    with open(HERE / "train-prompts.jsonl", "a") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
