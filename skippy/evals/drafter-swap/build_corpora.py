#!/usr/bin/env python3
"""Builds the per-workload prompt sets for the drafter-swap experiment.

    build_corpora.py --src DIR [--eval N] [--train N]

Writes `workloads.jsonl` (the eval prompts every arm measures) and, with `--train`,
`train-prompts.jsonl` (disjoint prompts whose target-model replies become drafter
training data). Every row carries `domain`, `id`, `split` and `prompt`.

Domains and sources (all public):
  html         HuggingFaceM4/WebSight v0.2 `llm_generated_idea` (CC-BY-4.0)
  code_review  bigcode/commitpackft Rust + TypeScript commits, shown as a unified diff (MIT)
  code_edit    bigcode/commitpackft: apply the commit subject to the old file, reply with the file
  rust_gen     nuprl/MultiPL-E humaneval-rs and mbpp-rs prompts (MIT)
  prose        skippy/evals/spec-prompts writing/roleplay/humanities/stem/reasoning/extraction
Agent transcripts are measured from skippy/evals/agent-transcripts as they are.

`--src` holds commitpackft's data/rust/data.jsonl and data/typescript/data.jsonl as
rust.jsonl and typescript.jsonl. WebSight and MultiPL-E come from the datasets-server API.
"""

import argparse
import difflib
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS = HERE.parent
ROWS_API = "https://datasets-server.huggingface.co/rows"
# Files above this are cut out: the reply has to fit the measured token budget.
MAX_FILE_CHARS = 5000
PROSE_CATEGORIES = {"writing", "roleplay", "humanities", "stem", "reasoning", "extraction"}


def rows(dataset, config, split, offset, length):
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
                return [r["row"] for r in json.load(response)["rows"]]
        except Exception:
            if attempt == 7:
                raise
            time.sleep(5 * (attempt + 1))


def stable_id(*parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


def html_prompts(n, rng):
    # Spread reads over the dataset instead of taking its first page.
    picked = []
    for offset in rng.sample(range(0, 1_900_000, 100), (n + 99) // 100 * 3):
        for row in rows("HuggingFaceM4/WebSight", "v0.2", "train", offset, 100):
            idea = (row.get("llm_generated_idea") or "").strip()
            if idea:
                picked.append(idea)
        if len(picked) >= n * 2:
            break
    rng.shuffle(picked)
    return [{"id": f"html-{stable_id(idea)}", "source": "HuggingFaceM4/WebSight@v0.2",
             "prompt": "Write a complete single-file HTML page using Tailwind CSS from the CDN for this "
                       f"website. Reply with only the HTML.\n\n{idea}"} for idea in picked]


def commits(src):
    out = []
    for lang in ("rust", "typescript"):
        for line in open(Path(src) / f"{lang}.jsonl"):
            row = json.loads(line)
            old, new = row["old_contents"], row["new_contents"]
            if not old.strip() or old == new or len(old) > MAX_FILE_CHARS or len(new) > MAX_FILE_CHARS:
                continue
            row["lang"] = lang
            out.append(row)
    return out


def review_prompt(row):
    diff = "".join(difflib.unified_diff(row["old_contents"].splitlines(True), row["new_contents"].splitlines(True),
                                        f"a/{row['old_file']}", f"b/{row['new_file']}"))
    return ("Review this change as a careful senior engineer. Point out bugs, risky edge cases and "
            "unclear code, quoting the lines you mean, then give a verdict.\n\n"
            f"Commit message: {row['subject']}\n\n```diff\n{diff}```")


def edit_prompt(row):
    fence = "rust" if row["lang"] == "rust" else "typescript"
    return (f"Apply this change to `{row['old_file']}`: {row['subject']}\n\n"
            f"Reply with the complete updated file in one code block and nothing else.\n\n"
            f"```{fence}\n{row['old_contents']}```")


def rust_prompts():
    out = []
    for config in ("humaneval-rs", "mbpp-rs"):
        offset = 0
        while True:
            page = rows("nuprl/MultiPL-E", config, "test", offset, 100)
            if not page:
                break
            for row in page:
                out.append({"id": f"rust-{row['name']}", "source": f"nuprl/MultiPL-E@{config}",
                            "prompt": "Complete this Rust function. Reply with the whole function in one "
                                      f"code block.\n\n```rust\n{row['prompt'].rstrip()}\n```"})
            offset += len(page)
            if len(page) < 100:
                break
    return out


def prose_prompts():
    items = [json.loads(line) for line in open(EVALS / "spec-prompts" / "spec-prompts.jsonl") if line.strip()]
    return [{"id": f"prose-{item['id']}", "source": f"spec-prompts:{item['source']}", "prompt": item["prompt"]}
            for item in items if item["category"] in PROSE_CATEGORIES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--eval", type=int, default=40)
    parser.add_argument("--train", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(20261113)
    pools = {}
    pools["html"] = html_prompts(args.eval + args.train, rng)
    rows_ = commits(args.src)
    rng.shuffle(rows_)
    # Review and edit draw from different commits, so no file appears in both.
    half = len(rows_) // 2
    pools["code_review"] = [{"id": f"review-{r['commit'][:12]}", "source": f"bigcode/commitpackft:{r['lang']}",
                             "prompt": review_prompt(r)} for r in rows_[:half]]
    pools["code_edit"] = [{"id": f"edit-{r['commit'][:12]}", "source": f"bigcode/commitpackft:{r['lang']}",
                           "prompt": edit_prompt(r)} for r in rows_[half:]]
    rust = rust_prompts()
    rng.shuffle(rust)
    pools["rust_gen"] = rust
    prose = prose_prompts()
    rng.shuffle(prose)
    pools["prose"] = prose

    with open(HERE / "workloads.jsonl", "w") as f:
        for domain, items in pools.items():
            for item in items[:args.eval]:
                f.write(json.dumps({"domain": domain, "split": "eval"} | item) + "\n")
            print(f"{domain:12s} eval {min(len(items), args.eval):4d}  pool {len(items)}")
    if args.train:
        with open(HERE / "train-prompts.jsonl", "w") as f:
            for domain, items in pools.items():
                for item in items[args.eval:args.eval + args.train]:
                    f.write(json.dumps({"domain": domain, "split": "train"} | item) + "\n")


if __name__ == "__main__":
    main()
