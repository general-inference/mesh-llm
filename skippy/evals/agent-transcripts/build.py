"""Builds agent-transcripts.jsonl from nebius/SWE-rebench-openhands-trajectories.

    curl -fLO https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/resolve/35455389ab51bf5e2306bfd436ef72d0f98bf882/trajectories.parquet
    uv run --with pyarrow python build.py trajectories.parquet

Each prompt is one real OpenHands history, cut just before an assistant turn and rendered
as text, followed by a request for the agent's next message. Selection does not depend on
any engine: trajectories are taken in SHA-256 order of their ID, at most one per
repository. Each fills the first unfilled history band it can reach, cut at its first
assistant turn whose history estimate lands inside the band.
"""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = {"dataset": "nebius/SWE-rebench-openhands-trajectories",
          "revision": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
          "file": "trajectories.parquet",
          "sha256": "14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1",
          "license": "CC-BY-4.0"}
# History bands in estimated tokens; 8 prompts each.
BANDS = [("4-8K", 4_000, 8_000), ("8-16K", 8_000, 16_000), ("16-32K", 16_000, 32_000), ("32-48K", 32_000, 48_000)]
PER_BAND = 8
# Code-heavy text runs near 3.6 characters per token for Qwen tokenizers; the measured
# prompt length is recorded at run time.
CHARS_PER_TOKEN = 3.6
INSTRUCTION = ("Below is the transcript of a coding agent working in a repository. Write the agent's next "
               "message in the same format, including its tool call.\n\n")
HERE = Path(__file__).resolve().parent


def render(message):
    role = message["role"]
    if role == "tool":
        return f"[tool {message.get('name') or ''}]\n{message.get('content') or ''}"
    text = f"[{role}]\n{message.get('content') or ''}"
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        text += f"\n<tool_call name=\"{function.get('name')}\">\n{function.get('arguments')}\n</tool_call>"
    return text


def order_key(trajectory_id):
    return hashlib.sha256(trajectory_id.encode()).hexdigest()


def main(path):
    parquet = pq.ParquetFile(path)
    ids = parquet.read(columns=["trajectory_id"]).column("trajectory_id").to_pylist()
    # Enough candidates to fill every band, read in one pass.
    wanted = set(sorted(ids, key=order_key)[:3000])
    rows = []
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=["trajectory_id", "instance_id", "repo", "trajectory"])
        rows += [row for row in table.to_pylist() if row["trajectory_id"] in wanted]
    rows.sort(key=lambda row: order_key(row["trajectory_id"]))

    filled = {band: [] for band, _, _ in BANDS}
    used_repos = set()
    for row in rows:
        if all(len(v) == PER_BAND for v in filled.values()):
            break
        if row["repo"] in used_repos:
            continue
        rendered = [render(message) for message in row["trajectory"]]
        prefix_chars = [0]
        for text in rendered:
            prefix_chars.append(prefix_chars[-1] + len(text) + 2)
        for band, low, high in BANDS:
            if len(filled[band]) == PER_BAND:
                continue
            cut = next((i for i, message in enumerate(row["trajectory"])
                        if i >= 3 and message["role"] == "assistant"
                        and low <= prefix_chars[i] / CHARS_PER_TOKEN <= high), None)
            if cut is None:
                continue
            prompt = INSTRUCTION + "\n\n".join(rendered[:cut])
            filled[band].append({"id": f"{row['trajectory_id']}@{cut}", "group": band, "category": "agent",
                                 "repo": row["repo"], "instance_id": row["instance_id"], "cut": cut,
                                 "est_tokens": round(len(prompt) / CHARS_PER_TOKEN), "prompt": prompt})
            used_repos.add(row["repo"])
            break
    if any(len(v) < PER_BAND for v in filled.values()):
        raise SystemExit({band: len(v) for band, v in filled.items()})

    items = [item for band, _, _ in BANDS for item in filled[band]]
    out = HERE / "agent-transcripts.jsonl"
    out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    provenance = {
        "source": SOURCE,
        "attribution": "Nebius, SWE-rebench-OpenHands-Trajectories (CC-BY-4.0); trajectories collected with "
                       "Qwen3-Coder-480B-A35B-Instruct and OpenHands v0.54.0.",
        "selection": __doc__.strip().split("\n\n", 2)[2],
        "bands": [{"name": band, "min_est_tokens": low, "max_est_tokens": high, "count": PER_BAND}
                  for band, low, high in BANDS],
        "chars_per_token_estimate": CHARS_PER_TOKEN,
        "items": len(items),
        "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(f"{len(items)} prompts, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
