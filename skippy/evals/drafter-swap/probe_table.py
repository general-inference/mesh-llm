#!/usr/bin/env python3
"""Summarises a swap-probe job: swap time, and drafting right after a swap by rows seen.

    probe_table.py JOB_ID

Reads the SWAP_PROBE lines from `jy logs`. Per probe and restart depth: the median
swap time, accepted tokens per verify window (all windows, and the first 8), and
decode tok/s, each pooled over prompts (tokens over time, accepted over windows).
"""

import json
import statistics
import subprocess
import sys
from collections import defaultdict


def main():
    log = subprocess.run(["jy", "logs", sys.argv[1]], capture_output=True, text=True).stdout
    groups = defaultdict(list)
    for line in log.splitlines():
        if "SWAP_PROBE " not in line:
            continue
        label, payload = line.split("SWAP_PROBE ", 1)[1].split(" ", 1)
        row = json.loads(payload)
        depth = row["restart_rows"]
        # A short prompt caps the depth: the new draft saw the whole prompt.
        if depth is not None and depth >= row["prompt_tokens"] - 1:
            depth = "whole prompt"
        groups[(label, depth)].append(row)
    # Each prompt's run without a swap, to pair swap rows against.
    base = {(label, r["id"]): r for (label, depth), rows in groups.items() if depth is None for r in rows}

    def time_s(r):
        return r["decode_ms"] / 1000

    print("| Probe | Rows seen after swap | Prompts | Swap ms (median) | Accepted / window | "
          "First 8 windows | Decode tok/s | Same prompts without swap tok/s | Ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    def order(key):
        label, depth = key
        return label, -1 if depth is None else 10**9 if depth == "whole prompt" else depth

    for (label, depth), rows in sorted(groups.items(), key=lambda kv: order(kv[0])):
        windows = sum(r["windows"] for r in rows)
        accepted = sum(r["accepted_tokens"] for r in rows)
        early_windows = sum(min(r["windows"], 8) for r in rows)
        early = sum(r["early_mean_accepted_per_window"] * min(r["windows"], 8) for r in rows)
        speed = sum(r["generated"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
        swaps = [r["swap_ms"] for r in rows if r["swap_ms"] is not None]
        paired = [base[(label, r["id"])] for r in rows if (label, r["id"]) in base]
        base_speed = sum(r["generated"] for r in paired) / sum(time_s(r) for r in paired) if paired else None
        print(f"| {label} | {'no swap' if depth is None else depth} | {len(rows)} | "
              f"{statistics.median(swaps) if swaps else 0:.3f} | {accepted / windows:.2f} | "
              f"{early / early_windows:.2f} | {speed:.1f} | "
              f"{f'{base_speed:.1f}' if base_speed else '–'} | "
              f"{f'{speed / base_speed:.2f}' if base_speed else '–'} |")
    out = [dict(r, probe=label) for (label, _), rows in groups.items() for r in rows]
    json.dump(out, open(f"out/swap-probe-{sys.argv[1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
