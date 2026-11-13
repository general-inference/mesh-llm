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
        groups[(label, row["restart_rows"])].append(row)
    print("| Probe | Rows seen after swap | Prompts | Swap ms (median) | Accepted / window | "
          "First 8 windows | Decode tok/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for (label, depth), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None
                                                                       else kv[0][1])):
        windows = sum(r["windows"] for r in rows)
        accepted = sum(r["accepted_tokens"] for r in rows)
        early_windows = sum(min(r["windows"], 8) for r in rows)
        early = sum(r["early_mean_accepted_per_window"] * min(r["windows"], 8) for r in rows)
        speed = sum(r["generated"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
        swaps = [r["swap_ms"] for r in rows if r["swap_ms"] is not None]
        print(f"| {label} | {'no swap' if depth is None else depth} | {len(rows)} | "
              f"{statistics.median(swaps) if swaps else 0:.3f} | {accepted / windows:.2f} | "
              f"{early / early_windows:.2f} | {speed:.1f} |")
    out = [dict(r, probe=label) for (label, _), rows in groups.items() for r in rows]
    json.dump(out, open(f"out/swap-probe-{sys.argv[1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
