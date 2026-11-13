#!/usr/bin/env python3
"""Step 4: end-to-end decode time for a mixed workload under different draft routers.

    compose.py --matrix matrix.json [--probe swap-probe-reload.json] [--mix mix.json]

Builds on measured numbers only:
  matrix.json  {"<workload>": {"off": tok/s, "generic": tok/s, "<specialist>": tok/s, ...}}
               every arm of a workload measured on one machine (run.sh)
  probe        swap-probe rows: swap time, and acceptance right after a swap by rows seen
  mix          {"<workload>": share of output tokens}; defaults to equal shares

Routers:
  generic      the shipped draft everywhere, never swaps
  oracle       the best measured draft per workload, swapping at every workload change
  metadata     the workload's own specialist when one exists, else generic (Max knows the
               task kind: routine, agent, file type), with a swap at every change
  classifier   metadata routing, but a share --misroute of requests get a random other draft
A request is `--reply` output tokens; workloads change between requests with probability
--switch. A swap costs --swap-cost-s (measured: resident swap plus replay of the kept
window), or with --probe its measured time plus the slower first windows of a cold restart.
"""

import argparse
import json
import random
import statistics


def swap_penalty_s(probe, reply_tokens, rows_seen):
    """Seconds a swap adds: the open, plus decode lost to weaker drafting right after it."""
    if not probe:
        return 0.0
    swaps = [r["swap_ms"] for r in probe if r.get("swap_ms") is not None]
    open_s = statistics.median(swaps) / 1000 if swaps else 0.0
    base = [r for r in probe if r["restart_rows"] is None]
    after = [r for r in probe if r["restart_rows"] == rows_seen] or \
        [r for r in probe if r["restart_rows"] is not None]
    if not base or not after:
        return open_s

    def rate(rows):
        return sum(r["generated"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
    slow = max(0.0, reply_tokens / rate(after) - reply_tokens / rate(base))
    return open_s + slow


def simulate(matrix, mix, router, args, probe, rng):
    workloads = list(mix)
    weights = [mix[w] for w in workloads]
    drafts = sorted({d for row in matrix.values() for d in row if d != "off"})
    attached, current, total_s, tokens, swaps = "generic", rng.choices(workloads, weights)[0], 0.0, 0, 0
    for _ in range(args.requests):
        if rng.random() < args.switch:
            current = rng.choices(workloads, weights)[0]
        row = matrix[current]
        if router == "generic":
            want = "generic"
        elif router == "oracle":
            want = max((d for d in row if d != "off"), key=lambda d: row[d])
        else:
            want = current if current in row else "generic"
            if router == "classifier" and rng.random() < args.misroute:
                want = rng.choice([d for d in drafts if d != want] or [want])
        if want != attached:
            swaps += 1
            total_s += args.swap_cost_s if args.swap_cost_s is not None else \
                swap_penalty_s(probe, args.reply, args.rows_seen)
            attached = want
        speed = row.get(attached) or row["generic"]
        total_s += args.reply / speed
        tokens += args.reply
    return tokens / total_s, swaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--probe")
    parser.add_argument("--mix")
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--reply", type=int, default=400)
    parser.add_argument("--switch", type=float, default=0.3)
    parser.add_argument("--misroute", type=float, default=0.1)
    parser.add_argument("--swap-cost-s", type=float,
                        help="fixed seconds per swap, in place of --probe")
    parser.add_argument("--rows-seen", type=int, default=256,
                        help="prompt rows the new draft sees before the reply (new turn's prefill)")
    args = parser.parse_args()
    matrix = json.load(open(args.matrix))
    probe = json.load(open(args.probe)) if args.probe else None
    mix = json.load(open(args.mix)) if args.mix else {w: 1.0 for w in matrix}
    print(f"mix {mix}  reply {args.reply} tok  switch p={args.switch}  misroute {args.misroute}")
    print("| Router | Decode tok/s | vs generic | Swaps per 100 requests |")
    print("|---|---:|---:|---:|")
    generic = None
    for router in ("generic", "metadata", "classifier", "oracle"):
        speed, swaps = simulate(matrix, mix, router, args, probe, random.Random(7))
        generic = generic or speed
        print(f"| {router} | {speed:.1f} | {speed / generic:.3f} | {100 * swaps / args.requests:.1f} |")


if __name__ == "__main__":
    main()
