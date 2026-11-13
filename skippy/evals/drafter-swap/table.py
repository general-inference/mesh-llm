#!/usr/bin/env python3
"""Prints the per-workload table from the jobs in out/<phase>-state.json.

    table.py [--phase headroom] [--state FILE]

Reads each job's DFLASH_BENCH lines from `jy logs`. Speedups divide by the same
job's `off` arm, so every ratio compares runs on one machine.
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arms(job_id):
    log = subprocess.run(["jy", "logs", job_id], capture_output=True, text=True).stdout
    out = {}
    for line in log.splitlines():
        if "DFLASH_BENCH " in line:
            row = json.loads(line.split("DFLASH_BENCH ", 1)[1])
            out[row["arm"]] = row
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="headroom")
    parser.add_argument("--state")
    args = parser.parse_args()
    state = json.loads(Path(args.state or HERE / "out" / f"{args.phase}-state.json").read_text())
    rows = []
    for job in state["jobs"]:
        found = arms(job["job_id"])
        for prefix in ("", "agent_"):
            off = found.get(f"{prefix}off")
            if not off or "groups" not in off:
                continue
            for group, base in off["groups"].items():
                if group == "all" and len(off["groups"]) > 2:
                    continue
                if group == "all" and prefix == "":
                    continue
                row = {"workload": f"agent {group}" if prefix else group, "node": job["node"],
                       "off": base["tok_s"]}
                for arm in ("dflash", "ngram"):
                    got = (found.get(f"{prefix}{arm}") or {}).get("groups", {}).get(group)
                    row[arm] = got["tok_s"] if got else None
                    row[f"{arm}_accept"] = got["accept"] if got else None
                rows.append(row)

    def f(value, spec):
        return format(value, spec) if value is not None else "–"

    print("| Workload | Off tok/s | DFlash2 tok/s | × | accept | n-gram tok/s | × | accept | Node |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        ratio = lambda arm: r[arm] / r["off"] if r[arm] and r["off"] else None  # noqa: E731
        print(f"| {r['workload']} | {f(r['off'], '.1f')} | {f(r['dflash'], '.1f')} | {f(ratio('dflash'), '.2f')} | "
              f"{f(r['dflash_accept'], '.2f')} | {f(r['ngram'], '.1f')} | {f(ratio('ngram'), '.2f')} | "
              f"{f(r['ngram_accept'], '.2f')} | {r['node']} |")
    (HERE / "out" / f"{args.phase}-table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
