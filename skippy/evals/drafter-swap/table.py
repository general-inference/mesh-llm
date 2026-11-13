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
    # Reruns of single arms (out/<phase>-<arms>-state.json) ran on the same node as the
    # job they complete; their arms fill in the ones that job lacks.
    extra = {}
    for path in sorted((HERE / "out").glob(f"{args.phase}-*-state.json")):
        for job in json.loads(path.read_text())["jobs"]:
            extra.setdefault(job["node"], {}).update(arms(job["job_id"]))
    rows = []
    for job in state["jobs"]:
        found = arms(job["job_id"])
        for arm, row in extra.get(job["node"], {}).items():
            if "groups" in row and "groups" not in found.get(arm, {}):
                found[arm] = row
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
                specs = sorted({a.removeprefix(prefix) for a in found
                                if a.startswith(f"{prefix}spec_") and "groups" in found[a]})
                for arm in ("dflash", "ngram", *specs):
                    got = (found.get(f"{prefix}{arm}") or {}).get("groups", {}).get(group)
                    row[arm] = got["tok_s"] if got else None
                    row[f"{arm}_accept"] = got["accept"] if got else None
                rows.append(row)

    def f(value, spec):
        return format(value, spec) if value is not None else "–"

    arms_shown = ["dflash", "ngram"] + sorted({k for r in rows for k in r if k.startswith("spec_")
                                                 and not k.endswith("_accept")})
    names = {"dflash": "DFlash2", "ngram": "n-gram"}
    print("| Workload | Off tok/s | " + " | ".join(f"{names.get(a, a)} tok/s | × | accept" for a in arms_shown)
          + " | Node |")
    print("|---|---:|" + "---:|---:|---:|" * len(arms_shown) + "---|")
    for r in rows:
        cells = []
        for arm in arms_shown:
            speed = r.get(arm)
            cells += [f(speed, ".1f"), f(speed / r["off"] if speed and r["off"] else None, ".2f"),
                      f(r.get(f"{arm}_accept"), ".2f")]
        print(f"| {r['workload']} | {f(r['off'], '.1f')} | " + " | ".join(cells) + f" | {r['node']} |")
    (HERE / "out" / f"{args.phase}-table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
