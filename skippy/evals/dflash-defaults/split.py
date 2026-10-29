#!/usr/bin/env python3
"""Measures a large model split across several machines, as jy formations.

    split.py plan   [--rev REV] [--stages 2] [--arms off,default,ngram] [--delays 0,20] [--out DIR]
    split.py submit [same]   one formation per (arm, wire delay) on the matrix's first N nodes
    split.py status [--out DIR]
    split.py table  [--out DIR]

Each formation builds REV on every member, serves split_matrix.json's layer package
across N machines (one coordinator, N-1 joiners; joiners copy the package from the coordinator) and measures single-stream decode on
the held-out spec-prompts split. The arm is fixed for a formation because joiners hold
an invite to the coordinator's mesh, so changing speculation means re-forming the split.
Uses the signed-in jy session for the formation API.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/general-inference/mesh-llm.git"
ARM_NAMES = {"off": "speculation off", "default": "package default", "ngram": "n-gram"}


def session():
    for path in (Path.home() / "Library/Application Support/jianyang/session.json",
                 Path.home() / ".config/jianyang/session.json"):
        if path.exists():
            data = json.loads(path.read_text())
            return data["api_url"].rstrip("/"), data["api_key"]
    sys.exit("no jy session; run `jy login`")


def api(method, path, body=None):
    url, key = session()
    request = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method, headers={"authorization": f"Bearer {key}",
                                                             "content-type": "application/json",
                                                             "user-agent": "dflash-split-bench"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def plan(args):
    matrix = json.loads((HERE / args.matrix).read_text())
    rev = subprocess.run(["git", "rev-parse", args.rev], cwd=HERE, check=True, capture_output=True,
                         text=True).stdout.strip()
    nodes = matrix["nodes"][: args.stages]
    if len(nodes) < args.stages:
        sys.exit(f"split_matrix.json lists {len(matrix['nodes'])} nodes, fewer than {args.stages}")
    formations = []
    for arm in args.arms.split(","):
        for delay in [int(d) for d in args.delays.split(",")]:
            key = f"split:{rev[:12]}:{args.stages}:{arm}:{delay}"
            spec = {
                "name": f"Split ×{args.stages} · {matrix['name']} · {ARM_NAMES[arm]} · {delay} ms/hop @ {rev[:9]}",
                "description": f"{matrix['name']} served across {args.stages} machines; single-stream decode on the "
                               f"held-out spec-prompts split with {ARM_NAMES[arm]} and {delay} ms injected per hop.",
                "priority": 20,
                "executor": "lab",
                "nodes": nodes,
                "runtime": {
                    "env": {"SPLIT_REPO": matrix["repo"], "SPLIT_REV": matrix["revision"],
                            "SPLIT_STAGES": str(args.stages), "SPLIT_ARM": arm,
                            "SPLIT_WIRE_DELAY_MS": str(delay), "SPLIT_LANES": str(matrix["lanes"]),
                            "SPLIT_CTX": str(matrix["ctx"]), "LLAMA_STAGE_USE_SCCACHE": "0",
                            "SPLIT_ALLOW_UNCERTIFIED": "1" if matrix.get("allow_uncertified") else "0"},
                    "workspace": {"repos": [{"url": REPO_URL, "rev": rev, "path": "mesh-llm"}],
                                  "build": "bash mesh-llm/skippy/evals/dflash-defaults/build.sh",
                                  "retain": "on_failure"},
                    "command": "bash mesh-llm/skippy/evals/dflash-defaults/split_run.sh",
                },
                "roles": [{"name": "coordinator", "count": 1, "nodes": nodes[:1]},
                          {"name": "joiner", "count": args.stages - 1, "nodes": nodes[1:]}],
                "rendezvous": {"mode": "peers", "leader_role": "coordinator"},
                "harness": {"name": "custom", "version": "split-defaults-1",
                            "trace": {"kind": "file", "ref": f"skippy/evals/spec-prompts/spec-prompts.jsonl@{rev[:9]}"},
                            "seed": 0, "warmup_requests": 1, "concurrency": [1], "request_count": 77},
                "budget": {"max_usd": 0, "max_duration_s": 6 * 3600},
                "labels": {"suite": "split-dflash", "model": matrix["id"], "stages": str(args.stages), "arm": arm,
                           "wire_delay_ms": str(delay), "rev": rev[:9]},
            }
            formations.append({"key": key, "stages": args.stages, "arm": arm, "delay": delay, "spec": spec})
            print(f"{key:44s} on {', '.join(nodes)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "formations.json").write_text(json.dumps(formations, indent=1))
    return rev, formations


def submit(args):
    rev, formations = plan(args)
    path = Path(args.out) / "split-state.json"
    state = json.loads(path.read_text()) if path.exists() else {"formations": []}
    for f in formations:
        reply = api("POST", "/v1/formations", {"spec": f["spec"], "idempotency_key": f["key"]})
        formation = reply.get("formation", reply)
        print(f"{f['key']:44s} {formation.get('id')}  {formation.get('status')}")
        state["formations"].append({k: f[k] for k in ("key", "stages", "arm", "delay")} | {"id": formation.get("id")})
        path.write_text(json.dumps(state, indent=1))


def formation_detail(formation_id):
    reply = api("GET", f"/v1/formations/{formation_id}")
    return reply.get("formation", reply), reply.get("members") or reply.get("formation", {}).get("members") or []


def status(args):
    state = json.loads((Path(args.out) / "split-state.json").read_text())
    for f in state["formations"]:
        formation, members = formation_detail(f["id"])
        roles = ", ".join(f"{m.get('role')}:{m.get('status')}@{m.get('node') or m.get('machine_node') or '?'}"
                          for m in members)
        print(f"{f['key']:44s} {formation.get('status'):10s} {roles}")


def coordinator_result(formation_id):
    _, members = formation_detail(formation_id)
    coordinator = next((m for m in members if m.get("role") == "coordinator"), None)
    if not coordinator or not coordinator.get("job_id"):
        return None, None
    log = subprocess.run(["jy", "logs", coordinator["job_id"]], capture_output=True, text=True).stdout
    result = layout = None
    for line in log.splitlines():
        line = line.removeprefix("[out] ")
        if line.startswith("DFLASH_BENCH "):
            result = json.loads(line[len("DFLASH_BENCH "):])
        elif line.startswith("SPLIT_LAYOUT "):
            layout = json.loads(line[len("SPLIT_LAYOUT "):])
    return result, layout


def table(args):
    state = json.loads((Path(args.out) / "split-state.json").read_text())
    cells, layouts = {}, {}
    for f in state["formations"]:
        result, layout = coordinator_result(f["id"])
        tok = ((result or {}).get("groups") or {}).get("all", {}).get("tok_s")
        cells[(f["stages"], f["delay"], f["arm"])] = tok
        if layout:
            layouts[f["stages"]] = layout.get("spans")
    arms = [a for a in ARM_NAMES if any(k[2] == a for k in cells)]
    lines = ["| Split | Layers per machine | Network | " + " | ".join(ARM_NAMES[a] for a in arms) + " | Best vs off |",
             "|" + "---|" * (len(arms) + 4)]
    for stages, delay in sorted({(k[0], k[1]) for k in cells}):
        row = {a: cells.get((stages, delay, a)) for a in arms}
        off = row.get("off")
        best = max(((v, a) for a, v in row.items() if v and a != "off"), default=(None, None))
        gain = f"{best[0] / off:.2f}× ({ARM_NAMES[best[1]]})" if best[0] and off else "—"
        network = "LAN" if delay == 0 else f"+{delay} ms/hop"
        lines.append(f"| {stages} Sparks | {layouts.get(stages, '—')} | {network} | "
                     + " | ".join("—" if row[a] is None else f"{row[a]:.1f}" for a in arms) + f" | {gain} |")
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "split-table.md").write_text(text)
    print("Decode tok/s, single stream, held-out spec-prompts.\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "submit", "status", "table"])
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--arms", default="off,default,ngram")
    parser.add_argument("--delays", default="0,20")
    parser.add_argument("--out", default="split-bench-out")
    parser.add_argument("--matrix", default="split_matrix.json", help="model and nodes to split across")
    args = parser.parse_args()
    {"plan": plan, "submit": submit, "status": status, "table": table}[args.command](args)


if __name__ == "__main__":
    main()
