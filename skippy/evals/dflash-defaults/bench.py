#!/usr/bin/env python3
"""Reproduces the DFlash default-speculation table on the lab through jy.

    bench.py plan   [--rev REV] [--only MODEL,...] [--machines LABEL,...] [--out DIR]
    bench.py submit [same]        submit one jy job per (model, machine) and record them
    bench.py status [--out DIR]   show each job's state
    bench.py table  [--out DIR]   read results back from the job logs and print the table
    bench.py record [--out DIR] [--label NAME]   store the results in MLflow (recorder.py)
    bench.py run    [same]        submit, wait, print the table, and record it when MLflow is set up
    bench.py compare REV_A REV_B  per-model change in decode speed between two recorded commits

Jobs build mesh-llm at REV and run skippy/evals/dflash-defaults/run.sh from that
checkout, so REV must contain this directory and be pushed to the repository URL.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/general-inference/mesh-llm.git"
TERMINAL = {"succeeded", "failed", "cancelled"}


def load_matrix():
    return json.loads((HERE / "matrix.json").read_text())


def git_rev(rev):
    return subprocess.run(["git", "rev-parse", rev], cwd=HERE, check=True, capture_output=True,
                          text=True).stdout.strip()


def plan(args):
    matrix = load_matrix()
    rev = git_rev(args.rev)
    only = set(args.only.split(",")) if args.only else None
    machines = set(args.machines.split(",")) if args.machines else None
    next_node = {label: 0 for label in matrix["machines"]}
    jobs = []
    for model in matrix["models"]:
        if only and model["id"] not in only:
            continue
        for label, entry in model["machines"].items():
            if "skip" in entry or (machines and label not in machines):
                continue
            machine = matrix["machines"][label]
            # Spread a machine kind's jobs over its nodes so they run in parallel.
            node = args.node or machine["nodes"][next_node[label] % len(machine["nodes"])]
            next_node[label] += 1
            draft = matrix["drafts"][entry["draft"]]
            env = {
                "LLAMA_STAGE_USE_SCCACHE": "0",
                "BENCH_SPLIT": matrix["prompt_split"],
                "BENCH_TARGET_URL": model["target"]["url"],
                "BENCH_TARGET_SHA256": model["target"]["sha256"],
                "BENCH_TARGET_BYTES": str(model["target"]["bytes"]),
                "BENCH_TARGET_FILE": model["target"]["url"].rsplit("/", 1)[1],
                "BENCH_DRAFT_URL": draft["url"],
                "BENCH_DRAFT_SHA256": draft["sha256"],
                "BENCH_DRAFT_BYTES": str(draft["bytes"]),
                "BENCH_DRAFT_FILE": draft["url"].rsplit("/", 1)[1],
                "BENCH_DEFAULT_DRAFT_FILE": draft["url"].rsplit("/", 1)[1] if entry["default"] else "",
                "BENCH_FIXED_BLOCKS": " ".join(str(k) for k in machine["fixed_blocks"]),
                "BENCH_AGENT": "1" if matrix.get("agent_transcripts") else "0",
            }
            spec = {"spec": {
                "name": f"DFlash defaults · {model['name']} · {label} · {node} @ {rev[:9]}",
                "description": "Decode tok/s with speculation off, the shipped default and fixed DFlash blocks, "
                               f"greedy and sampled, on the {matrix['prompt_split']} split of spec-prompts.",
                "priority": 20,
                "target": {"executor": "lab", "nodes": [node]},
                "harness": {"name": "custom", "version": "dflash-defaults-1",
                            "trace": {"kind": "file", "ref": f"skippy/evals/spec-prompts/spec-prompts.jsonl@{rev[:9]}"},
                            "seed": 0, "warmup_requests": 1, "concurrency": [1], "request_count": 77},
                "budget": {"max_usd": 0, "max_duration_s": 21600},
                "labels": {"suite": "dflash-defaults", "model": model["id"], "machine": label, "rev": rev[:9]},
                "runtime": {
                    # jy requires a registered package; jobs fetch and verify their own weights.
                    "model_package": {"ref": machine["placeholder_package"], "egress": "lab_only"},
                    "models": [],
                    "workspace": {"repos": [{"url": REPO_URL, "rev": rev, "path": "mesh-llm"}],
                                  "build": "bash mesh-llm/skippy/evals/dflash-defaults/build.sh",
                                  "retain": "on_failure"},
                    "env": env,
                    "command": "bash mesh-llm/skippy/evals/dflash-defaults/run.sh",
                },
            }}
            jobs.append({"key": f"{model['id']}@{label}", "model": model["id"], "machine": label,
                         "node": node, "spec": spec})
    out = Path(args.out)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    for job in jobs:
        (out / "specs" / f"{job['model']}@{job['machine'].replace(' ', '-')}.json").write_text(
            json.dumps(job["spec"], indent=1))
        print(f"{job['key']:32s} -> {job['node']}")
    return rev, jobs


def state_path(args):
    return Path(args.out) / "state.json"


def submit(args):
    rev, jobs = plan(args)
    state = {"rev": rev, "jobs": []}
    for job in jobs:
        spec_file = Path(args.out) / "specs" / f"{job['model']}@{job['machine'].replace(' ', '-')}.json"
        out = subprocess.run(["jy", "submit", str(spec_file)], capture_output=True, text=True)
        match = re.search(r"\bjob_[a-z0-9]+", out.stdout + out.stderr)
        if not match:
            sys.exit(f"submit failed for {job['key']}:\n{out.stdout}{out.stderr}")
        print(f"{job['key']:32s} {match.group(0)}")
        state["jobs"].append({k: job[k] for k in ("key", "model", "machine", "node")} | {"job_id": match.group(0)})
        state_path(args).write_text(json.dumps(state, indent=1))
    return state


def statuses(state):
    listed = json.loads(subprocess.run(["jy", "jobs", "--limit", "200"], check=True, capture_output=True,
                                       text=True).stdout)["jobs"]
    by_id = {job["id"]: job["status"] for job in listed}
    return {job["job_id"]: by_id.get(job["job_id"], "unknown") for job in state["jobs"]}


def status(args):
    state = json.loads(state_path(args).read_text())
    for job in state["jobs"]:
        print(f"{job['key']:32s} {job['job_id']}  {statuses(state)[job['job_id']]}")


def job_log(job_id):
    """Each arm's DFLASH_BENCH summary and the GPU state recorded just before it."""
    log = subprocess.run(["jy", "logs", job_id], capture_output=True, text=True).stdout
    arms, gpu = {}, {}
    for line in log.splitlines():
        line = line.removeprefix("[out] ")
        if line.startswith("DFLASH_BENCH "):
            arm = json.loads(line[len("DFLASH_BENCH "):])
            arms[arm["arm"]] = arm
        elif match := re.match(r"gpu: (\S+) (\d+) MHz, (\d+),", line):
            gpu[match.group(1)] = {"clock_mhz": int(match.group(2)), "temp_c": int(match.group(3))}
    return arms, gpu


def results(job_id):
    return job_log(job_id)[0]


def rate(arms, arm, group):
    value = ((arms.get(arm) or {}).get("groups") or {}).get(group, {}).get("tok_s")
    return value


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def table(args):
    matrix = load_matrix()
    state = json.loads(state_path(args).read_text())
    by_key = {job["key"]: job for job in state["jobs"]}
    collected = {}
    rows = ["| Machine | Model | Workload | Before | +DFlash | All together | Gain |",
            "|---|---|---|---|---|---|---|"]
    for model in matrix["models"]:
        for label, entry in model["machines"].items():
            if "skip" in entry:
                rows.append(f"| {label} | {model['name']} | — | n/a | n/a | n/a | {entry['skip']} |")
                continue
            job = by_key.get(f"{model['id']}@{label}")
            if not job:
                continue
            arms = collected[job["key"]] = results(job["job_id"])
            fixed = [arm for arm in arms if re.fullmatch(r"d\d+", arm) and rate(arms, arm, "all")]
            best = max(fixed, key=lambda arm: rate(arms, arm, "all"), default=None)
            note = "" if entry["default"] else " (off by default)"
            for workload, group, off, default, fixed_arm, suffix in [
                ("Prose", "prose", "off", "default", best, ""),
                ("Code", "code", "off", "default", best, ""),
                ("Math", "math", "off", "default", best, ""),
                ("**All**", "all", "off", "default", best, ""),
                ("**All**", "all", "off_t07", "default_t07", None, ", sampled"),
                *([("Agentic coding", "all", "agent_off", "agent_default", None, "")] if "agent_off" in arms else []),
            ]:
                before, after = rate(arms, off, group), rate(arms, default, group)
                gain = f"{after / before:.2f}×" if before and after else ("Running" if not arms else "—")
                bold = workload == "**All**"
                shown = fmt(after) + (note if bold else "")
                rows.append(f"| {label}{suffix} | {model['name']} | {workload} | {fmt(before)} | "
                            f"{fmt(rate(arms, fixed_arm, group)) if fixed_arm else '—'} | "
                            f"{'**' + shown + '**' if bold else shown} | {'**' + gain + '**' if bold else gain} |")
    agent_rows = ["| Machine | Model | History | Before | All together | Gain |", "|---|---|---|---|---|---|"]
    for model in matrix["models"]:
        for label in model["machines"]:
            arms = collected.get(f"{model['id']}@{label}")
            if not arms:
                continue
            groups = ((arms.get("agent_off") or {}).get("groups") or {})
            for band in [g for g in groups if g != "all"] + ["all"]:
                before, after = rate(arms, "agent_off", band), rate(arms, "agent_default", band)
                gain = f"{after / before:.2f}×" if before and after else "—"
                if band == "all":
                    agent_rows.append(f"| {label} | {model['name']} | **All** | {fmt(before)} | **{fmt(after)}** | **{gain}** |")
                else:
                    agent_rows.append(f"| {label} | {model['name']} | {band} | {fmt(before)} | {fmt(after)} | {gain} |")
    if len(agent_rows) > 2:
        rows += ["", "Coding-agent transcripts (real OpenHands histories, 64K context, greedy):", ""] + agent_rows
    out = Path(args.out)
    (out / "results.json").write_text(json.dumps({"rev": state["rev"], "jobs": state["jobs"], "arms": collected},
                                                 indent=1))
    text = "\n".join(rows) + "\n"
    (out / "table.md").write_text(text)
    print(f"mesh-llm @ {state['rev'][:9]}; decode tok/s on the held-out spec-prompts split.\n")
    print(text)
    return text


def record(args):
    import recorder
    state = json.loads(state_path(args).read_text())
    text = table(args)
    logs = {job["job_id"]: job_log(job["job_id"]) for job in state["jobs"]}
    recorder.record(state, load_matrix(), logs, args.label or args.rev, HERE, REPO_URL, text)


def compare(args):
    import recorder
    if len(args.revs) != 2:
        sys.exit("compare needs two commits")
    recorder.compare(git_rev(args.revs[0]), git_rev(args.revs[1]), load_matrix())


def run(args):
    state = submit(args)
    while True:
        current = statuses(state)
        pending = [job_id for job_id, s in current.items() if s not in TERMINAL]
        print(time.strftime("%H:%M:%S"), f"{len(current) - len(pending)}/{len(current)} jobs finished", flush=True)
        if not pending:
            break
        time.sleep(args.poll)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        record(args)
    else:
        table(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "submit", "status", "table", "record", "run", "compare"])
    parser.add_argument("revs", nargs="*", help="for compare: two mesh-llm commits")
    parser.add_argument("--rev", default="HEAD", help="mesh-llm commit to build and measure (pushed)")
    parser.add_argument("--only", help="comma-separated model ids from matrix.json")
    parser.add_argument("--machines", help="comma-separated machine labels from matrix.json")
    parser.add_argument("--out", default="dflash-bench-out", help="where specs, state and the table go")
    parser.add_argument("--poll", type=int, default=300, help="seconds between status checks in `run`")
    parser.add_argument("--label", help="free-form name for this commit in MLflow, e.g. main or a PR")
    parser.add_argument("--node", help="run every planned job on this node instead of the matrix's")
    args = parser.parse_args()
    commands = {"plan": plan, "submit": submit, "status": status, "table": table, "record": record,
                "run": run, "compare": compare}
    commands[args.command](args)


if __name__ == "__main__":
    main()
