"""Records bench.py results in MLflow and compares two commits, over MLflow's REST API.

Layout, in one experiment for every commit ever measured:

- one parent run per measured commit, tagged with the commit (`mesh_llm.commit`, and
  `mlflow.source.git.commit`, which the MLflow UI shows), its time and a free label;
- one child run per model, machine and arm, carrying the same commit tags so a filter
  such as `tags.mesh_llm.commit = '<sha>'` finds them without walking parents.

Runs are keyed by `gi.key` (commit, model, machine, arm), so recording again skips what
is already there. Credentials come from MLFLOW_TRACKING_URI and either
MLFLOW_TRACKING_USERNAME/PASSWORD or MLFLOW_TRACKING_TOKEN.
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request

EXPERIMENT = os.environ.get("DFLASH_BENCH_EXPERIMENT", "mesh-llm DFlash defaults — decode speed by commit")
ARM_LABELS = {"off": "speculation off", "default": "shipped default"}


class MLflow:
    def __init__(self):
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            raise SystemExit("MLFLOW_TRACKING_URI is not set")
        self.uri = uri.rstrip("/")
        self.headers = {"content-type": "application/json"}
        if os.environ.get("MLFLOW_TRACKING_TOKEN"):
            self.headers["authorization"] = "Bearer " + os.environ["MLFLOW_TRACKING_TOKEN"]
        elif os.environ.get("MLFLOW_TRACKING_USERNAME"):
            pair = f"{os.environ['MLFLOW_TRACKING_USERNAME']}:{os.environ.get('MLFLOW_TRACKING_PASSWORD', '')}"
            self.headers["authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()

    def call(self, method, path, body=None, raw=None, content_type=None):
        headers = dict(self.headers, **({"content-type": content_type} if content_type else {}))
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.uri + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read()
        return json.loads(text) if text else {}

    def experiment_id(self):
        name = urllib.parse.quote(EXPERIMENT)
        try:
            return self.call("GET", f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}")[
                "experiment"]["experiment_id"]
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        return self.call("POST", "/api/2.0/mlflow/experiments/create", {"name": EXPERIMENT})["experiment_id"]

    def search(self, experiment_id, filter_string):
        runs, token = [], None
        while True:
            body = {"experiment_ids": [experiment_id], "filter": filter_string, "max_results": 1000}
            if token:
                body["page_token"] = token
            page = self.call("POST", "/api/2.0/mlflow/runs/search", body)
            runs += page.get("runs", [])
            token = page.get("next_page_token")
            if not token:
                return runs

    def create(self, experiment_id, name, tags):
        run = self.call("POST", "/api/2.0/mlflow/runs/create", {
            "experiment_id": experiment_id, "run_name": name, "start_time": int(time.time() * 1000),
            "tags": [{"key": k, "value": str(v)} for k, v in tags.items()]})["run"]
        return run["info"]["run_id"], run["info"]["artifact_uri"]

    def log(self, run_id, params=None, metrics=None, tags=None):
        now = int(time.time() * 1000)
        self.call("POST", "/api/2.0/mlflow/runs/log-batch", {
            "run_id": run_id,
            "params": [{"key": k, "value": str(v)} for k, v in (params or {}).items()],
            "metrics": [{"key": k, "value": float(v), "timestamp": now, "step": 0}
                        for k, v in (metrics or {}).items() if v is not None],
            "tags": [{"key": k, "value": str(v)} for k, v in (tags or {}).items()]})

    def finish(self, run_id, status="FINISHED"):
        self.call("POST", "/api/2.0/mlflow/runs/update",
                  {"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)})

    def upload(self, artifact_uri, name, text):
        # Artifacts go through the tracking server's proxy (mlflow-artifacts:/<experiment>/<run>/artifacts).
        path = artifact_uri.removeprefix("mlflow-artifacts:")
        self.call("PUT", f"/api/2.0/mlflow-artifacts/artifacts{path}/{name}", raw=text.encode(),
                  content_type="application/octet-stream")


def commit_identity(rev, here):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=here, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    unix, subject = (git("show", "-s", "--format=%ct%x00%s", rev).split("\0") + ["", ""])[:2]
    return {"unix": int(unix) if unix else None, "subject": subject,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(unix))) if unix else ""}


def tag(run, key):
    return next((t["value"] for t in run["data"].get("tags", []) if t["key"] == key), None)


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def split_arm(arm):
    """(workload, base arm, sampled) for an arm name such as agent_default or d7 or off_t07."""
    sampled = arm.endswith("_t07")
    base = arm.removesuffix("_t07")
    workload = "agent" if base.startswith("agent_") else "spec-prompts"
    return workload, base.removeprefix("agent_"), sampled


def arm_label(arm):
    workload, base, sampled = split_arm(arm)
    label = ARM_LABELS.get(base) or f"fixed DFlash block {base.removeprefix('d')}"
    prefix = "agent transcripts · " if workload == "agent" else ""
    return prefix + label + (" · sampled T=0.7" if sampled else " · greedy")


def record(state, matrix, logs, label, here, repo_url, table_md):
    """`logs` maps job id to (arms, gpu) as parsed from the jy log."""
    mlflow = MLflow()
    experiment = mlflow.experiment_id()
    rev = state["rev"]
    commit = commit_identity(rev, here)
    commit_tags = {"mesh_llm.commit": rev, "mesh_llm.commit_short": rev[:9], "mesh_llm.label": label,
                   "mesh_llm.commit_time": commit["time"], "mlflow.source.git.commit": rev,
                   "mlflow.source.git.repoURL": repo_url, "eval.program": "dflash-defaults"}
    models = {model["id"]: model for model in matrix["models"]}
    matrix_text = json.dumps(matrix, indent=1, sort_keys=True)

    parent_key = f"{rev}:dflash-defaults"
    found = mlflow.search(experiment, f"tags.`gi.key` = '{parent_key}'")
    if found:
        parent_id, parent_artifacts = found[0]["info"]["run_id"], found[0]["info"]["artifact_uri"]
    else:
        parent_id, parent_artifacts = mlflow.create(experiment, f"mesh-llm @ {rev[:9]} · {commit['subject'][:70]}", {
            **commit_tags, "gi.key": parent_key, "eval.phase": "measurement",
            "mlflow.note.content": (
                f"Decode speed of mesh-llm {rev[:9]} ({commit['subject']}) with speculation off, the shipped "
                "DFlash default and fixed DFlash blocks, greedy and sampled, on the report split of "
                "skippy/evals/spec-prompts. Child runs hold one model, machine and arm each; speedups compare "
                "arms within the same jy job.")})
        mlflow.log(parent_id, params={"matrix_sha256": hashlib.sha256(matrix_text.encode()).hexdigest(),
                                      "prompt_split": matrix["prompt_split"]},
                   metrics={"commit_time_unix": commit["unix"]})

    existing = {tag(run, "gi.key") for run in mlflow.search(experiment, f"tags.`mesh_llm.commit` = '{rev}'")}
    created = 0
    for job in state["jobs"]:
        arms, gpu = logs[job["job_id"]]
        model = models[job["model"]]
        entry = model["machines"][job["machine"]]
        machine = matrix["machines"][job["machine"]]
        draft = matrix["drafts"][entry["draft"]]
        for arm, result in arms.items():
            key = f"{rev}:{job['model']}:{job['machine']}:{arm}"
            if key in existing:
                continue
            workload, base, sampled = split_arm(arm)
            off = arms.get(("agent_" if workload == "agent" else "") + ("off_t07" if sampled else "off")) or {}
            groups, off_groups = result.get("groups") or {}, off.get("groups") or {}
            metrics = {"commit_time_unix": commit["unix"],
                       "gpu_clock_mhz": gpu.get(arm, {}).get("clock_mhz"),
                       "gpu_temp_c": gpu.get(arm, {}).get("temp_c")}
            for g in groups:
                tok_s = (groups.get(g) or {}).get("tok_s")
                metrics[f"decode_tok_s_{g}"] = tok_s
                metrics[f"acceptance_{g}"] = (groups.get(g) or {}).get("accept")
                before = (off_groups.get(g) or {}).get("tok_s")
                if tok_s and before and base != "off":
                    metrics[f"speedup_vs_off_{g}"] = tok_s / before
                metrics[f"ttft_ms_p50_{g}"] = (groups.get(g) or {}).get("ttft_ms_p50")
            fixed = re.fullmatch(r"d(\d+)", base)
            uses_draft = bool(fixed) or (base == "default" and entry["default"])
            run_id, _ = mlflow.create(experiment, f"{model['name']} · {job['machine']} · {arm_label(arm)}", {
                **commit_tags, "gi.key": key, "mlflow.parentRunId": parent_id, "jy.job_id": job["job_id"],
                "hardware.node": job["node"], "eval.execution_state": "error" if "error" in result else "completed"})
            mlflow.log(run_id, params={
                "model": model["name"], "machine": job["machine"], "backend": machine["backend"],
                "node": job["node"], "arm": base, "workload": workload,
                "temperature": result.get("temp", 0.7 if sampled else 0),
                "fixed_block": fixed.group(1) if fixed else "", "default_speculation": entry["default"],
                "target_sha256": model["target"]["sha256"], "draft": draft["url"].split("/resolve/")[0] if uses_draft else "",
                "draft_sha256": draft["sha256"] if uses_draft else "",
                "prompts": "agent-transcripts" if workload == "agent" else f"spec-prompts/{matrix['prompt_split']}",
                "error": result.get("error", "")}, metrics=metrics)
            mlflow.finish(run_id, "FAILED" if "error" in result else "FINISHED")
            created += 1

    # Read the commit back before calling it verified.
    stored = mlflow.search(experiment, f"tags.`mesh_llm.commit` = '{rev}' and tags.`eval.program` = 'dflash-defaults'")
    expected = sum(len(arms) for arms, _ in logs.values()) + 1
    verified = len(stored) >= expected
    mlflow.log(parent_id, tags={"eval.verification_state": "verified" if verified else "requires review"})
    mlflow.upload(parent_artifacts, "table.md", table_md)
    mlflow.upload(parent_artifacts, "matrix.json", matrix_text)
    mlflow.upload(parent_artifacts, "jobs.json", json.dumps(state, indent=1))
    mlflow.log(parent_id, tags={"mlflow.note.content": f"## mesh-llm {rev[:9]} · {commit['subject']}\n\n"
                                                       f"{table_md.strip()}"})
    mlflow.finish(parent_id, "FINISHED" if verified else "FAILED")
    publish_dashboard(mlflow, experiment, rev, label, commit, parent_id, table_md, matrix)
    print(f"MLflow: {created} new runs for {rev[:9]} in '{EXPERIMENT}' "
          f"({len(stored)}/{expected} read back) — {mlflow.uri}/#/experiments/{experiment}/runs/{parent_id}")


def load_commit(mlflow, experiment, rev):
    rows = {}
    for run in mlflow.search(experiment, f"tags.`mesh_llm.commit` = '{rev}'"):
        params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
        metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
        if "arm" in params:
            sampled = params["temperature"] not in ("0", "0.0")
            rows[(params["model"], params["machine"], params.get("workload", "spec-prompts"), params["arm"],
                  sampled)] = metrics
    return rows


def compare_table(mlflow, experiment, rev_a, rev_b, matrix):
    """The shipped default's decode speed and gain per model, machine and workload, A against B,
    in the matrix's model and machine order so it reads like the main table."""
    models = [model["name"] for model in matrix["models"]]
    machines = list(matrix["machines"])
    rank = lambda name, names: names.index(name) if name in names else len(names)
    a, b = load_commit(mlflow, experiment, rev_a), load_commit(mlflow, experiment, rev_b)
    if not a or not b:
        raise SystemExit(f"no runs for {rev_a[:9] if not a else rev_b[:9]} in '{EXPERIMENT}'")
    lines = [f"| Machine | Model | Workload | {rev_a[:9]} | {rev_b[:9]} | Change | Gain vs off |",
             "|---|---|---|---|---|---|---|"]
    for model, machine, workload, arm, sampled in sorted(
            set(a) | set(b), key=lambda k: (rank(k[0], models), rank(k[1], machines), k[2] == "agent", k[4])):
        if arm != "default":
            continue
        key = (model, machine, workload, arm, sampled)
        before, after = a.get(key, {}).get("decode_tok_s_all"), b.get(key, {}).get("decode_tok_s_all")
        change = f"{100 * (after / before - 1):+.1f}%" if before and after else "—"
        gains = [m.get(key, {}).get("speedup_vs_off_all") for m in (a, b)]
        gain = " → ".join(f"{g:.2f}×" if g else "—" for g in gains)
        name = ("agent transcripts" if workload == "agent" else "all prompts") + (", sampled" if sampled else "")
        lines.append(f"| {machine} | {model} | {name} | {fmt(before)} | {fmt(after)} | **{change}** | {gain} |")
    return "\n".join(lines) + "\n"


def compare(rev_a, rev_b, matrix):
    mlflow = MLflow()
    print(f"Shipped default, decode tok/s: {rev_a[:9]} → {rev_b[:9]}\n")
    print(compare_table(mlflow, mlflow.experiment_id(), rev_a, rev_b, matrix))


def publish_dashboard(mlflow, experiment, rev, label, commit, parent_id, table_md, matrix):
    """Writes the latest commit's table, its change from the previous commit and an index of
    recorded commits into the experiment description, which MLflow shows as Markdown."""
    parents = [run for run in mlflow.search(experiment, "tags.`eval.program` = 'dflash-defaults'")
               if (tag(run, "gi.key") or "").endswith(":dflash-defaults")]
    parents.sort(key=lambda run: tag(run, "mesh_llm.commit_time") or "", reverse=True)
    link = lambda run_id: f"{mlflow.uri}/#/experiments/{experiment}/runs/{run_id}"
    previous = next((run for run in parents if tag(run, "mesh_llm.commit") != rev
                     and (tag(run, "mesh_llm.commit_time") or "") <= commit["time"]), None)
    sections = [
        f"## Latest: {label} · mesh-llm [{rev[:9]}]({link(parent_id)}) · {commit['subject']}",
        f"Committed {commit['time'][:10]}. Decode tok/s on held-out prompts, one request at a time. "
        "Before = speculation off. +DFlash = best fixed draft length. All together = the shipped default "
        "with no speculation config. Gain = All together ÷ Before, both from the same job.",
        table_md.strip(),
    ]
    if previous:
        prev = tag(previous, "mesh_llm.commit")
        sections += [f"## Change from [{prev[:9]}]({link(previous['info']['run_id'])}) "
                     f"({tag(previous, 'mesh_llm.label')})",
                     compare_table(mlflow, experiment, prev, rev, matrix).strip()]
    index = ["| Commit | Label | Committed | Verified |", "|---|---|---|---|"]
    for run in parents[:15]:
        index.append(f"| [{tag(run, 'mesh_llm.commit_short')}]({link(run['info']['run_id'])}) | "
                     f"{tag(run, 'mesh_llm.label')} | {(tag(run, 'mesh_llm.commit_time') or '')[:10]} | "
                     f"{tag(run, 'eval.verification_state') or '—'} |")
    sections += ["## Recorded commits", "\n".join(index)]
    note = "\n\n".join(sections)
    mlflow.call("POST", "/api/2.0/mlflow/experiments/set-experiment-tag",
                {"experiment_id": experiment, "key": "mlflow.note.content", "value": note})
