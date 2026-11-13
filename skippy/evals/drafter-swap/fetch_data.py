#!/usr/bin/env python3
"""Copies a finished datagen job's train-<domain>.jsonl artifacts into data/.

    fetch_data.py JOB_ID [JOB_ID ...]

Fine-tune jobs read the data from this branch at a pinned commit, because jy keeps
a job's workspace only when it fails. Prints each file's sha256 and row count.
"""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://pilot.jianyang.cloud/v1"
SESSION = Path.home() / "Library/Application Support/jianyang/session.json"


def get(url, key):
    # The API's edge refuses urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"authorization": f"Bearer {key}", "user-agent": "drafter-swap/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    key = json.loads(SESSION.read_text())["api_key"]
    (HERE / "data").mkdir(exist_ok=True)
    for job_id in sys.argv[1:]:
        job = json.loads(subprocess.run(["jy", "job", job_id], check=True, capture_output=True, text=True).stdout)
        runs = job.get("runs") or job.get("job", {}).get("runs")
        run_id = runs[-1]["id"]
        listing = json.loads(get(f"{API}/artifacts/{run_id}", key))["artifacts"]
        for artifact in listing:
            name = artifact["key"].rsplit("/", 1)[1]
            if not (name.startswith("train-") and name.endswith(".jsonl")):
                continue
            kind = artifact["key"].rsplit("/", 2)[1]
            body = get(f"{API}/artifacts/{run_id}/{kind}/{name}", key)
            if hashlib.sha256(body).hexdigest() != artifact["sha256"]:
                sys.exit(f"{name}: sha256 mismatch against the artifact record")
            (HERE / "data" / name).write_bytes(body)
            rows = body.count(b"\n")
            print(f"{name}  rows {rows}  sha256 {artifact['sha256']}  from {job_id}/{run_id}")


if __name__ == "__main__":
    main()
