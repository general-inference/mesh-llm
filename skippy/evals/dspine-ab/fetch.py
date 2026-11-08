"""Downloads every file in manifest.json to local disk and checks its SHA-256.

Files land in ROOT/<repo>/<path> (one pinned revision per repo). A file whose digest
does not match is downloaded again from scratch, twice at most. Never uses the NAS.
"""

import hashlib
import json
import os
import subprocess
import sys

manifest, root = sys.argv[1], sys.argv[2]
repos = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


bad = []
for m in json.load(open(manifest)):
    if repos and m["repo"] not in repos:
        continue
    path = os.path.join(root, m["repo"], m["path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = False
    for attempt in range(3):
        if os.path.exists(path) and os.path.getsize(path) == m["bytes"] and sha256(path) == m["sha256"]:
            ok = True
            break
        if os.path.exists(path):
            os.remove(path)
        subprocess.run(["curl", "-fL", "--retry", "10", "-sS", "-o", path, m["url"]], check=False)
    print(f"{'ok ' if ok else 'BAD'} {m['repo']}@{m['rev'][:9]} {m['path']} {m['sha256'][:12]}", flush=True)
    if not ok:
        bad.append(m["path"])
if bad:
    sys.exit(f"checksum failures: {bad}")
