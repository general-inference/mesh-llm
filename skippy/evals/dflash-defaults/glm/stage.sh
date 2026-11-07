#!/usr/bin/env bash
# Stages GLM-5.3-Flash as the au-mel-01 Sparks hold it (meshllm layers@d1b3be77 rewritten to
# match Unsloth a38483c8; manifest 1341078e) on a node that cannot copy it from them: download
# the Hugging Face package, rewrite it with glm_rename.py, and check every file against the
# manifest before installing it in the split harness's Hugging Face cache layout.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REV=d1b3be770eacf2cfef13e62866fefc2f3917616f
LOCAL_REV=1341078ef859d8ddf5cad491c619c463491fac82
SRC="$HOME/.cache/glm-src/$REV"
HUB="$HOME/.cache/split-bench-hf/hub/models--meshllm--GLM-5.3-Flash-UD-Q4_K_XL-layers"
SNAP="$HUB/snapshots/$LOCAL_REV"
[ -f "$SNAP/.verified" ] && { echo "already staged: $SNAP"; exit 0; }
BASE=https://huggingface.co/meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers/resolve/$REV
mkdir -p "$SRC" "$HUB/blobs" "$SNAP"
cd "$SRC"
curl -fsSL -o model-package.json "$BASE/model-package.json" || exit 1
python3 -c 'import json,sys; [print(e["path"], e["sha256"]) for e in json.load(open("model-package.json"))["artifact_catalog"]["entries"]]' > list.txt
fetch() { p="$1"; s="$2"; mkdir -p "$(dirname "$p")"
  for _ in 1 2 3; do
    [ -f "$p" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$s" ] && return 0
    rm -f "$p"; curl -fsSL --retry 10 -o "$p" "$BASE/$p"
  done; echo "FAILED $p"; return 1; }
export -f fetch; export BASE
T0=$(date +%s)
xargs -P 6 -n 2 bash -c 'fetch "$0" "$1"' < list.txt || exit 1
echo "source verified in $(( $(date +%s) - T0 ))s"
T0=$(date +%s)
OUT="$HOME/.cache/glm-src/renamed"
find . -name "*.gguf" | sed "s#^\./##" | sort | xargs -P 6 -n 8 python3 "$HERE/glm_rename.py" "$SRC" "$OUT" || exit 1
echo "rewritten in $(( $(date +%s) - T0 ))s"
cp "$HERE/model-package-1341078e.json" "$OUT/model-package.json"
cd "$OUT"
python3 - "$OUT" "$HUB" "$SNAP" <<'PY' || exit 1
import hashlib, json, os, sys
out, hub, snap = sys.argv[1:]
manifest = os.path.join(out, "model-package.json")
raw = open(manifest, "rb").read()
msha = hashlib.sha256(raw).hexdigest()
assert msha.startswith("1341078ef859d8dd"), msha
bad = []
entries = json.loads(raw)["artifact_catalog"]["entries"]
for e in entries:
    h = hashlib.sha256()
    with open(os.path.join(out, e["path"]), "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    if h.hexdigest() != e["sha256"]: bad.append(e["path"])
if bad: print("mismatch:", bad); sys.exit(1)
for e in entries + [{"path": "model-package.json", "sha256": msha}]:
    blob = os.path.join(hub, "blobs", e["sha256"]); link = os.path.join(snap, e["path"])
    os.replace(os.path.join(out, e["path"]), blob)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link): os.remove(link)
    os.symlink(os.path.relpath(blob, os.path.dirname(link)), link)
print("all", len(entries), "artifacts match manifest", msha[:16])
PY
touch "$SNAP/.verified"
echo "staged $SNAP"
