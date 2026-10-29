#!/usr/bin/env bash
# One member of a split formation: stages the layer package on local disk, then serves
# its share of the model. The coordinator measures once every stage has joined.
#
# Inputs (environment, set by split.py):
#   JIANYANG_ROLE              coordinator | joiner
#   JIANYANG_LEADER_HOST       the coordinator's LAN address (formation rendezvous)
#   SPLIT_REPO / SPLIT_REV     the layer package on Hugging Face
#   SPLIT_STAGES               how many machines the split must span before measuring
#   SPLIT_ARM                  off | default | ngram (speculation for this formation)
#   SPLIT_WIRE_DELAY_MS        injected per-hop delay, 0 for the plain LAN
#   SPLIT_LANES / SPLIT_CTX    parallel lanes and context per lane
#   SPLIT_ALLOW_UNCERTIFIED    1 to serve an architecture CI has not certified for splits
set -uo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/dflash-defaults"
ART="${JIANYANG_ARTIFACT_DIR:?}"
ROLE="${JIANYANG_ROLE:?}"
API=9447
CONSOLE=3232

TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
BIN=""
for candidate in "$TARGET_DIR/release/mesh-llm" target/release/mesh-llm; do
  [ -x "$candidate" ] && { BIN="$candidate"; break; }
done
[ -n "$BIN" ] || { echo "no release host build" >&2; exit 1; }
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"

# The shared store is slow to load from and has returned damaged copies of large files,
# so every member serves from its own verified copy in a private Hugging Face cache.
export HF_HOME="$HOME/.cache/split-bench-hf"
export HF_HUB_OFFLINE=1
REPO_DIR="models--${SPLIT_REPO//\//--}"
LOCAL="$HF_HOME/hub/$REPO_DIR/snapshots/$SPLIT_REV"
SHARED="${JIANYANG_STORE_HF:-/nonexistent}/hub/$REPO_DIR/snapshots/$SPLIT_REV"
mkdir -p "$LOCAL"
if [ ! -f "$LOCAL/.verified" ]; then
  T0=$(date +%s)
  if [ -f "$SHARED/model-package.json" ]; then
    echo "staging $SPLIT_REPO@${SPLIT_REV:0:9} from the shared store"
    rsync -rL --partial "$SHARED/" "$LOCAL/" || { echo "copy from $SHARED failed" >&2; exit 1; }
  else
    # Not on the shared store: fetch only the files the package manifest names.
    echo "downloading $SPLIT_REPO@${SPLIT_REV:0:9} from Hugging Face"
    BASE="https://huggingface.co/$SPLIT_REPO/resolve/$SPLIT_REV"
    curl -fsSL --retry 5 -o "$LOCAL/model-package.json" "$BASE/model-package.json" || exit 1
    python3 -c 'import json,sys; [print(e["path"]) for e in json.load(open(sys.argv[1]))["artifact_catalog"]["entries"]]' \
      "$LOCAL/model-package.json" | while read -r path; do
      mkdir -p "$LOCAL/$(dirname "$path")"
      curl -fsSL --retry 10 -C - -o "$LOCAL/$path" "$BASE/$path" || { echo "fetch failed: $path" >&2; exit 1; }
    done || exit 1
  fi
  SNAP="$LOCAL" python3 - <<'PY' || { echo "staged package failed verification" >&2; exit 1; }
import hashlib, json, os, sys
snap = os.environ["SNAP"]
manifest = json.load(open(os.path.join(snap, "model-package.json")))
bad = []
for entry in manifest["artifact_catalog"]["entries"]:
    path = os.path.join(snap, entry["path"])
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    except OSError:
        bad.append(entry["path"]); continue
    if h.hexdigest() != entry["sha256"]:
        bad.append(entry["path"])
if bad:
    print("bad files:", bad); sys.exit(1)
PY
  touch "$LOCAL/.verified"
  echo "package staged and verified in $(( $(date +%s) - T0 ))s"
fi

case "${SPLIT_ARM:?}" in
  off) ARM_FLAGS="--speculative-strategy disabled" ;;
  default) ARM_FLAGS="" ;;
  ngram) ARM_FLAGS="--speculative-strategy ngram-suffix" ;;
  *) echo "unknown arm $SPLIT_ARM" >&2; exit 1 ;;
esac
export MESH_LLM_BENCH_DOWNSTREAM_WIRE_DELAY_MS="${SPLIT_WIRE_DELAY_MS:-0}"
MODEL="$SPLIT_REPO@$SPLIT_REV"
SPLIT_FLAGS="--split"
[ "${SPLIT_ALLOW_UNCERTIFIED:-0}" = 1 ] && SPLIT_FLAGS="--split --allow-uncertified-split"
LOG="$ART/$ROLE.log"

if [ "$ROLE" = joiner ]; then
  LEAD="${JIANYANG_LEADER_HOST:?no leader host in the formation env}"
  TOKEN=""
  for _ in $(seq 1 360); do
    TOKEN=$(curl -sf -m 5 "http://$LEAD:$CONSOLE/api/status" 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)
    [ -n "$TOKEN" ] && break
    sleep 5
  done
  [ -n "$TOKEN" ] || { echo "no invite token from $LEAD after 30m" >&2; exit 1; }
  echo "joining the mesh at $LEAD"
  "$BIN" --log-format json --parallel "${SPLIT_LANES:-1}" $ARM_FLAGS serve --model "$MODEL" $SPLIT_FLAGS \
    --join "$TOKEN" --ctx-size "${SPLIT_CTX:-8192}" --port "$API" --console "$CONSOLE" --bind-port 7843 \
    >"$LOG" 2>&1 &
  SERVE=$!
  trap 'tail -60 "$LOG" >&2; kill "$SERVE" 2>/dev/null || true' EXIT
  # Serve until the coordinator goes away: it owns the measurement and the end of the run.
  MISSES=0
  while kill -0 "$SERVE" 2>/dev/null; do
    if curl -sf -m 5 "http://$LEAD:$CONSOLE/api/status" >/dev/null 2>&1; then MISSES=0
    else MISSES=$((MISSES + 1)); [ "$MISSES" -ge 6 ] && break; fi
    sleep 10
  done
  echo "joiner finished"
  exit 0
fi

"$BIN" --log-format json --parallel "${SPLIT_LANES:-1}" $ARM_FLAGS serve --model "$MODEL" $SPLIT_FLAGS --listen-all \
  --ctx-size "${SPLIT_CTX:-8192}" --port "$API" --console "$CONSOLE" --bind-port 7842 >"$LOG" 2>&1 &
SERVE=$!
trap 'rc=$?; [ $rc -ne 0 ] && tail -120 "$LOG" >&2; kill "$SERVE" 2>/dev/null || true' EXIT

MODEL_ID=""
for i in $(seq 1 360); do
  MODEL_ID=$(curl -sf -m 5 "http://127.0.0.1:$CONSOLE/api/status" 2>/dev/null | STAGES="$SPLIT_STAGES" python3 -c '
import json, os, sys
d = json.load(sys.stdin); st = (d.get("runtime") or {}).get("stages") or []
name = d.get("model_name") or (st[0].get("model_id") if st else "") or ""
print(name if d.get("llama_ready") and len(st) >= int(os.environ["STAGES"]) else "")' 2>/dev/null || true)
  [ -n "$MODEL_ID" ] && break
  [ $((i % 6)) -eq 0 ] && echo "waiting for $SPLIT_STAGES stages — $((i * 10))s"
  kill -0 "$SERVE" 2>/dev/null || { echo "coordinator died while forming" >&2; exit 1; }
  sleep 10
done
[ -n "$MODEL_ID" ] || { echo "the split never reached $SPLIT_STAGES stages" >&2; exit 1; }
SPANS=$(curl -sf -m 5 "http://127.0.0.1:$CONSOLE/api/status" | python3 -c '
import json, sys
st = (json.load(sys.stdin).get("runtime") or {}).get("stages") or []
print(" ".join("%s-%s" % (s.get("layer_start"), s.get("layer_end")) for s in sorted(st, key=lambda s: s["stage_index"])))' 2>/dev/null || true)
echo "split serving $MODEL_ID across $SPLIT_STAGES stages: layers $SPANS"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader | sed "s/^/gpu: $SPLIT_ARM /" || true

ARM="$SPLIT_ARM" PORT="$API" MODEL_ID="$MODEL_ID" TEMP=0 SPLIT=report \
  PROMPTS="$PWD/skippy/evals/spec-prompts/spec-prompts.jsonl" python3 "$HERE/measure.py" \
  || echo "DFLASH_BENCH {\"arm\": \"$SPLIT_ARM\", \"error\": \"measurement failed\"}"
echo "SPLIT_LAYOUT {\"stages\": $SPLIT_STAGES, \"spans\": \"$SPANS\", \"wire_delay_ms\": ${SPLIT_WIRE_DELAY_MS:-0}}"
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
