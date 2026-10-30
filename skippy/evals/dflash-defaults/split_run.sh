#!/usr/bin/env bash
# One member of a split formation: stages the layer package on local disk, then serves
# its share of the model. The coordinator measures once every stage has joined.
#
# Inputs (environment, set by split.py):
#   JIANYANG_ROLE              coordinator | joiner
#   JIANYANG_LEADER_HOST       the coordinator's LAN address (formation rendezvous)
#   SPLIT_REPO / SPLIT_REV     the layer package on Hugging Face, or
#   SPLIT_GGUF / _SHA256       a single GGUF already on every member, relative to $HOME
#   SPLIT_STAGES               how many machines the split must span before measuring
#   SPLIT_ARM                  off | default | ngram | dflash (speculation for this formation)
#   SPLIT_DRAFT_URL / _SHA256  the DFlash draft the dflash arm serves explicitly; every member
#                              holds it at the same path, since the final stage loads it
#   SPLIT_DFLASH_CAP           draft tokens per block for the dflash arm, empty for the default
#   SPLIT_DFLASH_FIXED         1 to pin blocks at the cap (adaptive sizing off), for profiling
#   SPLIT_LIMIT                measure only this many prompts, for profiling
#   SPLIT_WIRE_MBPS            simulated per-hop bandwidth limit, empty for none
#   SPLIT_PREFIX               1 to measure prefix-cache reuse on agent turns instead of decode speed
#   SPLIT_CODEC                activation wire codec for every arm (skippy.activation_codec), empty for the default
#   SPLIT_WIRE_DELAY_MS        injected per-hop delay, 0 for the plain LAN
#   SPLIT_LANES / SPLIT_CTX    parallel lanes and context per lane
#   SPLIT_ALLOW_UNCERTIFIED    1 to serve an architecture CI has not certified for splits
#   SPLIT_CUT                  layer boundaries for an even cut, e.g. "32" or "16,32,48"
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

# Every member serves from its own verified copy in a private Hugging Face cache. The
# shared model store returns I/O errors on writes this large, so the coordinator downloads
# the package once (or already holds it) and the joiners copy it from the coordinator.
export HF_HOME="$HOME/.cache/split-bench-hf"
export HF_HUB_OFFLINE=1
REPO_DIR="models--${SPLIT_REPO//\//--}"
LOCAL="$HF_HOME/hub/$REPO_DIR/snapshots/$SPLIT_REV"
# A real Hugging Face cache has blobs/ beside snapshots/, and the resolver requires it.
mkdir -p "$LOCAL" "$HF_HOME/hub/$REPO_DIR/blobs"
PKG_PORT=9446
LEAD="${JIANYANG_LEADER_HOST:-}"

# Fetch every file the manifest names from BASE (Hugging Face, or the coordinator on the
# LAN), then check each against the manifest's SHA-256 before trusting the copy.
fetch_package() {
  local base="$1"
  curl -fsSL --retry 5 -o "$LOCAL/model-package.json" "$base/model-package.json" || return 1
  python3 -c 'import json,sys; [print(e["path"]) for e in json.load(open(sys.argv[1]))["artifact_catalog"]["entries"]]' \
    "$LOCAL/model-package.json" | while read -r path; do
    mkdir -p "$LOCAL/$(dirname "$path")"
    curl -fsSL --retry 10 -C - -o "$LOCAL/$path" "$base/$path" || { echo "fetch failed: $path" >&2; exit 1; }
  done || return 1
  SNAP="$LOCAL" python3 - <<'PY' || { echo "staged package failed verification" >&2; return 1; }
import hashlib, json, os, sys
snap = os.environ["SNAP"]
bad = []
for entry in json.load(open(os.path.join(snap, "model-package.json")))["artifact_catalog"]["entries"]:
    h = hashlib.sha256()
    try:
        with open(os.path.join(snap, entry["path"]), "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    except OSError:
        bad.append(entry["path"]); continue
    if h.hexdigest() != entry["sha256"]:
        bad.append(entry["path"])
if bad:
    print("bad files:", bad); sys.exit(1)
PY
  # The resolver reads a Hugging Face cache: snapshot entries are links to blobs/<sha256>.
  SNAP="$LOCAL" python3 - <<'PY' || return 1
import json, os
snap = os.environ["SNAP"]
blobs = os.path.join(os.path.dirname(os.path.dirname(snap)), "blobs")
for entry in json.load(open(os.path.join(snap, "model-package.json")))["artifact_catalog"]["entries"]:
    path = os.path.join(snap, entry["path"])
    if os.path.islink(path):
        continue
    blob = os.path.join(blobs, entry["sha256"])
    os.replace(path, blob)
    os.symlink(os.path.relpath(blob, os.path.dirname(path)), path)
PY
  touch "$LOCAL/.verified"
}

if [ -n "${SPLIT_GGUF:-}" ]; then
  # A whole GGUF each member already holds: check it, and skip package staging.
  MODEL_FILE="$HOME/$SPLIT_GGUF"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${SPLIT_GGUF_SHA256:?}  $MODEL_FILE" | sha256sum -c --quiet || { echo "$MODEL_FILE failed verification" >&2; exit 1; }
  fi
  echo "serving the verified GGUF $MODEL_FILE"
elif [ ! -f "$LOCAL/.verified" ]; then
  T0=$(date +%s)
  if [ "$ROLE" = joiner ] && [ -n "$LEAD" ]; then
    # The coordinator holds a verified copy and serves it on the LAN; one internet download
    # per package instead of one per machine.
    for _ in $(seq 1 360); do
      curl -sf -m 5 -o /dev/null "http://$LEAD:$PKG_PORT/.verified" && break
      sleep 10
    done
    echo "copying $SPLIT_REPO@${SPLIT_REV:0:9} from the coordinator at $LEAD"
    fetch_package "http://$LEAD:$PKG_PORT" || exit 1
  else
    echo "downloading $SPLIT_REPO@${SPLIT_REV:0:9} from Hugging Face"
    fetch_package "https://huggingface.co/$SPLIT_REPO/resolve/$SPLIT_REV" || exit 1
  fi
  echo "package staged and verified in $(( $(date +%s) - T0 ))s"
fi
if [ "$ROLE" = coordinator ] && [ -z "${SPLIT_GGUF:-}" ]; then
  python3 -m http.server "$PKG_PORT" --directory "$LOCAL" >"$ART/package-server.log" 2>&1 &
  PKG_SERVER=$!
fi

if [ "${SPLIT_ARM:?}" = dflash ]; then
  DRAFT="$HOME/.cache/dflash-bench-models/${SPLIT_DRAFT_URL##*/}"
  mkdir -p "$(dirname "$DRAFT")"
  draft_ok() { [ "$( (sha256sum "$DRAFT" 2>/dev/null || shasum -a 256 "$DRAFT") | cut -d' ' -f1)" = "${SPLIT_DRAFT_SHA256:?}" ]; }
  for _ in 1 2 3; do
    draft_ok && break
    rm -f "$DRAFT"; curl -fsSL --retry 10 -o "$DRAFT" "${SPLIT_DRAFT_URL:?}" || true
  done
  draft_ok || { echo "draft $DRAFT failed verification" >&2; exit 1; }
  echo "draft verified: $DRAFT"
  DFLASH_TOML="$ART/split-dflash.toml"
  printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\n' "$DRAFT" > "$DFLASH_TOML"
  [ -n "${SPLIT_DFLASH_CAP:-}" ] && printf 'draft_max_tokens = %s\n' "$SPLIT_DFLASH_CAP" >> "$DFLASH_TOML"
fi

# One config file carries both the codec and the dflash arm's draft, since a serve takes one.
ARM_CONFIG="$ART/split-arm.toml"
: > "$ARM_CONFIG"
if [ -n "${SPLIT_CODEC:-}" ]; then
  printf '[defaults.skippy]\nactivation_codec = "%s"\n' "$SPLIT_CODEC" >> "$ARM_CONFIG"
fi
[ -n "${DFLASH_TOML:-}" ] && cat "$DFLASH_TOML" >> "$ARM_CONFIG"

case "${SPLIT_ARM:?}" in
  off) ARM_FLAGS="--speculative-strategy disabled" ;;
  default) ARM_FLAGS="" ;;
  ngram) ARM_FLAGS="--speculative-strategy ngram-suffix" ;;
  # Every member gets the config: the mesh, not jy, elects which node plans the split.
  dflash) ARM_FLAGS="" ;;
  *) echo "unknown arm $SPLIT_ARM" >&2; exit 1 ;;
esac
[ -s "$ARM_CONFIG" ] && ARM_FLAGS="$ARM_FLAGS --config $ARM_CONFIG"
export MESH_LLM_BENCH_DOWNSTREAM_WIRE_DELAY_MS="${SPLIT_WIRE_DELAY_MS:-0}"
[ -n "${SPLIT_WIRE_MBPS:-}" ] && export MESH_LLM_BENCH_DOWNSTREAM_WIRE_MBPS="$SPLIT_WIRE_MBPS"
[ "${SPLIT_DFLASH_FIXED:-0}" = 1 ] && export SKIPPY_DFLASH_ADAPTIVE=0
# Capacity-only placement packs a model that fits one node onto that node (63 of 64
# layers), which measures a split that barely exists. An explicit cut levels the
# stages; --auto-balance is what honours it, and its re-cut loop has nothing to fix on
# identical machines. The layout is logged before and after measuring to show if it did.
if [ -n "${SPLIT_CUT:-}" ]; then
  export MESH_LLM_AUTO_BALANCE_INITIAL_CUT="$SPLIT_CUT"
  BALANCE_FLAGS="--auto-balance"
fi
MODEL="${MODEL_FILE:-$SPLIT_REPO@$SPLIT_REV}"
SPLIT_FLAGS="--split ${BALANCE_FLAGS:-}"
[ "${SPLIT_ALLOW_UNCERTIFIED:-0}" = 1 ] && SPLIT_FLAGS="$SPLIT_FLAGS --allow-uncertified-split"
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
  trap 'grep -E "\"level\":\"error\"" "$LOG" | tail -5 >&2; tail -60 "$LOG" >&2; kill "$SERVE" 2>/dev/null || true' EXIT
  # Serve until the coordinator goes away: it owns the measurement and the end of the run.
  MISSES=0
  while kill -0 "$SERVE" 2>/dev/null; do
    # The mesh may elect this joiner to plan the split; if its launch fails the
    # coordinator only stands by, so the failure has to end the formation here.
    grep -qE 'model launch failed|Failed to start model' "$LOG" && { echo "the model failed to launch on this joiner" >&2; exit 1; }
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
# Only kill what was started: `kill 0` would signal this whole process group, the
# script included, and turn a finished run into a failed job.
trap 'rc=$?; [ $rc -ne 0 ] && tail -120 "$LOG" >&2; kill "$SERVE" ${PKG_SERVER:+"$PKG_SERVER"} 2>/dev/null || true' EXIT

MODEL_ID=""
# Joiners copy the package from this node first; a first copy of a ~190 GB package to
# several joiners at once takes over an hour, so wait up to three hours for the stages.
for i in $(seq 1 1080); do
  MODEL_ID=$(curl -sf -m 5 "http://127.0.0.1:$CONSOLE/api/status" 2>/dev/null | STAGES="$SPLIT_STAGES" python3 -c '
import json, os, sys
d = json.load(sys.stdin); st = (d.get("runtime") or {}).get("stages") or []
name = d.get("model_name") or (st[0].get("model_id") if st else "") or ""
# On identical machines the mesh elects the node with the smallest ID to run the split, which
# may be a joiner; this node then routes requests to it. Ready means every stage is ready.
ready = len(st) >= int(os.environ["STAGES"]) and all(s.get("state") == "ready" for s in st)
print(name if ready and (d.get("llama_ready") or d.get("serving_models")) else "")' 2>/dev/null || true)
  [ -n "$MODEL_ID" ] && break
  [ $((i % 6)) -eq 0 ] && echo "waiting for $SPLIT_STAGES stages — $((i * 10))s"
  kill -0 "$SERVE" 2>/dev/null || { echo "coordinator died while forming" >&2; exit 1; }
  # A failed launch leaves the node up in standby, so the wait would only time out.
  grep -qE 'model launch failed|Failed to start model' "$LOG" && { echo "the model failed to launch" >&2; exit 1; }
  sleep 10
done
[ -n "$MODEL_ID" ] || { echo "the split never reached $SPLIT_STAGES stages in 3 hours" >&2; exit 1; }
SPANS=$(curl -sf -m 5 "http://127.0.0.1:$CONSOLE/api/status" | python3 -c '
import json, sys
st = (json.load(sys.stdin).get("runtime") or {}).get("stages") or []
print(" ".join("%s-%s" % (s.get("layer_start"), s.get("layer_end")) for s in sorted(st, key=lambda s: s["stage_index"])))' 2>/dev/null || true)
echo "split serving $MODEL_ID across $SPLIT_STAGES stages: layers $SPANS"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader | sed "s/^/gpu: $SPLIT_ARM /" || true

if [ "${SPLIT_PREFIX:-0}" = 1 ]; then
  PORT="$API" MODEL_ID="$MODEL_ID" ART="$ART" LIMIT="${SPLIT_LIMIT:-8}" \
    PROMPTS="$PWD/skippy/evals/agent-transcripts/agent-transcripts.jsonl" python3 "$HERE/prefix.py" \
    || echo "PREFIX_BENCH {\"error\": \"measurement failed\"}"
else
ARM="$SPLIT_ARM${SPLIT_DFLASH_CAP:+-k$SPLIT_DFLASH_CAP}" LIMIT="${SPLIT_LIMIT:-0}" PORT="$API" MODEL_ID="$MODEL_ID" TEMP=0 SPLIT=report \
  PROMPTS="$PWD/skippy/evals/spec-prompts/spec-prompts.jsonl" python3 "$HERE/measure.py" \
  || echo "DFLASH_BENCH {\"arm\": \"$SPLIT_ARM\", \"error\": \"measurement failed\"}"
fi
# Errors from this node's serve log, so a failed arm explains itself in the job output.
grep -E '"level":"error"' "$LOG" | tail -5 | cut -c1-600 | sed 's/^/serve error: /' || true
SPANS_AFTER=$(curl -sf -m 5 "http://127.0.0.1:$CONSOLE/api/status" | python3 -c '
import json, sys
st = (json.load(sys.stdin).get("runtime") or {}).get("stages") or []
print(" ".join("%s-%s" % (s.get("layer_start"), s.get("layer_end")) for s in sorted(st, key=lambda s: s["stage_index"])))' 2>/dev/null || true)
echo "SPLIT_LAYOUT {\"stages\": $SPLIT_STAGES, \"spans\": \"$SPANS\", \"spans_after\": \"$SPANS_AFTER\", \"wire_delay_ms\": ${SPLIT_WIRE_DELAY_MS:-0}}"
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
