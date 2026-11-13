#!/usr/bin/env bash
# Headroom for per-workload drafters: one target, its shipped DFlash draft and n-gram
# suffix speculation, measured per workload on one machine. Run from the jy workspace
# root after dflash-defaults/build.sh.
#
# Arms, each a fresh server, greedy, thinking off:
#   off      no speculation; its replies are the reference and drafter training data
#   dflash   the pinned draft with adaptive blocks up to BENCH_DFLASH_CAP (the shipped default)
#   ngram    n-gram suffix speculation, no draft model
#
# Inputs (environment, set by submit.py):
#   BENCH_TARGET_URL / _SHA256 / _BYTES   target GGUF, pinned to a revision
#   BENCH_DRAFT_URL / _SHA256 / _BYTES    DFlash draft, pinned to a revision
#   BENCH_DFLASH_CAP                      draft tokens per block (default 7)
#   BENCH_DOMAINS                         comma-separated workloads.jsonl domains, and/or
#                                         "agent" for the agent transcripts at 64K context
#   BENCH_ARMS                            arms to run (default "off dflash ngram"); an arm
#                                         "spec_<name>" drafts with a fine-tuned draft
#   BENCH_SPECIALISTS                     "name=/abs/path.gguf ..." drafts already on this
#                                         machine (fine-tuned drafts are not published)
set -euo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/drafter-swap"
ART="${JIANYANG_ARTIFACT_DIR:?}"
CAP="${BENCH_DFLASH_CAP:-7}"
ARMS="${BENCH_ARMS:-off dflash ngram}"

export HF_HOME="$HOME/.cache/dflash-bench-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
MODELS="$HOME/.cache/dflash-bench-models"
mkdir -p "$HF_HUB_CACHE" "$MODELS"

TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
BIN=""
for candidate in "$TARGET_DIR/release/mesh-llm" target/release/mesh-llm; do
  [ -x "$candidate" ] && { BIN="$candidate"; break; }
done
[ -n "$BIN" ] || { echo "no release host build" >&2; exit 1; }
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"
[ -d "$MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" ] || { echo "no packaged runtime" >&2; exit 1; }

sha_ok() { # sha256 path
  [ -f "$2" ] || return 1
  local got
  got=$( (sha256sum "$2" 2>/dev/null || shasum -a 256 "$2") | cut -d' ' -f1)
  [ "$got" = "$1" ]
}
size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
fetch() { # url sha256 bytes path
  local url="$1" sha="$2" bytes="$3" path="$4"
  for attempt in 1 2 3; do
    sha_ok "$sha" "$path" && return 0
    [ "$attempt" -gt 1 ] && { echo "$path failed its checksum; downloading it again" >&2; rm -f "$path"; }
    [ "$(size_of "$path")" = "$bytes" ] && rm -f "$path"
    for _ in 1 2 3 4 5 6; do
      [ "$(size_of "$path")" = "$bytes" ] && break
      curl -fL --retry 10 -C - -sS -o "$path" "$url" || true
    done
  done
  sha_ok "$sha" "$path" || { echo "checksum mismatch for $path" >&2; rm -f "$path"; exit 1; }
}
gpu_state() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader | sed "s/^/gpu: $1 /"
  elif command -v pmset >/dev/null 2>&1; then
    pmset -g therm 2>/dev/null | tr -s ' \n' ' ' | sed "s/^/gpu: $1 /"; echo
  fi
}

TARGET="$MODELS/${BENCH_TARGET_URL##*/}"
DRAFT="$MODELS/${BENCH_DRAFT_URL##*/}"
fetch "${BENCH_TARGET_URL:?}" "${BENCH_TARGET_SHA256:?}" "${BENCH_TARGET_BYTES:?}" "$TARGET"
fetch "${BENCH_DRAFT_URL:?}" "${BENCH_DRAFT_SHA256:?}" "${BENCH_DRAFT_BYTES:?}" "$DRAFT"
echo "weights verified: $TARGET $DRAFT"

printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\ndraft_max_tokens = %s\n' \
  "$DRAFT" "$CAP" > "$ART/dflash.toml"

for pair in ${BENCH_SPECIALISTS:-}; do
  name="${pair%%=*}" path="${pair#*=}"
  [ -f "$path" ] || { echo "specialist draft $name missing: $path" >&2; exit 1; }
  echo "specialist $name: $path ($( (sha256sum "$path" 2>/dev/null || shasum -a 256 "$path") | cut -c1-16))"
  printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\ndraft_max_tokens = %s\n' \
    "$path" "$CAP" > "$ART/spec_$name.toml"
done

arm_args() {
  case "$1" in
    off) echo "--no-draft --speculative-strategy disabled" ;;
    dflash) echo "--config $ART/dflash.toml" ;;
    spec_*) echo "--config $ART/$1.toml" ;;
    # The suffix proposer requires 3 <= min <= max <= 64; 5/32 is SUFFIX_NGRAM_PROPOSER.md's example.
    ngram) echo "--no-draft --speculative-strategy ngram-suffix --speculative-ngram-min 5 --speculative-ngram-max 32" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

run_arm() { # label arm prompts split domains ctx
  local label="$1" arm="$2" prompts="$3" split="$4" domains="$5" ctx="$6"
  local port=$(( 9400 + $(printf '%s%s' "${JIANYANG_RUN_ID:-run}" "$label" | cksum | cut -d' ' -f1) % 500 ))
  gpu_state "$label"
  echo "=== $label: $arm on ${domains:-$split} (port $port, ctx $ctx)"
  # shellcheck disable=SC2046
  "$BIN" --log-format json --parallel 1 --port "$port" --console $(( port + 600 )) --headless $(arm_args "$arm") \
    serve --model "$TARGET" --ctx-size "$ctx" >"$ART/serve-$label.log" 2>&1 &
  local server=$! model_id="" t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    model_id=$(curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0].get("id","") if d else "")' 2>/dev/null || true)
    [ -n "$model_id" ] && break
    kill -0 "$server" 2>/dev/null || break
    sleep 2
  done
  if [ -z "$model_id" ]; then
    echo "$label: server published no model"; tail -40 "$ART/serve-$label.log" || true
    echo "DFLASH_BENCH {\"arm\": \"$label\", \"error\": \"server published no model\"}"
  else
    ARM="$label" PORT="$port" MODEL_ID="$model_id" TEMP=0 SPLIT="$split" PROMPTS="$prompts" DOMAINS="$domains" \
      python3 "$HERE/measure.py" || echo "DFLASH_BENCH {\"arm\": \"$label\", \"error\": \"measurement failed\"}"
  fi
  kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
  sleep 5
}

WORKLOAD_DOMAINS=$(echo "${BENCH_DOMAINS:?}" | tr ',' '\n' | grep -vx agent | paste -sd, - || true)
for arm in $ARMS; do
  if [ -n "$WORKLOAD_DOMAINS" ]; then
    run_arm "$arm" "$arm" "$HERE/workloads.jsonl" eval "$WORKLOAD_DOMAINS" 16384
  fi
  if echo ",$BENCH_DOMAINS," | grep -q ',agent,'; then
    run_arm "agent_$arm" "$arm" "$PWD/skippy/evals/agent-transcripts/agent-transcripts.jsonl" all "" 65536
  fi
done

echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
