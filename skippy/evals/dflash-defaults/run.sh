#!/usr/bin/env bash
# Runs one model's arms on one lab machine, from the jy workspace root after build.sh.
#
# Arms, in order: speculation off, the shipped default (no speculation config), then
# each fixed DFlash block in BENCH_FIXED_BLOCKS with adaptive sizing off, all greedy;
# then off and the default again at temperature 0.7. When BENCH_AGENT is 1, off and the
# default run once more over real coding-agent transcripts at 64K context (agent_*).
#
# Inputs (environment, set by bench.py):
#   BENCH_TARGET_URL / _SHA256 / _BYTES / _FILE   target GGUF, pinned to a revision
#   BENCH_DEFAULT_DRAFT_FILE   file name the default path fetches, or empty when the
#                              default serves this model without speculation
#   BENCH_DRAFT_URL / _SHA256 / _BYTES / _FILE   draft for the fixed arms, or empty
#   BENCH_FIXED_BLOCKS         space-separated draft lengths, e.g. "5 7"
#   BENCH_SPLIT                spec-prompts split (default: report)
#   BENCH_AGENT                1 to add the agent-transcript arms
set -euo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/dflash-defaults"
ART="${JIANYANG_ARTIFACT_DIR:?}"
SPLIT="${BENCH_SPLIT:-report}"
PROMPTS="$PWD/skippy/evals/spec-prompts/spec-prompts.jsonl"
CTX=8192

# The jy model store can hand back a damaged copy of a large file, so every job
# downloads its own weights and checks them, and the default draft lands in a
# private Hugging Face cache rather than a shared mount.
export HF_HOME="$HOME/.cache/dflash-bench-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE" "$HOME/.cache/dflash-bench-models"

TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
BIN=""
for candidate in "$TARGET_DIR/release/mesh-llm" target/release/mesh-llm; do
  [ -x "$candidate" ] && { BIN="$candidate"; break; }
done
[ -n "$BIN" ] || { echo "no release host build" >&2; exit 1; }
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"
[ -d "$MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" ] || { echo "no packaged runtime" >&2; exit 1; }

sha_ok() {
  if command -v sha256sum >/dev/null 2>&1; then echo "$1  $2" | sha256sum -c --quiet 2>/dev/null
  else echo "$1  $2" | shasum -a 256 -c --quiet 2>/dev/null; fi
}
size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
fetch() { # url sha256 bytes path
  local url="$1" sha="$2" bytes="$3" path="$4"
  if ! sha_ok "$sha" "$path"; then
    for _ in 1 2 3 4 5 6; do
      [ "$(size_of "$path")" = "$bytes" ] && break
      curl -fL --retry 10 -C - -sS -o "$path" "$url" || true
    done
  fi
  sha_ok "$sha" "$path" || { echo "checksum mismatch for $path" >&2; rm -f "$path"; exit 1; }
}
gpu_state() { # thermal throttling skews a baseline, so every arm records the clock it ran at
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader | sed "s/^/gpu: $1 /"
  elif command -v pmset >/dev/null 2>&1; then
    pmset -g therm 2>/dev/null | tr -s ' \n' ' ' | sed "s/^/gpu: $1 /"; echo
  fi
}

T0=$(date +%s)
TARGET="$HOME/.cache/dflash-bench-models/${BENCH_TARGET_FILE:?}"
fetch "${BENCH_TARGET_URL:?}" "${BENCH_TARGET_SHA256:?}" "${BENCH_TARGET_BYTES:?}" "$TARGET"
echo "target verified after $(( $(date +%s) - T0 ))s: $TARGET"
DRAFT=""
if [ -n "${BENCH_DRAFT_URL:-}" ]; then
  DRAFT="$HOME/.cache/dflash-bench-models/${BENCH_DRAFT_FILE:?}"
  fetch "$BENCH_DRAFT_URL" "${BENCH_DRAFT_SHA256:?}" "${BENCH_DRAFT_BYTES:?}" "$DRAFT"
  echo "fixed-arm draft verified: $DRAFT"
fi

serve() { # arm port extra-args... ; starts the server in the background
  local arm="$1" port="$2"; shift 2
  env ${ARM_ENV:-} "$BIN" --log-format json --parallel 1 --port "$port" --console $(( port + 600 )) --headless "$@" \
    serve --model "$TARGET" --ctx-size "$CTX" >"$ART/serve-$arm.log" 2>&1 &
  SERVER=$!
}

run_arm() { # arm extra-args...
  local arm="$1"; shift
  local port=$(( 9400 + $(printf '%s%s' "${JIANYANG_RUN_ID:-run}" "$arm" | cksum | cut -d' ' -f1) % 500 ))
  gpu_state "$arm"
  echo "=== arm $arm: $* (port $port)"
  serve "$arm" "$port" "$@"
  local model_id="" t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    model_id=$(curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0].get("id","") if d else "")' 2>/dev/null || true)
    [ -n "$model_id" ] && break
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 2
  done
  if [ -z "$model_id" ]; then
    echo "arm $arm: server published no model"; tail -40 "$ART/serve-$arm.log" || true
    echo "DFLASH_BENCH {\"arm\": \"$arm\", \"error\": \"server published no model\"}"
  else
    echo "arm $arm ready after $(( $(date +%s) - t0 ))s"
    ARM="$arm" PORT="$port" MODEL_ID="$model_id" TEMP="${MEASURE_TEMP:-0}" SPLIT="$SPLIT" PROMPTS="$PROMPTS" \
      python3 "$HERE/measure.py" || echo "DFLASH_BENCH {\"arm\": \"$arm\", \"error\": \"measurement failed\"}"
  fi
  kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true
  sleep 5
}

# The default path fetches its draft in the background and only uses it from the next
# load, so start the server once and wait for that file before measuring the default.
prime_default_draft() {
  [ -n "${BENCH_DEFAULT_DRAFT_FILE:-}" ] || return 0
  local found=""
  serve prime 9399
  local t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
    found=$(find -L "$HF_HUB_CACHE" -path '*snapshots*' -name "$BENCH_DEFAULT_DRAFT_FILE" 2>/dev/null | head -1)
    [ -n "$found" ] && break
    sleep 10
  done
  kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true
  sleep 5
  [ -n "$found" ] || { echo "default draft $BENCH_DEFAULT_DRAFT_FILE never arrived" >&2; exit 1; }
  echo "default draft cached after $(( $(date +%s) - t0 ))s: $found"
}

fixed_config() { # draft tokens -> config path
  printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\ndraft_max_tokens = %s\n' \
    "$DRAFT" "$1" > "$ART/dflash-$1.toml"
  echo "$ART/dflash-$1.toml"
}

prime_default_draft
ARM_ENV= run_arm off --no-draft --speculative-strategy disabled
ARM_ENV= run_arm default
if [ -n "$DRAFT" ]; then
  for k in ${BENCH_FIXED_BLOCKS:-}; do
    ARM_ENV="SKIPPY_DFLASH_ADAPTIVE=0" run_arm "d$k" --config "$(fixed_config "$k")"
  done
fi
MEASURE_TEMP=0.7 ARM_ENV= run_arm off_t07 --no-draft --speculative-strategy disabled
MEASURE_TEMP=0.7 ARM_ENV= run_arm default_t07
if [ "${BENCH_AGENT:-0}" = 1 ]; then
  # Transcripts reach 48K tokens of history; the server must hold them plus the reply.
  PROMPTS="$PWD/skippy/evals/agent-transcripts/agent-transcripts.jsonl" SPLIT=all CTX=65536
  ARM_ENV= run_arm agent_off --no-draft --speculative-strategy disabled
  ARM_ENV= run_arm agent_default
fi

echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
