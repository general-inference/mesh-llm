#!/usr/bin/env bash
# Generates drafter training data: the target answers train-prompts.jsonl for
# BENCH_DOMAINS, served with its shipped DFlash2 draft and BENCH_PARALLEL slots.
# Same inputs as run.sh. Run from the jy workspace root after dflash-defaults/build.sh.
set -euo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/drafter-swap"
ART="${JIANYANG_ARTIFACT_DIR:?}"
PARALLEL="${BENCH_PARALLEL:-4}"
MODELS="$HOME/.cache/dflash-bench-models"
export HF_HOME="$HOME/.cache/dflash-bench-hf"
mkdir -p "$MODELS"

TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
BIN="$TARGET_DIR/release/mesh-llm"
[ -x "$BIN" ] || { echo "no release host build" >&2; exit 1; }
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"

sha_ok() { [ -f "$2" ] && [ "$( (sha256sum "$2" 2>/dev/null || shasum -a 256 "$2") | cut -d' ' -f1)" = "$1" ]; }
fetch() { # url sha256 path
  for _ in 1 2 3; do
    sha_ok "$2" "$3" && return 0
    rm -f "$3"
    curl -fL --retry 10 -sS -o "$3" "$1" || true
  done
  sha_ok "$2" "$3" || { echo "checksum mismatch for $3" >&2; exit 1; }
}
TARGET="$MODELS/${BENCH_TARGET_URL##*/}"
DRAFT="$MODELS/${BENCH_DRAFT_URL##*/}"
fetch "${BENCH_TARGET_URL:?}" "${BENCH_TARGET_SHA256:?}" "$TARGET"
fetch "${BENCH_DRAFT_URL:?}" "${BENCH_DRAFT_SHA256:?}" "$DRAFT"
printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\ndraft_max_tokens = 7\n' "$DRAFT" > "$ART/dflash.toml"

PORT=9391
"$BIN" --log-format json --parallel "$PARALLEL" --port "$PORT" --console $(( PORT + 600 )) --headless \
  --config "$ART/dflash.toml" serve --model "$TARGET" --ctx-size $(( (4096 + ${BENCH_MAX_TOKENS:-1024}) * PARALLEL )) \
  >"$ART/serve-datagen.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
MODEL_ID=""
for _ in $(seq 450); do
  MODEL_ID=$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0].get("id","") if d else "")' 2>/dev/null || true)
  [ -n "$MODEL_ID" ] && break
  kill -0 "$SERVER" 2>/dev/null || { tail -40 "$ART/serve-datagen.log"; exit 1; }
  sleep 2
done
[ -n "$MODEL_ID" ] || { echo "server published no model"; tail -40 "$ART/serve-datagen.log"; exit 1; }

PORT="$PORT" MODEL_ID="$MODEL_ID" PROMPTS="$HERE/train-prompts.jsonl" DOMAINS="${BENCH_DOMAINS:?}" \
  CONCURRENCY="$PARALLEL" LIMIT="${BENCH_LIMIT:-0}" SHARD="${BENCH_SHARD:-}" \
  MAX_TOKENS="${BENCH_MAX_TOKENS:-1024}" python3 "$HERE/datagen.py"
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
