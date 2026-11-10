#!/usr/bin/env bash
# End-to-end check of a runtime setting through mesh-llm serving on this machine:
# prefill time of uncached long prompts and plain decode speed, one fresh server per arm.
# Speculation is off so prefill and target decode are measured on their own.
#
#   SB_MODEL  id from models.tsv
#   SB_ARMS   space-separated name=toml-body pairs; '\n' in a body is a newline.
#             e.g. 'base= ub2048=[defaults.model_fit]\nbatch=2048\nubatch=2048'
#   SB_CTX    context size (default 20480)
set -uo pipefail
cd mesh-llm
HERE="$PWD/skippy/evals/kernel-bench"
ART="${JIANYANG_ARTIFACT_DIR:-$PWD/kbench-out}"; mkdir -p "$ART"
TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
BIN=""
for c in "$TARGET_DIR/release/mesh-llm" target/release/mesh-llm; do [ -x "$c" ] && { BIN="$c"; break; }; done
[ -n "$BIN" ] || { echo "no release host build" >&2; exit 1; }
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"
export HF_HOME="$HOME/.cache/dflash-bench-hf"
MODELS_DIR="$HOME/.cache/dflash-bench-models"; mkdir -p "$MODELS_DIR"
sha() { (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
line=$(awk -v id="${SB_MODEL:?}" '$1==id' "$HERE/models.tsv"); url=$(echo "$line" | awk '{print $2}'); want=$(echo "$line" | awk '{print $3}')
M="$MODELS_DIR/${url##*/}"
if ! { [ -f "$M.sha256-ok" ] && [ "$(cat "$M.sha256-ok")" = "$want" ]; }; then
  [ -f "$M" ] && [ "$(sha "$M")" = "$want" ] || curl -fL --retry 5 -C - -sS -o "$M" "$url"
  [ "$(sha "$M")" = "$want" ] || { echo "sha mismatch $M" >&2; exit 1; }
  echo "$want" > "$M.sha256-ok"
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader
for pair in ${SB_ARMS:-base=}; do
  arm=${pair%%=*}; body=${pair#*=}
  cfg="$ART/arm-$arm.toml"; printf '%b\n' "$body" > "$cfg"
  port=$(( 9400 + $(printf '%s%s' "${JIANYANG_RUN_ID:-run}" "$arm" | cksum | cut -d' ' -f1) % 500 ))
  echo "=== arm $arm (port $port)"; cat "$cfg"
  LAUNCH=()
  if [ "${SB_NSYS:-0}" = 1 ] && command -v nsys >/dev/null 2>&1; then
    LAUNCH=(nsys launch --session-new="sb-$arm" -t cuda,nvtx,osrt --cuda-graph-trace=node)
  fi
  env RUST_LOG="${SB_RUST_LOG:-}" "${LAUNCH[@]}" "$BIN" --log-format json --parallel 1 --port "$port" --console $(( port + 600 )) --headless \
    --no-draft --speculative-strategy disabled --config "$cfg" \
    serve --model "$M" --ctx-size "${SB_CTX:-32768}" >"$ART/serve-$arm.log" 2>&1 &
  SERVER=$!
  model_id=""; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    model_id=$(curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0]["id"] if d else "")' 2>/dev/null || true)
    [ -n "$model_id" ] && break; kill -0 "$SERVER" 2>/dev/null || break; sleep 2
  done
  if [ -z "$model_id" ]; then echo "SERVE_BENCH {\"arm\":\"$arm\",\"error\":\"no model\"}"; tail -30 "$ART/serve-$arm.log"
  else
    if [ "${#LAUNCH[@]}" -gt 0 ]; then
      ARM="$arm" PORT="$port" MODEL_ID="$model_id" SB_PHASES=warm python3 "$HERE/serve_measure.py"
      nsys start --session="sb-$arm" -o "$HOME/.cache/kbench/nsys-serve-$arm" -f true
      ARM="$arm" PORT="$port" MODEL_ID="$model_id" SB_PHASES=long python3 "$HERE/serve_measure.py"
      nsys stop --session="sb-$arm"
      for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum osrt_sum; do
        echo "=== nsys serve $arm $rep"
        nsys stats -q --report $rep --format csv "$HOME/.cache/kbench/nsys-serve-$arm.nsys-rep" 2>/dev/null | head -25 | cut -c1-220
      done
    else
      ARM="$arm" PORT="$port" MODEL_ID="$model_id" python3 "$HERE/serve_measure.py"
    fi
  fi
  grep -oE '"n_ubatch[^,]*|n_ubatch=[0-9]+|n_batch=[0-9]+' "$ART/serve-$arm.log" | sort -u | head -4
  kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; sleep 5
done
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
echo "=== serve done"
