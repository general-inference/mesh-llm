#!/usr/bin/env bash
# Long-context ladder: one target at a 1M-token window (YaRN written into a copy of its
# GGUF), with and without its DFlash draft, from 32K up to the full window.
# Run from the jy workspace root after dflash-defaults/build.sh.
#
# Inputs (environment, set by submit.py):
#   BENCH_TARGET_URL / _SHA256 / _BYTES   target GGUF, pinned to a revision
#   BENCH_DRAFT_URL / _SHA256 / _BYTES    DFlash draft, pinned to a revision
#   BENCH_YARN_FACTOR / _ORIGINAL         rope scaling written into the target (4 x 262144)
#   BENCH_CTX                             context window (default 1048576)
#   BENCH_CACHE_TYPE                      KV cache type for target and draft (default f16)
#   BENCH_DEPTHS                          prompt lengths in tokens; "max" fills the window
#   BENCH_ARMS                            default "off dflash"
#   BENCH_DFLASH_CAP                      draft tokens per block (default 7)
#   BENCH_MAX_TOKENS                      tokens per reply (default 256)
set -euo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/long-context"
ART="${JIANYANG_ARTIFACT_DIR:?}"
CTX="${BENCH_CTX:-1048576}"
CACHE="${BENCH_CACHE_TYPE:-f16}"
ARMS="${BENCH_ARMS:-off dflash}"
MODELS="$HOME/.cache/dflash-bench-models"
mkdir -p "$MODELS"

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
  [ "$( (sha256sum "$2" 2>/dev/null || shasum -a 256 "$2") | cut -d' ' -f1)" = "$1" ]
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

SOURCE="$MODELS/${BENCH_TARGET_URL##*/}"
DRAFT="$MODELS/${BENCH_DRAFT_URL##*/}"
fetch "${BENCH_TARGET_URL:?}" "${BENCH_TARGET_SHA256:?}" "${BENCH_TARGET_BYTES:?}" "$SOURCE"
fetch "${BENCH_DRAFT_URL:?}" "${BENCH_DRAFT_SHA256:?}" "${BENCH_DRAFT_BYTES:?}" "$DRAFT"
echo "weights verified: $SOURCE $DRAFT"

FACTOR="${BENCH_YARN_FACTOR:-4}" ORIGINAL="${BENCH_YARN_ORIGINAL:-262144}"
# A factor of 0 serves the GGUF as published (native window, no YaRN), as a control.
TARGET="$SOURCE"
if [ "$FACTOR" != 0 ]; then
  TARGET="$MODELS/yarn${FACTOR}-${BENCH_TARGET_SHA256:0:12}-${SOURCE##*/}"
  [ -f "$TARGET" ] || python3 -I "$HERE/yarn_gguf.py" "$SOURCE" "$TARGET" --factor "$FACTOR" --original "$ORIGINAL"
fi
echo "target: $TARGET"

model_fit() {
  printf '[defaults.model_fit]\ncache_type_k = "%s"\ncache_type_v = "%s"\n' "$CACHE" "$CACHE"
  # BENCH_PROMPT_CACHE=false keeps finished prompts out of the host prompt cache.
  [ -n "${BENCH_PROMPT_CACHE:-}" ] && printf 'prompt_cache = %s\n' "$BENCH_PROMPT_CACHE"
  true
}
model_fit > "$ART/off.toml"
{ model_fit; printf '[defaults.speculative]\nstrategy = "dflash"\ndraft_model_path = "%s"\ndraft_max_tokens = %s\ndraft_cache_type_k = "%s"\ndraft_cache_type_v = "%s"\n' \
  "$DRAFT" "${BENCH_DFLASH_CAP:-7}" "$CACHE" "$CACHE"; } > "$ART/dflash.toml"

for arm in $ARMS; do
  port=$(( 9400 + $(printf '%s%s' "${JIANYANG_RUN_ID:-run}" "$arm" | cksum | cut -d' ' -f1) % 500 ))
  echo "=== $arm (port $port, ctx $CTX, cache $CACHE)"
  free -g 2>/dev/null | sed "s/^/mem before $arm: /" || true
  extra=""
  [ "$arm" = off ] && extra="--no-draft --speculative-strategy disabled"
  # shellcheck disable=SC2086
  "$BIN" --log-format json --parallel 1 --port "$port" --console $(( port + 600 )) --headless $extra \
    --config "$ART/$arm.toml" serve --model "$TARGET" --ctx-size "$CTX" >"$ART/serve-$arm.log" 2>&1 &
  server=$! model_id="" t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do
    model_id=$(curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0].get("id","") if d else "")' 2>/dev/null || true)
    [ -n "$model_id" ] && break
    kill -0 "$server" 2>/dev/null || break
    sleep 5
  done
  echo "$arm: load took $(( $(date +%s) - t0 ))s"
  grep -h -o -E '(n_ctx[a-z_]*|KV buffer size|RS buffer size|compute buffer size|freq_scale|yarn|rope scaling|n_ctx_train)[^"]{0,90}' \
    "$ART/serve-$arm.log" | sort | uniq -c | head -40 || true
  free -g 2>/dev/null | sed "s/^/mem loaded $arm: /" || true
  if [ -z "$model_id" ]; then
    echo "$arm: server published no model"; tail -60 "$ART/serve-$arm.log" || true
  else
    ARM="$arm" PORT="$port" MODEL_ID="$model_id" CTX="$CTX" REPO="$PWD" \
      DEPTHS="${BENCH_DEPTHS:-32768,131072,262144,524288,max}" MAX_TOKENS="${BENCH_MAX_TOKENS:-256}" \
      python3 "$HERE/ladder.py" || echo "$arm: ladder failed"
  fi
  kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
  sleep 10
done

echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
