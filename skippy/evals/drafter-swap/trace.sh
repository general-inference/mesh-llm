#!/usr/bin/env bash
# Per-token draft traces: which generated tokens came from accepted drafts, for the
# shipped draft and a fine-tuned one, on BENCH_DOMAINS of workloads.jsonl. Builds
# llama-spec-bench like swap_probe.sh and runs it in chat format, greedy.
#   BENCH_SPECIALISTS  "name=/abs/path.gguf" (one), on this machine
#   BENCH_DOMAINS      workloads.jsonl domains, e.g. "html,html_v2,html_ood"
#   BENCH_MAX_TOKENS   tokens per reply (default 512)
# Writes trace-<draft>.jsonl per draft (tokens, text pieces, from_draft flags).
set -euo pipefail

cd mesh-llm
HERE="$PWD/skippy/evals/drafter-swap"
ART="${JIANYANG_ARTIFACT_DIR:?}"
MODELS="$HOME/.cache/dflash-bench-models"
mkdir -p "$MODELS"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
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
NAME="${BENCH_SPECIALISTS%%=*}" SPEC="${BENCH_SPECIALISTS#*=}"
[ -f "$SPEC" ] || { echo "specialist draft missing: $SPEC" >&2; exit 1; }

cargo build --release -p llama-spec-bench --features dynamic-native-runtime
BENCH="$CARGO_TARGET_DIR/release/llama-spec-bench"

python3 - "$HERE/workloads.jsonl" "${BENCH_DOMAINS:?}" > "$ART/trace-prompts.jsonl" <<'EOF'
import json, sys
domains = set(sys.argv[2].split(","))
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row["domain"] in domains:
        category = row["domain"] + (f":{row['format']}" if row.get("format") else "")
        print(json.dumps({"id": row["id"], "category": category, "prompt": row["prompt"]}))
EOF
echo "trace prompts: $(wc -l < "$ART/trace-prompts.jsonl")"

for pair in "generic=$DRAFT" "$NAME=$SPEC"; do
  label="${pair%%=*}" draft="${pair#*=}"
  echo "=== trace: $label"
  "$BENCH" --target-model-path "$TARGET" --draft-model-path "$draft" --dflash --chat --allow-mismatch \
    --prompt-corpus "$ART/trace-prompts.jsonl" --speculative-window 7 \
    --max-new-tokens "${BENCH_MAX_TOKENS:-512}" --ctx-size 16384 \
    --trace-out "$ART/trace-$label.jsonl" --json-out "$ART/spec-bench-$label.json" 2> "$ART/trace-$label.log" \
    | tail -20
done

echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
