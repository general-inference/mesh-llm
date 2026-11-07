#!/usr/bin/env bash
# splitarch-infer: single-node DFlash trace for offline replay of pipelined speculation.
# Runs llama-spec-bench (DFlash, fixed block) over the 77 chat and 32 agent prompts and writes
# every verify window (draft tokens, target predictions, timing) to $JIANYANG_ARTIFACT_DIR.
set -uo pipefail
ART="${JIANYANG_ARTIFACT_DIR:?}"
cd mesh-llm
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
export PATH="$HOME/.cargo/bin:$PATH"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes"
ls -la dist/native-runtimes || { echo "no native runtime bundle" >&2; exit 1; }
cargo build --release -p llama-spec-bench 2>&1 | tail -5
BIN="$CARGO_TARGET_DIR/release/llama-spec-bench"
[ -x "$BIN" ] || { echo "no llama-spec-bench build" >&2; exit 1; }

fetch() { # url sha path
  local url="$1" sha="$2" path="$3"
  mkdir -p "$(dirname "$path")"
  for _ in 1 2 3; do
    [ -f "$path" ] && echo "$sha  $path" | sha256sum -c --quiet && return 0
    rm -f "$path.part"; curl -fsSL --retry 10 -o "$path.part" "$url" && mv "$path.part" "$path"
  done
  echo "$sha  $path" | sha256sum -c --quiet
}
TARGET="$HOME/.cache/dflash-bench-models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="$HOME/.cache/dflash-bench-models/Qwen3.6-27B-DFlash.Q8_0.gguf"
fetch "$TARGET_URL" "$TARGET_SHA256" "$TARGET" || { echo "target failed verification" >&2; exit 1; }
fetch "$DRAFT_URL" "$DRAFT_SHA256" "$DRAFT" || { echo "draft failed verification" >&2; exit 1; }
echo "models verified"

python3 - "$ART" <<'PY'
import json, sys
art = sys.argv[1]
def render(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"
chat = [json.loads(l) for l in open("skippy/evals/spec-prompts/spec-prompts.jsonl") if l.strip()]
chat = [c for c in chat if c["split"] == "report"]
agent = [json.loads(l) for l in open("skippy/evals/agent-transcripts/agent-transcripts.jsonl") if l.strip()]
with open(f"{art}/corpus-chat.jsonl", "w") as f:
    for c in chat: f.write(json.dumps({"id": c["id"], "category": c["category"], "prompt": render(c["prompt"])}) + "\n")
with open(f"{art}/corpus-agent.jsonl", "w") as f:
    for c in agent: f.write(json.dumps({"id": c["id"], "category": c["group"], "prompt": render(c["prompt"])}) + "\n")
print("chat", len(chat), "agent", len(agent))
PY
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader || true
for SET in ${TRACE_SETS:-chat agent}; do
  T0=$(date +%s)
  SPEC_SKIP_BASELINE=1 SPEC_TRACE_OUT="$ART/trace-$SET.jsonl" "$BIN" \
    --target-model-path "$TARGET" --draft-model-path "$DRAFT" --dflash \
    --prompt-corpus "$ART/corpus-$SET.jsonl" --max-new-tokens "${MAX_NEW:-256}" \
    --speculative-window "${WINDOW:-15}" --ctx-size 65536 --allow-mismatch \
    --json-out "$ART/report-$SET.json" > "$ART/bench-$SET.log" 2>&1
  echo "set $SET rc=$? in $(( $(date +%s) - T0 ))s; windows $(grep -c '"window"' "$ART/trace-$SET.jsonl" 2>/dev/null)"
  tail -25 "$ART/bench-$SET.log"
done
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
