#!/usr/bin/env bash
# Step 3: what a draft swap costs. Builds llama-spec-bench against the packaged
# runtime and runs `--swap-probe` on one machine:
#   resident  DFlash2 swapped for a second open copy of itself: the restart's effect
#             on acceptance by rows seen, and the cost of exchanging open drafts
#   replay    resident, with the last 2048 feature rows replayed into the incoming draft
#   load      the same, opening the incoming draft from its file at every swap
#   cross     DFlash2 and the original DFlash draft, both open, swapped back and forth
# Prompts: agent transcripts (4-48K tokens) and the eval code/HTML prompts. Same
# weight inputs as run.sh, plus BENCH_ALT_DRAFT_URL / _SHA256 / _BYTES.
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
ALT="$MODELS/${BENCH_ALT_DRAFT_URL##*/}"
fetch "${BENCH_TARGET_URL:?}" "${BENCH_TARGET_SHA256:?}" "$TARGET"
fetch "${BENCH_DRAFT_URL:?}" "${BENCH_DRAFT_SHA256:?}" "$DRAFT"
fetch "${BENCH_ALT_DRAFT_URL:?}" "${BENCH_ALT_DRAFT_SHA256:?}" "$ALT"

cargo build --release -p llama-spec-bench --features dynamic-native-runtime
BENCH="$CARGO_TARGET_DIR/release/llama-spec-bench"

# Probe prompts: every agent transcript but the longest band, plus 8 each of code
# edit, code review and HTML from the eval set.
python3 - "$HERE/workloads.jsonl" "$PWD/skippy/evals/agent-transcripts/agent-transcripts.jsonl" \
  > "$ART/probe-prompts.jsonl" <<'EOF'
import json, sys
work = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
agent = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
rows = [dict(a, category=f"agent:{a['group']}") for a in agent if a["group"] != "32-48K"][:16]
for domain in ("code_edit", "code_review", "html"):
    rows += [w for w in work if w["domain"] == domain][:8]
for r in rows:
    print(json.dumps({"id": r["id"], "category": r.get("category") or r.get("domain"), "prompt": r["prompt"]}))
EOF
echo "probe prompts: $(wc -l < "$ART/probe-prompts.jsonl")"

probe() { # label swap-draft [extra args]; BENCH_PROBES limits which labels run
  local label="$1" swap="$2"; shift 2
  case " ${BENCH_PROBES:-resident replay load cross} " in *" $label "*) ;; *) return 0 ;; esac
  echo "=== swap probe: $label"
  "$BENCH" --target-model-path "$TARGET" --draft-model-path "$DRAFT" --dflash --swap-probe \
    --swap-draft-model-path "$swap" --prompt-corpus "$ART/probe-prompts.jsonl" "$@" \
    --speculative-window 7 --max-new-tokens 256 --ctx-size 40960 \
    --swap-restart-rows "${BENCH_RESTART_ROWS:-0,16,64,256,1024,2048}" \
    --json-out "$ART/swap-probe-$label.json" 2> "$ART/swap-probe-$label.log" \
    | sed "s/^SWAP_PROBE /SWAP_PROBE $label /"
}
probe resident "$DRAFT" --swap-resident
BENCH_RESTART_ROWS=0,256 probe replay "$DRAFT" --swap-resident --swap-replay-rows 2048
BENCH_RESTART_ROWS=0,256 probe load "$DRAFT"
BENCH_RESTART_ROWS=0,256,2048 probe cross "$ALT" --swap-resident

echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
