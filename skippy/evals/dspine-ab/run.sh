#!/usr/bin/env bash
# DSpine vs DFlash2 vs autoregressive on SGLang 0.5.22 (official cu130 arm64 image) on one
# DGX Spark. Run from the jy workspace root. Inputs (environment):
#   ARMS       space-separated: ar dflash2 dspine            (default: all three)
#   WORKLOADS  space-separated: chat agent                   (default: both)
#   CONCS      space-separated concurrencies                 (default: "1 8")
#   LIMIT      first N prompts only (smoke)                  (default: all)
#   DFLASH2_REPLAYSSM  1 to give DFlash2 ReplaySSM rollback as DSpine has (default 1)
set -euo pipefail
HERE="$PWD/mesh-llm/skippy/evals/dspine-ab"
EVALS="$PWD/mesh-llm/skippy/evals"
ART="${JIANYANG_ARTIFACT_DIR:?}"
W="$HOME/.cache/dspine-ab"
mkdir -p "$W/hf" "$W/home" "$W/out"
IMAGE="lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae"  # v0.5.22-cu130
DSPINE_REV=6450af726375cb3c6bac4de95d98ebc8df99fbd4
ARMS="${ARMS:-ar dflash2 dspine}"
WORKLOADS="${WORKLOADS:-chat agent}"
CONCS="${CONCS:-1 8}"
NAME="dspine-ab-${JIANYANG_RUN_ID:-manual}"
PORT=$(( 31000 + $(printf '%s' "$NAME" | cksum | cut -d' ' -f1) % 2000 ))
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

gpu_state() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader | sed "s/^/gpu: $1 /"; }

echo "=== fetch + verify weights (local disk, sha256 per file)"
NEED="Qwen/Qwen3.8-27B"
case " $ARMS " in *" dflash2 "*) NEED="$NEED,z-lab/Qwen3.8-27B-DFlash2";; esac
case " $ARMS " in *" dspine "*) NEED="$NEED,tencent/Qwen3.8-27B-DSpine-b16";; esac
python3 "$HERE/fetch.py" "$HERE/manifest.json" "$W/hf" "$NEED" | tee "$ART/fetch.log" | grep -c '^ok' | sed 's/^/verified files: /'

echo "=== image"
docker pull -q "$IMAGE"
docker image inspect "$IMAGE" --format '{{.Id}}'

echo "=== DSpine @ $DSPINE_REV"
if [ ! -d "$W/DSpine/.git" ]; then git clone -q https://github.com/Tencent/DSpine.git "$W/DSpine"; fi
git -C "$W/DSpine" fetch -q origin
git -C "$W/DSpine" checkout -q -f "$DSPINE_REV"
git -C "$W/DSpine" clean -qfdx -e '*.egg-info' || true
# The only change: let the draft's native sliding-window attention run on FA4 as well as
# FA3. SGLang refuses FA3 on SM12x (GB10); its FA4 path has an SM12x kernel set, and the
# plugin's own worker check already accepts fa_impl_ver 4. The target itself must use
# flashinfer: SGLang allows only flashinfer/triton/trtllm_mha for hybrid GDN on SM12x.
sed -i 's/^BACKENDS = frozenset(("fa3",))$/BACKENDS = frozenset(("fa3", "fa4"))/' "$W/DSpine/sglang_dspine/layers/attention.py"
git --no-pager -C "$W/DSpine" diff --stat
git --no-pager -C "$W/DSpine" diff > "$ART/dspine-gb10.patch"

COMMON=(--model-path /w/hf/Qwen/Qwen3.8-27B --tp-size 1 --dtype bfloat16
  --attention-backend "${TARGET_ATTN:-flashinfer}" --linear-attn-backend triton
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 64
  --mem-fraction-static "${MEM_FRACTION:-0.80}" --context-length 65536 --max-running-requests 8
  --cuda-graph-max-bs-decode 8 --sampling-backend flashinfer --trust-remote-code
  --host 127.0.0.1 --port "$PORT")

arm_flags() {
  case "$1" in
    ar) echo "";;
    dflash2)
      echo "--speculative-algorithm DFLASH --speculative-draft-model-path /w/hf/z-lab/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 --speculative-draft-attention-backend fa4"
      [ "${DFLASH2_REPLAYSSM:-1}" = 1 ] && echo "--enable-linear-replayssm-spec" || true;;
    dspine)
      echo "--speculative-algorithm DSPINE --speculative-draft-model-path /w/hf/tencent/Qwen3.8-27B-DSpine-b16 --speculative-num-draft-tokens 16 --speculative-draft-attention-backend fa4 --enable-linear-replayssm-spec --min-free-slots-delay 1";;
  esac
}

for arm in $ARMS; do
  cleanup
  gpu_state "$arm-before"
  FLAGS=$(arm_flags "$arm" | tr '\n' ' ')
  echo "=== arm $arm: sglang serve ${COMMON[*]} $FLAGS"
  docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 16g \
    -e HOME=/w/home -e HF_HOME=/w/home/hf -e HF_HUB_OFFLINE=1 -e TRITON_CACHE_DIR=/w/home/triton \
    -v "$W:/w" -v "$EVALS:/evals:ro" "$IMAGE" \
    bash -c "pip install -q --no-deps --no-build-isolation -e /w/DSpine >/w/out/pip-$arm.log 2>&1 || cat /w/out/pip-$arm.log; python3 -c 'import sglang,torch,flashinfer;print(\"versions\",sglang.__version__,torch.__version__,torch.version.cuda,flashinfer.__version__,torch.cuda.get_device_capability())'; exec sglang serve ${COMMON[*]} $FLAGS" >/dev/null
  t0=$(date +%s); up=0
  while [ $(( $(date +%s) - t0 )) -lt 3600 ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || break
    sleep 10
  done
  docker logs "$NAME" > "$ART/serve-$arm.log" 2>&1 || true
  if [ "$up" != 1 ]; then
    echo "arm $arm: server never became healthy"; tail -60 "$ART/serve-$arm.log"
    echo "DSPINE_AB {\"label\": \"$arm\", \"error\": \"server not healthy\"}"
    continue
  fi
  echo "arm $arm healthy after $(( $(date +%s) - t0 ))s"
  grep -E "versions|DSPINE ready|DSPine|max_total_num_tokens|KV Cache|cuda graph|Capture" "$ART/serve-$arm.log" | head -20 || true
  for wl in $WORKLOADS; do
    if [ "$wl" = chat ]; then P=/evals/spec-prompts/spec-prompts.jsonl; S=report; else P=/evals/agent-transcripts/agent-transcripts.jsonl; S=all; fi
    for c in $CONCS; do
      curl -sf -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || curl -sf "http://127.0.0.1:$PORT/flush_cache" >/dev/null || echo "flush_cache failed"
      gpu_state "$arm-$wl-c$c"
      docker exec "$NAME" python3 /evals/dspine-ab/client.py --port "$PORT" --prompts "$P" --split "$S" \
        --tokenizer /w/hf/Qwen/Qwen3.8-27B --concurrency "$c" --label "$arm-$wl" \
        --out "/w/out/${NAME}.jsonl" ${LIMIT:+--limit "$LIMIT"} \
        || echo "DSPINE_AB {\"label\": \"$arm-$wl\", \"concurrency\": $c, \"error\": \"client failed\"}"
      gpu_state "$arm-$wl-c$c-after"
    done
  done
  docker logs "$NAME" > "$ART/serve-$arm.log" 2>&1 || true
  cleanup
  sleep 10
done
docker run --rm -v "$W:/w" "$IMAGE" chmod -R a+rwX /w/out /w/home /w/DSpine >/dev/null 2>&1 || true
cp "$W/out/${NAME}.jsonl" "$ART/results.jsonl" 2>/dev/null || true
echo '{"points": [], "provenance": {"stack_commit": "'"${JIANYANG_COMMIT:-}"'"}}' > "${JIANYANG_RESULT_PATH:-/dev/null}"
