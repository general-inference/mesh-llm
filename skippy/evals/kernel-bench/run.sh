#!/usr/bin/env bash
# Measures every built arm on this machine with llama-bench, interleaved per
# model so arms see the same thermal state. Emits KBENCH lines and JSONL
# artifacts; KB_NSYS=1 adds an Nsight Systems kernel summary for the base arm.
#
#   KB_MODELS   space-separated ids from models.tsv (default: qwen3.5-9b)
#   KB_ARMS     arms built by build.sh (default: base)
#   KB_BENCH    llama-bench shape sets: verify prefill decode (default: all)
#   KB_REPS     repetitions per shape (default 3)
set -uo pipefail
cd mesh-llm
HERE="$PWD/skippy/evals/kernel-bench"
CACHE="$HOME/.cache/kbench"
MODELS_DIR="$HOME/.cache/dflash-bench-models"
ART="${JIANYANG_ARTIFACT_DIR:-$PWD/kbench-out}"
mkdir -p "$ART" "$MODELS_DIR"
REPS="${KB_REPS:-3}"
if [ "$(uname -s)" = Darwin ]; then
  sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
  sysctl -n machdep.cpu.brand_string
else
  export PATH="/usr/local/cuda/bin:$PATH"
  sha() { sha256sum "$1" | cut -d' ' -f1; }
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
fi

fetch() { # id -> path
  local line url want f
  line=$(awk -v id="$1" '$1==id' "$HERE/models.tsv")
  [ -n "$line" ] || { echo "unknown model $1" >&2; return 1; }
  url=$(echo "$line" | awk '{print $2}'); want=$(echo "$line" | awk '{print $3}')
  f="$MODELS_DIR/${url##*/}"
  if [ -f "$f" ] && [ -f "$f.sha256-ok" ] && [ "$(cat "$f.sha256-ok")" = "$want" ]; then echo "$f"; return 0; fi
  if [ ! -f "$f" ] || [ "$(sha "$f")" != "$want" ]; then
    for attempt in 1 2 3; do
      curl -fL --retry 5 -sS -C - -o "$f.part" "$url" >&2 && { mv "$f.part" "$f"; break; }
      echo "curl exit $? on attempt $attempt for $url" >&2; sleep 10
    done
  fi
  [ "$(sha "$f")" = "$want" ] || { echo "sha mismatch for $f" >&2; rm -f "$f"; return 1; }
  echo "$want" > "$f.sha256-ok"; echo "$f"
}

bench() { # arm model_path tag args...
  local arm=$1 m=$2 tag=$3; shift 3
  local out="$ART/${tag}.${arm}.jsonl"
  "$CACHE/build-$arm/bin/llama-bench" -m "$m" -ngl 99 -fa 1 -r "$REPS" -o jsonl "$@" >>"$out" 2>>"$ART/stderr.${arm}.log"
  python3 - "$out" "$arm" "$tag" <<'PY'
import json,sys
for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip()):
    print("KBENCH", json.dumps({"arm":sys.argv[2],"set":sys.argv[3],"model":r["model_filename"].rsplit("/",1)[-1],
      "pp":r["n_prompt"],"tg":r["n_gen"],"depth":r.get("n_depth",0),"ub":r.get("n_ubatch"),
      "tok_s":round(r["avg_ts"],2),"sd":round(r["stddev_ts"],2)}))
PY
  cat "$out" >>"$ART/all.jsonl"; rm -f "$out"
}

for ID in ${KB_MODELS-qwen3.5-9b}; do
  M=$(fetch "$ID") || { echo "KBENCH_FAIL fetch $ID"; continue; }
  for SET in ${KB_BENCH:-verify prefill decode}; do
    for ARM in ${KB_ARMS:-base}; do
      case $SET in
        verify)  bench "$ARM" "$M" "$ID-verify"  -p 1,2,4,8,16,32,64 -n 0 -d 0,4096 ;;
        verifyfine) bench "$ARM" "$M" "$ID-verifyfine" -p 1,2,3,4,5,6,7,8,10,12,16 -n 0 ;;
        prefill) bench "$ARM" "$M" "$ID-prefill" -p 512,2048,8192 -n 0 -ub 512,2048 ;;
        ubatch)  bench "$ARM" "$M" "$ID-ubatch" -p 2048,8192 -n 0 -b 8192 -ub 512,1024,2048,4096,8192 ;;
        # Rollback snapshots (needs the bench-rs-seq arm). A batch must be longer than
        # the snapshot count + 1, so 16- and 32-token verifies with 0, 7 and 14 slots.
        rsverify) for RS in 0 7 14; do LLAMA_BENCH_RS_SEQ=$RS bench "$ARM" "$M" "$ID-rs$RS" -p 16,32 -n 0 -d 0,4096; done ;;
        longctx) bench "$ARM" "$M" "$ID-longctx" -p 2048 -n 0 -d 0,8192,16384,32768 -r 2
                 bench "$ARM" "$M" "$ID-longctx-tg" -p 0 -n 64 -d 0,8192,16384,32768 -r 2 ;;
        decode)  bench "$ARM" "$M" "$ID-decode"  -p 0 -n 128 -d 0,4096 ;;
      esac
    done
  done
  if [ "${KB_NSYS:-0}" = 1 ] && command -v nsys >/dev/null 2>&1; then
    # No -d: a depth fill is a prefill and swamps the small shapes. Graph nodes are
    # traced one by one, otherwise decode shows up as opaque graph launches.
    IFS=';' read -r -a SHAPES <<<"${KB_NSYS_SHAPES:-verify16:-p 16 -n 0 -r 20;decode:-p 0 -n 128 -r 3;prefill:-p 2048 -n 0 -r 2}"
    for SHAPE in "${SHAPES[@]}"; do
      NAME=${SHAPE%%:*}; ARGS=${SHAPE#*:}
      nsys profile -t cuda,nvtx --cuda-graph-trace=node -o "$CACHE/nsys-$ID-$NAME" -f true "$CACHE/build-${KB_NSYS_ARM:-base}/bin/llama-bench" -m "$M" -ngl 99 -fa 1 $ARGS >/dev/null 2>&1
      nsys stats -q --report cuda_gpu_kern_sum --format csv "$CACHE/nsys-$ID-$NAME.nsys-rep" > "$ART/nsys-$ID-$NAME.csv" 2>/dev/null
      echo "=== nsys $ID $NAME top kernels"
      python3 - "$ART/nsys-$ID-$NAME.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(l for l in open(sys.argv[1]) if "," in l))
tot=sum(float(r["Total Time (ns)"]) for r in rows) or 1
for r in rows[:25]:
    print(f'{100*float(r["Total Time (ns)"])/tot:5.1f}%  n={r["Instances"]:>7}  avg={float(r["Avg (ns)"])/1e3:8.1f}us  {r["Name"][:150]}')
PY
    done
  fi
done
grep -h '"avg_ts"' "$ART/all.jsonl" >/dev/null 2>&1 && python3 - "$ART/all.jsonl" "${JIANYANG_RESULT_PATH:-/dev/null}" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
json.dump({"points":[{"concurrency":1,"requests_completed":len(rows),"duration_s":1,
  "throughput":{"output_tok_s":max(r["avg_ts"] for r in rows)}}],"provenance":{}},open(sys.argv[2],"w"))
PY
# Device-to-host copy bandwidth (CUDA only).
case " ${KB_BENCH:-} " in *" d2h "*)
  nvcc -O2 -o "$CACHE/d2h" "$HERE/d2h.cu" && for MB in 1 13 64 256; do "$CACHE/d2h" $MB 8; done ;;
esac
# Correctness against the CPU backend for the ops an arm changes.
case " ${KB_BENCH:-} " in *" correct "*)
  for ARM in ${KB_ARMS:-base}; do
    "$CACHE/build-$ARM/bin/test-backend-ops" test -b CUDA0 -o "${KB_OPS:-MUL_MAT,MUL_MAT_ID}" > "$ART/correct.$ARM.txt" 2>&1
    echo "KBENCH_CORRECT $ARM exit=$? $(grep -cE '\bOK\b' "$ART/correct.$ARM.txt") ok $(grep -cE 'FAIL' "$ART/correct.$ARM.txt") fail"
    grep -E "FAIL" "$ART/correct.$ARM.txt" | head -20
  done ;;
esac
# Per-shape matmul timings (mmbench) for every arm, interleaved shape by shape.
case " ${KB_BENCH:-} " in *" mm "*)
  grep -v '^#' "$HERE/shapes.tsv" | while read -r T MR KD E U C MODEL; do
    for N in ${KB_MM_N:-512 1024 2048}; do
      for ARM in ${KB_ARMS:-base}; do
        r=$("$CACHE/build-$ARM/bin/mmbench" "$T" "$MR" "$KD" "$N" "$E" "$U" 2>/dev/null | grep MMBENCH)
        echo "MMB $ARM $MODEL $C $r"
      done
    done
  done | tee "$ART/mm.txt" ;;
esac
# Op microbenchmarks last: a full MUL_MAT sweep takes most of an hour.
case " ${KB_BENCH:-} " in *" ops "*)
  for ARM in ${KB_ARMS:-base}; do
    "$CACHE/build-$ARM/bin/test-backend-ops" perf -o "${KB_OPS:-MUL_MAT}" 2>&1 | grep -v "not supported" > "$ART/ops.$ARM.txt"
    echo "KBENCH_OPS $ARM $(wc -l < "$ART/ops.$ARM.txt") lines"
  done ;;
esac
echo "=== run done"
