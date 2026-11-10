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
    curl -fL --retry 5 -C - -o "$f.part" "$url" >&2 && mv "$f.part" "$f"
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

for ID in ${KB_MODELS:-qwen3.5-9b}; do
  M=$(fetch "$ID") || { echo "KBENCH_FAIL fetch $ID"; continue; }
  for SET in ${KB_BENCH:-verify prefill decode}; do
    for ARM in ${KB_ARMS:-base}; do
      case $SET in
        verify)  bench "$ARM" "$M" "$ID-verify"  -p 1,2,4,8,16,32,64 -n 0 -d 0,4096 ;;
        prefill) bench "$ARM" "$M" "$ID-prefill" -p 512,2048,8192 -n 0 -ub 512,2048 ;;
        decode)  bench "$ARM" "$M" "$ID-decode"  -p 0 -n 128 -d 0,4096 ;;
        ops)     "$CACHE/build-$ARM/bin/test-backend-ops" perf -o "${KB_OPS:-MUL_MAT}" 2>&1 | grep -v "not supported" | tail -n 400 > "$ART/ops.$ARM.txt"; echo "KBENCH_OPS $ARM $(wc -l < "$ART/ops.$ARM.txt") lines" ;;
      esac
    done
  done
  if [ "${KB_NSYS:-0}" = 1 ] && command -v nsys >/dev/null 2>&1; then
    for SHAPE in "verify16:-p 16 -n 0 -d 4096 -r 10" "decode:-p 0 -n 64 -d 4096 -r 2" "prefill:-p 2048 -n 0 -r 2"; do
      NAME=${SHAPE%%:*}; ARGS=${SHAPE#*:}
      nsys profile -t cuda,nvtx -o "$CACHE/nsys-$ID-$NAME" -f true "$CACHE/build-base/bin/llama-bench" -m "$M" -ngl 99 -fa 1 $ARGS >/dev/null 2>&1
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
echo "=== run done"
