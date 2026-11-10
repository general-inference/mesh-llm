#!/usr/bin/env bash
# Builds llama-bench from the pinned + patched llama.cpp tree, once per arm.
# An arm is "base" or a '+'-joined list of patch names from ./patches applied
# on top of the patch queue. Sources and build dirs persist under ~/.cache so
# a repeat run only recompiles what an arm changed.
set -euo pipefail
cd mesh-llm
HERE="$PWD/skippy/evals/kernel-bench"
CACHE="$HOME/.cache/kbench"
mkdir -p "$CACHE"
export LLAMA_WORKDIR="$CACHE/llama-src"
scripts/prepare-llama.sh pinned >"$CACHE/prepare.log" 2>&1 || { tail -40 "$CACHE/prepare.log"; exit 1; }
if [ "$(uname -s)" = Darwin ]; then
  BACKEND_FLAGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
  JOBS=$(sysctl -n hw.ncpu)
else
  export PATH="/usr/local/cuda/bin:$PATH"
  BACKEND_FLAGS=(-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON "-DCMAKE_CUDA_ARCHITECTURES=${KB_CUDA_ARCH:-121a-real}")
  JOBS=$(nproc)
fi
for ARM in ${KB_ARMS:-base}; do
  SRC="$CACHE/src-$ARM"; BUILD="$CACHE/build-$ARM"
  mkdir -p "$SRC"
  rsync -a --checksum --delete --exclude .git --exclude '/build/' "$LLAMA_WORKDIR/" "$SRC/"
  if [ "$ARM" != base ]; then
    for P in ${ARM//+/ }; do
      echo "=== $ARM: apply $P"
      patch -p1 -d "$SRC" --forward --no-backup-if-mismatch < "$HERE/patches/$P.patch"
    done
  fi
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
    -DGGML_NATIVE=ON "${BACKEND_FLAGS[@]}" >"$CACHE/cmake-$ARM.log" 2>&1 || { tail -40 "$CACHE/cmake-$ARM.log"; exit 1; }
  echo "=== build $ARM"
  if ! cmake --build "$BUILD" -j "$JOBS" --target llama-bench test-backend-ops >"$CACHE/build-$ARM.log" 2>&1; then
    grep -E "error" "$CACHE/build-$ARM.log" | head -40
    echo "KBENCH_BUILD_FAIL $ARM"; exit 1
  fi
  echo "objects rebuilt for $ARM: $(grep -c "Building" "$CACHE/build-$ARM.log" || true)"
  c++ -O2 -std=c++17 -I"$SRC/ggml/include" "$HERE/mmbench.cpp" -L"$BUILD/bin" -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin" -o "$BUILD/bin/mmbench" || { echo "KBENCH_BUILD_FAIL mmbench $ARM"; exit 1; }
done
echo "=== build done"
