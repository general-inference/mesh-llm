#!/usr/bin/env bash
# Builds the release host and this machine's native runtime, from the jy workspace root.
set -eu
if [ "$(uname -s)" = Darwin ] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
  echo "holding this Mac awake for the build (caffeinate pid $!)"
fi
if [ -f "$HOME/.cargo/env" ]; then
  . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1 && command -v rustup >/dev/null 2>&1; then
  CARGO_BIN="$(rustup which cargo 2>/dev/null || true)"
  if [ -n "$CARGO_BIN" ] && [ -x "$CARGO_BIN" ]; then
    export PATH="$(dirname "$CARGO_BIN"):$PATH"
    echo "cargo found through rustup: $CARGO_BIN"
  fi
fi
export SCCACHE_DIR="${SCCACHE_DIR:-$HOME/.cache/sccache}"
if [ -z "${SCCACHE_CACHE_SIZE:-}" ]; then
  FREE_GB=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || true)
  QUARTER=$(( ${FREE_GB:-0} / 4 ))
  if [ "$QUARTER" -lt 8 ]; then QUARTER=8; fi
  if [ "$QUARTER" -gt 48 ]; then QUARTER=48; fi
  HALF_FREE=$(( ${FREE_GB:-0} / 2 ))
  if [ "$QUARTER" -gt "$HALF_FREE" ]; then QUARTER="$HALF_FREE"; fi
  if [ "$QUARTER" -lt 1 ]; then QUARTER=1; fi
  export SCCACHE_CACHE_SIZE="${QUARTER}G"
fi
echo "sccache ceiling ${SCCACHE_CACHE_SIZE} (${FREE_GB:-?}G free)"
NODE_BIN=$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1 || true)
[ -n "$NODE_BIN" ] && export PATH="$NODE_BIN:$PATH"
NODE_VER=24.20.0
NODE_DIR="$HOME/.cache/gi-node/node-v$NODE_VER-linux-arm64"
NODE_MAJOR=$(node -v 2>/dev/null | sed 's/^v//; s/\..*//' || true)
if ! command -v pnpm >/dev/null 2>&1 || [ "${NODE_MAJOR:-0}" -lt 24 ]; then
  if [ ! -x "$NODE_DIR/bin/pnpm" ]; then
    mkdir -p "$HOME/.cache/gi-node"
    curl -fsSL "https://nodejs.org/dist/v$NODE_VER/node-v$NODE_VER-linux-arm64.tar.xz" | tar -xJ -C "$HOME/.cache/gi-node"
    PATH="$NODE_DIR/bin:$PATH" npm install -g pnpm@10.34.5
  fi
  export PATH="$NODE_DIR/bin:$PATH"
fi
echo "node $(node -v) pnpm $(pnpm -v)"
if ! command -v patchelf >/dev/null 2>&1; then
  PE="$HOME/.cache/gi-patchelf"
  [ -x "$PE/bin/patchelf" ] || { python3 -m venv "$PE" && "$PE/bin/pip" -q install patchelf; }
  export PATH="$PE/bin:$PATH"
fi
echo "patchelf $(patchelf --version)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/mesh-llm-target}"
mkdir -p "$CARGO_TARGET_DIR"
export LLAMA_WORKDIR="${LLAMA_WORKDIR:-$HOME/.cache/mesh-llm-llama-src}"
export MESH_LLM_LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$HOME/.cache/mesh-llm-llama-build}"
mkdir -p "$LLAMA_WORKDIR" "$MESH_LLM_LLAMA_BUILD_ROOT"
cd mesh-llm
if [ "$(uname -s)" = Darwin ]; then BACKEND=metal
elif command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda
else BACKEND=cpu; fi
SCV="$(sccache --version 2>/dev/null | awk '{print $2}' || true)"
export LLAMA_STAGE_USE_SCCACHE=0
echo "llama stage builds without sccache (nvcc wrapping is broken on this fleet)"
echo "  sccache ${SCV:-absent} still caches the Rust half through cargo's own wrapper"
MISSING=""
for tool in cargo just; do
  command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
if [ -n "$MISSING" ]; then
  echo "this node cannot build mesh-llm: no$MISSING on PATH for $(whoami)" >&2
  echo "PATH=$PATH" >&2
  command -v rustup >/dev/null 2>&1 && \
    echo "rustup is installed; 'rustup which cargo' says: $(rustup which cargo 2>&1)" >&2
  exit 1
fi
echo "building release host and $BACKEND runtime"
just release-host-build
just release-runtime-build "$BACKEND"
TARGET_CEILING_GB="${TARGET_CEILING_GB:-40}"
TARGET_KB=$(du -sk "$CARGO_TARGET_DIR" 2>/dev/null | cut -f1 || true)
TARGET_GB=$(( ${TARGET_KB:-0} / 1048576 ))
if [ "$TARGET_GB" -gt "$TARGET_CEILING_GB" ]; then
  echo "target ${TARGET_GB}G over ${TARGET_CEILING_GB}G ceiling — resetting"
  rm -rf "$CARGO_TARGET_DIR"
  mkdir -p "$CARGO_TARGET_DIR"
else
  echo "target ${TARGET_GB}G within ${TARGET_CEILING_GB}G ceiling — kept"
fi
LLAMA_CEILING_GB="${LLAMA_CEILING_GB:-25}"
LLAMA_KB=$(du -sk "$MESH_LLM_LLAMA_BUILD_ROOT" 2>/dev/null | cut -f1 || true)
LLAMA_GB=$(( ${LLAMA_KB:-0} / 1048576 ))
if [ "$LLAMA_GB" -gt "$LLAMA_CEILING_GB" ]; then
  echo "llama build ${LLAMA_GB}G over ${LLAMA_CEILING_GB}G ceiling — resetting"
  rm -rf "$MESH_LLM_LLAMA_BUILD_ROOT"
  mkdir -p "$MESH_LLM_LLAMA_BUILD_ROOT"
else
  echo "llama build ${LLAMA_GB}G within ${LLAMA_CEILING_GB}G ceiling — kept"
fi
command -v sccache >/dev/null 2>&1 && sccache -s 2>/dev/null || true
