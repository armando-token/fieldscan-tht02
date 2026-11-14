#!/usr/bin/env bash
# Unified 3-daemon launcher: llama-server VL (:8091), Audio (:8092), FieldScan FastAPI (:8088).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA_DIR="$ROOT_DIR/bin/llama-b8763"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$LLAMA_DIR:${DYLD_LIBRARY_PATH:-}"
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
export FIELDSCAN_PORT="${FIELDSCAN_PORT:-8088}"
export FIELDSCAN_VL_URL="${FIELDSCAN_VL_URL:-http://127.0.0.1:8091}"
export FIELDSCAN_AUDIO_URL="${FIELDSCAN_AUDIO_URL:-http://127.0.0.1:8092}"

LLAMA_SERVER="$LLAMA_DIR/llama-server"
VL_PID=""
AUDIO_PID=""
APP_PID=""

cleanup() {
  local pids=()
  [[ -n "${APP_PID:-}" ]] && pids+=("$APP_PID")
  [[ -n "${VL_PID:-}" ]] && pids+=("$VL_PID")
  [[ -n "${AUDIO_PID:-}" ]] && pids+=("$AUDIO_PID")
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

if [[ "$(uname -s)" == "Darwin" ]]; then
  GPU_NGL="${FIELDSCAN_GPU_LAYERS:-99}"
else
  GPU_NGL="${FIELDSCAN_GPU_LAYERS:-99}"
fi

CPU_THREADS="${FIELDSCAN_CPU_THREADS:-}"
if [[ -z "$CPU_THREADS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    CPU_THREADS="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    CPU_THREADS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  else
    CPU_THREADS=4
  fi
fi

VL_MODEL="$ROOT_DIR/models/LFM2.5-VL-1.6B-Q4_0.gguf"
VL_MMPROJ="$ROOT_DIR/models/mmproj-LFM2.5-VL-1.6B-Q8_0.gguf"
AUDIO_MODEL="$ROOT_DIR/models/LFM2.5-Audio-1.5B-Q4_0.gguf"
AUDIO_MMPROJ="$ROOT_DIR/models/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"

if [[ -x "$LLAMA_SERVER" && -f "$VL_MODEL" && -f "$VL_MMPROJ" ]]; then
  echo "[start_all] Starting LFM2.5-VL on 127.0.0.1:8091 (ngl=$GPU_NGL)"
  "$LLAMA_SERVER" \
    -m "$VL_MODEL" \
    --mmproj "$VL_MMPROJ" \
    --host 127.0.0.1 --port 8091 \
    -c 4096 -ngl "$GPU_NGL" -t "$CPU_THREADS" \
    --reasoning off >"$LOG_DIR/llama_vl.log" 2>&1 &
  VL_PID=$!
else
  echo "[start_all] WARNING: VL llama-server skipped (binary or GGUF models missing)"
fi

if [[ -x "$LLAMA_SERVER" && -f "$AUDIO_MODEL" && -f "$AUDIO_MMPROJ" ]]; then
  echo "[start_all] Starting LFM2.5-Audio on 127.0.0.1:8092 (ngl=$GPU_NGL)"
  "$LLAMA_SERVER" \
    -m "$AUDIO_MODEL" \
    --mmproj "$AUDIO_MMPROJ" \
    --host 127.0.0.1 --port 8092 \
    -c 2048 -ngl "$GPU_NGL" -t "$CPU_THREADS" \
    --reasoning off >"$LOG_DIR/llama_audio.log" 2>&1 &
  AUDIO_PID=$!
else
  echo "[start_all] WARNING: Audio llama-server skipped (binary or GGUF models missing)"
fi

vl_ready=0
audio_ready=0
if [[ -z "${VL_PID:-}" ]]; then vl_ready=1; fi
if [[ -z "${AUDIO_PID:-}" ]]; then audio_ready=1; fi

for _ in $(seq 1 60); do
  if [[ "$vl_ready" -eq 0 ]] && curl -fsS --noproxy '*' "http://127.0.0.1:8091/health" >/dev/null 2>&1; then
    vl_ready=1
    echo "[start_all] VL health OK"
  fi
  if [[ "$audio_ready" -eq 0 ]] && curl -fsS --noproxy '*' "http://127.0.0.1:8092/health" >/dev/null 2>&1; then
    audio_ready=1
    echo "[start_all] Audio health OK"
  fi
  if [[ "$vl_ready" -eq 1 && "$audio_ready" -eq 1 ]]; then
    break
  fi
  sleep 0.5
done

if [[ "$vl_ready" -eq 0 || "$audio_ready" -eq 0 ]]; then
  echo "[start_all] WARNING: llama-server health poll timed out; FastAPI deterministic fallbacks remain available"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
PY="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

echo "[start_all] Starting uvicorn fieldscan.server:app on 127.0.0.1:$FIELDSCAN_PORT"
"$PY" -m uvicorn fieldscan.server:app --host 127.0.0.1 --port "$FIELDSCAN_PORT" &
APP_PID=$!
wait "$APP_PID"