#!/usr/bin/env bash
# One-time online asset staging for FieldScan TZ THT-02 (offline thereafter).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p models bin static/vendor exports outbox logs assets/manual_pages
touch exports/.gitkeep outbox/.gitkeep

OS="$(uname -s)"
ARCH="$(uname -m)"
LLAMA_TAG="b8763"
LLAMA_DIR="$ROOT_DIR/bin/llama-b8763"
mkdir -p "$LLAMA_DIR"

echo "[download_assets] Staging for ${OS}/${ARCH} (llama.cpp ${LLAMA_TAG})"

download_vendor() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "  keep $out"
    return 0
  fi
  echo "  fetch $out"
  curl -fsSL "$url" -o "$out"
}

# --- Frontend vendor (Three.js r170, uPlot 1.6.31, Open Props 1.7.10) ---
download_vendor "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js" \
  "$ROOT_DIR/static/vendor/three.module.min.js"
download_vendor "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js" \
  "$ROOT_DIR/static/vendor/OrbitControls.js"
download_vendor "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/utils/BufferGeometryUtils.js" \
  "$ROOT_DIR/static/vendor/BufferGeometryUtils.js"
download_vendor "https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.esm.js" \
  "$ROOT_DIR/static/vendor/uPlot.esm.js"
download_vendor "https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css" \
  "$ROOT_DIR/static/vendor/uPlot.min.css"
download_vendor "https://cdn.jsdelivr.net/npm/open-props@1.7.10/open-props.min.css" \
  "$ROOT_DIR/static/vendor/open-props.min.css"

# --- llama.cpp binary or compile ---
stage_llama() {
  if [[ -x "$LLAMA_DIR/llama-server" ]]; then
    echo "[download_assets] llama-server already present"
    return 0
  fi
  if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    local tarball="llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz"
    curl -fsSL -o "/tmp/${tarball}" \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${tarball}" || true
    if [[ -f "/tmp/${tarball}" ]]; then
      tar -xzf "/tmp/${tarball}" -C "$LLAMA_DIR" --strip-components=1 || tar -xzf "/tmp/${tarball}" -C "$LLAMA_DIR"
      return 0
    fi
  elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    local tarball="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
    curl -fsSL -o "/tmp/${tarball}" \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${tarball}" || true
    if [[ -f "/tmp/${tarball}" ]]; then
      tar -xzf "/tmp/${tarball}" -C "$LLAMA_DIR" --strip-components=1 || tar -xzf "/tmp/${tarball}" -C "$LLAMA_DIR"
      return 0
    fi
  fi

  # Linux aarch64 (Dragonwing IQ-9075) or fallback: compile from source if toolchain exists
  if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    local src="/tmp/llama.cpp-${LLAMA_TAG}"
    if [[ ! -d "$src" ]]; then
      git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$src" \
        || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$src"
    fi
    cmake -S "$src" -B "$src/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON || \
      cmake -S "$src" -B "$src/build" -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
    cmake --build "$src/build" -j"$(nproc)" --target llama-server
    cp -a "$src/build/bin/llama-server" "$LLAMA_DIR/" 2>/dev/null || cp -a "$src/build/llama-server" "$LLAMA_DIR/"
    find "$src/build" -name 'libmtmd.so*' -o -name 'libllama.so*' -o -name 'libggml*.so*' | while read -r lib; do
      cp -a "$lib" "$LLAMA_DIR/" || true
    done
  else
    echo "[download_assets] WARNING: no cmake/g++ — skipping llama-server compile (FastAPI deterministic fallbacks still work)"
  fi
}
stage_llama

# --- HuggingFace GGUF / Piper voices via Python ---
export PATH="${HOME}/.local/bin:${PATH}"
python3 - <<'PY'
from pathlib import Path
import sys
root = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
# script is executed with cwd = ROOT_DIR
root = Path.cwd()
models = root / "models"
models.mkdir(exist_ok=True)
try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    print(f"[download_assets] huggingface_hub unavailable: {exc}", file=sys.stderr)
    sys.exit(0)

def pull(repo, name, dest_name=None):
    dest = models / (dest_name or name)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  keep {dest.name}")
        return
    print(f"  hf {repo}/{name}")
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=str(models))
    p = Path(path)
    if dest_name and p.name != dest_name:
        p.rename(dest)

try:
    # Note: HF mmproj VL filename uses lowercase 'b' in 1.6b; we flatten to plan name.
    pull("LiquidAI/LFM2.5-VL-1.6B-GGUF", "LFM2.5-VL-1.6B-Q4_0.gguf")
    pull("LiquidAI/LFM2.5-VL-1.6B-GGUF", "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf", "mmproj-LFM2.5-VL-1.6B-Q8_0.gguf")
    pull("LiquidAI/LFM2.5-Audio-1.5B-GGUF", "LFM2.5-Audio-1.5B-Q4_0.gguf")
    pull("LiquidAI/LFM2.5-Audio-1.5B-GGUF", "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf")
    pull("LiquidAI/LFM2.5-Audio-1.5B-GGUF", "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf")
    pull("LiquidAI/LFM2.5-Audio-1.5B-GGUF", "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf")
    pull("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx")
    pull("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json", "en_US-lessac-medium.onnx.json")
except Exception as exc:
    print(f"[download_assets] WARNING model pull incomplete: {exc}", file=sys.stderr)
PY

echo "[download_assets] Done. Vendor + models staged under $ROOT_DIR"
