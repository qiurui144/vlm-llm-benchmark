#!/usr/bin/env bash
# Run on an internet-connected staging host to pre-fetch:
#   1. vLLM and its dependencies as wheels → wheels/
#   2. Reference Qwen model weights → models/
#
# After this, tar up the whole vlm-llm-benchmark/ directory and ship it to your
# air-gapped GPU host.
#
# Note: the 235B-FP8 model is ~240 GB; skip it via MODEL_SET unless you have disk.
#
# Usage:
#   MODEL_SET=minimal  bash scripts/prepare_offline.sh   # VLM primary only  (~16 GB)
#   MODEL_SET=standard bash scripts/prepare_offline.sh   # VLM ×2 + LLM-30B  (~80 GB)
#   MODEL_SET=full     bash scripts/prepare_offline.sh   # all 4 models      (~320 GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELS_DIR="$PKG_ROOT/wheels"
MODELS_DIR="$PKG_ROOT/models"

MODEL_SET="${MODEL_SET:-standard}"

echo "====================================="
echo " 离线包准备 (MODEL_SET=$MODEL_SET)"
echo "====================================="

# ─────────────────────────────────────────
# Step 1: 下载 vLLM wheels
# ─────────────────────────────────────────
echo
echo "[1/2] 下载 vLLM + 依赖 wheels 到 $WHEELS_DIR"
mkdir -p "$WHEELS_DIR"

# vLLM 0.8.x 支持 Qwen3-VL 系列；同时下基准脚本用的依赖
pip download \
    --dest "$WHEELS_DIR" \
    --platform manylinux2014_x86_64 \
    --python-version 310 \
    --only-binary=:all: \
    "vllm>=0.8.0" \
    "httpx>=0.27" \
    "pyyaml>=6.0" \
    "Pillow>=10.0" \
    "pynvml>=11.5" \
    || {
        echo "[WARN] --only-binary 严格模式失败，尝试宽松模式（可能拉源码包）"
        pip download --dest "$WHEELS_DIR" \
            "vllm>=0.8.0" "httpx>=0.27" "pyyaml>=6.0" "Pillow>=10.0" "pynvml>=11.5"
    }

echo "[OK] wheels 数量: $(ls "$WHEELS_DIR" | wc -l)"

# ─────────────────────────────────────────
# Step 2: 下载模型权重
# ─────────────────────────────────────────
echo
echo "[2/2] 下载 Qwen 模型权重 到 $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# huggingface-cli 在预备机上安装
pip install -q "huggingface-hub[cli]>=0.22"

download_model() {
    local repo="$1"
    local dir="$MODELS_DIR/$(basename "$repo")"
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        echo "[SKIP] $repo 已存在"
        return
    fi
    echo "[DOWN] $repo -> $dir"
    huggingface-cli download "$repo" \
        --local-dir "$dir" \
        --local-dir-use-symlinks False \
        --exclude "*.bin" "*.pt"   # 只要 safetensors 减体积
}

case "$MODEL_SET" in
    minimal)
        download_model "Qwen/Qwen3-VL-8B-Instruct"
        ;;
    standard)
        download_model "Qwen/Qwen3-VL-8B-Instruct"
        download_model "Qwen/Qwen2.5-VL-7B-Instruct"
        download_model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        ;;
    full)
        download_model "Qwen/Qwen3-VL-8B-Instruct"
        download_model "Qwen/Qwen2.5-VL-7B-Instruct"
        download_model "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        download_model "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
        ;;
    *)
        echo "[ERROR] 未知 MODEL_SET=$MODEL_SET (支持: minimal/standard/full)"
        exit 1
        ;;
esac

echo
echo "====================================="
echo " 完成"
echo "====================================="
echo "总体积: $(du -sh "$PKG_ROOT" | awk '{print $1}')"
echo "Bundle command:"
echo "  cd $(dirname "$PKG_ROOT") && tar czf vlm-llm-benchmark.tar.gz $(basename "$PKG_ROOT")/"
