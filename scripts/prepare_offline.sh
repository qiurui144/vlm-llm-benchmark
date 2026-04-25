#!/usr/bin/env bash
# 预备机（有网）打包脚本
#
# 作用: 在一台能访问 HuggingFace + PyPI 的机器上，提前下载:
#   1. vLLM 及其依赖的 wheels → wheels/
#   2. Qwen3-VL + Qwen3 LLM 模型权重 → models/
# 然后把整个 qwen3vl_eval/ 打成 tar.gz 丢给 DGX
#
# 注意: 235B 模型权重约 240GB，如果硬盘放不下，可以跳过（MODEL_SET 控制）
#
# 使用:
#   MODEL_SET=minimal bash scripts/prepare_offline.sh   # 只下 VLM 首选（~16GB）
#   MODEL_SET=standard bash scripts/prepare_offline.sh  # VLM 双模 + LLM 30B（~80GB）
#   MODEL_SET=full bash scripts/prepare_offline.sh      # 全部四模型（~320GB）

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
echo "打包命令: "
echo "  cd $(dirname "$PKG_ROOT") && tar czf qwen3vl_eval.tar.gz qwen3vl_eval/"
