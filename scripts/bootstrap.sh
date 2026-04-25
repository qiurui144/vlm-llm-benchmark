#!/usr/bin/env bash
# First-time bootstrap on a GPU host (e.g. DGX).
#
# Does, in order:
#   1. (Optional) ZeroTier install + network join
#   2. Create venv + activate
#   3. Install vLLM offline from ./wheels (or fall back to online install)
#   4. Symlink ./models/* into the HuggingFace cache layout
#
# Prerequisites: vlm-llm-benchmark/ unpacked at $PKG_ROOT, Python 3.10+ available.
#
# Usage:
#   cd vlm-llm-benchmark
#   sudo bash scripts/bootstrap.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELS_DIR="$PKG_ROOT/wheels"
MODELS_DIR="$PKG_ROOT/models"

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] 请 sudo bash $0"
    exit 1
fi

# ─────────────────────────────────────────
# Step 1: ZeroTier 组网
# ─────────────────────────────────────────
echo "[1/4] ZeroTier"
bash "$SCRIPT_DIR/setup_zerotier.sh"

# ─────────────────────────────────────────
# Step 2: Python 环境
# ─────────────────────────────────────────
echo
echo "[2/4] Python 环境检查"
PY_BIN="${PYTHON:-python3}"
if ! "$PY_BIN" -c 'import sys; assert sys.version_info >= (3,10)' &>/dev/null; then
    echo "[ERROR] 需要 Python 3.10+，当前 $($PY_BIN --version 2>&1)"
    echo "        DGX OS 默认自带 3.10，可 export PYTHON=/usr/bin/python3.10"
    exit 2
fi
$PY_BIN -m venv "$PKG_ROOT/.venv"
source "$PKG_ROOT/.venv/bin/activate"
pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────
# Step 3: 离线装 vLLM
# ─────────────────────────────────────────
echo
echo "[3/4] 离线安装 vLLM"
if [[ -d "$WHEELS_DIR" && -n "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]]; then
    pip install --no-index --find-links "$WHEELS_DIR" \
        vllm httpx pyyaml Pillow pynvml
else
    echo "[WARN] wheels/ 目录为空，改走在线安装"
    pip install -r "$PKG_ROOT/requirements.txt"
    pip install "vllm>=0.8.0"
fi

# ─────────────────────────────────────────
# Step 4: 模型权重软链 → HF cache
# ─────────────────────────────────────────
echo
echo "[4/4] 链接模型权重"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
mkdir -p "$HF_CACHE"

if [[ -d "$MODELS_DIR" ]]; then
    for d in "$MODELS_DIR"/*/; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        # Qwen/Qwen3-VL-8B-Instruct → models--Qwen--Qwen3-VL-8B-Instruct
        hf_name="models--Qwen--$name"
        target="$HF_CACHE/$hf_name"
        if [[ -e "$target" ]]; then
            echo "[SKIP] $target 已存在"
            continue
        fi
        # 手动搭建 HF cache 目录布局: snapshots/main/* 指向真实权重
        snap="$target/snapshots/main"
        mkdir -p "$snap"
        for f in "$d"/*; do
            ln -sf "$f" "$snap/$(basename "$f")"
        done
        echo "[LINK] $name → $target"
    done
fi

echo
echo "============================================="
echo " bootstrap 完成"
echo "============================================="
echo "下一步:"
echo "  source $PKG_ROOT/.venv/bin/activate"
echo "  bash $PKG_ROOT/run.sh   # 跑基准"
