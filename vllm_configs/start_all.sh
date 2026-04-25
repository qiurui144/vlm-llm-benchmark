#!/usr/bin/env bash
# 按 models.yaml 启动基准模型
#
# 默认只启动 VLM 首选（Qwen3-VL-8B-Instruct）。其他模型按需取消注释。
# 多模型同机并行：确保显存够（A100-80G 可以同时跑 VLM+LLM-30B）

set -euo pipefail
cd "$(dirname "$0")"
source ./launch_helpers.sh || true

# ═══ VLM ═══

# 🌟 VLM 首选：Qwen3-VL-8B-Instruct (BF16, ~20GB)
_launch "Qwen/Qwen3-VL-8B-Instruct"              8001  ""    8192  bfloat16

# 📍 VLM 备选基线：Qwen2.5-VL-7B-Instruct (BF16, ~18GB)
# _launch "Qwen/Qwen2.5-VL-7B-Instruct"            8002  ""    8192  bfloat16

# ═══ LLM (text-only) ═══

# 🌟 LLM 首选：Qwen3-30B-A3B-Instruct-2507-FP8 (MoE 激活 3B, ~35GB)
# _launch "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"   9001  fp8   16384 auto

# 🌟🌟🌟 LLM 旗舰：Qwen3-235B-A22B-Instruct-2507-FP8 (需 8×H100, ~240GB)
# _launch "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" 9002  fp8   16384 auto

echo
echo "═════════════════════════════════════════════"
echo "  默认仅启动 Qwen3-VL-8B-Instruct（端口 8001）"
echo "  需启多模型请编辑本文件取消注释"
echo "═════════════════════════════════════════════"
