#!/usr/bin/env bash
# 顶层一键入口 — DGX 上解压 + bootstrap 后跑这个
#
# 前置（bootstrap.sh 已处理）：
#   - Python 3.10+ venv 激活 (.venv)
#   - vLLM 已装（离线 wheels）
#   - 模型权重已链到 HF_HOME cache
#   - ZeroTier 已组网
#
# 流程：
#   1. 检查 vLLM 可用
#   2. 启动 vLLM（默认 Qwen3-VL-8B-Instruct 端口 8001）
#   3. 等模型就绪
#   4. 跑 benchmark 矩阵
#   5. 收尾打印报告路径

set -euo pipefail
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "  vlm-llm-benchmark  (vLLM serving harness)"
echo "═══════════════════════════════════════════════"

# ── 1. 环境自检 ────────────────────────────────
echo
echo "[1/4] 环境自检"
if [[ -d .venv ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "  [info] 激活 .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
pip install -q -r requirements.txt
if ! python -c "import vllm" 2>/dev/null; then
  echo "  ✗ vLLM 未装，先跑 sudo bash scripts/bootstrap.sh"
  exit 1
fi
echo "  ✓ vLLM $(python -c 'import vllm; print(vllm.__version__)')"

# ── 2. 启动 vLLM ──────────────────────────────
echo
echo "[2/4] 启动 vLLM（默认 Qwen3-VL-8B-Instruct）"
echo "    → 要换模型，编辑 vllm_configs/start_all.sh 取消注释"
mkdir -p output/logs output/reports
bash vllm_configs/start_all.sh

# ── 3. 跑 benchmark ──────────────────────────
echo
echo "[3/4] 基准测试"
MODEL="${BENCHMARK_MODEL:-qwen3-vl-8b-instruct}"
SKIP="${BENCHMARK_SKIP:-stability}"  # stability 要 30 min，默认跳
python run_benchmark.py --model "$MODEL" --skip "$SKIP" || EXIT_CODE=$?
EXIT_CODE=${EXIT_CODE:-0}

# ── 4. 收尾 ──────────────────────────────────
echo
echo "[4/4] 完成"
echo "═══════════════════════════════════════════════"
echo "  报告: $(pwd)/output/reports/"
echo "  退出码: $EXIT_CODE  (0=PASS, 1=WARN, 2=FAIL)"
echo "═══════════════════════════════════════════════"
echo
echo "其他用法："
echo "  # 换模型跑："
echo "    python run_benchmark.py --model qwen2.5-vl-7b-fp16"
echo "    python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8"
echo
echo "  # 含 30 min 稳定性："
echo "    python run_benchmark.py --model qwen3-vl-8b-instruct"
echo
echo "  # 停所有 vLLM："
echo "    bash vllm_configs/launch_helpers.sh stop"

exit "$EXIT_CODE"
