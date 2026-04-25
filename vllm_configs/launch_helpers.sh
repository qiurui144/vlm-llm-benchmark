#!/usr/bin/env bash
# vLLM 启动辅助函数
# 每个模型一个脚本，共用这里的 _launch 函数
#
# 前置：
#   pip install vllm>=0.7.0  # FP8/FP4 需要 >= 0.7
#   export HF_HOME=/data/hf_cache  # 模型缓存目录（DGX 需足够空间）
#
# 端口分配（见 models.yaml）：
#   8001-8007  VLM
#   9001-9003  LLM text-only

set -euo pipefail

_launch() {
  local model_repo="$1"
  local port="$2"
  local quantization="${3:-}"           # 空=原生 / fp8 / fp4 / awq_marlin
  local max_model_len="${4:-8192}"
  local dtype="${5:-auto}"               # auto / half / bfloat16 / fp8

  local args=(
    --model "$model_repo"
    --port "$port"
    --host 0.0.0.0
    --max-model-len "$max_model_len"
    --trust-remote-code
    --gpu-memory-utilization 0.85
  )

  if [[ -n "$quantization" ]]; then
    args+=(--quantization "$quantization")
  fi
  if [[ "$dtype" != "auto" ]]; then
    args+=(--dtype "$dtype")
  fi

  # VLM 需要 limit_mm_per_prompt 参数（vLLM 0.6+）
  if [[ "$model_repo" == *"VL"* || "$model_repo" == *"vl"* ]]; then
    args+=(--limit-mm-per-prompt '{"image":3}')
  fi

  echo "[vllm] 启动 $model_repo → port $port"
  echo "       args: ${args[*]}"
  nohup vllm serve "${args[@]}" \
    > "$(dirname "$0")/../output/logs/vllm_${port}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$(dirname "$0")/../output/logs/vllm_${port}.pid"
  echo "[vllm] PID=$pid, 日志: output/logs/vllm_${port}.log"
  echo "[vllm] 等待模型加载（约 30-180 秒取决于模型大小 + GPU）..."

  # 等就绪
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
      echo "[vllm] ✓ 就绪 (${i}×5s=$(( i*5 ))s)"
      return 0
    fi
    sleep 5
  done
  echo "[vllm] ✗ 超时 5 分钟未就绪，检查日志"
  return 1
}

_stop_all() {
  for pidfile in $(dirname "$0")/../output/logs/vllm_*.pid; do
    [[ -f "$pidfile" ]] || continue
    local pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[vllm] stop PID=$pid"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
  # 兜底
  pkill -f "vllm serve" 2>/dev/null || true
}

case "${1:-}" in
  stop) _stop_all ;;
  *) echo "usage: $0 stop  (实际启动脚本见同目录其他 .sh)" ;;
esac
