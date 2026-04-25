"""vLLM OpenAI 兼容客户端 + 共享工具

所有 benchmark 脚本共用的：
- 调用 vLLM 的同步 / 异步客户端（带 VL 图片输入）
- 获取 usage（prompt_tokens / completion_tokens）
- 流式捕获 TTFT
- 从 models.yaml 加载配置

依赖：openai>=1.40 (chat.completions.create with stream=True)
    httpx
    pyyaml
    pynvml（可选，不装就跳 VRAM 监控）
    Pillow（图片 base64）
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 模型配置
# ────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    name: str
    hf_repo: str
    port: int
    vram_estimate_gb: float
    role: str
    quantization: Optional[str] = None
    hardware_min: str = "A100-40G"
    task_type: str = "vlm"             # vlm | text_only
    notes: str = ""

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def is_vlm(self) -> bool:
        return self.task_type != "text_only"


def load_models(yaml_path: Path | str = "models.yaml") -> list[ModelConfig]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [ModelConfig(**m) for m in data.get("models", [])]


def load_benchmarks_config(yaml_path: Path | str = "models.yaml") -> dict:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", {})


# ────────────────────────────────────────────────────────────
# 请求 payload 构造
# ────────────────────────────────────────────────────────────


def encode_image_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{b64}"


VLM_PROMPT = """你是法律证据分类助手。分析图片内容，严格按 JSON 格式输出（只输出 JSON）：

{
  "category": "communication|financial|contract|identity|legal_document|receipt|other",
  "subcategory": "具体子类（如 wechat_chat_with_transfer / bank_statement 等）",
  "description": "一句话描述图片内容",
  "key_entities": ["人名/金额/日期/银行卡号等关键实体"],
  "key_facts": ["2-5 条关键事实"]
}"""


TEXT_PROMPT = """你是法律文本分析助手。简要分析以下内容（200 字以内），输出 JSON：

{
  "summary": "一句话核心内容",
  "entities": ["人名/金额/日期"],
  "legal_risks": ["可能的法律风险点"]
}

内容："""


# ────────────────────────────────────────────────────────────
# 同步客户端（用 httpx 直调，避免 openai SDK 的额外开销）
# ────────────────────────────────────────────────────────────


@dataclass
class InferResult:
    model: str
    ok: bool = False
    error: str = ""
    # 输出
    content: str = ""
    parsed_json: Optional[dict] = None
    # Token
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    # 时延（ms）
    latency_ms: float = 0.0
    ttft_ms: float = 0.0               # 流式场景
    # 吞吐
    tokens_per_sec: float = 0.0        # output_tokens / (latency_ms / 1000)

    def looks_truncated(self, max_tokens_requested: int) -> bool:
        """判断是否被截断（eval_count 接近 max_tokens）"""
        return (
            self.finish_reason == "length"
            or (max_tokens_requested > 0 and self.output_tokens >= max_tokens_requested * 0.95)
        )


def infer_sync(
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout_s: float = 600.0,
) -> InferResult:
    """单次同步推理（非流式），返回 token 统计 + 延迟 + 解析内容"""
    messages: list[dict] = []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.hf_repo,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    try:
        r = httpx.post(url, json=payload, timeout=timeout_s, headers={"Authorization": "Bearer EMPTY"})
        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    if r.status_code != 200:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"HTTP {r.status_code}: {r.text[:200]}",
            latency_ms=elapsed,
        )

    data = r.json()
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {}) or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    # 尝试解析 JSON（失败不算致命）
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0

    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=content,
        parsed_json=parsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=choice.get("finish_reason", ""),
        latency_ms=elapsed,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 流式客户端（测 TTFT）
# ────────────────────────────────────────────────────────────


def infer_stream(
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout_s: float = 600.0,
) -> InferResult:
    """流式推理，返回 TTFT + 总延迟 + usage"""
    messages: list[dict] = []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.hf_repo,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},   # vLLM 流式里带 usage
    }
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    ttft_ms = 0.0
    content_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    finish_reason = ""
    first_token_seen = False

    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout_s,
                         headers={"Authorization": "Bearer EMPTY", "Accept": "text/event-stream"}) as r:
            if r.status_code != 200:
                return InferResult(
                    model=model_cfg.name,
                    ok=False,
                    error=f"HTTP {r.status_code}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk_data = line[5:].strip()
                if chunk_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue
                # 捕获 TTFT
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content", "")
                    if piece and not first_token_seen:
                        ttft_ms = (time.monotonic() - t0) * 1000
                        first_token_seen = True
                    if piece:
                        content_parts.append(piece)
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                # usage 在最后一条 chunk（include_usage）
                usage = chunk.get("usage")
                if usage:
                    input_tokens = int(usage.get("prompt_tokens", 0))
                    output_tokens = int(usage.get("completion_tokens", 0))

        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    content = "".join(content_parts)
    # 解析 JSON（同 sync）
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0

    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=content,
        parsed_json=parsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        latency_ms=elapsed,
        ttft_ms=ttft_ms,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 异步客户端（并发测试）
# ────────────────────────────────────────────────────────────


async def infer_async(
    client: httpx.AsyncClient,
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> InferResult:
    """异步版本，供并发测试调用"""
    messages: list[dict] = []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.hf_repo,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload,
                              headers={"Authorization": "Bearer EMPTY"})
        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    if r.status_code != 200:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"HTTP {r.status_code}",
            latency_ms=elapsed,
        )
    data = r.json()
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {}) or {}
    output_tokens = int(usage.get("completion_tokens", 0))
    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0
    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=choice.get("message", {}).get("content", ""),
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=output_tokens,
        finish_reason=choice.get("finish_reason", ""),
        latency_ms=elapsed,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 健康检查
# ────────────────────────────────────────────────────────────


def wait_model_ready(model_cfg: ModelConfig, timeout_s: float = 300.0) -> bool:
    """等 vLLM serve 的 /v1/models 返回 200"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{model_cfg.base_url}/models", timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ────────────────────────────────────────────────────────────
# VRAM 监控（可选）
# ────────────────────────────────────────────────────────────


def get_vram_info(device_index: int = 0) -> dict:
    """使用 pynvml 读 GPU 显存；pynvml 未装则返回空 dict"""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "used_mb": mem.used // (1024**2),
            "total_mb": mem.total // (1024**2),
            "used_ratio": mem.used / mem.total,
        }
    except Exception:
        return {}


# ────────────────────────────────────────────────────────────
# 统计工具
# ────────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def summarize_latencies(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    s = sorted(latencies_ms)
    return {
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "count": len(s),
    }
