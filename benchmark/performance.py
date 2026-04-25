"""性能 benchmark — TTFT / 吞吐 / 并发稳定性 / 长时间稳定性

每个 benchmark 独立函数，可单独调用也可被 run_all 串跑。
"""

from __future__ import annotations

import asyncio
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Optional

import httpx

from common import (
    ModelConfig,
    TEXT_PROMPT,
    VLM_PROMPT,
    encode_image_data_url,
    infer_async,
    infer_stream,
    infer_sync,
    summarize_latencies,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 1. TTFT（首 token 延迟）—— 用流式请求
# ────────────────────────────────────────────────────────────


def run_ttft(
    model_cfg: ModelConfig,
    fixtures_dir: Path,
    samples: int = 5,
) -> dict:
    """每模型跑 samples 次流式请求，统计 TTFT P50/P95"""
    image_files = sorted(fixtures_dir.glob("*.jpg")) if model_cfg.is_vlm else []
    ttfts: list[float] = []
    total_lats: list[float] = []
    errors = 0

    for i in range(samples):
        image = image_files[i % len(image_files)] if image_files else None
        result = infer_stream(
            model_cfg,
            prompt=VLM_PROMPT if model_cfg.is_vlm else TEXT_PROMPT + "请简述劳动合同法第三十九条",
            image_path=image,
            max_tokens=200,   # 短输出让 TTFT 更纯粹
        )
        if result.ok and result.ttft_ms > 0:
            ttfts.append(result.ttft_ms)
            total_lats.append(result.latency_ms)
        else:
            errors += 1
        logger.info("  [TTFT %d/%d] %s  ttft=%.0fms  total=%.0fms",
                    i + 1, samples, model_cfg.name, result.ttft_ms, result.latency_ms)

    return {
        "benchmark": "ttft",
        "model": model_cfg.name,
        "samples": samples,
        "ttft_ms_stats": summarize_latencies(ttfts),
        "total_latency_ms_stats": summarize_latencies(total_lats),
        "errors": errors,
        "error_rate": errors / samples if samples else 0,
    }


# ────────────────────────────────────────────────────────────
# 2. 吞吐（Throughput）—— 单并发持续 N 秒
# ────────────────────────────────────────────────────────────


def run_throughput(
    model_cfg: ModelConfig,
    fixtures_dir: Path,
    duration_s: float = 60.0,
) -> dict:
    """在 duration_s 内持续单并发请求，统计 output tokens 总数 + TPS"""
    image_files = sorted(fixtures_dir.glob("*.jpg")) if model_cfg.is_vlm else []

    deadline = time.monotonic() + duration_s
    total_output = 0
    total_input = 0
    latencies = []
    tps_per_req = []
    errors = 0
    n = 0

    while time.monotonic() < deadline:
        image = image_files[n % len(image_files)] if image_files else None
        result = infer_sync(
            model_cfg,
            prompt=VLM_PROMPT if model_cfg.is_vlm else TEXT_PROMPT + "解释合同违约金 vs 定金",
            image_path=image,
            max_tokens=400,
        )
        n += 1
        if result.ok:
            total_output += result.output_tokens
            total_input += result.input_tokens
            latencies.append(result.latency_ms)
            if result.tokens_per_sec > 0:
                tps_per_req.append(result.tokens_per_sec)
        else:
            errors += 1

    wall_duration_s = duration_s if n > 0 else 1
    avg_tps = total_output / wall_duration_s

    return {
        "benchmark": "throughput",
        "model": model_cfg.name,
        "duration_s": duration_s,
        "requests": n,
        "errors": errors,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "aggregate_tps": avg_tps,           # 整体吞吐 = output tokens / wall time
        "per_request_tps_stats": {
            "p50": statistics.median(tps_per_req) if tps_per_req else 0,
            "p95": sorted(tps_per_req)[int(len(tps_per_req) * 0.95)] if tps_per_req else 0,
        },
        "latency_stats_ms": summarize_latencies(latencies),
    }


# ────────────────────────────────────────────────────────────
# 3. 并发稳定性（Concurrency）—— 阶梯并发
# ────────────────────────────────────────────────────────────


async def _concurrency_probe(
    model_cfg: ModelConfig,
    fixtures_dir: Path,
    concurrency: int,
    duration_s: float,
) -> dict:
    """指定并发数持续 duration_s 秒"""
    image_files = sorted(fixtures_dir.glob("*.jpg")) if model_cfg.is_vlm else []
    deadline = time.monotonic() + duration_s
    results: list = []

    async with httpx.AsyncClient(timeout=300.0,
                                  limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        async def worker(worker_id: int):
            i = 0
            while time.monotonic() < deadline:
                image = image_files[(worker_id + i) % max(len(image_files), 1)] if image_files else None
                r = await infer_async(client, model_cfg,
                                      prompt=VLM_PROMPT if model_cfg.is_vlm else TEXT_PROMPT + "列举诉讼时效例外",
                                      image_path=image,
                                      max_tokens=300)
                results.append(r)
                i += 1
                # 小随机延迟避免同步洪峰
                await asyncio.sleep(random.uniform(0.01, 0.05))

        await asyncio.gather(*[worker(w) for w in range(concurrency)])

    ok = sum(1 for r in results if r.ok)
    latencies_ok = [r.latency_ms for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in results if r.ok)

    return {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "total_requests": len(results),
        "success": ok,
        "success_rate": ok / len(results) if results else 0,
        "aggregate_tps": out_tokens / duration_s,
        "latency_stats_ms": summarize_latencies(latencies_ok),
    }


def run_concurrency(
    model_cfg: ModelConfig,
    fixtures_dir: Path,
    concurrencies: list[int] = (1, 5, 10, 30, 50),
    duration_s: float = 60.0,
) -> dict:
    """阶梯并发，每个并发度跑 duration_s 秒"""
    steps = []
    for c in concurrencies:
        logger.info("  [并发 %d] %s...", c, model_cfg.name)
        step_result = asyncio.run(_concurrency_probe(model_cfg, fixtures_dir, c, duration_s))
        steps.append(step_result)

    return {
        "benchmark": "concurrency",
        "model": model_cfg.name,
        "steps": steps,
    }


# ────────────────────────────────────────────────────────────
# 4. 长时间稳定性 —— N 分钟持续跑，观察延迟漂移
# ────────────────────────────────────────────────────────────


def run_stability(
    model_cfg: ModelConfig,
    fixtures_dir: Path,
    duration_s: float = 1800.0,   # 30 分钟
    sample_interval_s: float = 5.0,
) -> dict:
    """连续 duration_s 秒每 sample_interval_s 发一次请求

    判定：前 5 分钟 vs 最后 5 分钟 P95 延迟的比值 < 1.30 → PASS
    """
    image_files = sorted(fixtures_dir.glob("*.jpg")) if model_cfg.is_vlm else []
    deadline = time.monotonic() + duration_s
    samples: list[dict] = []
    n = 0

    while time.monotonic() < deadline:
        image = image_files[n % max(len(image_files), 1)] if image_files else None
        result = infer_sync(
            model_cfg,
            prompt=VLM_PROMPT if model_cfg.is_vlm else TEXT_PROMPT + "评估合同违约",
            image_path=image,
            max_tokens=200,
        )
        samples.append({
            "ts_offset_s": time.monotonic() - (deadline - duration_s),
            "ok": result.ok,
            "latency_ms": result.latency_ms,
            "output_tokens": result.output_tokens,
        })
        n += 1
        await_time = sample_interval_s - (time.monotonic() % sample_interval_s)
        time.sleep(max(0.1, await_time) if await_time < sample_interval_s else sample_interval_s)

    # 前 5 min vs 最后 5 min
    first_window_end = 300.0
    last_window_start = duration_s - 300.0
    first_lats = [s["latency_ms"] for s in samples
                  if s["ok"] and s["ts_offset_s"] <= first_window_end]
    last_lats = [s["latency_ms"] for s in samples
                 if s["ok"] and s["ts_offset_s"] >= last_window_start]

    first_p95 = summarize_latencies(first_lats)["p95"]
    last_p95 = summarize_latencies(last_lats)["p95"]
    drift_ratio = (last_p95 / first_p95) if first_p95 > 0 else 1.0

    total_errors = sum(1 for s in samples if not s["ok"])

    return {
        "benchmark": "stability",
        "model": model_cfg.name,
        "duration_s": duration_s,
        "total_samples": len(samples),
        "errors": total_errors,
        "error_rate": total_errors / len(samples) if samples else 0,
        "first_5min_p95_ms": first_p95,
        "last_5min_p95_ms": last_p95,
        "latency_drift_ratio": drift_ratio,
        "drift_verdict": "PASS" if drift_ratio < 1.30 else "WARN",
        "all_samples": samples,  # 大对象，生成报告时可选择裁剪
    }
