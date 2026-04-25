"""准确性评估 —— 跑 golden set 判定分类/实体/事实识别

只对 VLM 模型有意义（text_only 模型跳过）。

输出 JSON 字段：
  per_case: 每个 case 的判定明细
  aggregate: precision/recall/token 分布
  acceptance: 按 expectations.json 的 acceptance_criteria 判 PASS/FAIL
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from common import (
    InferResult,
    ModelConfig,
    VLM_PROMPT,
    infer_sync,
    summarize_latencies,
)

logger = logging.getLogger(__name__)


def _normalize(s) -> str:
    if not isinstance(s, str):
        return str(s)
    return s.replace(" ", "").replace(",", "").replace("，", "").replace("¥", "").lower()


def judge_case(case: dict, pred: InferResult) -> dict:
    """对单 case 打分，返回判定字典"""
    parsed = pred.parsed_json or {}

    category_correct = _normalize(parsed.get("category", "")) == _normalize(case["expected_category"])

    pred_blob = _normalize(json.dumps(parsed, ensure_ascii=False))
    entity_hits = sum(1 for e in case.get("must_identify_entities", []) if _normalize(e) in pred_blob)
    fact_hits = sum(1 for f in case.get("must_identify_facts", []) if _normalize(f[:8]) in pred_blob)

    # must_not_say 检查（红线，不可命中）
    must_not_violations = [
        f for f in case.get("must_not_say", []) if _normalize(f) in pred_blob
    ]

    # token 预算检查
    budget = case.get("token_budget", {})
    in_min, in_max = budget.get("input", [0, 99999])
    out_min, out_max = budget.get("output", [0, 99999])
    input_in_range = in_min <= pred.input_tokens <= in_max
    output_in_range = out_min <= pred.output_tokens <= out_max
    possibly_truncated = pred.looks_truncated(max_tokens_requested=800)

    return {
        "case_id": case["id"],
        "image": case["image"],
        "category_correct": category_correct,
        "entity_hits": entity_hits,
        "entity_total": len(case.get("must_identify_entities", [])),
        "fact_hits": fact_hits,
        "fact_total": len(case.get("must_identify_facts", [])),
        "must_not_violations": must_not_violations,
        "input_tokens": pred.input_tokens,
        "output_tokens": pred.output_tokens,
        "input_in_range": input_in_range,
        "output_in_range": output_in_range,
        "possibly_truncated": possibly_truncated,
        "latency_ms": pred.latency_ms,
        "ok": pred.ok,
        "error": pred.error,
        "predicted_category": parsed.get("category", ""),
        "predicted_description": parsed.get("description", "")[:100],
    }


def run_accuracy(
    model_cfg: ModelConfig,
    golden: dict,
    fixtures_dir: Path,
) -> dict:
    """跑完整 accuracy 评估"""
    if not model_cfg.is_vlm:
        logger.info("skip %s (task_type=text_only)", model_cfg.name)
        return {"skipped": True, "reason": "text_only model"}

    cases = golden.get("cases", [])
    per_case: list[dict] = []

    for i, case in enumerate(cases, 1):
        image_path = fixtures_dir / case["image"]
        if not image_path.exists():
            logger.warning("  [%d/%d] 跳过 %s：图片不存在", i, len(cases), case["image"])
            continue
        logger.info("  [%d/%d] %s → %s", i, len(cases), case["id"], model_cfg.name)
        pred = infer_sync(
            model_cfg,
            prompt=VLM_PROMPT,
            image_path=image_path,
            max_tokens=800,
        )
        per_case.append(judge_case(case, pred))

    # 聚合
    n = len(per_case) or 1
    correct = sum(1 for r in per_case if r["category_correct"])
    entity_hit_total = sum(r["entity_hits"] for r in per_case)
    entity_total = sum(r["entity_total"] for r in per_case) or 1
    fact_hit_total = sum(r["fact_hits"] for r in per_case)
    fact_total = sum(r["fact_total"] for r in per_case) or 1
    errors = sum(1 for r in per_case if not r["ok"])
    truncations = sum(1 for r in per_case if r["possibly_truncated"])
    violations = sum(len(r["must_not_violations"]) for r in per_case)

    input_tokens = [r["input_tokens"] for r in per_case if r["ok"]]
    output_tokens = [r["output_tokens"] for r in per_case if r["ok"]]
    latencies = [r["latency_ms"] for r in per_case if r["ok"]]

    aggregate = {
        "total_cases": n,
        "category_precision": correct / n,
        "entity_recall": entity_hit_total / entity_total if entity_total > 0 else 0.0,
        "fact_recall": fact_hit_total / fact_total if fact_total > 0 else 0.0,
        "error_rate": errors / n,
        "truncation_rate": truncations / n,
        "must_not_say_violations": violations,
        "input_tokens_stats": {
            "avg": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            "min": min(input_tokens) if input_tokens else 0,
            "max": max(input_tokens) if input_tokens else 0,
        },
        "output_tokens_stats": {
            "avg": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
            "min": min(output_tokens) if output_tokens else 0,
            "max": max(output_tokens) if output_tokens else 0,
        },
        "latency_stats_ms": summarize_latencies(latencies),
    }

    # 验收判定
    acc = golden.get("acceptance_criteria", {})
    verdict_reasons = []
    if aggregate["category_precision"] < acc.get("category_precision_min", 0.8):
        verdict_reasons.append(
            f"FAIL: 分类 precision {aggregate['category_precision']*100:.1f}% < "
            f"{acc.get('category_precision_min')*100:.0f}%"
        )
    if aggregate["entity_recall"] < acc.get("entity_recall_min", 0.6):
        verdict_reasons.append(
            f"FAIL: 实体 recall {aggregate['entity_recall']*100:.1f}% < "
            f"{acc.get('entity_recall_min')*100:.0f}%"
        )
    if aggregate["error_rate"] > acc.get("error_rate_max", 0.05):
        verdict_reasons.append(f"FAIL: 错误率 {aggregate['error_rate']*100:.1f}% 超标")
    if aggregate["must_not_say_violations"] > acc.get("must_not_say_violation_max", 0):
        verdict_reasons.append(
            f"FAIL: must_not_say 违反 {aggregate['must_not_say_violations']} 次（红线）"
        )
    if aggregate["truncation_rate"] > acc.get("output_token_truncation_rate_max", 0.1):
        verdict_reasons.append(f"WARN: 输出截断率 {aggregate['truncation_rate']*100:.1f}%")

    verdict = "FAIL" if any(r.startswith("FAIL") for r in verdict_reasons) else (
        "WARN" if verdict_reasons else "PASS"
    )

    return {
        "benchmark": "accuracy",
        "model": model_cfg.name,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "aggregate": aggregate,
        "per_case": per_case,
    }
