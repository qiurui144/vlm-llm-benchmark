"""主 benchmark 入口 —— 对 models.yaml 中声明的模型跑完整测试套件

使用：
  python run_benchmark.py --model qwen3-vl-8b-instruct
  python run_benchmark.py --model all              # 跑所有已启动的模型
  python run_benchmark.py --model qwen3-vl-8b-instruct \\
      --skip stability                             # 跳过 30 分钟稳定性

输出：
  output/reports/{model}_{timestamp}.json       # 机器可读
  output/reports/{model}_{timestamp}.md          # 人类可读
  output/reports/matrix_{timestamp}.md           # 所有模型对比表（all 模式）

退出码：
  0 全部 PASS
  1 有 WARN
  2 任一模型 FAIL
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from common import (
    ModelConfig,
    get_vram_info,
    load_benchmarks_config,
    load_models,
    wait_model_ready,
)

sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
from benchmark.accuracy import run_accuracy
from benchmark.performance import (
    run_concurrency,
    run_stability,
    run_throughput,
    run_ttft,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
GOLDEN = ROOT / "golden" / "expectations.json"
REPORTS = ROOT / "output" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def run_all_for_model(
    model_cfg: ModelConfig,
    golden: dict,
    skip: set[str],
) -> dict:
    """对单个模型跑全量 benchmark"""
    results: dict = {
        "model": model_cfg.name,
        "hf_repo": model_cfg.hf_repo,
        "quantization": model_cfg.quantization,
        "hardware_min": model_cfg.hardware_min,
        "timestamp": datetime.datetime.now().isoformat(),
        "vram_snapshot": get_vram_info(),
        "benchmarks": {},
    }

    # 预检：端点是否就绪
    logger.info("检查 %s (port %d) 就绪...", model_cfg.name, model_cfg.port)
    if not wait_model_ready(model_cfg, timeout_s=60.0):
        logger.error("  %s 未就绪，跳过", model_cfg.name)
        results["error"] = "model_not_ready"
        return results

    # 准确性
    if "accuracy" not in skip:
        logger.info("▶ accuracy")
        results["benchmarks"]["accuracy"] = run_accuracy(model_cfg, golden, FIXTURES)

    # TTFT
    if "ttft" not in skip:
        logger.info("▶ ttft")
        results["benchmarks"]["ttft"] = run_ttft(model_cfg, FIXTURES, samples=5)

    # 吞吐
    if "throughput" not in skip:
        logger.info("▶ throughput")
        results["benchmarks"]["throughput"] = run_throughput(
            model_cfg, FIXTURES, duration_s=60.0
        )

    # 并发
    if "concurrency" not in skip:
        logger.info("▶ concurrency")
        results["benchmarks"]["concurrency"] = run_concurrency(
            model_cfg, FIXTURES,
            concurrencies=[1, 5, 10, 30, 50],
            duration_s=60.0,
        )

    # 稳定性（默认 30min 太长，加 skip 开关）
    if "stability" not in skip:
        logger.info("▶ stability (30 min)")
        results["benchmarks"]["stability"] = run_stability(
            model_cfg, FIXTURES, duration_s=1800.0, sample_interval_s=5.0
        )

    results["vram_after"] = get_vram_info()
    return results


def render_markdown(result: dict) -> str:
    """单模型报告 Markdown"""
    m = result["model"]
    bm = result.get("benchmarks", {})
    acc = bm.get("accuracy") or {}
    ttft = bm.get("ttft") or {}
    tp = bm.get("throughput") or {}
    con = bm.get("concurrency") or {}
    stab = bm.get("stability") or {}
    vram = result.get("vram_after", {})

    lines = [
        f"# {m} Benchmark",
        "",
        f"- HF repo: `{result.get('hf_repo','')}`",
        f"- Quantization: `{result.get('quantization')}`",
        f"- Hardware min: {result.get('hardware_min','')}",
        f"- Time: {result.get('timestamp','')}",
        f"- VRAM after run: {vram.get('used_mb','?')}MB / {vram.get('total_mb','?')}MB"
        if vram else "- VRAM: 未采集",
        "",
        "## 准确性",
        "",
    ]
    if acc.get("skipped"):
        lines.append(f"- SKIPPED: {acc.get('reason')}")
    elif acc:
        agg = acc.get("aggregate", {})
        lines += [
            f"- **判定**: {acc.get('verdict','?')}",
            f"- 分类 precision: {agg.get('category_precision',0)*100:.1f}%",
            f"- 实体 recall: {agg.get('entity_recall',0)*100:.1f}%",
            f"- 事实 recall: {agg.get('fact_recall',0)*100:.1f}%",
            f"- must_not_say 违反: {agg.get('must_not_say_violations',0)}",
            f"- 输出 token 平均: {agg.get('output_tokens_stats',{}).get('avg',0):.0f}",
            f"- 截断率: {agg.get('truncation_rate',0)*100:.1f}%",
            f"- 错误率: {agg.get('error_rate',0)*100:.1f}%",
        ]
        for r in acc.get("verdict_reasons", []):
            lines.append(f"  - {r}")
    lines += ["", "## 首 Token (TTFT)", ""]
    if ttft:
        s = ttft.get("ttft_ms_stats", {})
        lines += [
            f"- P50 TTFT: {s.get('p50',0):.0f} ms",
            f"- P95 TTFT: {s.get('p95',0):.0f} ms",
            f"- 样本数: {ttft.get('samples',0)}  错误率: {ttft.get('error_rate',0)*100:.1f}%",
        ]
    lines += ["", "## 吞吐量", ""]
    if tp:
        lines += [
            f"- 聚合 TPS: **{tp.get('aggregate_tps',0):.1f}** tokens/s",
            f"- 每请求 P50 TPS: {tp.get('per_request_tps_stats',{}).get('p50',0):.1f}",
            f"- 请求数: {tp.get('requests',0)}  错误: {tp.get('errors',0)}",
            f"- 总输入 tokens: {tp.get('total_input_tokens',0)}",
            f"- 总输出 tokens: {tp.get('total_output_tokens',0)}",
        ]
    lines += ["", "## 并发稳定性", ""]
    if con and con.get("steps"):
        lines += ["| 并发 | 成功率 | P50 ms | P95 ms | 聚合 TPS |", "|---|---|---|---|---|"]
        for step in con["steps"]:
            s = step.get("latency_stats_ms", {})
            lines.append(
                f"| {step['concurrency']} "
                f"| {step['success_rate']*100:.1f}% "
                f"| {s.get('p50',0):.0f} | {s.get('p95',0):.0f} "
                f"| {step.get('aggregate_tps',0):.1f} |"
            )
    lines += ["", "## 长时间稳定性（30 min）", ""]
    if stab:
        lines += [
            f"- 判定: {stab.get('drift_verdict','?')}",
            f"- 前 5min P95: {stab.get('first_5min_p95_ms',0):.0f} ms",
            f"- 最后 5min P95: {stab.get('last_5min_p95_ms',0):.0f} ms",
            f"- 漂移比: {stab.get('latency_drift_ratio',1):.2f}×",
            f"- 错误率: {stab.get('error_rate',0)*100:.1f}%",
            f"- 样本数: {stab.get('total_samples',0)}",
        ]
    return "\n".join(lines)


def render_matrix(all_results: list[dict]) -> str:
    """多模型横向对比"""
    lines = [
        "# 模型矩阵对比",
        "",
        "| Model | 分类 precision | 实体 recall | TTFT P95 | TPS | 并发 50 成功率 | 稳定性漂移 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        name = r["model"]
        bm = r.get("benchmarks", {})
        acc = bm.get("accuracy", {}).get("aggregate", {}) if bm.get("accuracy") else {}
        ttft_p95 = bm.get("ttft", {}).get("ttft_ms_stats", {}).get("p95", 0)
        tps = bm.get("throughput", {}).get("aggregate_tps", 0)
        con_steps = bm.get("concurrency", {}).get("steps", [])
        con_50 = next((s for s in con_steps if s.get("concurrency") == 50), {})
        drift = bm.get("stability", {}).get("latency_drift_ratio", 0)

        lines.append(
            f"| {name} "
            f"| {acc.get('category_precision',0)*100:.1f}% "
            f"| {acc.get('entity_recall',0)*100:.1f}% "
            f"| {ttft_p95:.0f}ms "
            f"| {tps:.1f} "
            f"| {con_50.get('success_rate',0)*100:.1f}% "
            f"| {drift:.2f}× |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="模型名（见 models.yaml）或 all")
    parser.add_argument("--skip", default="",
                        help="逗号分隔跳过的 benchmark: accuracy,ttft,throughput,concurrency,stability")
    parser.add_argument("--golden", default=str(GOLDEN))
    args = parser.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    models = load_models(ROOT / "models.yaml")
    if args.model != "all":
        models = [m for m in models if m.name == args.model]
        if not models:
            logger.error("未知模型: %s。可选: %s", args.model,
                         [m.name for m in load_models(ROOT / 'models.yaml')])
            return 2

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: list[dict] = []
    has_fail = False
    has_warn = False

    for m in models:
        logger.info("═══ %s ═══", m.name)
        result = run_all_for_model(m, golden, skip)
        all_results.append(result)

        # 判定
        acc = result.get("benchmarks", {}).get("accuracy", {})
        if acc.get("verdict") == "FAIL":
            has_fail = True
        elif acc.get("verdict") == "WARN":
            has_warn = True

        # 保存单模型报告
        stem = f"{m.name}_{timestamp}"
        (REPORTS / f"{stem}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=_default),
            encoding="utf-8",
        )
        (REPORTS / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
        logger.info("报告保存: %s", REPORTS / f"{stem}.md")

    # 矩阵报告
    if len(all_results) > 1:
        matrix_md = render_matrix(all_results)
        (REPORTS / f"matrix_{timestamp}.md").write_text(matrix_md, encoding="utf-8")
        print("\n" + matrix_md)

    return 2 if has_fail else (1 if has_warn else 0)


if __name__ == "__main__":
    sys.exit(main())
