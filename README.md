# vlm-llm-benchmark

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/vlm-llm-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/vlm-llm-benchmark/actions/workflows/ci.yml)

A small, reproducible benchmark harness for evaluating **VLM** (vision-language) and **LLM** (text) models served via **vLLM** — across 6 dimensions on a single high-end GPU node.

Built for the question: *"Can model X replace model Y in production without quality regression?"*

---

## What it measures

| Dimension | What | Why it matters |
|---|---|---|
| **Accuracy** | Classification precision, entity recall, fact recall, **must-not-say violations** against a golden set | Catches digit-shift errors (e.g. ¥120 vs ¥1200) that pure perplexity misses |
| **TTFT** | First-token latency P50 / P95 (streaming) | UX baseline — anything > 2s feels broken |
| **Throughput** | Aggregate tokens-per-second under sustained load | Capacity planning |
| **Concurrency** | Success rate + P50/P95 across 1 / 5 / 10 / 30 / 50 concurrent requests | Production load shape |
| **Stability** | 30-min sustained run; latency drift between first 5 min and last 5 min | Memory leaks, KV-cache thrashing |
| **Token budget** | Input/output token distribution + truncation rate | Cost monitoring + silent truncation detection |

Pass/Warn/Fail is determined by thresholds in `golden/expectations.json::acceptance_criteria` — exit code `0` PASS / `1` WARN / `2` FAIL, ready for CI consumption.

---

## Reference model matrix (`models.yaml`)

The harness is provider-agnostic — anything serving an **OpenAI-compatible** endpoint works (vLLM, sglang, lmdeploy, llama.cpp server, Ollama 0.21+, …). Out of the box we ship a 4-model reference matrix:

| Role | Model | Quant | Port | VRAM | Min HW |
|---|---|---|---|---|---|
| 🌟 VLM primary | Qwen3-VL-8B-Instruct | BF16 | 8001 | 20 GB | A100-40G |
| 📍 VLM baseline | Qwen2.5-VL-7B-Instruct | BF16 | 8002 | 18 GB | A100-40G |
| 🌟 LLM primary | Qwen3-30B-A3B-Instruct-2507-FP8 (MoE) | FP8 | 9001 | 35 GB | H100-80G |
| 🌟🌟🌟 LLM flagship | Qwen3-235B-A22B-Instruct-2507-FP8 (MoE) | FP8 | 9002 | 240 GB | 8×H100-80G |

**No-downgrade design**: if you have DGX-class hardware, run real models. The flagship MoE entries activate only ~3B / ~22B params per forward pass, so they're competitive with much smaller dense models on latency while preserving quality.

Drop in your own models by appending to `models.yaml` — only `(name, hf_repo, port, role)` are required; other fields are documentation hints.

---

## Quick start

### Prerequisites

- Linux (Ubuntu 22.04 or 24.04 tested) with **CUDA-capable GPU**
- Python 3.10+
- ~50 GB free disk for the default 80-GB-of-models matrix (or 16 GB for the minimal set)

### 3-step deploy

```bash
# 1. On a machine with internet — download all artifacts
git clone https://github.com/qiurui144/vlm-llm-benchmark.git
cd vlm-llm-benchmark
MODEL_SET=standard bash scripts/prepare_offline.sh
# MODEL_SET options:
#   minimal  (~16 GB) — VLM primary only
#   standard (~80 GB) — VLM ×2 + LLM-30B  [recommended]
#   full    (~320 GB) — all 4 models including 235B

# 2. (Optional) bundle for offline transfer to an air-gapped GPU host
tar czf vlm-llm-benchmark-bundle.tar.gz vlm-llm-benchmark/
scp vlm-llm-benchmark-bundle.tar.gz dgx:/data/

# 3. On the GPU host
cd /path/to/vlm-llm-benchmark
sudo bash scripts/bootstrap.sh   # installs vLLM, links models to HF cache
bash run.sh                      # default: VLM primary, skips 30-min stability
```

### Targeted runs

```bash
# Replace baseline with candidate — the core "can X replace Y?" question
bash vllm_configs/start_all.sh   # uncomment baseline in start_all.sh first
python run_benchmark.py --model qwen2.5-vl-7b-fp16  --skip stability
python run_benchmark.py --model qwen3-vl-8b-instruct --skip stability
cat output/reports/matrix_*.md

# LLM concurrency sweep only
python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,stability

# Flagship 235B smoke test (needs 8×H100)
python run_benchmark.py --model qwen3-235b-a22b-instruct-2507-fp8 \
    --skip concurrency,stability
```

---

## Bring your own data

The repo ships **no fixture images** by design — VLM benchmarks need real-world screenshots / scans / photos that often contain PII. See [`fixtures/README.md`](fixtures/README.md) for guidance on:

- What images go where (one per `golden/expectations.json::cases[].image`)
- How to author your own golden-set entries (`must_identify_entities`, `must_identify_facts`, **`must_not_say`**)
- Why `.gitignore` excludes binary fixtures by default

The shipped `golden/expectations.json` is a synthetic 9-case demo. Replace it with your own ground truth to evaluate against your domain.

---

## Repository layout

```
vlm-llm-benchmark/
├── run.sh                    # one-liner entry point
├── run_benchmark.py          # main scheduler
├── models.yaml               # model matrix (edit this to add/remove models)
├── common.py                 # vLLM client + shared utilities
├── requirements.txt          # httpx / pyyaml / Pillow / pynvml
├── benchmark/
│   ├── accuracy.py           # golden-set driven accuracy
│   └── performance.py        # TTFT / throughput / concurrency / stability
├── vllm_configs/
│   ├── launch_helpers.sh     # vllm serve helper functions
│   └── start_all.sh          # batch model startup (default: VLM primary only)
├── scripts/
│   ├── prepare_offline.sh    # internet host: pull wheels + models
│   ├── bootstrap.sh          # GPU host: install vLLM, link models
│   └── setup_zerotier.sh     # OPTIONAL: ZeroTier VPN for remote deploy
├── fixtures/
│   └── README.md             # bring-your-own-data guide
├── golden/
│   └── expectations.json     # acceptance criteria + demo cases
└── .github/workflows/ci.yml  # lint / syntax / shellcheck
```

---

## Optional: deploying to a remote air-gapped GPU host via ZeroTier

If you want to ship to a remote DGX through a flat L2 VPN, the bundled `scripts/setup_zerotier.sh` automates ZeroTier install and joining a network you've created at [my.zerotier.com](https://my.zerotier.com):

```bash
ZEROTIER_NETWORK_ID=<your-16-hex-id> sudo -E bash scripts/setup_zerotier.sh
# Then approve the new node at https://my.zerotier.com/network/<your-id>
```

This is entirely optional — direct SSH or `scp` works just as well.

---

## FAQ

**Q: Why both VLM and LLM in one repo?**
A: Many teams replace both at the same major model upgrade (e.g. Qwen2.5 → Qwen3). Keeping them in one harness lets you compare against a shared baseline and run cross-modal accept criteria.

**Q: Why vLLM specifically?**
A: It's the de-facto OpenAI-compatible serving stack with strong continuous-batching, paged-attention, and FP8 / AWQ support. The harness itself only talks HTTP, so you can point it at any compatible endpoint — but the launch scripts assume vLLM.

**Q: Can I run on consumer GPUs (4090 / 3090 / 7900 XT)?**
A: The 235B flagship — no. The 30B-A3B FP8 — barely (RTX 4090 is 24 GB, FP8 wants ~35). The 8B VLMs — yes, with care (bf16 → fp16). Adjust `models.yaml::quantization` and `dtype` accordingly.

**Q: My model is OpenAI-compatible but not on HuggingFace.**
A: Set `hf_repo: null` in `models.yaml` and skip `prepare_offline.sh` — point `start_all.sh` directly at your endpoint URL.

**Q: How do I add a new benchmark dimension?**
A: New file under `benchmark/`, register in `run_benchmark.py::BENCHMARKS`, add thresholds to `models.yaml::benchmarks`. See `CONTRIBUTING.md`.

---

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Interactions in this project are governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

The maintainers prefer small, focused PRs over sweeping refactors. New model adapters, hardware configs, and benchmark dimensions are especially welcome. **Never commit real PII to fixtures/.**

## License

[Apache License 2.0](LICENSE)

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) — the serving stack that makes this all reasonable
- [Qwen](https://github.com/QwenLM/Qwen3) — the reference model family used in the default matrix
- [HuggingFace Hub](https://huggingface.co) — model distribution
