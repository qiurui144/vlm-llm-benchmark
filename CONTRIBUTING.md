# Contributing to vlm-llm-benchmark

Thanks for your interest in improving this benchmark. This project aims to be a small, focused, reproducible harness — not a kitchen sink. Contributions that keep it that way are most welcome.

## What we want

- **New model adapters** in `models.yaml` (any OpenAI-compatible endpoint works out of the box)
- **New benchmark dimensions** beyond the 6 currently shipped (e.g. cost-per-1k-tokens, energy efficiency, retrieval RAG quality)
- **More golden-set patterns** in `golden/expectations.json` — we ship a small synthetic demo; reference patterns for new domains (medical, financial, legal, retail) are valuable
- **Bug fixes** with a regression test
- **Hardware-specific configs** in `vllm_configs/` (currently A100/H100 focused — Ada / MI300X / Habana welcome)

## What we don't want

- **Real PII in `fixtures/`**. Never commit real chat screenshots, ID photos, contracts, or any image with identifiable people. The repo's `.gitignore` excludes binary fixtures by default — keep it that way.
- **Vendor-locked code**. The harness talks OpenAI-compatible HTTP. Don't add hard dependencies on a specific provider's SDK in the core path.
- **Dependency creep**. `requirements.txt` is intentionally tiny (httpx + pyyaml + Pillow + pynvml). New deps need justification.

## Workflow

1. Open an issue first for non-trivial changes — a 5-line discussion can save a 500-line PR.
2. Fork, create a feature branch (`feat/short-description`), do your work.
3. Run `ruff check .` and `python -m py_compile $(git ls-files '*.py')` locally before pushing.
4. Open a PR. Describe **what** changed, **why**, and **how you tested** it.
5. Be patient — this is a side project for the maintainers; reviews may take a few days.

## Coding style

- Python: 4-space indent, `ruff` for lint, no unused imports, type hints on public functions.
- Shell: `set -euo pipefail`, prefer `shellcheck`-clean scripts.
- YAML: 2-space indent.
- Comments in English. Minimal — explain *why*, not *what* (the code already says what).

## Tests

- Every new benchmark dimension should ship a synthetic-data unit test in `tests/` (TBD — we'll add the harness when the first contribution arrives).
- For golden-set patterns, ship a synthetic example users can reproduce without your private data.

## DCO / Sign-off

Not required. A clear PR description and clean commits are enough.

## Security

If you find a security issue (e.g. a way the harness could leak credentials or write outside its working dir), please email the maintainer privately rather than opening a public issue. Contact info is in the repo profile.

## License

By contributing, you agree your contributions are licensed under the Apache License 2.0 (same as the project).
