# Fixtures

This directory is intentionally empty in the upstream repo. **You bring your own data.**

## Why empty

VLM benchmarks need real-world images (chat screenshots, scanned forms, ID photos, contract pages, etc.) — but those images often contain PII. Shipping them in a public repo is a privacy violation. So this repo only ships the **methodology** (`golden/expectations.json`) and **harness** (`benchmark/accuracy.py`), and you provide the actual images.

## What goes here

For each entry in `golden/expectations.json::cases`, place a JPEG/PNG file matching its `image` field. With the demo expectations file shipped:

```
fixtures/
├── 0.jpg   ← chat_01_transfer_1200
├── 1.jpg   ← chat_02
├── 2.jpg   ← chat_03
├── 3.jpg   ← chat_04
├── 4.jpg   ← chat_05_with_transfer
├── 5.jpg   ← chat_06_with_transfer
├── 10.jpg  ← chat_07_dense
├── 11.jpg  ← chat_08
└── 12.jpg  ← chat_09
```

## How to author your own golden set

1. Collect 5–20 real images representative of your target task.
2. For each, decide what the "correct" model output should contain:
   - `must_identify_entities` — names / amounts / orgs the model **must** mention.
   - `must_identify_facts` — semantic facts (e.g. "reconciliation", "balance_diff_1200").
   - `must_not_say` — wrong outputs that look plausible (e.g. ¥120 vs ¥1200 — digit shift; FAIL if model says any of these).
3. Edit `golden/expectations.json` accordingly. The harness (`benchmark/accuracy.py`) will compute `category_precision`, `entity_recall`, `fact_recall`, `must_not_say_violation_count` against your ground truth.

## Privacy reminder

`.gitignore` excludes `fixtures/*.jpg` `fixtures/*.png` etc. by default — never commit real PII to a public fork.
