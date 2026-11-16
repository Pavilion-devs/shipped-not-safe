---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 f7d11179d94e82ef4d74b634c8adea46ca32bd97425fa0a417af4618fcd9dffe
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---

# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-16T10:52:37.426414Z
- **Wall clock time**: 0.4s
- **Pipeline config hash**: `43c2c536a09832a0`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 300 | 300 | — | 0 | 300 |
| SchemaGate | 300 | 249 | 0 | 51 | 249 |
| ExactDeduplicator | 249 | 249 | — | — | 249 |
| MinHashDeduplicator | 249 | 249 | — | — | 249 |
| TextCleaner | 249 | 249 | — | — | 249 |
| AlpacaExporter | — | 249 | — | — | 249 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 300/0 passed (N/A), 0 rejected
- **SchemaGate**: 249/300 passed (83.0%), 51 rejected


**Rejection reasons:**
  - `below_min_tokens:9`: 36
  - `below_min_tokens:8`: 14
  - `below_min_tokens:7`: 1

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 249

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.29
- **python**: 3.12.3

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (249) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
