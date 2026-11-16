# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-16T10:52:43.268990Z
- **Wall clock time**: 0.5s
- **Pipeline config hash**: `4d240079b8078b6c`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 350 | 350 | — | 0 | 350 |
| SchemaGate | 350 | 289 | 0 | 61 | 289 |
| ExactDeduplicator | 289 | 289 | — | — | 289 |
| MinHashDeduplicator | 289 | 289 | — | — | 289 |
| TextCleaner | 289 | 289 | — | — | 289 |
| AlpacaExporter | — | 289 | — | — | 289 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 350/0 passed (N/A), 0 rejected
- **SchemaGate**: 289/350 passed (82.6%), 61 rejected


**Rejection reasons:**
  - `below_min_tokens:9`: 32
  - `below_min_tokens:8`: 12
  - `below_min_tokens:6`: 11
  - `below_min_tokens:7`: 6

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 289

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
