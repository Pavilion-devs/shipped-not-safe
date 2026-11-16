---
license: cc-by-4.0
language: [en]
tags: [safety, refusal, over-refusal, quantization, gguf, tiny-aya, lexsi-hackathon, auditkit, curatorkit]
pretty_name: "Shipped != Safe: Tiny Aya quantization safety audit"
---

# Shipped ≠ Safe: eval sets and full run logs

Lexsi AI Safety & Alignment Hackathon, Track 2a. Does Tiny Aya Global's safety behaviour survive the
low-bit GGUF files people actually deploy?

## Contents

- `eval/harmful.jsonl`: 249 prompts a safe model should refuse (JailbreakBench JBB-Behaviors harmful, MIT; XSTest contrast/unsafe, CC-BY-4.0)
- `eval/benign.jsonl`: 289 borderline-but-harmless prompts; refusing them is over-refusal (XSTest safe, CC-BY-4.0; JBB-Behaviors benign, MIT)
- `eval/gsm8k.jsonl`: 100 GSM8K test questions (MIT), capability check
- `eval/_work/curated_*`: CuratorKIT outputs (clean, exact + MinHash dedup, schema gate) with manifest, checksums, provenance and rejected rows
- `results/runs/<variant>/<set>.json`: AuditKIT RunResults with **every generation** and per-sample scores
- `results/results.md` / `results.json`: paired-bootstrap deltas vs the official Q4_K_M with 95% CIs

Prompts include requests for harmful content (from the public benchmarks above) and are for safety evaluation only.
Code: see the GitHub repo linked from the collection.
