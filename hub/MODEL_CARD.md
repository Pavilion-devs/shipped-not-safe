---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [gguf, quantization, safety, tiny-aya, lexsi-hackathon]
---

# Shipped ≠ Safe: quantization recipes for the Tiny Aya Global safety audit

Lexsi AI Safety & Alignment Hackathon (Track 2a), Round 1.

Audited files:

- `CohereLabs/tiny-aya-global-GGUF` → `tiny-aya-global-q4_k_m.gguf` (official, reference)
- Q3_K_M and Q2_K, requantized from that file with `llama-quantize --allow-requantize` (llama.cpp b12359)

The low-bit weights are fully reproducible from the official file with the recipe in `recipe/`
(`run_round1.sh`); their sha256s are in the dataset repo (`results/gguf_sha256.txt`). They were not
re-uploaded in Round 1 because of upload bandwidth. Round 2 adds repaired checkpoints (SafeTune QReSafe /
AAQ, CircuitKIT circuit-aware mixed precision) as real weights.

Licence: inherits CC-BY-NC-4.0 and the Cohere Labs Acceptable Use Policy from Tiny Aya. Safety research use.
