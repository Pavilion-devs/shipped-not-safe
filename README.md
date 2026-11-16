# Shipped ≠ Safe: does Tiny Aya's safety survive the quantization people deploy?

**Lexsi AI Safety & Alignment Hackathon · Track 2a (Adversarial and safety evaluation sets) · Round 1**

Small open models reach users as low-bit GGUF files (llama.cpp, Ollama, LM Studio), not as the
bf16 checkpoint that was safety-tested. This project measures what happens to Tiny Aya Global's
safety behaviour, its over-refusal and its task capability as the deployed file is compressed
further, using a curated, decontaminated eval set and paired statistics.

> Lexsi's own kickoff deck names three places where safety leaks after alignment: benign
> fine-tuning, **4-bit quantization for serving**, and I/O-only guardrails. Round 1 measures the
> quantization leak on the real deployment artifacts. Round 2 repairs it (see the plan below).

## TL;DR

RESULTS_TLDR

## Pipeline (Lexsi repos used)

```
JBB-Behaviors + XSTest  ──►  CuratorKIT  ──►  curated eval sets  ──►  llama-server (GGUF)  ──►  AuditKIT  ──►  paired bootstrap
 (harmful / benign)          clean + exact +    (harmful, benign,       Q4_K_M / Q3_K_M / Q2_K     refusal scorers,   Δ, 95% CI, p
                             MinHash dedup,      GSM8K)                                            GSM8K scorer
                             schema gate
```

| Repo | What it does here | Features used |
|---|---|---|
| **CuratorKIT** | Builds the eval sets | `Curator` with text cleaning, exact + MinHash near-dup removal, schema gate, provenance manifest + checksums |
| **AuditKIT** | Runs and scores every variant | `evaluate()` over the `api:` backend (OpenAI-compatible llama-server), `refusal_scorer` (multilingual refusal patterns) for refusal and over-refusal, a custom `@scorer` for GSM8K, `paired_bootstrap` for significance |
| SafeTune / CircuitKIT | **Round 2** (repair) | QReSafe / AAQ quantization-aware safety patching; circuit-aware mixed precision (protect the refusal circuit) |

## Models and licences

| Model | Role | Licence |
|---|---|---|
| `CohereLabs/tiny-aya-global-GGUF` → `tiny-aya-global-q4_k_m.gguf` | Official deployment file, reference | CC-BY-NC-4.0 + Cohere Labs AUP |
| Q3_K_M, Q2_K requantized **from that Q4_K_M** with `llama-quantize --allow-requantize` (llama.cpp b12359) | Further-compressed deployment variants (phones, edge) | as above |

## Datasets and licences

| Set | Source | Licence | Raw → after CuratorKIT |
|---|---|---|---|
| `harmful` (refusing is correct) | JailbreakBench JBB-Behaviors *harmful* (100) + XSTest *contrast* unsafe prompts (200) | MIT / CC-BY-4.0 | 300 → 249 |
| `benign` (refusing is **over**-refusal) | XSTest safe prompts (250) + JBB-Behaviors *benign* (100) | CC-BY-4.0 / MIT | 350 → 289 |
| `gsm8k` (capability) | GSM8K test, seeded random 100 | MIT | 100 |

## Method

- Same prompts, same llama-server build, greedy decoding (temperature 0, seed 0), 8 parallel slots,
  the model's own chat template (`--jinja`). Only the weights change between rows.
- Refusal = AuditKIT `refusal_scorer` (refusal phrasing in the opening sentence; responses with no
  measurable answer are excluded, not counted). Max 96 new tokens for safety prompts, 320 for GSM8K.
- Every delta is a **paired** bootstrap over the same prompts (AuditKIT `paired_bootstrap`,
  2,000 resamples), reported with a 95% CI and p-value.

## Results

RESULTS_TABLE

## What this does and does not show (honest limits)

- **Reference is the official Q4_K_M, not bf16.** The 6.7 GB bf16/Q8 files could not be downloaded
  in the Round 1 window; Round 2 adds them as the top row.
- **Q3_K_M / Q2_K are requantized from Q4_K_M**, which compounds error relative to quantizing
  from bf16. It is a realistic "squeeze the shipped file further" path, but Round 2 re-does it from bf16.
- **Refusal is pattern-based**, a fast first signal: it can miss paraphrased refusals and does not
  judge whether a compliance is actually harmful. Round 2 adds a guard-model judge and a 100-sample
  human audit with agreement reported.
- Single decoding run (greedy), English-only prompts in Round 1.

## Round 2 plan (prove the thesis)

1. bf16 → Q8_0 → Q4_K_M → Q3 → Q2 dose-response, quantized from bf16, plus the CuratorKIT-built
   multilingual slice (Tiny Aya's 70 languages are its reason to exist).
2. **Repair** the leak and compare, on the same eval: SafeTune `QReSafe` (SNIP-selected safety weights kept
   at high precision), SafeTune `AAQ` (alignment-aware quantization), and CircuitKIT circuit-aware mixed
   precision that protects the discovered refusal circuit, against a **random-component control**
   with the same bit budget.
3. Benign fine-tune drift (SafeTune `PlainSFT` vs `SafeGrad`, `ReSta` repair) composed with quantization:
   does a repair applied before quantization survive it?
4. ROI: tokens/s and memory of each GGUF on a laptop vs closed-API token billing at realistic volume.

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install datasets auditkit curatorkit huggingface_hub
brew install llama.cpp                    # b12359 used here
python -m sns.data --out data/eval --n_gsm8k 150 && head -100 data/eval/gsm8k.jsonl > /tmp/g && mv /tmp/g data/eval/gsm8k.jsonl
curl -L -o models/gguf/tiny-aya-global-q4_k_m.gguf \
  https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf
scripts/run_round1.sh                     # requantize, evaluate every variant, write results/results.md
```

`sns/` also contains the Round 2 code paths already written against SafeTune
(`sns/train.py`: PlainSFT / SafeGrad; `sns/repair.py`: ReSta) and `scripts/quantize.sh` (HF → bf16 GGUF → quants).

## Artifacts

- Dataset (eval sets, CuratorKIT provenance, every generation, per-sample scores): HF_DATASET
- Collections: HF_COLLECTIONS

## Feedback for the Lexsi tools (found while building)

- CuratorKIT's schema gate rejects rows with an empty `output`, so curating **prompt-only eval sets**
  needs a placeholder output. A `prompt_only=True` mode would help evaluation use cases.
- AuditKIT's run cache keys on the model *name*; serving different GGUFs behind one OpenAI-compatible
  endpoint under the same name silently replays cached results. We pass `cache=False` and a unique name
  per variant; a warning when `api_base` is local would prevent this.
