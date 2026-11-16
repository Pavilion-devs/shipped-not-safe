Shipped ≠ Safe: does Tiny Aya's safety survive the quantization people deploy? (Track 2a)

Small open models reach users as low-bit GGUF files in llama.cpp, Ollama or LM Studio, not as the bf16
checkpoint that was safety-tested. Lexsi's kickoff deck names 4-bit quantization for serving as one of
the three places safety leaks after alignment. We measure that leak on Cohere's official Tiny Aya Global
GGUF and on further-compressed variants (Q3_K_M, Q2_K) that phone and edge deployments use.

Pipeline: CuratorKIT builds the eval sets from public benchmarks (JailbreakBench JBB-Behaviors, XSTest,
GSM8K) using text cleaning, exact and MinHash near-duplicate removal and a schema gate, with a
provenance manifest. The result is 249 harmful prompts a safe model should refuse, 289 borderline benign
prompts where refusing is over-refusal, and 100 GSM8K questions for capability. Every variant is served
by the same llama-server build with greedy decoding and the model's own chat template, so only the weights
differ. AuditKIT runs the evaluation over its OpenAI-compatible api: backend: multilingual refusal scorers
for refusal and over-refusal, plus a custom GSM8K scorer. Each change is reported as a paired bootstrap
over identical prompts (AuditKIT paired_bootstrap) with a 95% CI and p-value. We report three metrics
together (safety, over-refusal, capability) so that "safer" cannot just mean "refuses everything".

Results: RESULTS_ONE_LINE

Everything is public: the curated eval sets with CuratorKIT provenance, every generation and
per-sample score, the quantization recipe with sha256s, and code that reproduces the table in one
command on a laptop. Limits are stated plainly. The Round 1 reference is the official Q4_K_M (bf16 could
not be fetched in time), low-bit variants are requantized from it, and refusal is pattern-based.

Round 2 plan (prove the thesis): a bf16 → Q8 → Q4 → Q3 → Q2 dose-response including a multilingual slice;
then repair the leak with SafeTune QReSafe and AAQ and CircuitKIT circuit-aware mixed precision that
protects the discovered refusal circuit, against a random-component control at the same bit budget;
then compose with fine-tuning drift (SafeTune PlainSFT vs SafeGrad, ReSta repair). Code paths for these
are already in the repo. We also report tool feedback found while building (CuratorKIT prompt-only
curation, an AuditKIT cache-by-name pitfall).
