#!/usr/bin/env bash
# Round 1 audit: official Tiny Aya Global Q4_K_M (what people deploy) vs further-compressed
# Q3_K_M / Q2_K requantized from it. Same prompts, same server, greedy decoding.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source .venv/bin/activate
G=models/gguf
Q4=$G/tiny-aya-global-q4_k_m.gguf
if [ ! -f "$Q4" ]; then cat $G/part.* > "$Q4"; fi
test "$(stat -f %z "$Q4")" = "2143977056" || { echo "Q4_K_M size mismatch"; exit 1; }
for Q in Q3_K_M Q2_K; do
  [ -f "$G/tiny-aya-global-$Q-from-q4km.gguf" ] || llama-quantize --allow-requantize "$Q4" "$G/tiny-aya-global-$Q-from-q4km.gguf" "$Q" > "$G/$Q.quantize.log" 2>&1
done
shasum -a 256 $G/*.gguf > results/gguf_sha256.txt
python -m sns.evaluate --gguf "$Q4" --name q4_k_m
python -m sns.evaluate --gguf "$G/tiny-aya-global-Q2_K-from-q4km.gguf" --name q2_k
python -m sns.report --ref q4_k_m
