#!/usr/bin/env bash
# HF checkpoint -> GGUF bf16 -> Q8_0 (reference) + Q4_K_M (what people ship), then drop the bf16 file.
# Usage: scripts/quantize.sh <hf_dir_or_repo_snapshot> <name> [extra quant types...]
set -euo pipefail
SRC="$1"; NAME="$2"; shift 2
EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/models/gguf"; mkdir -p "$OUT"
BF16="$OUT/$NAME-bf16.gguf"
python "$ROOT/third_party/llama.cpp/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$BF16"
for Q in Q8_0 Q4_K_M "${EXTRA[@]}"; do
  llama-quantize "$BF16" "$OUT/$NAME-$Q.gguf" "$Q" >"$OUT/$NAME-$Q.quantize.log" 2>&1
  echo "wrote $OUT/$NAME-$Q.gguf ($(du -h "$OUT/$NAME-$Q.gguf" | cut -f1))"
done
rm -f "$BF16"
