#!/usr/bin/env bash
# Sweep concurrency against a Fireworks (or any OpenAI-compatible) endpoint and
# write one JSON per point into results/. Then run: python plot.py results
#
# Usage:
#   export FIREWORKS_API_KEY=fw_xxx
#   ./sweep.sh                          # defaults: gpt-oss-120b, 8k/1k
#   MODEL=accounts/fireworks/models/gpt-oss-120b ISL=8000 OSL=1000 ./sweep.sh
#   CONCURRENCY="1 4 16 64 256" ./sweep.sh
set -euo pipefail

MODEL="${MODEL:-accounts/fireworks/models/gpt-oss-120b}"
BASE_URL="${BASE_URL:-https://api.fireworks.ai/inference/v1}"
ISL="${ISL:-8000}"
OSL="${OSL:-1000}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32 64 128 256}"
TOKENIZER="${TOKENIZER:-openai/gpt-oss-120b}"
IGNORE_EOS="${IGNORE_EOS:-1}"          # 1 = pin OSL to exactly $OSL (recommended). 0 = natural length.
REASONING_EFFORT="${REASONING_EFFORT:-}"  # gpt-oss: low|medium|high|max|none. Match InferenceMax.
EXTRA="${EXTRA:-}"                     # e.g. EXTRA="--cache-warm"

[[ "$IGNORE_EOS" == "1" ]] && EXTRA="$EXTRA --ignore-eos"
[[ -n "$REASONING_EFFORT" ]] && EXTRA="$EXTRA --reasoning-effort $REASONING_EFFORT"

if [[ -z "${FIREWORKS_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "set FIREWORKS_API_KEY first (export FIREWORKS_API_KEY=fw_...)" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="results/${STAMP}"
mkdir -p "$OUTDIR"
echo "writing to $OUTDIR" >&2
echo "model=$MODEL  ISL=$ISL  OSL=$OSL  concurrency=[$CONCURRENCY]" >&2

for C in $CONCURRENCY; do
  uv run fw_bench.py \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --concurrency "$C" \
    --tokenizer "$TOKENIZER" \
    --output "$OUTDIR/c$(printf '%04d' "$C").json" \
    $EXTRA \
    >/dev/null || echo "  point C=$C FAILED, continuing" >&2
done

echo "" >&2
echo "done. plot with:  uv run plot.py $OUTDIR" >&2
