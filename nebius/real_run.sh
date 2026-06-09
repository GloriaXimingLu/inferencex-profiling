#!/usr/bin/env bash
# Node-side: serve gpt-oss-120b with EAGLE3 spec-decode AND prefix caching ON,
# then replay the REAL-TOKEN tau-bench schedules and measure realized cache-hit
# + spec acceptance on realistic traffic.
set -uo pipefail
RES=/opt/results; mkdir -p "$RES"
export MODEL="${MODEL:-openai/gpt-oss-120b}"
export PREFIX_CACHING=on
export SPEC="${SPEC:-nvidia/gpt-oss-120b-Eagle3-v3}" SPEC_NUM="${SPEC_NUM:-7}"
export MAXLEN="${MAXLEN:-40960}" MAXSEQS="${MAXSEQS:-64}"
CONC="${CONC:-12}"; MAXSIMS="${MAXSIMS:-150}"
source /tmp/exp_lib.sh

predownload
# also fetch the EAGLE3 draft head
sudo docker run --rm --network host -v "$HFC":/hfcache -e HF_HUB_CACHE=/hfcache \
  -e HF_HOME=/hfcache --entrypoint bash "$IMAGE" \
  -lc "hf download '$SPEC' >/dev/null 2>&1 || huggingface-cli download '$SPEC' >/dev/null 2>&1" \
  > "$RES/spec_head_dl.log" 2>&1 || true

start_server || { echo "SERVER FAILED"; exit 1; }
# Ubuntu 24.04 is PEP-668 externally-managed; --break-system-packages, apt fallback.
python3 -c "import aiohttp" 2>/dev/null || python3 -m pip install --break-system-packages -q aiohttp 2>/dev/null || sudo apt-get install -y -qq python3-aiohttp 2>/dev/null
python3 -c "import aiohttp" 2>/dev/null || { echo "AIOHTTP INSTALL FAILED"; exit 1; }

for f in /tmp/real_schedule/*.jsonl; do
  part=$(basename "$f" .jsonl)
  echo "===== replay $part  conc=$CONC maxsims=$MAXSIMS ====="
  python3 /tmp/real_client.py --schedule "$f" --base "http://localhost:$PORT" \
    --concurrency "$CONC" --max-sims "$MAXSIMS" --out "$RES/real_${part}.json" 2>&1 | tail -14
done
echo "===== REAL RUN COMPLETE ====="
