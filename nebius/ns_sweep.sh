#!/usr/bin/env bash
# Node-side: validate the num_speculative_tokens hypothesis. Serve gpt-oss-120b
# with EAGLE3 at num_spec in {0(vanilla),1,3,5,7}, and at each setting measure
# throughput + acceptance (rate, length, per-position) at C=1 and C=64.
# Confirms whether a shallower draft (num_spec~3) beats the default 7.
set -uo pipefail
export MODEL="${MODEL:-openai/gpt-oss-120b}"
export PREFIX_CACHING=off ISL=8000 OSL=1000 MAXLEN=10240 MAXSEQS=256
SPEC_HEAD=nvidia/gpt-oss-120b-Eagle3-v3
source /tmp/exp_lib.sh

predownload
sudo docker run --rm --network host -v "$HFC":/hfcache -e HF_HUB_CACHE=/hfcache \
  -e HF_HOME=/hfcache --entrypoint bash "$IMAGE" \
  -lc "hf download '$SPEC_HEAD' >/dev/null 2>&1 || huggingface-cli download '$SPEC_HEAD' >/dev/null 2>&1" \
  > "$RES/spec_head_dl.log" 2>&1 || true

runcells(){ run_measured "${1}_c1" 1 10; run_measured "${1}_c64" 64 640; }

# vanilla baseline (no spec) for a same-node throughput reference
unset SPEC SPEC_NUM
start_server && runcells ns0

# num_spec sweep
for SN in 1 3 5 7; do
  export SPEC="$SPEC_HEAD" SPEC_NUM="$SN"
  start_server && runcells "ns${SN}"
done
echo "===== NS SWEEP COMPLETE ====="
