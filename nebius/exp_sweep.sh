#!/usr/bin/env bash
# Driver: Fireworks-aligned throughput-vs-interactivity sweep.
# gpt-oss-120b, 1xH200, TP=1, ISL=8k/OSL=1k, C=1..256. conc x12 (2x warmup
# dropped + 10x measured), ignore_eos. One persistent server, swept concurrency.
set -uo pipefail
cd "$(dirname "$0")"
source ./exp_lib.sh

predownload
start_server || { echo "server failed to start"; exit 1; }

for C in 1 2 4 8 16 32 64 128 256; do
  run_measured "sweep_c${C}" "$C" $(( C * 10 ))
done

echo "===== SWEEP COMPLETE ====="
column -t -s$'\t' "$RUNS_TSV"
