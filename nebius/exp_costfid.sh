#!/usr/bin/env bash
# Driver: profiling cost vs fidelity (parameterized).
# Reuses the persistent server (run after exp_sweep.sh, or it starts its own).
# Method:
#   - Ground-truth long run with --save-detailed (per-request arrays) per concurrency
#     -> reference metrics + offline bootstrap of latency-tail convergence.
#   - Real repeated short runs at budgets B in BUDGETS (num_prompts = B*conc), REPEATS each
#     -> real run-to-run variance + bias of SYSTEM THROUGHPUT vs the long reference.
#
# Env knobs (defaults reproduce the original C=64/C=4 run):
#   CONCS="64 4"   BUDGETS="1 2 5 10"   REPEATS=5   GT_MULT=40
# For the expensive high-concurrency regime, e.g.:
#   CONCS=256 GT_MULT=15 REPEATS=3 bash exp_costfid.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./exp_lib.sh

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  predownload; start_server || { echo "server failed"; exit 1; }
fi

CONCS="${CONCS:-64 4}"
BUDGETS="${BUDGETS:-1 2 5 10}"
REPEATS="${REPEATS:-5}"
GT_MULT="${GT_MULT:-40}"

for C in $CONCS; do
  run_measured "gt_c${C}" "$C" $(( C * GT_MULT )) detailed     # reference (long, detailed)
  for B in $BUDGETS; do                                        # ascending => partial data still useful
    for R in $(seq 1 "$REPEATS"); do
      run_measured "cf_c${C}_b${B}_r${R}" "$C" $(( C * B ))
    done
  done
done

echo "===== COST-FIDELITY COMPLETE (CONCS=$CONCS) ====="
column -t -s$'\t' "$RUNS_TSV"
