#!/usr/bin/env bash
# Driver: profiling cost vs fidelity.
# Reuses the persistent server (run after exp_sweep.sh, or it starts its own).
# Method:
#   - Ground-truth long run with --save-detailed (per-request arrays) at C=64 and
#     C=4 -> reference metrics + offline bootstrap of latency-tail convergence.
#   - Real repeated short runs at budgets B in {1,2,5,10}x conc, R=5 each ->
#     real run-to-run variance + bias of SYSTEM THROUGHPUT vs the long reference.
set -uo pipefail
cd "$(dirname "$0")"
source ./exp_lib.sh

# start server only if not already running
if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  predownload; start_server || { echo "server failed"; exit 1; }
fi

REPEATS="${REPEATS:-5}"
BUDGETS="${BUDGETS:-1 2 5 10}"

# ---- C=64 (high-throughput regime) ----
run_measured "gt_c64" 64 $(( 64 * 40 )) detailed     # ~2560 measured, reference
for B in $BUDGETS; do
  for R in $(seq 1 "$REPEATS"); do
    run_measured "cf_c64_b${B}_r${R}" 64 $(( 64 * B ))
  done
done

# ---- C=4 (low-concurrency / high-interactivity regime) ----
run_measured "gt_c4" 4 $(( 4 * 100 )) detailed       # ~400 measured, reference
for B in $BUDGETS; do
  for R in $(seq 1 "$REPEATS"); do
    run_measured "cf_c4_b${B}_r${R}" 4 $(( 4 * B ))
  done
done

echo "===== COST-FIDELITY COMPLETE ====="
column -t -s$'\t' "$RUNS_TSV"
