#!/usr/bin/env bash
# Driver: replay all 4 source-model parts of ONE domain against a gpt-oss server.
# One node per domain (uniform server config). Env:
#   DOMAIN=airline|retail|telecom|banking_knowledge  CONC=<int>  NTRACES=<int>
set -uo pipefail
cd "$(dirname "$0")"
source ./tau_lib.sh

DOMAIN="${DOMAIN:?set DOMAIN}"; CONC="${CONC:-64}"; NTRACES="${NTRACES:-200}"
# MODELS overridable (space-separated) to split a domain across nodes; default all 4.
read -r -a MODELS <<< "${MODELS:-claude-opus-4-5 gemini-3-pro gpt-5-2 qwen3.5-397b-a17b-think}"

predownload
start_server || { echo "server failed"; exit 1; }

for M in "${MODELS[@]}"; do
  run_part "${M}__${DOMAIN}" "$CONC" "$NTRACES"
done

echo "===== TAU DOMAIN COMPLETE: $DOMAIN ====="
column -t -s$'\t' "$RUNS_TSV"
