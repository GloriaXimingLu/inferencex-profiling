#!/usr/bin/env bash
# Runs ON the H200 node. Calibrates per-cell wall time for two InferenceX vLLM
# endpoints. Times: model download (once/model), server startup (container ->
# /health 200), and benchmark phase (ready -> container exit). Robust to vLLM
# version (polls /health rather than parsing logs).
set -uo pipefail

REPO=/opt/inferencex
HFC=/opt/hfcache
RES=/opt/results
PORT=8888
export RANDOM_RANGE_RATIO=0.8
HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "$RES"
TSV="$RES/timings.tsv"
[ -f "$TSV" ] || echo -e "label\tmodel\timage\ttp\tconc\tisl\tosl\tdownload_s\tstartup_s\tbench_s\ttotal_s\tstatus" > "$TSV"

dlog(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- model download, timed once per model ----
declare -A DL_DONE
predownload(){ # image model
  local image="$1" model="$2" key="$2"
  [ -n "${DL_DONE[$key]:-}" ] && { echo "${DL_DONE[$key]}"; return; }
  dlog "PULL image $image" >&2
  sudo docker pull -q "$image" >/dev/null 2>&1 || true
  dlog "DOWNLOAD model $model" >&2
  local t0 t1; t0=$(date +%s)
  sudo docker run --rm --network host \
    -v "$HFC":/hfcache -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint bash "$image" -lc "hf download '$model' >/dev/null 2>&1 || huggingface-cli download '$model' >/dev/null 2>&1" \
    > "$RES/dl_${model//\//_}.log" 2>&1
  t1=$(date +%s); DL_DONE[$key]=$((t1-t0))
  dlog "DOWNLOAD $model took ${DL_DONE[$key]}s" >&2
  echo "${DL_DONE[$key]}"
}

# ---- one benchmark cell ----
run_cell(){ # label image model script tp conc isl osl dp_attn ep max_model_len
  local label="$1" image="$2" model="$3" script="$4" tp="$5" conc="$6" isl="$7" osl="$8" dpa="${9:-false}" ep="${10:-1}" mml="${11:-}"
  [ -z "$mml" ] && mml=$((isl+osl+256))
  local log="$RES/${label}.log"
  local dl; dl=$(predownload "$image" "$model")
  dlog "CELL $label : tp=$tp conc=$conc isl=$isl osl=$osl dpa=$dpa"
  # kill anything on the port, clear ready marker
  rm -f "/tmp/${label}.ready"
  local t0; t0=$(date +%s)
  # background health poller -> records first 200 timestamp
  ( while ! curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do sleep 2; done; date +%s > "/tmp/${label}.ready" ) &
  local poller=$!
  local status=ok
  timeout 5400 sudo docker run --rm --gpus all --network host --ipc=host --shm-size=32g \
    --entrypoint bash \
    -v "$REPO":/workspace -w /workspace \
    -v "$HFC":/hfcache \
    -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL="$model" -e TP="$tp" -e CONC="$conc" -e ISL="$isl" -e OSL="$osl" \
    -e DP_ATTENTION="$dpa" -e EP_SIZE="$ep" -e MAX_MODEL_LEN="$mml" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" -e PORT="$PORT" \
    -e RESULT_FILENAME="$label" -e RUN_EVAL=false -e EVAL_ONLY=false \
    "$image" "$script" > "$log" 2>&1 || status="fail/timeout"
  local t_done; t_done=$(date +%s)
  kill "$poller" 2>/dev/null
  local t_ready startup bench
  if [ -f "/tmp/${label}.ready" ]; then
    t_ready=$(cat "/tmp/${label}.ready"); startup=$((t_ready-t0)); bench=$((t_done-t_ready))
  else
    t_ready=""; startup=""; bench=""; [ "$status" = ok ] && status="no-ready"
  fi
  local total=$((t_done-t0))
  echo -e "${label}\t${model}\t${image}\t${tp}\t${conc}\t${isl}\t${osl}\t${dl}\t${startup}\t${bench}\t${total}\t${status}" >> "$TSV"
  dlog "CELL $label DONE: startup=${startup}s bench=${bench}s total=${total}s status=$status"
}

GPTOSS_IMG=vllm/vllm-openai:v0.22.0
GPTOSS=openai/gpt-oss-120b
GPTOSS_SH=benchmarks/single_node/fixed_seq_len/gptoss_fp4_h200.sh
DSV4_IMG=vllm/vllm-openai:v0.21.0
DSV4=deepseek-ai/DeepSeek-V4-Pro
DSV4_SH=benchmarks/single_node/fixed_seq_len/dsv4_fp8_h200.sh

# --- fast endpoint: gpt-oss-120b (TP8) ---
run_cell gptoss_tp8_c4_1k1k   "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 8 4  1024 1024
run_cell gptoss_tp8_c64_1k1k  "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 8 64 1024 1024
run_cell gptoss_tp8_c4_8k1k   "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 8 4  8192 1024

# --- slow endpoint: DeepSeek-V4-Pro (TP8, pure TP) ---
run_cell dsv4_tp8_c1_1k1k     "$DSV4_IMG" "$DSV4" "$DSV4_SH" 8 1  1024 1024 false 1
run_cell dsv4_tp8_c64_1k1k    "$DSV4_IMG" "$DSV4" "$DSV4_SH" 8 64 1024 1024 false 1
run_cell dsv4_tp8_c4_8k1k     "$DSV4_IMG" "$DSV4" "$DSV4_SH" 8 4  8192 1024 false 1

echo "===== CALIBRATION COMPLETE ====="
column -t -s$'\t' "$TSV"
