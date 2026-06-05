#!/usr/bin/env bash
# Runs ON a 1xH200 node. Calibrates per-cell wall time for gpt-oss-120b (the
# one real InferenceX H200 vLLM config that fits on a single H200, TP=1).
# Times model download (once), server startup (container -> /health 200), and
# benchmark phase (ready -> container exit). Version-independent (/health poll).
set -uo pipefail
REPO=/opt/inferencex; HFC=/opt/hfcache; RES=/opt/results; PORT=8888
export RANDOM_RANGE_RATIO=0.8
HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "$RES"; TSV="$RES/timings.tsv"
[ -f "$TSV" ] || echo -e "label\tmodel\timage\ttp\tconc\tisl\tosl\tdownload_s\tstartup_s\tbench_s\ttotal_s\tstatus" > "$TSV"
dlog(){ echo "[$(date +%H:%M:%S)] $*"; }

GPTOSS_IMG=vllm/vllm-openai:v0.22.0
GPTOSS=openai/gpt-oss-120b
GPTOSS_SH=benchmarks/single_node/fixed_seq_len/gptoss_fp4_h200.sh

DL_S=""
predownload(){ # image model
  [ -n "$DL_S" ] && return
  dlog "PULL $1" ; sudo docker pull -q "$1" >/dev/null 2>&1 || true
  dlog "DOWNLOAD $2"; local t0; t0=$(date +%s)
  sudo docker run --rm --network host -v "$HFC":/hfcache \
    -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e HF_TOKEN="$HF_TOKEN" \
    --entrypoint bash "$1" -lc "hf download '$2' >/dev/null 2>&1 || huggingface-cli download '$2' >/dev/null 2>&1" \
    > "$RES/dl.log" 2>&1
  DL_S=$(( $(date +%s) - t0 )); dlog "DOWNLOAD took ${DL_S}s"
}

run_cell(){ # label image model script tp conc isl osl
  local label="$1" image="$2" model="$3" script="$4" tp="$5" conc="$6" isl="$7" osl="$8"
  local mml=$((isl+osl+256)) log="$RES/${label}.log"
  predownload "$image" "$model"
  dlog "CELL $label : tp=$tp conc=$conc isl=$isl osl=$osl"
  rm -f "/tmp/${label}.ready"
  local t0; t0=$(date +%s)
  ( while ! curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do sleep 2; done; date +%s > "/tmp/${label}.ready" ) &
  local poller=$! status=ok
  timeout 3600 sudo docker run --rm --gpus all --network host --ipc=host --shm-size=16g \
    --entrypoint bash -v "$REPO":/workspace -w /workspace -v "$HFC":/hfcache \
    -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL="$model" -e TP="$tp" -e CONC="$conc" -e ISL="$isl" -e OSL="$osl" \
    -e DP_ATTENTION=false -e EP_SIZE=1 -e MAX_MODEL_LEN="$mml" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" -e PORT="$PORT" \
    -e RESULT_FILENAME="$label" -e RUN_EVAL=false -e EVAL_ONLY=false \
    "$image" "$script" > "$log" 2>&1 || status="fail/timeout"
  local t_done; t_done=$(date +%s); kill "$poller" 2>/dev/null
  local t_ready startup bench
  if [ -f "/tmp/${label}.ready" ]; then
    t_ready=$(cat "/tmp/${label}.ready"); startup=$((t_ready-t0)); bench=$((t_done-t_ready))
  else startup=""; bench=""; [ "$status" = ok ] && status="no-ready"; fi
  echo -e "${label}\t${model}\t${image}\t${tp}\t${conc}\t${isl}\t${osl}\t${DL_S}\t${startup}\t${bench}\t$((t_done-t0))\t${status}" >> "$TSV"
  dlog "CELL $label DONE: startup=${startup}s bench=${bench}s status=$status"
}

run_cell gptoss_tp1_c4_1k1k  "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 1 4  1024 1024
run_cell gptoss_tp1_c64_1k1k "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 1 64 1024 1024
run_cell gptoss_tp1_c4_8k1k  "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 1 4  8192 1024
run_cell gptoss_tp1_c64_8k1k "$GPTOSS_IMG" "$GPTOSS" "$GPTOSS_SH" 1 64 8192 1024
echo "===== CALIBRATION COMPLETE ====="; column -t -s$'\t' "$TSV"
