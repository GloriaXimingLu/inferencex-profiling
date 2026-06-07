#!/usr/bin/env bash
# Node-side experiment library for Fireworks-aligned vLLM profiling.
# Starts ONE persistent gpt-oss server and runs measured benchmark cells against
# it, capturing the comprehensive metric set per run:
#   - client JSON   (benchmark_serving --save-result): throughput, TTFT/TPOT/ITL/E2E pctiles
#   - /metrics M0,M1 (counter deltas): prefix-cache, preemptions, spec-decode, prefill/decode, queue
#   - 1 Hz samples  (gauges over the run): KV-cache usage, running/waiting requests
#   - server.log    (CUDA-graph capture/fallback, warnings)
#   - wall time     (cost)
set -uo pipefail

REPO=/opt/inferencex; RES=/opt/results; HFC=/opt/hfcache
PORT=8888; CTR=ix-vllm
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
ISL="${ISL:-8000}"; OSL="${OSL:-1000}"; MAXLEN="${MAXLEN:-10240}"
RANGE="${RANGE:-0.8}"; MAXSEQS="${MAXSEQS:-256}"
PREFIX_CACHING="${PREFIX_CACHING:-off}"   # off = InferenceX recipe (random tokens => ~0 hit)
mkdir -p "$RES"
RUNS_TSV="$RES/runs.tsv"
[ -f "$RUNS_TSV" ] || echo -e "label\tconc\tnum_prompts\twarmups\twall_s\tt0\tt1\tdetailed" > "$RUNS_TSV"
dlog(){ echo "[$(date +%H:%M:%S)] $*"; }

# gauge lines to sample at 1 Hz (names verified against live /metrics on first start)
SAMPLE_RE='num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage|num_preemptions|prefix_cache'

predownload(){
  dlog "pull $IMAGE"; sudo docker pull -q "$IMAGE" >/dev/null 2>&1 || true
  if ! ls "$HFC"/*"${MODEL##*/}"* >/dev/null 2>&1 && ! ls "$HFC"/models--* >/dev/null 2>&1; then
    dlog "download $MODEL"
    sudo docker run --rm --network host -v "$HFC":/hfcache \
      -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache --entrypoint bash "$IMAGE" \
      -lc "hf download '$MODEL' >/dev/null 2>&1 || huggingface-cli download '$MODEL' >/dev/null 2>&1" \
      > "$RES/download.log" 2>&1
  fi
}

start_server(){
  sudo docker rm -f "$CTR" >/dev/null 2>&1 || true
  local pc=""; [ "$PREFIX_CACHING" = off ] && pc="--no-enable-prefix-caching" || pc="--enable-prefix-caching"
  dlog "start server: $MODEL TP=1 max-num-seqs=$MAXSEQS max-model-len=$MAXLEN prefix-caching=$PREFIX_CACHING"
  sudo docker run -d --name "$CTR" --gpus all --network host --ipc=host --shm-size=16g \
    -v "$REPO":/workspace -w /workspace -v "$HFC":/hfcache -v "$RES":/results \
    -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e VLLM_MXFP4_USE_MARLIN=1 -e TORCH_CUDA_ARCH_LIST=9.0 \
    --entrypoint bash "$IMAGE" -lc "
      pip install -q datasets pandas 2>/dev/null;
      vllm serve '$MODEL' --host 0.0.0.0 --port $PORT \
        --tensor-parallel-size 1 --max-num-seqs $MAXSEQS \
        --max-model-len $MAXLEN --gpu-memory-utilization 0.9 \
        --max-cudagraph-capture-size 2048 --max-num-batched-tokens 8192 \
        $pc > /results/server.log 2>&1
    " >/dev/null
  dlog "waiting for /health ..."
  local t0; t0=$(date +%s)
  while ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! sudo docker ps --format '{{.Names}}' | grep -q "^$CTR$"; then
      dlog "SERVER CONTAINER DIED"; tail -40 "$RES/server.log"; return 1
    fi
    sleep 3
  done
  dlog "server ready in $(( $(date +%s) - t0 ))s"
  # one-time: dump full /metrics so we can confirm exact metric names available
  curl -s "http://localhost:$PORT/metrics" > "$RES/metrics_catalog.prom"
  dlog "metric names captured: $(grep -c '^vllm:' "$RES/metrics_catalog.prom") vllm series"
}

# run_measured <label> <conc> <num_prompts> [detailed]
run_measured(){
  local label="$1" conc="$2" np="$3" detailed="${4:-}"
  local warm=$(( 2 * conc ))
  local mre="$RES/$label"
  dlog "RUN $label : conc=$conc measured=$np warmup=$warm detailed=${detailed:-no}"
  curl -s "http://localhost:$PORT/metrics" > "$mre.m0"
  # 1 Hz sampler: vLLM gauges (KV usage, running/waiting) + GPU power/util/mem
  ( while true; do
      printf 'ts %s\n' "$(date +%s.%N)"
      nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | sed 's/^/gpu /'
      curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "$SAMPLE_RE"
      sleep 1
    done ) > "$mre.samples" 2>/dev/null &
  local sampler=$!
  local det=(); [ "$detailed" = detailed ] && det=(--save-detailed)
  local t0; t0=$(date +%s.%N)
  sudo docker exec "$CTR" python3 /workspace/utils/bench_serving/benchmark_serving.py \
    --model "$MODEL" --backend vllm --base-url "http://0.0.0.0:$PORT" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio "$RANGE" --num-prompts "$np" --max-concurrency "$conc" \
    --request-rate inf --ignore-eos --num-warmups "$warm" \
    --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '50,90,95,99' \
    --save-result "${det[@]}" --result-dir /results --result-filename "$label.json" \
    --disable-tqdm > "$mre.client.log" 2>&1
  local rc=$?; local t1; t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null
  curl -s "http://localhost:$PORT/metrics" > "$mre.m1"
  tail -n 600 "$RES/server.log" > "$mre.serverlog" 2>/dev/null
  local wall; wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")
  echo -e "${label}\t${conc}\t${np}\t${warm}\t${wall}\t${t0}\t${t1}\t${detailed:-no}" >> "$RUNS_TSV"
  dlog "RUN $label DONE rc=$rc wall=${wall}s"
  [ $rc -ne 0 ] && { dlog "client error tail:"; tail -8 "$mre.client.log"; }
  return 0
}
