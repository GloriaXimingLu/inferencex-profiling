#!/usr/bin/env bash
# Node-side tau-bench replay library. Serves gpt-oss-120b with PREFIX CACHING ON
# (the whole point — measure realized vs analytical-ideal cache-hit rate on a
# realistic, prefix-cache-friendly workload) and replays tau-bench parts via
# client.py, capturing the comprehensive server-side metric set per part.
set -uo pipefail
TAU=/opt/tau; RES=/opt/results; HFC=/opt/hfcache
PORT=8888; CTR=ix-vllm
IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
MAXLEN="${MAXLEN:-131072}"; MAXSEQS="${MAXSEQS:-256}"
mkdir -p "$RES"
RUNS_TSV="$RES/tau_runs.tsv"
[ -f "$RUNS_TSV" ] || echo -e "part\tconc\tntraces\twall_s\tt0\tt1" > "$RUNS_TSV"
dlog(){ echo "[$(date +%H:%M:%S)] $*"; }
SAMPLE_RE='num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage|num_preemptions|prefix_cache'

predownload(){
  dlog "pull $IMAGE"; sudo docker pull -q "$IMAGE" >/dev/null 2>&1 || true
  if ! ls "$HFC"/models--* >/dev/null 2>&1; then
    dlog "download $MODEL"
    sudo docker run --rm --network host -v "$HFC":/hfcache \
      -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache --entrypoint bash "$IMAGE" \
      -lc "hf download '$MODEL' >/dev/null 2>&1 || huggingface-cli download '$MODEL' >/dev/null 2>&1" \
      > "$RES/download.log" 2>&1
  fi
}

start_server(){
  sudo docker rm -f "$CTR" >/dev/null 2>&1 || true
  dlog "start server: $MODEL prefix-caching=ON max-model-len=$MAXLEN"
  sudo docker run -d --name "$CTR" --gpus all --network host --ipc=host --shm-size=16g \
    -v "$HFC":/hfcache -v "$RES":/results \
    -e HF_HUB_CACHE=/hfcache -e HF_HOME=/hfcache -e VLLM_MXFP4_USE_MARLIN=1 -e TORCH_CUDA_ARCH_LIST=9.0 \
    --entrypoint bash "$IMAGE" -lc "
      vllm serve '$MODEL' --host 0.0.0.0 --port $PORT \
        --tensor-parallel-size 1 --max-num-seqs $MAXSEQS \
        --max-model-len $MAXLEN --gpu-memory-utilization 0.92 \
        --max-num-batched-tokens 8192 --enable-prefix-caching > /results/server.log 2>&1
    " >/dev/null
  dlog "waiting for /health ..."; local t0; t0=$(date +%s)
  while ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sudo docker ps --format '{{.Names}}' | grep -q "^$CTR$" || { dlog "SERVER DIED"; tail -40 "$RES/server.log"; return 1; }
    sleep 3
  done
  dlog "server ready in $(( $(date +%s) - t0 ))s"
  curl -s "http://localhost:$PORT/metrics" > "$RES/metrics_catalog.prom"
}

# run_part <part> <conc> <ntraces>
run_part(){
  local part="$1" conc="$2" nt="$3"; local m="$RES/$part"
  dlog "PART $part : conc=$conc ntraces=$nt"
  curl -s "http://localhost:$PORT/metrics" > "$m.m0"
  ( while true; do printf 'ts %s\n' "$(date +%s.%N)"
      nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | sed 's/^/gpu /'
      curl -s "http://localhost:$PORT/metrics" 2>/dev/null | grep -E "$SAMPLE_RE"
      sleep 2; done ) > "$m.samples" 2>/dev/null &
  local sampler=$!
  local t0; t0=$(date +%s.%N)
  python3 "$TAU/client.py" --part "$part" \
    --endpoint "http://localhost:$PORT/v1" --model "$MODEL" \
    --metrics-url "http://localhost:$PORT/metrics" \
    --concurrency "$conc" --ntraces "$nt" --timeout 600 \
    > "$m.client.json" 2> "$m.client.err"
  local rc=$?; local t1; t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null
  curl -s "http://localhost:$PORT/metrics" > "$m.m1"
  tail -n 800 "$RES/server.log" > "$m.serverlog" 2>/dev/null
  echo -e "${part}\t${conc}\t${nt}\t$(awk "BEGIN{printf \"%.1f\",$t1-$t0}")\t${t0}\t${t1}" >> "$RUNS_TSV"
  dlog "PART $part DONE rc=$rc"
  [ $rc -ne 0 ] && { dlog "client err:"; tail -5 "$m.client.err"; }
}
