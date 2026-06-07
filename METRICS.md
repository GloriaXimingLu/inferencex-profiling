# Comprehensive vLLM Profiling Metrics

A survey of the metrics worth recording per benchmark cell, *where each comes from*,
and *how we capture it* inside the InferenceX run harness. Designed so every metric
is recorded automatically during each run (no manual steps).

## Capture architecture (how, not just what)

A benchmark "cell" = one `(model, ISL/OSL, TP, concurrency)` run. `nebius/exp_lib.sh`
wraps each `benchmark_serving` invocation (`run_measured`) with four capture channels:

| Channel | Mechanism | Captures |
|---|---|---|
| **Client JSON** | `benchmark_serving --save-result [--save-detailed]` | request/throughput + latency distributions (TTFT/TPOT/ITL/E2E) |
| **`/metrics` snapshots** | `curl :8888/metrics` before (`m0`) & after (`m1`) the measured window → counter **deltas** | prefix-cache, preemptions, prefill/decode/queue time, spec-decode, iteration tokens |
| **1 Hz sampler** | background loop sampling `/metrics` gauges + `nvidia-smi` every second | KV-cache usage %, running/waiting requests, GPU power/util/mem (time-series → mean & max) |
| **Server log** | `tail server.log` | CUDA-graph capture/eager-fallback, preemption warnings |

The counter-delta design is key: vLLM's Prometheus counters are cumulative, so
`m1 − m0` gives exactly the work done in the measured window (warmup excluded, since
snapshots bracket only the measured call). `analyze_metrics.py` parses all four
channels into one per-run record; it **matches metric names by regex**, so it's robust
to vLLM version differences in metric naming.

---

## The metric set

### A. Throughput & latency (client-side — `benchmark_serving` JSON)
| Metric | Definition | JSON key | Why it matters |
|---|---|---|---|
| Request throughput | completed reqs / wall-time | `request_throughput` | system capacity (req/s) |
| **System throughput** | output tokens / wall-time | `output_throughput` | **plot Y-axis**; the headline tok/s |
| Total token throughput | (in+out) tokens / wall-time | `total_token_throughput` | prefill-inclusive load |
| **TTFT** | time to first token | `mean/median/std/p50/p90/p95/p99_ttft_ms` | prefill latency / responsiveness |
| **TPOT** | time per output token | `*_tpot_ms` | decode speed; **interactivity = 1000/mean_tpot** (plot X) |
| ITL | inter-token latency | `*_itl_ms` | decode smoothness/jitter |
| E2E latency | full request latency | `*_e2el_ms` | user-visible total |
| Completed / errors | success count | `completed` | run validity |

### B. Scheduler & memory (server-side — `/metrics` counter deltas m1−m0)
| Metric | Derivation | vLLM series (regex) | Why it matters |
|---|---|---|---|
| **Prefix-cache hit rate** | Δhits / Δqueries | `prefix_cache.*hit`, `prefix_cache.*(quer\|total)` | cache effectiveness — **≈0 on random tokens; the headline metric for the tau-bench realistic workload** (PDF #2) |
| **Preemptions** | Δcount | `num_preemptions` | KV-cache pressure — requests evicted & recomputed |
| **Prefill cost** | Δsum/Δcount | `request_prefill_time_seconds_{sum,count}` | avg prefill ms/request |
| **Decode cost** | Δsum/Δcount | `request_decode_time_seconds_{sum,count}` | avg decode ms/request |
| **Scheduler wait** | Δsum/Δcount | `request_queue_time_seconds_{sum,count}` | time queued before running |
| Avg iteration tokens | Δsum/Δcount | `iteration_tokens_total_{sum,count}` | effective batch size (tokens/step) |
| **Spec-decode accept rate** | Δaccepted/Δdraft | `spec_decode.*accept`, `spec_decode.*draft` | speculative efficiency — **n/a for vanilla gpt-oss; present when an MTP/draft model is enabled** (Fireworks reported 67%) |

### C. Live pressure gauges (server-side — 1 Hz samples → mean & max)
| Metric | vLLM series | Why it matters |
|---|---|---|
| **KV-cache usage %** | `gpu_cache_usage_perc` / `kv_cache_usage` | KV-cache pressure over the run (mean & peak) |
| Running requests | `num_requests_running` | actual concurrent batch size achieved |
| Waiting requests | `num_requests_waiting` | queue depth under load |

### D. GPU hardware (host — `nvidia-smi`, 1 Hz)
| Metric | Source | Why it matters |
|---|---|---|
| Power draw (mean/max W) | `power.draw` | energy; perf-per-watt |
| SM utilization (%) | `utilization.gpu` | compute saturation |
| Memory used (MB) | `memory.used` | footprint headroom |

### E. Derived / log-parsed
| Metric | Derivation | Why it matters |
|---|---|---|
| **Energy / output token (J/tok)** | mean_power × wall / output_tokens | efficiency (InferenceX "tokens per MW" cousin) |
| **CUDA-graph captured / eager fallback** | grep `server.log` for capture vs "falling back to eager" | a silent eager fallback tanks decode perf — a top InferenceX failure mode |
| Server startup time | container start → `/health` 200 | fixed per-cell overhead (matters for the full-matrix budget) |

---

## Notes & honesty
- **Prefix-cache & spec-decode read ~0 / n/a here** by design: the Fireworks-aligned
  workload sends random tokens with prefix caching off, and vanilla gpt-oss has no draft
  model. Both are wired up and will light up under (a) the tau-bench realistic workload
  and (b) an MTP-enabled model — exactly the two follow-ups your team flagged.
- Exact vLLM Prometheus names drift across versions; the analyzer's regex matching plus
  the per-run `metrics_catalog.prom` dump (full `/metrics` at server start) make the
  pipeline self-documenting and version-robust.
- Client TTFT/TPOT (channel A) and server-side histograms (channel B) are captured
  independently, giving a built-in cross-check.
