# vLLM Profiling Experiments — Fireworks Comparison, Metrics, Cost-vs-Fidelity

**Date:** 2026-06-07 · **Hardware:** 1× NVIDIA H200 (Nebius) · **Model:** gpt-oss-120b (FP4/MXFP4), vLLM v0.22.0
**Workload:** random tokens, ISL≈8000 / OSL=1000, `ignore_eos`, prefix-caching off (InferenceX recipe)

Raw data in `nebius/results_exp/`; plots `fireworks_compare.png`, `cost_fidelity.png`; full
metric table `metrics_table.tsv`. Reproduce via `nebius/exp_*.sh` (see `PROFILING_RUNBOOK.md`).

---

## 1. Throughput-vs-interactivity: vLLM vs Fireworks

**Setup is aligned with the Fireworks benchmark.** InferenceX's `run_benchmark_serving`
already uses `--ignore-eos` (exactly OSL tokens) and `--num-warmups 2*conc` (warmup requests
run then **discarded**) on top of `10*conc` measured prompts — i.e. **`conc*12` total = 2×
warmup dropped + 10× measured**, the same recipe Fireworks described. (Cross-check: co-worker's
vLLM C=4 wall time 105 s = our independently measured 105 s.)

### The curve (`fireworks_compare.png`)
X = interactivity (per-user output speed, tok/s/user = 1000/mean_TPOT); Y = system throughput
(output tok/s). Our vLLM sweep overlaid with the co-worker's Fireworks points (digitized from
the report — happy to drop in exact values if you share the raw numbers).

| C | ours: interactivity | ours: sys-tput | Fireworks: interactivity | Fireworks: sys-tput |
|--:|--:|--:|--:|--:|
| 1 | 232 | 214 | 310 | 250 |
| 2 | 190 | 349 | 270 | 410 |
| 4 | 140 | 520 | 255 | 800 |
| 8 | 96 | 722 | 178 | 1160 |
| 16 | 63 | 946 | 113 | 1570 |
| 32 | 40 | 1212 | 68 | 1960 |
| 64 | 24 | 1455 | 35 | 2300 |
| 128 | 13.5 | 1647 | 33 | 2180 |
| 256 | 8.5 | 1722 | 33 | 2300 |

**Fireworks is faster on both axes at every concurrency** — ~1.3–1.6× higher system throughput
at matched load *and* higher per-user speed. Two drivers: (1) **speculative decoding**
(Fireworks ~67% accept; our vanilla gpt-oss has no draft model) lifts both axes; (2) at high
concurrency our single H200 becomes **queue-bound** (see §2) while Fireworks plateaus gracefully
at ~33 tok/s/user. To close the gap head-to-head, enable an MTP/draft model in vLLM and rerun.

---

## 2. Comprehensive metrics across the concurrency sweep

~30 metrics/run are captured (client JSON + vLLM `/metrics` counter deltas + 1 Hz GPU/gauge
sampler + server log); see `metrics_table.tsv` for the full set and `METRICS.md` for how each is
recorded. The table below is gpt-oss-120b, 1×H200, ISL≈8k/OSL=1k, C=1→256.

### A. Throughput & latency
| C | sys-tput (tok/s) | interactivity (tok/s/user) | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | mean ITL (ms) |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 214 | 232 | 343 | 441 | 4.3 | 4.3 |
| 2 | 349 | 190 | 354 | 592 | 5.3 | 5.3 |
| 4 | 520 | 140 | 403 | 1,183 | 7.1 | 7.1 |
| 8 | 722 | 96 | 477 | 2,469 | 10.4 | 10.4 |
| 16 | 946 | 63 | 650 | 4,714 | 16.0 | 16.0 |
| 32 | 1,212 | 40 | 916 | 8,795 | 25.0 | 25.0 |
| 64 | 1,455 | 24 | 1,526 | 17,368 | 41.7 | 41.8 |
| 128 | 1,647 | 13.5 | 2,656 | 35,112 | 74.0 | 74.2 |
| 256 | 1,722 | 8.5 | 25,547 | 113,217 | 118.3 | 118.6 |

### B. Scheduler, memory, hardware, efficiency
| C | KV usage max (%) | preemptions | prefill/req (ms) | decode/req (ms) | queue/req (ms) | batch (tok/step) | power (W) | GPU util (%) | energy (J/tok) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.5 | 0 | 351 | 3,914 | 0 | 9 | 316 | 80 | 1.48 |
| 2 | 1.4 | 0 | 332 | 4,804 | 0 | 17 | 382 | 85 | 1.10 |
| 4 | 2.4 | 0 | 392 | 6,377 | 11 | 35 | 463 | 88 | 0.89 |
| 8 | 4.3 | 0 | 428 | 9,185 | 128 | 69 | 534 | 91 | 0.74 |
| 16 | 8.1 | 0 | 471 | 13,819 | 341 | 139 | 592 | 92 | 0.63 |
| 32 | 16.0 | 0 | 509 | 21,896 | 693 | 273 | 641 | 95 | 0.53 |
| 64 | 31.6 | 0 | 578 | 36,903 | 1,376 | 537 | 665 | 96 | 0.46 |
| 128 | 62.9 | 0 | 665 | 64,304 | 2,866 | 1,095 | 672 | 97 | 0.41 |
| 256 | 100.0 | 129 | 758 | 103,600 | 25,349 | 1,736 | 674 | 98 | 0.39 |

### Per-metric takeaways (the trend that matters)
- **System throughput** — rises steeply to C≈32, then flattens: +18% from C=64→256 (1,455→1,722) for an 8× concurrency increase. **The Pareto knee is around C=32–64**; beyond it you buy almost no throughput but pay heavily in latency.
- **Interactivity (per-user speed)** — monotonic decline, ~halving every ~4× concurrency (232→8.5). This is the throughput-for-latency trade the curve in §1 traces out.
- **mean TTFT** — flat (<1 s) up to C=32, then blows up (25.5 s at C=256). Time-to-first-token is the first thing users feel when the box saturates.
- **p99 TTFT** — blows up far earlier and harder (441 ms → 113 s). **Tail TTFT is the most sensitive saturation signal** — already >17 s at C=64.
- **mean TPOT / ITL** — rise ~27× (4.3→118 ms) as the decode batch shares the GPU; ITL≈TPOT means decode is steady, not bursty. This sets the per-user "stream feel."
- **KV-cache usage** — climbs to 100% only at C=256; there's headroom below that. Hitting the ceiling is what triggers preemptions.
- **Preemptions** — exactly 0 until KV saturates, then 129 at C=256. **A direct KV-pressure / over-subscription alarm** — nonzero means the engine is evicting & recomputing.
- **Prefill time/req** — sublinear (351→758 ms); prefill is compute-bound and batches efficiently.
- **Decode time/req** — explodes (3.9 s → 104 s); this is the dominant component of end-to-end latency under load.
- **Queue time/req** — 0 → 25 s; the other half of the TTFT blowup — requests waiting for a scheduler slot once saturated.
- **Batch size (tokens/step)** — 9 → 1,736 (~190×); this is the mechanism that buys throughput, and why energy/token falls.
- **GPU power** — 316 → 674 W (96% of the 700 W TDP); tracks utilization, useful for perf-per-watt and thermal budgeting.
- **GPU utilization** — already 80% at C=1, 98% at C=256; the H200 is well-fed even at low load (this is a 120B model).
- **Energy/token** — **1.48 → 0.39 J/tok (3.8× more efficient)**; the strongest argument for running at higher concurrency — better tokens-per-watt — as long as the latency budget allows.

> Note: `prefix-cache hit = 0` (random tokens) and `spec-decode = n/a` (no draft model) by
> design; both are captured and will light up under the tau-bench workload and an MTP model.

---

## 3. Profiling cost vs fidelity

**Question:** if we cap a benchmark cell at a short budget, how close is the measurement to a
full run, and how repeatable is it? We answer it two ways because two kinds of metrics behave
differently:

- **System throughput** is a property of a real wall-clock window (concurrency × rate over
  time), so we measure it with **real repeated runs**: at each budget (`num_prompts = B×conc`)
  we run the cell **5 times independently** against the same warm server and compare to a long
  ground-truth run (`conc×40`).
  - **bias** = |mean(estimate) − ground_truth| / ground_truth — systematic error of a short run.
  - **run-to-run CV** = std/mean across the 5 repeats — *if you run the same short benchmark
    twice, how much does the number move?* (Coefficient of variation; 0.4% ≈ ±0.4% noise.)
- **Latency percentiles** are per-request properties, so we measure their convergence by
  **bootstrap**: resample N requests from the ground-truth run's per-request TTFT array,
  recompute the statistic, 300× → the sampling error of a budget-N measurement, without paying
  for N separate runs. (`cost_s` below = N / completion-rate, the measured-window equivalent.)

### System throughput converges fast and is low-variance (C=64, ref = 1,461 tok/s)
| budget | ≈ cost | throughput bias | run-to-run CV |
|---|--:|--:|--:|
| conc×1 | ~2.2 min | 2.5% | 0.4% |
| conc×2 | ~2.9 min | 1.9% | 0.6% |
| conc×5 | ~4.8 min | 1.0% | 0.2% |
| **conc×10 (InferenceX default)** | **~8.2 min** | **0.35%** | **0.19%** |

(C=4 is similar from ~1 min: conc×10 → 1.9% bias; only the tiny conc×1 run is noisy at 7%.)

### Latency tails converge slowly — and that's the real cost driver (C=64, bootstrap)
| budget | ≈ cost | mean TTFT error | **p99 TTFT error** |
|---|--:|--:|--:|
| conc×1 | ~42 s | 1.3% | **57%** |
| conc×2 | ~84 s | 1.1% | **51%** |
| conc×5 | ~3.5 min | 0.3% | **27%** |
| conc×10 | ~7 min | 0.1% | **13%** |

(Ground truth: 2,560 requests, mean TTFT 718 ms, p99 TTFT 11.6 s.)

### Takeaway
- **Throughput and mean latency are cheap** — within ~1–2% in under a minute, with <1%
  run-to-run noise. The InferenceX default (conc×10) is already well past the knee (0.35%
  throughput bias); a **~5-min cap (≈conc×5) still lands within ~1%**. Throughput/interactivity
  sweeps can be **cut 2–4×** with negligible loss.
- **Tail latency (p99) is expensive** — still **~13% off even at conc×10 (~7 min)** and needs
  the long run to pin down. **Policy:** run the throughput/interactivity curve cheaply
  (conc×2–5), and reserve long runs only for the subset of cells where p99/tail SLOs are
  reported.

### C=256 (saturated regime — ref = 1,723 tok/s)
| budget | ≈ cost | throughput bias | run-to-run CV | p99 TTFT error |
|---|--:|--:|--:|--:|
| conc×1 | ~7 min | 6.8% | 0.10% | ~6% |
| conc×2 | ~9 min | 2.3% | 0.09% | ~6% |
| conc×5 | ~16 min | 1.3% | 0.24% | ~15% |
| conc×10 | ~27 min | 0.12% | 0.05% | ~12% |

Two effects flip at high concurrency: (1) **throughput needs a bit more budget** to converge
(6.8% bias at conc×1 vs 2.5% at C=64) because the saturated server has a longer startup
transient — but it's *extremely* repeatable (CV ≤0.24%); (2) **p99 TTFT converges far better
than at C=64** (~6–15% vs 57%), simply because each budget at C=256 contains 4× more requests —
tail fidelity tracks the absolute request count, not the wall-clock budget. (The absolute p99
of 67 s is itself a saturation artifact.) See the 3-column `cost_fidelity.png`.

---

## 4. Realistic workload — tau-bench replay (all 16 model×domain parts)

Replays Sierra τ-bench agent traffic (real prompt/output lengths + prefix-sharing structure,
synthetic tokens) against gpt-oss-120b **with prefix caching ON**, using the co-worker's
`tau-bench-replay` toolkit. The point: does vLLM's prefix cache realize the *analytical ideal*
hit rate on a realistic, cache-friendly workload — and how do the two regimes (light chat vs
long-context RAG) behave? Light/medium domains at concurrency 64, banking (long-context) at 4.
Plot: `tau_compare.png`; table: `nebius/results_tau/tau_table.tsv`.

### How the replay works (and what "model" means here)
- **We do not send real text — we send synthetic token IDs.** For serving metrics (TTFT,
  decode, throughput, prefix-cache hit) only *sequence lengths* and *prefix-sharing structure*
  drive the engine; token values are irrelevant. From each τ-bench trajectory we keep only the
  *shape* of each call (input len, output len, agent vs. user-simulator stream, per-conversation
  order) and synthesize tokens that reproduce the exact structure — **one shared system block +
  a per-conversation growing body** — with IDs in a fixed range (`--id-lo/--id-hi`, kept below
  the served model's vocab). Tokenizer-independent, so it recreates the same cache-reuse on any
  served model.
- **Replay loop:** each conversation is a closed-loop session run **strictly sequentially**
  (call k+1 only after k returns → the append-only prompt keeps the prefix cache warm), with
  **many sessions concurrent** (`--concurrency`) to load the server. Output length is **forced**
  to the recorded value (`max_tokens=min_tokens`, `ignore_eos`). Fully deterministic — only the
  server's *timing* varies, which is what we measure; realized hit rate is compared to the
  analytical ideal in `stats/`.
- **Two distinct "model" roles:** (1) **source models** define the *workload shape* only —
  `gpt-5-2`, `claude-opus-4-5`, `gemini-3-pro`, `qwen3.5-397b-a17b-think` (× 4 domains = the 16
  parts); (2) the **served model** actually benchmarked on the GPU is **gpt-oss-120b** (FP4,
  1×H200, prefix caching ON). Because tokens are synthetic, the served model is independent of
  which source-model trace is replayed.

| domain | source model | realized hit | ideal (+sys) | gap | out tok/s | TTFT p95 (ms) |
|---|---|--:|--:|--:|--:|--:|
| airline | claude-opus-4-5 | 0.914 | 0.915 | 0.1 | 1,782 | 232 |
| airline | gemini-3-pro | 0.922 | 0.924 | 0.2 | 2,018 | 148 |
| airline | gpt-5-2 | 0.894 | 0.897 | 0.3 | 1,998 | 199 |
| airline | qwen3.5 | 0.920 | 0.927 | 0.7 | 1,943 | 201 |
| retail | claude-opus-4-5 | 0.921 | 0.922 | 0.1 | 1,816 | 244 |
| retail | gemini-3-pro | 0.934 | 0.936 | 0.2 | 2,095 | 146 |
| retail | gpt-5-2 | 0.909 | 0.912 | 0.3 | 1,944 | 192 |
| retail | qwen3.5 | 0.924 | 0.927 | 0.3 | 1,907 | 195 |
| telecom | claude-opus-4-5 | 0.973 | 0.975 | 0.2 | 1,687 | 154 |
| telecom | gemini-3-pro | 0.975 | 0.974 | −0.1 | n/a* | n/a* |
| telecom | gpt-5-2 | 0.968 | 0.969 | 0.1 | 1,721 | 145 |
| telecom | qwen3.5 | 0.973 | 0.978 | 0.5 | 1,821 | 157 |
| banking | claude-opus-4-5 | 0.926 | 0.947 | **2.1** | 453 | 278 |
| banking | gemini-3-pro | 0.940 | 0.966 | **2.6** | 442 | 406 |
| banking | gpt-5-2 | 0.875 | 0.956 | **8.1** | n/a* | n/a* |
| banking | qwen3.5 | 0.898 | 0.939 | **4.1** | 496 | 230 |

\* client-side latency/throughput missing for 2 parts whose outliers crashed the client
(gemini-telecom has a 65k-token output; gpt-5-2-banking a 271k-token input > gpt-oss's 128k
context); their server-side hit rate (from `/metrics`) still landed.

### Findings
- **Light & medium chat (airline / retail / telecom): vLLM realizes the analytical ideal hit
  rate almost exactly — gap <1% everywhere.** Prefix caching "just works"; the realized number
  you'd measure equals what the workload structure allows.
- **Long-context RAG (banking_knowledge): realized falls 2–8% below ideal.** The huge contexts
  (mean 17–62k, up to 271k tokens) create KV pressure, so cached prefix blocks get evicted before
  reuse. The gap is worst for gpt-5-2-banking (8.1%, the largest contexts). **This is the
  actionable result: prefix-cache benefit degrades exactly where contexts are longest** — more KV
  (or higher cache priority for long shared prefixes) would recover it.
- **Two throughput regimes, ~4× apart:** light/medium chat sustains ~1,700–2,100 output tok/s at
  ~150–250 ms TTFT; long-context RAG drops to ~450 tok/s (prefill-bound) at ~280–410 ms TTFT.
- **Tooling note (worth telling the co-worker):** `tau-bench-replay/client.py` reads
  `vllm:gpu_prefix_cache_{hits,queries}_total`, but vLLM v0.22 renamed these to
  `vllm:prefix_cache_{hits,queries}_total` — so its `server_prefix_cache` returns `None`. We
  recovered the headline metric from independent `/metrics` snapshots; the one-line fix is to add
  the new name to the regex in `scrape_prefix_cache`.

---

## Cost of this study
~$6 (1×H200, ~1.7 h) for the sweep + C=64/C=4 cost-fidelity; C=256 adds ~$10. Nodes torn down.
