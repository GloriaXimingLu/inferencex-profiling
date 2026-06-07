# vLLM Profiling Experiments — Fireworks Alignment, Metrics, Cost-vs-Fidelity

**Date:** 2026-06-07 · **Hardware:** 1× NVIDIA H200 (Nebius) · **Model:** gpt-oss-120b (FP4/MXFP4), vLLM v0.22.0
**Workload:** random tokens, ISL≈8000 / OSL=1000, `ignore_eos`, prefix-caching off (InferenceX recipe)

Four experiments requested by the co-worker. All ran on a single H200; raw data in
`nebius/results_exp/`, plots `fireworks_compare.png` and `cost_fidelity.png`, full metric
table `metrics_table.tsv`. Reproduce via `nebius/exp_*.sh` (see `PROFILING_RUNBOOK.md`).

---

## 1. GPU availability: non-preemptible VMs

**Tested the hypothesis empirically.** Non-preemptible **8×H200 still fails** to schedule
(`NotEnoughResources` / VM schedule timeout) across eu-north1, eu-west1, us-central1 —
same as before. So *"non-preemptible → more GPUs"* **did not hold for full 8-GPU nodes.**
Single **1×H200** schedules fine (occasionally contended in eu-north1 because the company
projects are **shared** — a co-worker's instance was running there; used eu-west1 instead).

**Takeaway:** all single-GPU work (these four experiments) is unaffected. For the full
8×H200 matrix, on-demand capacity remains the blocker; use a reservation/capacity-block
or keep retrying across regions. A failed create costs $0.

---

## 2. Fireworks-aligned throughput-vs-interactivity plot

**Alignment was already in place.** InferenceX's `run_benchmark_serving` already passes
`--ignore-eos` and `--num-warmups $((2*conc))` (warmup requests are run then **discarded**),
on top of the measured `num_prompts` (gpt-oss uses `conc*10`). That is **exactly** the
Fireworks recipe: `conc*12` total = `2*conc` warmup (dropped) + `10*conc` measured. The
"we used conc×10" was the *measured* portion; the warmup was already there. Our harness
(`exp_lib.sh`) replicates it precisely. (Sanity check: co-worker's vLLM C=4 8k/1k wall time
= 105 s; our independent calibration measured 105 s.)

### Result — vLLM (ours) vs Fireworks (`fireworks_compare.png`)

| C | ours: interactivity (tok/s/user) | ours: sys-tput (tok/s) | Fireworks: interactivity | Fireworks: sys-tput |
|--:|--:|--:|--:|--:|
| 1 | 232 | 214 | 310 | 250 |
| 4 | 140 | 520 | 255 | 800 |
| 16 | 63 | 946 | 113 | 1570 |
| 64 | 24 | 1455 | 35 | 2300 |
| 256 | 8.5 | 1722 | 33 | 2300 |

**Fireworks is faster on both axes at every concurrency** — ~1.3–1.6× higher system
throughput at matched load, and higher per-user speed. Two drivers:
1. **Speculative decoding** (Fireworks reported 67% acceptance; our vanilla gpt-oss has no
   draft model). Spec-decode raises both interactivity (more tokens/step) and throughput.
2. **High-concurrency saturation on a single H200.** Our vanilla vLLM becomes **queue-bound**
   at high C: p99 TTFT 35 s @ C=128 and **113 s @ C=256**, KV-cache hits **100%** at C=256
   (→ **129 preemptions**), and throughput plateaus at ~1722 tok/s. Fireworks plateaus at
   ~33 tok/s/user / ~2300 tok/s — a more graceful saturation.

This is the apples-to-(almost)-apples comparison the co-worker wanted; the spec-decode
caveat is the headline difference. To close it, enable an MTP/draft model in vLLM and rerun.

---

## 3. Comprehensive metrics (survey + recording)

Full design in **`METRICS.md`**; per-run values in **`metrics_table.tsv`**. We capture
**~30 metrics/run** across four channels wired into every cell (`exp_lib.sh`):
client JSON, vLLM `/metrics` counter deltas (m1−m0), 1 Hz gauge+GPU sampler, server log.
The `/metrics` parser matched the **real vLLM v0.22.0 names** (359 series; verified against a
per-run `metrics_catalog.prom` dump).

**Highlights across the sweep (C=1 → 256):**
| Metric | C=1 | C=64 | C=256 | Notes |
|---|--:|--:|--:|---|
| System throughput (tok/s) | 214 | 1455 | 1722 | saturates |
| Interactivity (tok/s/user) | 232 | 24 | 8.5 | degrades under load |
| p99 TTFT (ms) | 441 | 17,368 | 113,217 | queue-bound at high C |
| KV-cache usage (max %) | 0.5 | 32 | **100** | KV pressure at C=256 |
| Preemptions | 0 | 0 | **129** | recompute under KV pressure |
| Prefill / decode / queue (avg ms) | 351 / 3914 / 0 | 947 / 32k / 5.4k | — | queue dominates at load |
| GPU power (mean W) | 316 | 665 | 674 | → 700 W TDP |
| GPU util (mean %) | 80 | ~93 | ~95 | compute-saturated |

- **Prefix-cache hit = 0** and **spec-decode = n/a**, *by design* (random tokens, prefix
  caching off, no draft model). Both are **wired and verified present in `/metrics`**; they
  light up under (a) the tau-bench realistic workload (your PDF #2) and (b) an MTP model.
- The **counter-delta approach beat log-parsing**: it caught the 129 preemptions at C=256
  even though the messages had scrolled out of the log tail. (Caveat: `cudagraph_captured`
  is detected from the log tail, so it only shows for the first few runs after startup —
  CUDA-graphs *were* captured once at startup and **no eager fallback** occurred in any run.)

---

## 4. Profiling cost vs fidelity (`cost_fidelity.png`)

**Method.** One long `--save-detailed` ground-truth run per cell (C=64 high-throughput,
C=4 high-interactivity), plus repeated independent short runs at budgets `B×conc` measured
prompts (B∈{1,2,5,10}, 5 repeats). Fidelity for **system throughput** = bias of the mean vs
the long reference + run-to-run **CV**; for **latency tails** = bootstrap convergence of
mean/p99 TTFT from the ground-truth per-request array.

### System throughput converges fast and is low-variance
| C=64 budget | ≈ cost | throughput bias | run-to-run CV |
|---|--:|--:|--:|
| conc×1 | ~2.2 min | 2.5% | 0.4% |
| conc×2 | ~2.9 min | 1.9% | 0.6% |
| conc×5 | ~4.8 min | 1.0% | 0.2% |
| **conc×10 (InferenceX default)** | **~8.2 min** | **0.35%** | **0.19%** |

(C=4 is similar from ~1 min: conc×10 → 1.9% bias; only the tiny conc×1 run is noisy at 7%.)

### Tail latency converges slowly
p99 TTFT needs **far more** samples than throughput or mean latency — at short budgets its
bootstrap error is tens of percent, while mean TTFT and throughput are already within a few
percent (bottom panels of the figure).

### Recommendation
- **Throughput / mean-latency sweeps can be cut 2–4×** in wall time with <2.5% fidelity
  loss. The InferenceX default (conc×10) sits comfortably past the throughput "knee"
  (0.35% bias) — and a ~5 min cap (≈conc×5) still lands within ~1%.
- **Keep long runs only where p99 tails matter.** A practical policy: run cells at conc×2–5
  for the throughput/interactivity curve, and reserve conc×10+ (or longer) for the subset
  where tail SLOs are reported.

---

## Cost of this study
One 1×H200 for ~1.7 h (download + sweep + cost-fidelity) ≈ **$6**. Failed 8×H200 probes: $0.
Node torn down; nothing billing.
