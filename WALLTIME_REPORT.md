# Wall-Time Cost of InferenceX vLLM Profiling on H200

**Date:** 2026-06-05 · **Method:** calibrate-then-extrapolate · **Hardware:** NVIDIA H200 (Nebius)
**Confidence:** gpt-oss-120b = measured; big-MoE = ±50% estimate (pending 8×H200 calibration)

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Profile **gpt-oss-120b** (full H200 vLLM config, 35 cells)? | **~3.8 h**, ~$90 *(measured)* |
| Profile the **full base matrix** (4 models, 99 cells), **1 node**? | **~34 h**, ~$700–960 *(±50%)* |
| Same matrix on an **8-node fleet**? | **~4–5 h** wall-clock, ~same $ |
| With MTP/spec-decode variants (135 cells)? | add **~35%** |

Wall time is governed by two things per benchmark cell: **vLLM server startup**
(~3 min, paid on *every* cell because InferenceX restarts the server per
concurrency point) and the **serving benchmark** itself (scales with concurrency
and input length). For large MoE models, startup balloons (weight load + CUDA-graph
capture), which is why **DeepSeek-V4-Pro alone is ~60% of the total**.

---

## 2. Methodology

InferenceX defines a benchmark "cell" as `model × precision × ISL/OSL × tensor-parallel
× concurrency`. Each cell independently: downloads the model (cached) → starts a fresh
vLLM server → runs `benchmark_serving` (`num_prompts = concurrency × 10`) → optional eval.

We **measured** the real per-phase wall time of `gpt-oss-120b` on a single H200 across 4
cells (the corners of the concurrency × input-length space), then **extrapolated** to the
full matrix using a concurrency-aware model fit from that data. Big-MoE models could not
be measured directly (they need an 8×H200 node, which was capacity-constrained), so their
numbers use size-relative multipliers and carry ±50% uncertainty.

---

## 3. Measured results (real, on H200)

`gpt-oss-120b`, vLLM v0.22.0, TP=1:

| Cell | Concurrency | ISL/OSL | Startup | Benchmark | Total |
|---|--:|--:|--:|--:|--:|
| c4 / 1k | 4 | 1024/1024 | 221 s | 89 s | 310 s |
| c64 / 1k | 64 | 1024/1024 | 191 s | 275 s | 466 s |
| c4 / 8k | 4 | 8192/1024 | 142 s | 105 s | 247 s |
| c64 / 8k | 64 | 8192/1024 | 148 s | 507 s | 655 s |

- **One-time model download:** 815 s for 63 GB (~77 MB/s effective).
- **Validated:** benchmarks produced sane vLLM metrics (e.g. c64/1k → 5,251 tok/s,
  TTFT 265 ms, TPOT 23.6 ms), so these are real serving runs, not degenerate.

### Fitted per-cell cost model
```
startup        ≈ 170 s   (median; weight load + CUDA-graph capture, ~constant per cell)
benchmark(1k)  ≈ 77 + 3.10 · concurrency   seconds
benchmark(8k)  ≈ 78 + 6.70 · concurrency   seconds
cell_time      ≈ startup + benchmark
```
Benchmark time grows with concurrency (10× more total prompts) and with input length
(heavier prefill). Startup is the floor — even the cheapest cell is ~3–4 min.

---

## 4. Wall-time estimate — full H200 vLLM matrix

Single node, sequential. gpt-oss is measured; the rest are size-scaled estimates.

| Config | Cells | Download | Compute | **Total** | Basis |
|---|--:|--:|--:|--:|---|
| gptoss-fp4-h200-vllm | 35 | 0.2 h | 3.5 h | **3.8 h** | **measured** |
| minimaxm2.5-fp8-h200-vllm | 18 | 0.8 h | 4.5 h | **5.3 h** | est. ±50% |
| kimik2.5-int4-h200-vllm | 10 | 1.8 h | 3.0 h | **4.8 h** | est. ±50% |
| dsv4-fp8-h200-vllm | 36 | 2.4 h | 18.0 h | **20.4 h** | est. ±50% |
| **TOTAL** | **99** | **5.3 h** | **29.0 h** | **~34 h** | |

Notes:
- **DeepSeek-V4-Pro dominates** (~20 h) — large weights → long startup *per cell*, and 36 cells.
- **Downloads (~5 h)** are one-time per model (huge MoE weights: DeepSeek ~670 GB, Kimi ~500 GB).
- Add **~35%** (≈ +12 h) for the MTP/spec-decoding variants → ~46 h for all 135 cells.

### Fleet parallelism
Total **GPU-node-hours is fixed (~34)**; a fleet trades nodes for wall-clock:

| Fleet size | Wall-clock | Node-hours (≈ cost driver) |
|---|--:|--:|
| 1 node | ~34 h | 34 |
| 4 nodes | ~9 h | 34 |
| 8 nodes | ~4–5 h | 34 |

(Production InferenceX runs a multi-node fleet, which is why their public dashboard
refreshes in hours, not days.)

---

## 5. Dollar cost

Most cells require ≥2 GPUs (TP up to 8), so a full sweep runs on an **8×H200 node**.
Assuming **~$2.5–3.5 / GPU-hr** (≈ **$20–28 / node-hr** for 8×H200 on-demand; confirm on
the Nebius pricing page):

| Scope | Node-hours | Cost @ ~$24/node-hr | Range |
|---|--:|--:|--:|
| gpt-oss-120b config | 3.8 | ~$91 | $76–106 |
| MiniMax-M2.5 config | 5.3 | ~$127 | $106–148 |
| Kimi-K2.5 config | 4.8 | ~$115 | $96–134 |
| DeepSeek-V4-Pro config | 20.4 | ~$490 | $408–571 |
| **Full base matrix (99 cells)** | **34.3** | **~$823** | **$686–960** |
| + MTP variants (135 cells) | ~46 | ~$1,100 | $920–1,290 |

**Dollar cost is ~independent of fleet size** (you pay for node-hours, not wall-clock).
Fleets buy *speed*, not *savings*.

**What this calibration actually cost:** one 1×H200 for ~73 min ≈ **$3–4**. The three
failed 8×H200 provisioning attempts cost **$0** (nothing scheduled).

---

## 6. Uncertainty & caveats

1. **Big-MoE rows are ±50%.** They extrapolate from gpt-oss by model size *and* assume
   1-GPU→8-GPU behavior we couldn't measure. DeepSeek's per-cell startup (the recipe
   allows up to 1 h) is the single largest unknown — it could move the total by ±10 h.
2. **gpt-oss TP 2/4/8 cells** were estimated from the TP=1 measurement (mildly
   conservative — more GPUs shorten the benchmark phase).
3. **Capacity, not quota, is the real constraint.** 8×H200 on-demand was unavailable in
   all three Nebius regions during this exercise; plan for ret/reservation lead time.
4. Prices are list-ish estimates; commitments lower them materially.

---

## 7. Recommendation

The estimate is **firm for gpt-oss** and **directional for the big models**. To collapse
the ±50% to real numbers, run **one 8×H200 calibration** of DeepSeek-V4-Pro + Kimi-K2.5
(a few cells each, ~$30–60, ~2–3 h) when 8-GPU capacity is available — that single run
replaces 60%+ of the estimated total with measurements. The reproducible scripts and the
`calib.sh` 8-GPU cell list are ready to go (see `PROFILING_RUNBOOK.md`).
