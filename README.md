# vLLM Profiling on Nebius (InferenceX)

Reproducible setup + measurements for estimating the **wall-time cost of profiling
vLLM performance** on NVIDIA H200 GPUs using [SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
benchmarks on Nebius.

## Start here

| If you want to… | Read |
|---|---|
| **Reproduce the profiling yourself** | [`PROFILING_RUNBOOK.md`](PROFILING_RUNBOOK.md) |
| **See the wall-time / cost numbers** | [`WALLTIME_REPORT.md`](WALLTIME_REPORT.md) |
| **Fireworks comparison, metrics, cost-vs-fidelity** | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| **The comprehensive metric set & how it's recorded** | [`METRICS.md`](METRICS.md) |

## Headline results

**Wall-time / cost (see `WALLTIME_REPORT.md`)**
- **gpt-oss-120b** full H200 vLLM config: **~3.8 h / ~$90** *(measured on real H200)*
- **Full base matrix** (4 models, 99 cells): **~34 h** on 1 node / **~4–5 h** on 8 nodes, **~$700–960** *(big-MoE rows ±50%)*

**Experiments (see `EXPERIMENTS.md`)**
- **vs Fireworks**: our vanilla vLLM trails Fireworks ~1.3–1.6× on throughput *and* interactivity at matched concurrency — the speculative-decoding gap (Fireworks 67% accept; vanilla gpt-oss none). At C=256 a single H200 is queue-bound (p99 TTFT 113 s, KV 100%, 129 preemptions).
- **Metrics**: ~30 metrics/run captured (client + vLLM `/metrics` + GPU + logs); see `METRICS.md`.
- **Cost vs fidelity**: system throughput is within **~0.35%** at the InferenceX default (conc×10) and within ~1% at a 5-min cap; **p99 tail latency** is the only metric needing the long run.

## What's in here

```
PROFILING_RUNBOOK.md   step-by-step guide (CLI auth → provision → run → analyze → tear down)
WALLTIME_REPORT.md     the wall-time + dollar cost report, with methodology & caveats
EXPERIMENTS.md         Fireworks comparison, metrics, cost-vs-fidelity findings
METRICS.md             comprehensive metric set + how each is recorded in InferenceX
nebius/
  cloud-init.yaml      injects your SSH key (REPLACE the placeholder key before use)
  node_setup.sh        installs docker + nvidia-container-toolkit, clones InferenceX
  calib_1g.sh / calib.sh   wall-time calibration cells (1×H200 / 8×H200)
  analyze.py           turns measured timings.tsv into the full-matrix estimate
  exp_lib.sh           persistent server + run_measured (captures full metric set/run)
  exp_sweep.sh         Fireworks throughput-vs-interactivity sweep (C=1..256)
  exp_costfid.sh       cost-vs-fidelity experiment (ground-truth + budget sweep)
  analyze_metrics.py   parse runs → comprehensive metrics_table.tsv
  plot_fireworks.py    throughput-vs-interactivity plot (overlays Fireworks)
  plot_costfid.py      cost-vs-fidelity plot (bootstrap + real variance)
  results/             wall-time calibration data
  results_exp/         experiment data: metrics_table.tsv, plots (.png), per-run JSON
```

> **Note:** the `InferenceX/` benchmark repo is **not** vendored here — clone it yourself
> (`git clone https://github.com/SemiAnalysisAI/InferenceX`) as the runbook describes.

## Prerequisites (summary)

Membership in your company's (billing-enabled) Nebius tenant, the Nebius CLI,
and `ssh`/`scp`/`python3` locally. Full details in the runbook.
