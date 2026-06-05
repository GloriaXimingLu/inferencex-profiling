# vLLM Profiling on Nebius (InferenceX)

Reproducible setup + measurements for estimating the **wall-time cost of profiling
vLLM performance** on NVIDIA H200 GPUs using [SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
benchmarks on Nebius.

## Start here

| If you want to… | Read |
|---|---|
| **Reproduce the profiling yourself** | [`PROFILING_RUNBOOK.md`](PROFILING_RUNBOOK.md) |
| **See the wall-time / cost numbers** | [`WALLTIME_REPORT.md`](WALLTIME_REPORT.md) |

## Headline results

- **gpt-oss-120b** full H200 vLLM config: **~3.8 h / ~$90** *(measured on real H200)*
- **Full base matrix** (4 models, 99 cells): **~34 h** on 1 node / **~4–5 h** on 8 nodes, **~$700–960** *(big-MoE rows ±50%)*
- Wall time is startup-dominated (vLLM restarts per concurrency point); **DeepSeek-V4-Pro is ~60% of the total**.

## What's in here

```
PROFILING_RUNBOOK.md   step-by-step guide (CLI auth → provision → run → analyze → tear down)
WALLTIME_REPORT.md     the wall-time + dollar cost report, with methodology & caveats
nebius/
  cloud-init.yaml      injects your SSH key (REPLACE the placeholder key before use)
  node_setup.sh        installs docker + nvidia-container-toolkit, clones InferenceX
  calib_1g.sh          calibration cells for a 1×H200 (gpt-oss-120b)
  calib.sh             calibration cells for an 8×H200 (gpt-oss + DeepSeek-V4-Pro)
  analyze.py           turns measured timings.tsv into the full-matrix estimate
  results/             raw measurements (timings.tsv + per-cell vLLM logs) — auditable
```

> **Note:** the `InferenceX/` benchmark repo is **not** vendored here — clone it yourself
> (`git clone https://github.com/SemiAnalysisAI/InferenceX`) as the runbook describes.

## Prerequisites (summary)

Membership in your company's (billing-enabled) Nebius tenant, the Nebius CLI,
and `ssh`/`scp`/`python3` locally. Full details in the runbook.
