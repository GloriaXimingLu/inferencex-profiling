# fw-inferencemax-bench

Black-box, InferenceMax-style serving benchmark for an OpenAI-compatible endpoint
(defaults to **Fireworks** + **gpt-oss-120b**, ISL≈8000 / OSL=1000).

It sweeps concurrency and produces the throughput-vs-interactivity Pareto curve,
plus TTFT / TPOT / E2E percentiles — the client-observable half of what
InferenceMax measures. It does **not** require installing vLLM.

## Why this is "comparable" (and where it isn't)

- ✅ Same shape of result: throughput vs interactivity (1/TPOT) across concurrency.
- ✅ Uses **server-reported `usage`** as the source of truth for token counts, so a
  tokenizer mismatch or early stopping cannot inflate throughput.
- ✅ **Cache-cold by default** (unique random prefix per request) → real prefill.
- ⚠️ On **serverless** you can't get tok/s **per GPU** or true $/token (multi-tenant,
  unknown hardware). For apples-to-apples-ish numbers, run a **dedicated** Fireworks
  deployment with a known GPU count and pass `--gpus N` / `--price-gpu-hr` to plot.py.
- ⚠️ Fireworks' serving stack/version is opaque; this measures the *provider*, not a
  specific framework build. Don't publish these as official InferenceMax results.

## Setup

Uses [uv](https://docs.astral.sh/uv/). Dependencies are declared inline in the
scripts (PEP 723), so there's **no venv to create** — `uv run` builds an ephemeral
env and caches it on first run.

```bash
cd fw-inferencemax-bench
export FIREWORKS_API_KEY=fw_...        # get this from the Fireworks console
```

That's it. (Don't have uv? `curl -LsSf https://astral.sh/uv/install.sh | sh`)

By default prompt sizing uses tiktoken `o200k_base`. For exact gpt-oss tokenization,
add `transformers` to the `dependencies` block at the top of `fw_bench.py` — token
*counts* come from the server either way, so this only affects input-length sizing.

## Run

Single point:
```bash
uv run fw_bench.py --concurrency 32 --input-len 8000 --output-len 1000
```

Full sweep + plot (`sweep.sh` calls `uv run` for you):
```bash
./sweep.sh                              # writes results/<timestamp>/c0001.json ...
uv run plot.py results/<timestamp>      # prints table + writes pareto.png
```

Dedicated deployment (recommended for comparability), e.g. 8 GPUs at $X/GPU-hr:
```bash
BASE_URL=https://<your-dedicated-endpoint>/v1 \
MODEL=accounts/<you>/models/gpt-oss-120b ./sweep.sh
uv run plot.py results/<timestamp> --gpus 8 --price-gpu-hr 2.90 --csv
```

## Knobs

| flag / env | meaning |
|---|---|
| `MODEL`, `BASE_URL`, `ISL`, `OSL` | model, endpoint, input/output token targets |
| `CONCURRENCY="1 4 16 64 256"` | which concurrency points to sweep |
| `--cache-warm` (via `EXTRA`) | reuse identical prompt to MEASURE prefix-cache effect |
| `--ignore-eos` (via `EXTRA`) | try to force full output length (backend may ignore) |
| `--gpus N` (plot) | divide throughput → tok/s/GPU |
| `--price-gpu-hr` (plot) | compute $/1M tokens |

## Reading the curve

Each point = one concurrency level. Low concurrency → fast per user (high
interactivity), low total throughput. High concurrency → slow per user, high
throughput. Overlay InferenceMax's published gpt-oss-120b curve to see where
Fireworks lands at a given interactivity target.

## Gotchas already handled / to watch

- **Prefix cache**: busted by default; use `--cache-warm` to characterize it.
- **Short outputs**: throughput normalized by *actual* `completion_tokens`.
- **Network/region**: TTFT includes round-trip — run the client near the region.
- **Rate limits / autoscaling** on serverless can make "throughput" jump as you're
  spread across replicas; that's provider scaling, not per-GPU perf. Prefer dedicated.
