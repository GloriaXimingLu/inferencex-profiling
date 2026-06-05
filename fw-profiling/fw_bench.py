#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.30",
#   "tiktoken>=0.7",
#   # For exact gpt-oss tokenization add "transformers>=4.44" (heavier).
#   # Token *counts* come from the server regardless, so tiktoken sizing is fine.
# ]
# ///
"""
fw_bench.py — single-concurrency-point load test against an OpenAI-compatible
endpoint (default: Fireworks), in the spirit of InferenceMax's serving benchmark.

What it measures, per concurrency point:
  - TTFT  (time to first token / prefill)            p50/p90/p99
  - TPOT  (time per output token / decode speed)     p50/p90/p99
  - E2E   (end to end latency)                        p50/p90/p99
  - System output throughput (tok/s)  over the saturated window
  - Interactivity (tok/s/user = 1 / median TPOT)

Key correctness choices (the gotchas that silently break black-box benchmarks):
  - Uses the SERVER-reported `usage` (prompt_tokens / completion_tokens) as the
    ground truth for token counts, so a client/server tokenizer mismatch and
    early-stopping (ignore_eos not honored) cannot inflate throughput.
  - Prepends a unique random prefix to every prompt to defeat automatic prompt
    caching, so you measure real prefill (cache-cold). Use --cache-warm to flip.
  - Holds concurrency fixed (request_rate = inf, capped in-flight) and reports
    throughput over the saturated window only, dropping warmup requests.

Run one point; use sweep.sh to sweep concurrency and plot.py to draw the curve.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time
import uuid

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("pip install openai>=1.0  (see requirements.txt)")


# ----------------------------- prompt building ------------------------------ #

def _make_tokenizer(name: str):
    """Return a callable encode(str)->list[int] and decode(list[int])->str,
    plus a vocab size, trying transformers then tiktoken then a word fallback."""
    if name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
            vocab = tok.vocab_size
            return (lambda s: tok.encode(s, add_special_tokens=False),
                    lambda ids: tok.decode(ids),
                    vocab, f"transformers:{name}")
        except Exception as e:
            print(f"[warn] transformers tokenizer '{name}' unavailable ({e}); "
                  f"falling back to tiktoken", file=sys.stderr)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (enc.encode, enc.decode, enc.n_vocab, "tiktoken:o200k_base")
    except Exception:
        # crude word fallback: ~1.3 tokens per word
        return (lambda s: s.split(),
                lambda ids: " ".join(map(str, ids)),
                50000, "wordcount-fallback")


def build_prompt(n_tokens: int, encode, decode, vocab: int) -> str:
    """Build a pseudo-random prompt of approximately n_tokens tokens.
    Random content => unique => no cross-request prefix-cache hits."""
    # sample random token ids in a safe interior range to dodge special tokens
    lo, hi = 100, max(101, vocab - 100)
    ids = [random.randint(lo, hi) for _ in range(max(1, n_tokens))]
    try:
        text = decode(ids)
    except Exception:
        text = " ".join(random.choices(string.ascii_lowercase, k=n_tokens))
    return text


def cache_buster() -> str:
    return f"req-{uuid.uuid4().hex} "


# ------------------------------- one request -------------------------------- #

def _gen_piece(delta):
    """Return any newly generated text from a stream delta, counting BOTH the
    answer (`content`) and reasoning-model thinking (`reasoning_content`) so that
    TTFT/TPOT cover the real generation window, matching usage.completion_tokens."""
    txt = getattr(delta, "content", None)
    if txt:
        return txt
    rc = getattr(delta, "reasoning_content", None)
    if rc:
        return rc
    extra = getattr(delta, "model_extra", None) or {}
    return extra.get("reasoning_content") or extra.get("reasoning")


async def one_request(client, model, prompt, max_tokens, temperature,
                      ignore_eos, reasoning_effort, results, errors):
    start = time.perf_counter()
    first_tok_t = None
    last_tok_t = None
    usage = None
    extra = {}
    if ignore_eos:
        # Fireworks honors ignore_eos; with max_tokens this forces a FIXED output
        # length (no min_tokens on Fireworks, so ignore_eos is the lever).
        extra["ignore_eos"] = True
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra or None,
        )
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if _gen_piece(delta):
                if first_tok_t is None:
                    first_tok_t = now
                last_tok_t = now
    except Exception as e:
        errors.append(str(e))
        return

    end = time.perf_counter()
    if first_tok_t is None or usage is None:
        errors.append("no content or no usage returned")
        return

    out_tok = usage.completion_tokens
    in_tok = usage.prompt_tokens
    # how much of the output was reasoning (gpt-oss etc.), if the backend reports it
    reasoning_tok = None
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tok = getattr(details, "reasoning_tokens", None)
    ttft = first_tok_t - start
    decode_time = last_tok_t - first_tok_t
    tpot = decode_time / (out_tok - 1) if out_tok > 1 else None
    e2e = end - start
    results.append({
        "start": start, "end": end,
        "ttft": ttft, "tpot": tpot, "e2e": e2e,
        "in_tok": in_tok, "out_tok": out_tok, "reasoning_tok": reasoning_tok,
    })


# --------------------------------- sweep ------------------------------------ #

async def run_point(args):
    encode, decode, vocab, tok_name = _make_tokenizer(args.tokenizer)
    print(f"[info] tokenizer: {tok_name}; vocab≈{vocab}", file=sys.stderr)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                         max_retries=0, timeout=args.timeout)

    # pre-build prompts (each unique). cache-warm reuses one shared body.
    shared = build_prompt(args.input_len, encode, decode, vocab)
    prompts = []
    for _ in range(args.num_prompts):
        if args.cache_warm:
            prompts.append(shared)                    # identical => cache hits
        else:
            prompts.append(cache_buster() +
                           build_prompt(args.input_len, encode, decode, vocab))

    results, errors = [], []
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(p):
        async with sem:
            await one_request(client, args.model, p, args.output_len,
                              args.temperature, args.ignore_eos,
                              args.reasoning_effort, results, errors)

    print(f"[info] concurrency={args.concurrency} prompts={args.num_prompts} "
          f"ISL≈{args.input_len} OSL={args.output_len} ...", file=sys.stderr)
    t0 = time.perf_counter()
    await asyncio.gather(*(guarded(p) for p in prompts))
    wall = time.perf_counter() - t0

    if not results:
        sys.exit(f"[fatal] all requests failed. first error: "
                 f"{errors[0] if errors else 'unknown'}")

    # Drop warmup = the first `warmup` requests to be ADMITTED (ramp-up while the
    # concurrency pipe fills / caches are cold). Sort by START, not end: dropping
    # the first-to-FINISH would delete the fastest/shortest requests and bias both
    # latency (up) and throughput (down) — wrong, especially without --ignore-eos.
    # Sorting by start also keeps the throughput window below consistent with the
    # token sum (same set). No tail/drain drop, matching InferenceMAX.
    results.sort(key=lambda r: r["start"])
    measured = results[args.warmup:] if len(results) > args.warmup else results

    def pct(xs, p):
        xs = sorted(v for v in xs if v is not None)
        if not xs:
            return None
        k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return xs[k]

    ttfts = [r["ttft"] for r in measured]
    tpots = [r["tpot"] for r in measured if r["tpot"] is not None]
    e2es = [r["e2e"] for r in measured]

    # saturated-window throughput: sum output tokens over the measured set,
    # divided by the wall-clock span of that set.
    win_start = min(r["start"] for r in measured)
    win_end = max(r["end"] for r in measured)
    window = max(1e-9, win_end - win_start)
    out_tokens = sum(r["out_tok"] for r in measured)
    sys_tput = out_tokens / window

    med_tpot = statistics.median(tpots) if tpots else None
    interactivity = (1.0 / med_tpot) if med_tpot else None

    out = {
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "input_len_target": args.input_len,
        "output_len_target": args.output_len,
        "cache_warm": args.cache_warm,
        "ignore_eos": args.ignore_eos,
        "num_prompts": args.num_prompts,
        "completed": len(results),
        "errors": len(errors),
        "warmup_dropped": len(results) - len(measured),
        "measured_requests": len(measured),
        "actual_in_tok_median": statistics.median(r["in_tok"] for r in measured),
        "actual_out_tok_median": statistics.median(r["out_tok"] for r in measured),
        "reasoning_tok_median": (statistics.median(
            r["reasoning_tok"] for r in measured if r["reasoning_tok"] is not None)
            if any(r["reasoning_tok"] is not None for r in measured) else None),
        "ttft_s": {"p50": pct(ttfts, 50), "p90": pct(ttfts, 90), "p99": pct(ttfts, 99)},
        "tpot_s": {"p50": pct(tpots, 50), "p90": pct(tpots, 90), "p99": pct(tpots, 99)},
        "e2e_s":  {"p50": pct(e2es, 50),  "p90": pct(e2es, 90),  "p99": pct(e2es, 99)},
        "system_output_tok_per_s": sys_tput,
        "interactivity_tok_per_s_per_user": interactivity,
        "wall_s": wall,
        "first_errors": errors[:3],
    }

    print(json.dumps(out, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[info] wrote {args.output}", file=sys.stderr)

    # one-line human summary to stderr
    def ms(x):
        return f"{x*1000:.0f}ms" if x is not None else "n/a"
    print(f"[done] C={args.concurrency:>4}  "
          f"TTFT p50={ms(out['ttft_s']['p50'])} p99={ms(out['ttft_s']['p99'])}  "
          f"TPOT p50={ms(out['tpot_s']['p50'])}  "
          f"sys={sys_tput:.0f} tok/s  "
          f"user={interactivity:.1f} tok/s  "
          f"errs={len(errors)}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="accounts/fireworks/models/gpt-oss-120b")
    p.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    p.add_argument("--api-key", default=os.environ.get("FIREWORKS_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--input-len", type=int, default=8000)
    p.add_argument("--output-len", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="total requests issued; default = concurrency * 12 "
                        "(InferenceMAX measures 10x after dropping 2x warmup)")
    p.add_argument("--warmup", type=int, default=None,
                   help="requests dropped from stats; default = concurrency * 2 "
                        "(matches InferenceMAX --num-warmups)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tokenizer", default="openai/gpt-oss-120b",
                   help="HF id for prompt sizing; falls back to tiktoken o200k_base")
    p.add_argument("--ignore-eos", action="store_true",
                   help="force full output length: with max_tokens this pins OSL "
                        "(Fireworks honors ignore_eos; there is no min_tokens)")
    p.add_argument("--reasoning-effort", default=None,
                   help="for reasoning models (gpt-oss): low|medium|high|max|none "
                        "or an int token budget. Match InferenceMax's setting.")
    p.add_argument("--cache-warm", action="store_true",
                   help="reuse identical prompt to MEASURE prefix-cache behavior")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--output", default=None, help="write result JSON here")
    a = p.parse_args()
    if not a.api_key:
        sys.exit("set FIREWORKS_API_KEY (or pass --api-key)")
    if a.num_prompts is None:
        # InferenceMAX: 10x concurrency measured + 2x concurrency warmup (dropped).
        a.num_prompts = a.concurrency * 12
    if a.warmup is None:
        a.warmup = a.concurrency * 2
    return a


if __name__ == "__main__":
    asyncio.run(run_point(parse_args()))
