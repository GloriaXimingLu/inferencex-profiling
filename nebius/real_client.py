#!/usr/bin/env python3
"""Replay a REAL-TOKEN tau-bench schedule against a vLLM server and measure
prefix-cache hit rate + speculative-decode acceptance on realistic traffic.

Sends reconstructed prompt TEXT (system policy + real conversation history) to
/v1/completions (the server tokenizes with gpt-oss's tokenizer). Closed-loop:
each simulation issues its calls sequentially (keeps the cache warm); many
simulations run concurrently to load the server. Scrapes /metrics before/after
for prefix_cache_{hits,queries}_total and spec_decode_num_{accepted,draft}_tokens.

Usage: real_client.py --schedule F.jsonl --base http://localhost:8888
                      --concurrency 8 --max-sims 200 --out out.json
"""
import argparse, asyncio, json, time, re, urllib.request
import aiohttp

def scrape(base):
    try:
        txt = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception as e:
        return {"error": str(e)}
    def s(pat):
        tot = 0.0; hit = False
        for m in re.finditer(pat + r"\S*\s+([\d.eE+]+)", txt):
            tot += float(m.group(1)); hit = True
        return tot if hit else None
    return {
        "pc_hits": s(r"vllm:(?:gpu_)?prefix_cache_hits_total"),
        "pc_queries": s(r"vllm:(?:gpu_)?prefix_cache_queries_total"),
        "spec_accepted": s(r"vllm:spec_decode_num_accepted_tokens_total"),
        "spec_draft": s(r"vllm:spec_decode_num_draft_tokens_total"),
    }

def prompt_for(rec, i):
    t = rec["turns"][i]
    sysp = rec["agent_system"] if t["stream"] == "agent" else rec["user_system"]
    return sysp + "".join(x["text"] for x in rec["turns"][:i])

def msgs_for(rec, i):
    """Chat-format: policy as system, the real history as a user turn. Puts the
    reasoning model in its native chat mode (vs raw /v1/completions)."""
    t = rec["turns"][i]
    sysp = rec["agent_system"] if t["stream"] == "agent" else rec["user_system"]
    hist = "".join(x["text"] for x in rec["turns"][:i])
    return [{"role": "system", "content": sysp}, {"role": "user", "content": hist}]

async def one_call(sess, base, model, payload, out_len, stats, chat):
    if chat:
        url = base + "/v1/chat/completions"
        body = {"model": model, "messages": payload, "max_tokens": max(1, out_len),
                "temperature": 0.0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
    else:
        url = base + "/v1/completions"
        body = {"model": model, "prompt": payload, "max_tokens": max(1, out_len),
                "temperature": 0.0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; n = 0
    try:
        async with sess.post(url, json=body) as r:
            if r.status != 200:
                stats["errors"] += 1
                stats["last_status"] = r.status
                return
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                try:
                    obj = json.loads(data)
                    u = obj.get("usage")
                    if u and u.get("completion_tokens"):
                        n = u["completion_tokens"]
                except Exception:
                    pass
    except Exception as e:
        stats["errors"] += 1
        return
    now = time.perf_counter()
    stats["ttft"].append(((first or now) - t0) * 1000)
    stats["e2e"].append((now - t0) * 1000)
    stats["out_tokens"] += n or out_len
    stats["calls"] += 1

async def run_sim(sess, base, model, rec, stats, sem, chat):
    async with sem:
        for i, t in enumerate(rec["turns"]):
            if t["is_call"]:
                payload = msgs_for(rec, i) if chat else prompt_for(rec, i)
                await one_call(sess, base, model, payload, t["output_len"], stats, chat)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--base", default="http://localhost:8888")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-sims", type=int, default=200)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--chat", action="store_true", help="use /v1/chat/completions (native chat mode)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # vLLM validates the model name — fetch the actually-served id.
    try:
        md = json.loads(urllib.request.urlopen(a.base + "/v1/models", timeout=10).read())
        model = md["data"][0]["id"]
    except Exception:
        model = a.model
    print("using model:", model)
    recs = [json.loads(l) for l in open(a.schedule)][:a.max_sims]
    stats = {"ttft": [], "e2e": [], "out_tokens": 0, "calls": 0, "errors": 0}
    m0 = scrape(a.base)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=64 * 1024 * 1024) as sess:
        await asyncio.gather(*(run_sim(sess, a.base, model, r, stats, sem, a.chat) for r in recs))
    wall = time.perf_counter() - t0
    m1 = scrape(a.base)
    def pctl(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    def delta(k):
        try: return m1[k] - m0[k]
        except Exception: return None
    hit = q = acc = dr = None
    if delta("pc_queries"):
        q = delta("pc_queries"); hit = delta("pc_hits")
    if delta("spec_draft"):
        dr = delta("spec_draft"); acc = delta("spec_accepted")
    res = {
        "schedule": a.schedule, "concurrency": a.concurrency, "sims": len(recs),
        "calls": stats["calls"], "errors": stats["errors"], "wall_s": round(wall, 1),
        "output_tok_per_s": round(stats["out_tokens"] / wall) if wall else 0,
        "ttft_ms": {"p50": round(pctl(stats["ttft"], .5)), "p95": round(pctl(stats["ttft"], .95)),
                    "p99": round(pctl(stats["ttft"], .99))},
        "realized_prefix_cache_hit": round(hit / q, 4) if (hit is not None and q) else None,
        "spec_acceptance": round(acc / dr, 4) if (acc is not None and dr) else None,
        "m0": m0, "m1": m1,
    }
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("calls", "errors", "output_tok_per_s",
        "ttft_ms", "realized_prefix_cache_hit", "spec_acceptance")}, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
