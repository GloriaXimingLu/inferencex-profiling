#!/usr/bin/env python3
"""Parse one experiment's raw artifacts into a comprehensive per-run metric table.

For each run (row in runs.tsv) combines:
  client JSON   -> throughput, TTFT/TPOT/ITL/E2E (mean/median/std/p50/p90/p95/p99)
  m0/m1 deltas  -> prefix-cache hit, preemptions, prefill/decode/queue time, spec-decode,
                   avg iteration tokens, server-side TTFT/TPOT
  .samples      -> KV-cache usage %, running/waiting requests, GPU power/util/mem (mean & max)
  .serverlog    -> CUDA-graph captured? eager-fallback? preemption warnings

Usage: python analyze_metrics.py <results_dir> [--prefix sweep_]
Writes <results_dir>/metrics_table.tsv and prints a readable summary.
"""
import sys, os, re, json, glob, csv, statistics as st

def parse_prom(path):
    """Prometheus text -> dict. Counters/gauges summed across label sets.
    Histograms: keep <name>_sum and <name>_count aggregated."""
    d = {}
    if not os.path.exists(path): return d
    with open(path) as f:
        for line in f:
            if not line or line[0] == '#': continue
            m = re.match(r'([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+naN]+)', line.strip())
            if not m: continue
            name, val = m.group(1), m.group(3)
            try: v = float(val)
            except ValueError: continue
            d[name] = d.get(name, 0.0) + v
    return d

def find(d, *patterns):
    """Sum all metric values whose name matches ANY regex pattern."""
    tot, hit = 0.0, False
    for k, v in d.items():
        if any(re.search(p, k) for p in patterns):
            tot += v; hit = True
    return tot if hit else None

def delta(m1, m0, *patterns):
    a, b = find(m1, *patterns), find(m0, *patterns)
    if a is None or b is None: return None
    return a - b

def parse_samples(path):
    """Return dict of series -> list of floats. Lines: 'ts <t>', 'gpu p,u,m,mem',
    or 'vllm:metric{...} value'."""
    series = {"kv":[], "running":[], "waiting":[], "power":[], "util":[], "mem":[]}
    if not os.path.exists(path): return series
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('gpu '):
                parts = [p.strip() for p in line[4:].split(',')]
                try:
                    series["power"].append(float(parts[0]))
                    series["util"].append(float(parts[1]))
                    series["mem"].append(float(parts[3]))
                except (ValueError, IndexError): pass
            elif 'cache_usage' in line:
                m = re.search(r'\s([-\d.eE+]+)$', line)
                if m: series["kv"].append(float(m.group(1)))
            elif 'num_requests_running' in line:
                m = re.search(r'\s([-\d.eE+]+)$', line)
                if m: series["running"].append(float(m.group(1)))
            elif 'num_requests_waiting' in line:
                m = re.search(r'\s([-\d.eE+]+)$', line)
                if m: series["waiting"].append(float(m.group(1)))
    return series

def mm(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs: return (None, None)
    return (round(sum(xs)/len(xs), 2), round(max(xs), 2))

def parse_serverlog(path):
    if not os.path.exists(path): return {}
    txt = open(path, errors='ignore').read().lower()
    return {
        "cudagraph_captured": ("capturing cudagraph" in txt or "capturing cuda graph" in txt
                               or "graph capturing finished" in txt),
        "eager_fallback": ("falling back to eager" in txt or "fallback to eager" in txt
                            or "disabling cudagraph" in txt),
        "preempt_warn": txt.count("preempt"),
    }

def pct(d, metric, p):
    return d.get(f"p{p}_{metric}_ms") or d.get(f"percentile_{p}_{metric}_ms")

def analyze_run(rdir, label):
    cj = os.path.join(rdir, f"{label}.json")
    rec = {"label": label}
    if os.path.exists(cj):
        c = json.load(open(cj))
        rec.update({
            "completed": c.get("completed"),
            "req_tput": round(c.get("request_throughput", 0), 3),
            "out_tput": round(c.get("output_throughput", 0), 1),       # system throughput (plot Y)
            "tot_tput": round(c.get("total_token_throughput", 0), 1),
            "mean_ttft_ms": round(c.get("mean_ttft_ms", 0), 1),
            "p99_ttft_ms": round(c.get("p99_ttft_ms", c.get("p99.0_ttft_ms", 0) or 0), 1),
            "mean_tpot_ms": round(c.get("mean_tpot_ms", 0), 2),
            "p99_tpot_ms": round(c.get("p99_tpot_ms", c.get("p99.0_tpot_ms", 0) or 0), 2),
            "mean_itl_ms": round(c.get("mean_itl_ms", 0), 2),
            "mean_e2e_ms": round(c.get("mean_e2el_ms", 0), 1),
        })
        if rec["mean_tpot_ms"]:
            rec["interactivity_tok_s"] = round(1000.0 / rec["mean_tpot_ms"], 1)  # plot X
    # /metrics counter deltas
    m0 = parse_prom(os.path.join(rdir, f"{label}.m0"))
    m1 = parse_prom(os.path.join(rdir, f"{label}.m1"))
    if m0 and m1:
        hits = delta(m1, m0, r'prefix_cache.*hit')
        quer = delta(m1, m0, r'prefix_cache.*(quer|total)')
        rec["prefix_hit_rate"] = round(hits/quer, 4) if (hits and quer) else 0.0
        rec["preemptions"] = delta(m1, m0, r'num_preemptions')
        pf_s = delta(m1, m0, r'prefill_time_seconds_sum'); pf_n = delta(m1, m0, r'prefill_time_seconds_count')
        rec["prefill_ms_avg"] = round(1000*pf_s/pf_n, 1) if (pf_s and pf_n) else None
        dc_s = delta(m1, m0, r'decode_time_seconds_sum'); dc_n = delta(m1, m0, r'decode_time_seconds_count')
        rec["decode_ms_avg"] = round(1000*dc_s/dc_n, 1) if (dc_s and dc_n) else None
        q_s = delta(m1, m0, r'queue_time_seconds_sum'); q_n = delta(m1, m0, r'queue_time_seconds_count')
        rec["queue_ms_avg"] = round(1000*q_s/q_n, 1) if (q_s and q_n) else None
        it_s = delta(m1, m0, r'iteration_tokens.*_sum'); it_n = delta(m1, m0, r'iteration_tokens.*_count')
        rec["iter_tokens_avg"] = round(it_s/it_n, 1) if (it_s and it_n) else None
        acc = delta(m1, m0, r'spec_decode.*accept'); drf = delta(m1, m0, r'spec_decode.*(draft|num_draft)')
        rec["spec_accept_rate"] = round(acc/drf, 3) if (acc and drf) else None
    # gauges
    s = parse_samples(os.path.join(rdir, f"{label}.samples"))
    rec["kv_mean"], rec["kv_max"] = mm([x*100 for x in s["kv"]])  # usage fraction -> %
    rec["running_mean"], rec["running_max"] = mm(s["running"])
    rec["waiting_mean"], rec["waiting_max"] = mm(s["waiting"])
    rec["power_mean_w"], rec["power_max_w"] = mm(s["power"])
    rec["gpu_util_mean"], _ = mm(s["util"])
    rec["mem_used_mb"], _ = mm(s["mem"])
    # energy per output token (J/tok) = avg_power(W) * duration(s) / output_tokens
    # serverlog
    rec.update(parse_serverlog(os.path.join(rdir, f"{label}.serverlog")))
    return rec

def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "nebius/results_exp"
    prefix = None
    if "--prefix" in sys.argv: prefix = sys.argv[sys.argv.index("--prefix")+1]
    labels = []
    runs_tsv = os.path.join(rdir, "runs.tsv")
    if os.path.exists(runs_tsv):
        for row in csv.DictReader(open(runs_tsv), delimiter='\t'):
            labels.append(row["label"])
    else:
        labels = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(rdir, "*.json"))]
    if prefix: labels = [l for l in labels if l.startswith(prefix)]
    recs = [analyze_run(rdir, l) for l in labels]
    if not recs: print("no runs found"); return
    cols = ["label","conc" if False else "out_tput","interactivity_tok_s","mean_ttft_ms","p99_ttft_ms",
            "mean_tpot_ms","mean_itl_ms","prefix_hit_rate","preemptions","prefill_ms_avg","decode_ms_avg",
            "queue_ms_avg","iter_tokens_avg","spec_accept_rate","kv_mean","kv_max","running_mean",
            "waiting_max","power_mean_w","gpu_util_mean","cudagraph_captured","eager_fallback"]
    out = os.path.join(rdir, "metrics_table.tsv")
    allcols = sorted({k for r in recs for k in r})
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=["label"]+[c for c in allcols if c!="label"], delimiter='\t')
        w.writeheader()
        for r in sorted(recs, key=lambda x: x["label"]): w.writerow(r)
    print(f"wrote {out}  ({len(recs)} runs)\n")
    # readable summary (subset)
    show = [c for c in cols if c in allcols]
    print("\t".join(show))
    for r in sorted(recs, key=lambda x: x.get("out_tput") or 0):
        print("\t".join(str(r.get(c, "")) for c in show))

if __name__ == "__main__":
    main()
