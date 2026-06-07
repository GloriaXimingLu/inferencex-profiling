#!/usr/bin/env python3
"""Compile the 16-part tau-bench replay results into a comparison table + plots.

For each part (model__domain) combines:
  client.json (tau-bench-replay client) -> realized prefix-cache hit rate, TTFT/E2E
      percentiles, req/s, output tok/s, requests
  m0/m1 /metrics deltas -> server prefill/decode/queue time, preemptions
  .samples -> KV-cache usage (max), GPU power (mean)
  stats/<part>.json (from the repo) -> analytical ideal cache-hit rate (intra/+sys),
      prompt-token mean
Key comparison: REALIZED prefix-cache hit rate (under load) vs the ANALYTICAL IDEAL.

Usage: python analyze_tau.py <results_tau_dir> <repo_stats_dir>
"""
import sys, os, re, json, glob, csv

def parse_prom(path):
    d = {}
    if not os.path.exists(path): return d
    for line in open(path):
        if not line or line[0] == '#': continue
        m = re.match(r'([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+naN]+)', line.strip())
        if not m: continue
        try: d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError: pass
    return d

def delta(m1, m0, *pats):
    def f(d):
        t=None
        for k,v in d.items():
            if any(re.search(p,k) for p in pats): t=(t or 0)+v
        return t
    a,b=f(m1),f(m0)
    return None if (a is None or b is None) else a-b

def samp_max(path, key):
    if not os.path.exists(path): return None
    vals=[]
    for line in open(path):
        if key in line:
            m=re.search(r'\s([-\d.eE+]+)$', line.strip())
            if m: vals.append(float(m.group(1)))
    return round(max(vals),3) if vals else None

def samp_gpu_mean(path, idx):
    if not os.path.exists(path): return None
    vals=[]
    for line in open(path):
        if line.startswith('gpu '):
            p=[x.strip() for x in line[4:].split(',')]
            try: vals.append(float(p[idx]))
            except (ValueError,IndexError): pass
    return round(sum(vals)/len(vals),1) if vals else None

def main():
    rdir = sys.argv[1] if len(sys.argv)>1 else "nebius/results_tau"
    sdir = sys.argv[2] if len(sys.argv)>2 else "tau-bench-replay/stats"
    rows=[]
    # iterate over parts that have an m0 snapshot (so the 2 parts whose client.json
    # crashed on outliers are still included via server-side metrics)
    for m0p in sorted(glob.glob(os.path.join(rdir,"*.m0"))):
        part=os.path.basename(m0p)[:-3]
        if "__" not in part: continue
        model,domain=part.split("__",1)
        cj=os.path.join(rdir,f"{part}.client.json")
        c={}
        try:
            if os.path.getsize(cj)>0: c=json.load(open(cj))
        except Exception: c={}
        st=os.path.join(sdir,f"{part}.json")
        ideal_sys=ideal_intra=prompt_mean=None
        if os.path.exists(st):
            s=json.load(open(st))
            ideal_sys=round(s.get("ideal_cache_hit_rate_with_sys",0),3)
            ideal_intra=round(s.get("ideal_cache_hit_rate_intra",0),3)
            prompt_mean=s.get("prompt_tokens",{}).get("mean")
        m0=parse_prom(m0p); m1=parse_prom(os.path.join(rdir,f"{part}.m1"))
        # realized prefix-cache hit from MY counter deltas (client.py used the wrong
        # metric name for vLLM v0.22 -> recompute here). Token-level hits/queries.
        dh=delta(m1,m0,r'^vllm:prefix_cache_hits_total$',r'gpu_prefix_cache_hits')
        dq=delta(m1,m0,r'^vllm:prefix_cache_queries_total$',r'gpu_prefix_cache_queries')
        realized=round(dh/dq,4) if (dh and dq) else None
        def avg_ms(name):
            s=delta(m1,m0,name+r'_sum'); n=delta(m1,m0,name+r'_count')
            return round(1000*s/n,1) if (s and n) else None
        rows.append({
            "part":part,"domain":domain,"model":model,
            "prompt_mean":prompt_mean,"requests":c.get("requests"),
            "realized_hit":realized,"ideal_hit_sys":ideal_sys,"ideal_hit_intra":ideal_intra,
            "ttft_p50_ms":c.get("ttft_ms",{}).get("p50"),"ttft_p95_ms":c.get("ttft_ms",{}).get("p95"),
            "e2e_p50_ms":c.get("e2e_ms",{}).get("p50"),
            "out_tok_s":c.get("output_tok_per_s"),"req_s":c.get("req_per_s"),
            "prefill_ms":avg_ms(r'request_prefill_time_seconds'),
            "decode_ms":avg_ms(r'request_decode_time_seconds'),
            "queue_ms":avg_ms(r'request_queue_time_seconds'),
            "preempt":delta(m1,m0,r'num_preemptions'),
            "kv_max_pct":(lambda v: round(v*100,1) if v is not None else None)(samp_max(os.path.join(rdir,f"{part}.samples"),"cache_usage")),
            "power_w":samp_gpu_mean(os.path.join(rdir,f"{part}.samples"),0),
        })
    if not rows: print("no client.json found in",rdir); return
    out=os.path.join(rdir,"tau_table.tsv")
    cols=["part","domain","model","prompt_mean","requests","realized_hit","ideal_hit_sys",
          "ideal_hit_intra","ttft_p50_ms","ttft_p95_ms","e2e_p50_ms","out_tok_s","req_s",
          "prefill_ms","decode_ms","queue_ms","preempt","kv_max_pct","power_w"]
    with open(out,"w") as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t'); w.writeheader()
        for r in sorted(rows,key=lambda x:(x["domain"],x["model"])): w.writerow(r)
    print(f"wrote {out} ({len(rows)} parts)\n")
    print(f"{'part':44}{'realized':>9}{'ideal+sys':>10}{'gap':>7}{'ttftP95':>9}{'out_tok/s':>10}{'kv%':>6}")
    for r in sorted(rows,key=lambda x:(x["domain"],x["model"])):
        rz=r["realized_hit"]; idl=r["ideal_hit_sys"]
        gap = round((idl-rz)*100,1) if (rz is not None and idl is not None) else None
        print(f"{r['part']:44}{str(rz):>9}{str(idl):>10}{str(gap):>7}{str(r['ttft_p95_ms']):>9}{str(r['out_tok_s']):>10}{str(r['kv_max_pct']):>6}")

    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("(matplotlib unavailable:",e,")"); return
    dom_order=["airline","retail","telecom","banking_knowledge"]
    col={"airline":"tab:blue","retail":"tab:green","telecom":"tab:orange","banking_knowledge":"tab:red"}
    valid=[r for r in rows if r["realized_hit"] is not None and r["ideal_hit_sys"] is not None]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6))
    # (1) realized vs ideal hit rate
    for r in valid:
        ax1.scatter(r["ideal_hit_sys"]*100,r["realized_hit"]*100,c=col.get(r["domain"],"gray"),s=70)
        ax1.annotate(r["model"][:8],(r["ideal_hit_sys"]*100,r["realized_hit"]*100),fontsize=6)
    lo=min([r["realized_hit"]*100 for r in valid]+[r["ideal_hit_sys"]*100 for r in valid])-2
    ax1.plot([lo,100],[lo,100],"k--",alpha=.4,label="realized = ideal")
    for d in dom_order: ax1.scatter([],[],c=col[d],label=d)
    ax1.set_xlabel("analytical ideal hit rate (+sys) %"); ax1.set_ylabel("realized hit rate %")
    ax1.set_title("Prefix-cache: realized (under load) vs ideal"); ax1.legend(fontsize=7); ax1.grid(alpha=.3)
    # (2) throughput vs prompt size, colored by domain
    for r in rows:
        if r["out_tok_s"] and r["prompt_mean"]:
            ax2.scatter(r["prompt_mean"],r["out_tok_s"],c=col.get(r["domain"],"gray"),s=70)
            ax2.annotate(r["model"][:8],(r["prompt_mean"],r["out_tok_s"]),fontsize=6)
    ax2.set_xscale("log"); ax2.set_xlabel("mean prompt tokens (log)"); ax2.set_ylabel("output tok/s")
    ax2.set_title("Throughput vs prompt size (workload regime)"); ax2.grid(alpha=.3)
    for d in dom_order: ax2.scatter([],[],c=col[d],label=d)
    ax2.legend(fontsize=7)
    p=os.path.join(rdir,"tau_compare.png"); fig.tight_layout(); fig.savefig(p,dpi=140)
    print("\nwrote",p)

if __name__=="__main__":
    main()
