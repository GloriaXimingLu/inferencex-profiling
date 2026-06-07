#!/usr/bin/env python3
"""Banking (long-context RAG): realized prefix-cache hit at C=4 (1xH200) vs C=64
(8xH200) vs the analytical ideal — shows the C=4 gap was a KV limitation, not
concurrency. Reads m0/m1 counter deltas from results_tau (C=4) and
results_tau_c64 (C=64), ideal from tau-bench-replay/stats.

Usage: python plot_banking_conc.py
"""
import os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

MODELS = ["claude-opus-4-5", "gemini-3-pro", "gpt-5-2", "qwen3.5-397b-a17b-think"]
SHORT = {"claude-opus-4-5":"claude", "gemini-3-pro":"gemini", "gpt-5-2":"gpt-5-2", "qwen3.5-397b-a17b-think":"qwen"}

def realized(d, part):
    def g(suf, key):
        f = os.path.join(d, f"{part}.{suf}"); s=0; hit=False
        if not os.path.exists(f): return None
        for l in open(f):
            if l.startswith(key): s+=float(l.split()[-1]); hit=True
        return s if hit else None
    h0=g("m0","vllm:prefix_cache_hits_total"); h1=g("m1","vllm:prefix_cache_hits_total")
    q0=g("m0","vllm:prefix_cache_queries_total"); q1=g("m1","vllm:prefix_cache_queries_total")
    if None in (h0,h1,q0,q1) or (q1-q0)<=0: return None
    return (h1-h0)/(q1-q0)

def main():
    ideal, c4, c64 = [], [], []
    for m in MODELS:
        part=f"{m}__banking_knowledge"
        s=json.load(open(f"tau-bench-replay/stats/{part}.json"))
        ideal.append(s["ideal_cache_hit_rate_with_sys"]*100)
        r4=realized("nebius/results_tau", part); r64=realized("nebius/results_tau_c64", part)
        c4.append(r4*100 if r4 else 0); c64.append(r64*100 if r64 else 0)
    print(f"{'model':10}{'ideal':>8}{'C=4':>8}{'C=64':>8}{'gap@4':>8}{'gap@64':>8}")
    for i,m in enumerate(MODELS):
        print(f"{SHORT[m]:10}{ideal[i]:>8.1f}{c4[i]:>8.1f}{c64[i]:>8.1f}{ideal[i]-c4[i]:>8.1f}{ideal[i]-c64[i]:>8.1f}")
    import numpy as np
    x=np.arange(len(MODELS)); w=0.27
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.bar(x-w, ideal, w, label="analytical ideal", color="lightgray")
    ax.bar(x,   c4,   w, label="realized C=4 (1×H200)", color="tab:orange")
    ax.bar(x+w, c64,  w, label="realized C=64 (8×H200)", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS])
    ax.set_ylabel("prefix-cache hit rate (%)"); ax.set_ylim(80, 100)
    ax.set_title("Banking (long-context RAG): realized prefix-cache hit vs ideal\nC=4 on 1×H200 vs C=64 on 8×H200 (16× KV)")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
    for i in range(len(MODELS)):
        ax.annotate(f"{ideal[i]-c64[i]:.1f}", (x[i]+w, c64[i]), textcoords="offset points",
                    xytext=(0,3), ha='center', fontsize=7, color="tab:blue")
    out="nebius/results_tau_c64/banking_conc_compare.png"; fig.tight_layout(); fig.savefig(out,dpi=140)
    print("wrote", out)

if __name__=="__main__":
    main()
