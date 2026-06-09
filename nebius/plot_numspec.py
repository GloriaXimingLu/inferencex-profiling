#!/usr/bin/env python3
"""num_speculative_tokens validation: throughput-vs-interactivity curves for
vanilla(0) / num_spec 1 / 3 / 7, on the §1 workload (gpt-oss-120B, ISL~8k OSL=1k,
1xH200). Confirms a shallower draft (num_spec=1) beats the default 7.
Reads results_exp (vanilla), results_specns1, results_specns3, results_spec (7).
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = [(0, "nebius/results_exp", "vanilla (no spec)", "tab:gray"),
       (1, "nebius/results_specns1", "num_spec=1", "tab:green"),
       (3, "nebius/results_specns3", "num_spec=3", "tab:blue"),
       (7, "nebius/results_spec", "num_spec=7 (default)", "tab:red")]
CS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def g(f, k):
    s = 0; h = False
    try:
        for l in open(f):
            if l.startswith('#') or 'per_pos' in l or k+'{' not in l: continue
            s += float(l.split()[-1]); h = True
    except: pass
    return s if h else None

def cell(d, c):
    try: j = json.load(open(f"{d}/sweep_c{c}.json"))
    except: return None
    tput = j.get("output_throughput"); tpot = j.get("mean_tpot_ms")
    a0 = g(f"{d}/sweep_c{c}.m0", "vllm:spec_decode_num_accepted_tokens_total")
    a1 = g(f"{d}/sweep_c{c}.m1", "vllm:spec_decode_num_accepted_tokens_total")
    dr0 = g(f"{d}/sweep_c{c}.m0", "vllm:spec_decode_num_draft_tokens_total")
    dr1 = g(f"{d}/sweep_c{c}.m1", "vllm:spec_decode_num_draft_tokens_total")
    acc = (a1-a0)/(dr1-dr0) if (a1 and dr1 and dr1-dr0 > 0) else None
    return {"tput": tput, "inter": 1000/tpot if tpot else None, "acc": acc}

fig, ax = plt.subplots(figsize=(8.5, 6))
print(f"{'C':>4} | " + " ".join(f"{l.split()[0][:8]:>14}" for _, _, l, _ in CFG))
rows = {sn: {} for sn, *_ in CFG}
for sn, d, lab, col in CFG:
    xs, ys = [], []
    for c in CS:
        r = cell(d, c)
        if not r or not r["tput"]: continue
        rows[sn][c] = r
        xs.append(r["inter"]); ys.append(r["tput"])
    ax.plot(xs, ys, "o-", color=col, label=lab)
for c in CS:
    cells_str = []
    for sn, *_ in CFG:
        r = rows[sn].get(c)
        if r: cells_str.append(f"{round(r['tput'])}" + (f"({round(r['acc']*100)}%)" if r['acc'] else ""))
        else: cells_str.append("-")
    print(f"{c:>4} | " + " ".join(f"{x:>14}" for x in cells_str))
ax.set_xlabel("Interactivity — per-user output speed (tok/s/user)")
ax.set_ylabel("System throughput (output tok/s)")
ax.set_title("gpt-oss-120B + EAGLE3 | num_speculative_tokens sweep (1xH200, ISL~8k OSL=1k)\n"
             "num_spec=1 wins everywhere; the default num_spec=7 is worst (below vanilla mid-load)")
ax.grid(True, alpha=.3); ax.legend()
out = "nebius/results_specns1/num_spec_sweep.png"
fig.tight_layout(); fig.savefig(out, dpi=140); print("\nwrote", out)
