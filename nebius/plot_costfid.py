#!/usr/bin/env python3
"""Profiling cost vs fidelity.

Two complementary views, per concurrency cell (C=64 high-tput, C=4 high-interactivity):

(A) SYSTEM THROUGHPUT — real run-to-run variance.
    Repeated independent measured runs (cf_*_b{B}_r{R}) at budgets B (num_prompts=B*conc).
    For each budget: cost = mean wall-clock seconds; fidelity = relative error of mean
    output_tput vs the long ground-truth run (gt_*), plus run-to-run CV (std/mean).

(B) LATENCY TAILS — statistical convergence via bootstrap on the ground-truth
    --save-detailed per-request TTFT array. Subsample N requests, bootstrap mean &
    p99 TTFT -> relative error + 95% CI half-width vs the full estimate. Cost(N) =
    N / (measured completion rate).

Produces cost_fidelity.png (4 panels) + prints a table.
Usage: python plot_costfid.py <results_dir> [--node-usd-per-hr 3.5]
"""
import sys, os, csv, json, glob, statistics as st, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(0)

def load_runs(rdir):
    runs = {}
    for row in csv.DictReader(open(os.path.join(rdir,"runs.tsv")), delimiter='\t'):
        runs[row["label"]] = float(row["wall_s"])
    return runs

def cj(rdir, label):
    p = os.path.join(rdir, f"{label}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def real_throughput(rdir, runs, conc):
    """budget -> (cost_s_mean, relerr, cv) for output_tput vs ground truth."""
    gt = cj(rdir, f"gt_c{conc}")
    if not gt: return {}, None
    ref = gt["output_throughput"]
    by_b = {}
    for lbl, wall in runs.items():
        m = lbl.startswith(f"cf_c{conc}_b")
        if not m: continue
        b = int(lbl.split("_b")[1].split("_r")[0])
        c = cj(rdir, lbl)
        if not c: continue
        by_b.setdefault(b, {"tput":[], "cost":[]})
        by_b[b]["tput"].append(c["output_throughput"]); by_b[b]["cost"].append(wall)
    out = {}
    for b, d in sorted(by_b.items()):
        mean = st.mean(d["tput"]); cv = (st.pstdev(d["tput"])/mean if mean else 0)
        out[b] = (st.mean(d["cost"]), abs(mean-ref)/ref, cv)
    return out, ref

def bootstrap_latency(rdir, runs, conc, metric="ttft"):
    gt = cj(rdir, f"gt_c{conc}")
    if not gt or "ttfts" not in gt: return None
    xs = [t*1000 for t in gt["ttfts"]]      # ms
    M = len(xs); wall = runs.get(f"gt_c{conc}", 0)
    rate = M/wall if wall else 1.0
    full_mean = st.mean(xs); full_p99 = sorted(xs)[int(.99*M)-1]
    Ns = [n for n in [conc, conc*2, conc*5, conc*10, conc*20, conc*40, M] if n <= M]
    rows = []
    for N in sorted(set(Ns)):
        means, p99s = [], []
        for _ in range(200):
            samp = [random.choice(xs) for _ in range(N)]
            means.append(st.mean(samp)); p99s.append(sorted(samp)[max(0,int(.99*N)-1)])
        def stats(arr, full):
            mu = st.mean(arr); ci = 1.96*st.pstdev(arr)
            return abs(mu-full)/full, ci/full
        e_mean, ci_mean = stats(means, full_mean)
        e_p99, ci_p99 = stats(p99s, full_p99)
        rows.append((N, N/rate, e_mean, ci_mean, e_p99, ci_p99))
    return rows, rate

def detect_concs(rdir):
    cs = []
    for p in glob.glob(os.path.join(rdir, "gt_c*.json")):
        try: cs.append(int(os.path.basename(p)[4:-5]))
        except ValueError: pass
    return sorted(set(cs), reverse=True)  # high concurrency first

def main():
    rdir = sys.argv[1] if len(sys.argv)>1 else "nebius/results_exp"
    usd_hr = 3.5
    if "--node-usd-per-hr" in sys.argv: usd_hr = float(sys.argv[sys.argv.index("--node-usd-per-hr")+1])
    runs = load_runs(rdir)
    concs = detect_concs(rdir)
    n = len(concs)
    fig, axes = plt.subplots(2, n, figsize=(6.5*n, 10), squeeze=False)

    for col, conc in enumerate(concs):
        # (A) throughput real variance
        ax = axes[0][col]
        rt, ref = real_throughput(rdir, runs, conc)
        if rt:
            bs = sorted(rt); cost=[rt[b][0] for b in bs]; err=[rt[b][1]*100 for b in bs]; cv=[rt[b][2]*100 for b in bs]
            ax.plot(cost, err, "o-", label="bias |est-ref|/ref %")
            ax.plot(cost, cv, "s--", label="run-to-run CV %")
            for b,c0 in zip(bs,cost): ax.annotate(f"{b}x", (c0, max(rt[b][1],rt[b][2])*100), fontsize=7)
            ax.axvline(300, color="grey", ls=":", alpha=.6); ax.text(300, ax.get_ylim()[1]*.8, " 5 min", fontsize=8, color="grey")
        ax.set_title(f"C={conc}: system throughput fidelity vs cost (ref={ref and round(ref)} tok/s)")
        ax.set_xlabel("measurement wall-clock cost (s)"); ax.set_ylabel("error / CV (%)")
        ax.grid(True, alpha=.3); ax.legend(fontsize=8)
        secax = ax.secondary_xaxis('top', functions=(lambda s: s/3600*usd_hr, lambda d: d*3600/usd_hr))
        secax.set_xlabel(f"cost (USD @ ${usd_hr}/hr)", fontsize=8)

        # (B) latency-tail bootstrap convergence
        ax = axes[1][col]
        bl = bootstrap_latency(rdir, runs, conc)
        if bl:
            rows, rate = bl
            cost=[r[1] for r in rows]
            ax.plot(cost, [r[2]*100 for r in rows], "o-", label="mean TTFT rel-err %")
            ax.plot(cost, [r[3]*100 for r in rows], "o:", alpha=.6, label="mean TTFT 95%CI %")
            ax.plot(cost, [r[4]*100 for r in rows], "s-", label="p99 TTFT rel-err %")
            ax.plot(cost, [r[5]*100 for r in rows], "s:", alpha=.6, label="p99 TTFT 95%CI %")
            ax.axvline(300, color="grey", ls=":", alpha=.6)
        ax.set_title(f"C={conc}: latency-tail convergence (bootstrap)")
        ax.set_xlabel("equivalent measurement cost (s)"); ax.set_ylabel("error (%)")
        ax.grid(True, alpha=.3); ax.legend(fontsize=8)

    out = os.path.join(rdir, "cost_fidelity.png")
    fig.suptitle("Profiling cost vs fidelity — GPT-OSS-120B 1xH200 (ISL~8k OSL=1k)", fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out, dpi=140)
    print("wrote", out)

if __name__ == "__main__":
    main()
