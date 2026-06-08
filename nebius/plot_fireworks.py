#!/usr/bin/env python3
"""Throughput-vs-interactivity Pareto curve (the InferenceMax/Fireworks plot).

X = interactivity = per-user output speed (tok/s/user) = 1000/mean_tpot_ms
Y = system throughput = output token throughput (tok/s)
Our vLLM sweep (sweep_c*) overlaid with Fireworks reference points digitized
from the co-worker's report (GPT-OSS-120B, ISL~8k OSL=1k, 1xH200).

Usage: python plot_fireworks.py <results_dir>
"""
import sys, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fireworks reference, digitized from the report's plot (approx; spec-decode ON).
FW = {1:(310,250), 2:(270,410), 4:(255,800), 8:(178,1160), 16:(113,1570),
      32:(68,1960), 64:(35,2300), 128:(33,2180), 256:(33,2300)}

def load(rdir):
    tbl = os.path.join(rdir, "metrics_table.tsv")
    rows = list(csv.DictReader(open(tbl), delimiter='\t'))
    pts = {}
    for r in rows:
        if not r["label"].startswith("sweep_c"): continue
        c = int(r["label"].split("_c")[1])
        try:
            x = float(r["interactivity_tok_s"]); y = float(r["out_tput"])
        except (ValueError, KeyError): continue
        pts[c] = (x, y)
    return pts

def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "nebius/results_exp"
    spec_dir = sys.argv[sys.argv.index("--spec-dir")+1] if "--spec-dir" in sys.argv else None
    ours = load(rdir)
    spec = load(spec_dir) if spec_dir else {}
    fig, ax = plt.subplots(figsize=(8,6))
    # Fireworks reference
    fc = sorted(FW); fx=[FW[c][0] for c in fc]; fy=[FW[c][1] for c in fc]
    ax.plot(fx, fy, "o--", color="tab:orange", alpha=.7, label="Fireworks (spec-decode, digitized)")
    for c in fc: ax.annotate(f"C={c}", FW[c], fontsize=7, color="tab:orange")
    # Ours — vanilla
    if ours:
        oc = sorted(ours); ax.plot([ours[c][0] for c in oc], [ours[c][1] for c in oc],
                                   "o-", color="tab:blue", label="vLLM (ours, no spec-decode)")
        for c in oc: ax.annotate(f"C={c}", ours[c], fontsize=7, color="tab:blue")
    # Ours — with EAGLE3 speculative decoding
    if spec:
        sc = sorted(spec); ax.plot([spec[c][0] for c in sc], [spec[c][1] for c in sc],
                                   "o-", color="tab:green", label="vLLM + EAGLE3 spec-decode (ours)")
        for c in sc: ax.annotate(f"C={c}", spec[c], fontsize=7, color="tab:green")
    ax.set_xlabel("Interactivity — per-user output speed (tok/s/user)")
    ax.set_ylabel("System throughput (output tok/s)")
    ax.set_title("GPT-OSS-120B | ISL~8k OSL=1k | 1x H200 — vLLM vs Fireworks")
    ax.grid(True, alpha=.3); ax.legend()
    out = os.path.join(rdir, "fireworks_compare.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print("wrote", out)
    for name, d in [("vanilla", ours), ("+spec", spec)]:
        if d:
            print(f"\nC   interactivity  sys_tput   ({name})")
            for c in sorted(d): print(f"{c:<4}{d[c][0]:>12}{d[c][1]:>11}")

if __name__ == "__main__":
    main()
