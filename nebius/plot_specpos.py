#!/usr/bin/env python3
"""Per-draft-position EAGLE3 acceptance for gpt-oss-120B — explains why the
headline accept-rate (accepted/draft, averaged over num_spec=7) looks low even
though first-token acceptance ~matches Fireworks. Reads results_spec/*.m0,*.m1.

Usage: python plot_specpos.py
"""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_spec")

def parse(f):
    d = {"pos": {}}
    for l in open(f):
        if l.startswith("#"): continue
        if "spec_decode_num_drafts_total{" in l: d["drafts"] = float(l.split()[-1])
        elif "spec_decode_num_accepted_tokens_total{" in l: d["acc"] = float(l.split()[-1])
        elif "spec_decode_num_draft_tokens_total{" in l: d["draft_tok"] = float(l.split()[-1])
        elif "per_pos_total{" in l:
            p = int(re.search(r'position="(\d+)"', l).group(1)); d["pos"][p] = float(l.split()[-1])
    return d

def cell(name):
    a, b = parse(f"{RES}/{name}.m0"), parse(f"{RES}/{name}.m1")
    dr = b["drafts"] - a["drafts"]
    pos = {p: (b["pos"].get(p, 0) - a["pos"].get(p, 0)) / dr for p in sorted(b["pos"])}
    return pos, (b["acc"] - a["acc"]) / dr, (b["acc"] - a["acc"]) / (b["draft_tok"] - a["draft_tok"])

cells = [("sweep_c1", "C=1"), ("sweep_c64", "C=64"), ("sweep_c256", "C=256")]
fig, ax = plt.subplots(figsize=(9, 5.5))
positions = list(range(7)); w = 0.26
for i, (nm, lab) in enumerate(cells):
    pos, al, rate = cell(nm)
    ys = [pos.get(p, 0) * 100 for p in positions]
    ax.bar([p + (i - 1) * w for p in positions], ys, w,
           label=f"{lab}  (accept-len {al:.2f} tok/step, rate {rate*100:.0f}%)")
ax.axhline(67, ls="--", color="tab:orange", alpha=.7)
ax.text(4.2, 68.5, "Fireworks reported ~67%", color="tab:orange", fontsize=9)
ax.set_xlabel("draft position (num_speculative_tokens = 7)")
ax.set_ylabel("acceptance at this position (%)")
ax.set_title("gpt-oss-120B + EAGLE3-v3: per-position acceptance\n"
             "first-token ~60–69% (≈Fireworks); deep positions rarely hit → low 7-way average")
ax.set_xticks(positions); ax.grid(True, axis="y", alpha=.3); ax.legend()
out = f"{RES}/spec_acceptance_per_pos.png"
fig.tight_layout(); fig.savefig(out, dpi=140); print("wrote", out)
