#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.7"]
# ///
"""
plot.py — draw the InferenceMax-style Pareto curve from a results dir.

  x-axis: interactivity  = per-user decode speed (tok/s/user) = 1 / median TPOT
  y-axis: system output throughput (tok/s)   [optionally /GPU]

Each point is one concurrency level. Up-and-to-the-right is better.

Usage:
  python plot.py results/20260605-120000
  python plot.py results/<dir> --gpus 8        # divide throughput by #GPUs
  python plot.py results/<dir> --price-gpu-hr 2.90 --gpus 8   # adds $/1M tok
  python plot.py results/<dir> --csv            # also dump a table
"""
import argparse
import glob
import json
import os
import sys


def load(d):
    pts = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(f) as fh:
            pts.append(json.load(fh))
    pts.sort(key=lambda p: p["concurrency"])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--gpus", type=int, default=1,
                    help="divide system throughput by this (dedicated deploys)")
    ap.add_argument("--price-gpu-hr", type=float, default=None,
                    help="$/GPU-hour to compute $/1M tokens")
    ap.add_argument("--label", default=None,
                    help="model display name for the title, e.g. 'GPT OSS 120B' "
                         "(defaults to the model string in the result JSON)")
    ap.add_argument("--accelerator", default=None,
                    help="GPU type for the title, e.g. 'H200'. Combined with --gpus "
                         "this renders as '1x H200'.")
    ap.add_argument("--out", default=None, help="png path (default <dir>/pareto.png)")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    pts = load(args.results_dir)
    if not pts:
        sys.exit(f"no *.json in {args.results_dir}")

    rows = []
    for p in pts:
        sys_tput = p["system_output_tok_per_s"]
        per_gpu = sys_tput / args.gpus
        inter = p["interactivity_tok_per_s_per_user"]
        cost = None
        if args.price_gpu_hr and sys_tput > 0:
            cost = (args.price_gpu_hr * args.gpus) / (sys_tput * 3600) * 1e6
        rows.append({
            "C": p["concurrency"],
            "interactivity": inter,
            "sys_tput": sys_tput,
            "per_gpu": per_gpu,
            "ttft_p50_ms": (p["ttft_s"]["p50"] or 0) * 1000,
            "ttft_p99_ms": (p["ttft_s"]["p99"] or 0) * 1000,
            "tpot_p50_ms": (p["tpot_s"]["p50"] or 0) * 1000,
            "in_tok": p["actual_in_tok_median"],
            "out_tok": p["actual_out_tok_median"],
            "errors": p["errors"],
            "cost_per_mtok": cost,
        })

    # table
    hdr = (f"{'C':>5} {'interact':>9} {'sysTput':>9} {'perGPU':>9} "
           f"{'TTFTp50':>8} {'TTFTp99':>8} {'TPOTp50':>8} "
           f"{'in':>6} {'out':>5} {'err':>4}")
    if args.price_gpu_hr:
        hdr += f" {'$/Mtok':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['C']:>5} {r['interactivity']:>9.1f} {r['sys_tput']:>9.0f} "
                f"{r['per_gpu']:>9.0f} {r['ttft_p50_ms']:>8.0f} {r['ttft_p99_ms']:>8.0f} "
                f"{r['tpot_p50_ms']:>8.1f} {r['in_tok']:>6.0f} {r['out_tok']:>5.0f} "
                f"{r['errors']:>4}")
        if args.price_gpu_hr:
            line += f" {r['cost_per_mtok']:>8.2f}" if r['cost_per_mtok'] else f" {'n/a':>8}"
        print(line)

    if args.csv:
        import csv
        csv_path = os.path.join(args.results_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[info] pip install matplotlib to render the PNG curve", file=sys.stderr)
        return

    xs = [r["interactivity"] for r in rows]
    ys = [r["per_gpu"] if args.gpus > 1 else r["sys_tput"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, "-o")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"C={r['C']}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Interactivity — per-user output speed (tok/s/user)")
    ylabel = "Throughput per GPU (tok/s)" if args.gpus > 1 else "System throughput (tok/s)"
    ax.set_ylabel(ylabel)
    p0 = pts[0]
    model_name = args.label or p0["model"]
    if args.accelerator:
        hw = f"{args.gpus}x {args.accelerator}"
    elif args.gpus > 1:
        hw = f"{args.gpus} GPU"
    else:
        hw = None
    title = f"{model_name}  |  ISL≈{p0['input_len_target']} OSL={p0['output_len_target']}"
    if hw:
        title += f"  |  {hw}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = args.out or os.path.join(args.results_dir, "pareto.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
