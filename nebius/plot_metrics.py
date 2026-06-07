#!/usr/bin/env python3
"""Visualize the comprehensive metric trends across the concurrency sweep.

Reads metrics_table.tsv (sweep_c* rows) and produces a 2x3 panel figure
(metrics_sweep.png), x-axis = concurrency (log), making the §2 tables digestible.

Usage: python plot_metrics.py <results_dir>
"""
import sys, os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "nebius/results_exp"
    rows = {}
    for r in csv.DictReader(open(os.path.join(rdir, "metrics_table.tsv")), delimiter='\t'):
        if not r["label"].startswith("sweep_c"): continue
        try: rows[int(r["label"].split("_c")[1])] = r
        except ValueError: pass
    C = sorted(rows)
    def col(k, f=float):
        out = []
        for c in C:
            v = rows[c].get(k, "")
            try: out.append(f(v))
            except (ValueError, TypeError): out.append(None)
        return out
    out_tput = col("out_tput"); interact = col("interactivity_tok_s")
    mttft = col("mean_ttft_ms"); p99ttft = col("p99_ttft_ms"); mtpot = col("mean_tpot_ms")
    kv = col("kv_max"); preempt = col("preemptions")
    power = col("power_mean_w"); util = col("gpu_util_mean")
    batch = col("iter_tokens_avg"); queue = col("queue_ms_avg")
    prefill = col("prefill_ms_avg"); decode = col("decode_ms_avg")
    energy = [ (p/t if (p and t) else None) for p,t in zip(power, out_tput) ]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    def logx(a): a.set_xscale("log", base=2); a.set_xticks(C); a.set_xticklabels(C); a.grid(alpha=.3); a.set_xlabel("concurrency")

    # 1: throughput vs interactivity (the core tradeoff)
    a = ax[0][0]; a.plot(C, out_tput, "o-", color="tab:blue", label="system throughput (tok/s)")
    a.set_ylabel("system throughput (tok/s)", color="tab:blue"); a.tick_params(axis='y', colors="tab:blue")
    a2 = a.twinx(); a2.plot(C, interact, "s--", color="tab:orange", label="interactivity")
    a2.set_ylabel("interactivity (tok/s/user)", color="tab:orange"); a2.tick_params(axis='y', colors="tab:orange")
    a.set_title("Throughput rises & saturates; interactivity falls"); logx(a)

    # 2: latency degradation (log y)
    a = ax[0][1]
    a.plot(C, mttft, "o-", label="mean TTFT"); a.plot(C, p99ttft, "s-", label="p99 TTFT")
    a.plot(C, mtpot, "^-", label="mean TPOT")
    a.set_yscale("log"); a.set_ylabel("latency (ms, log)"); a.legend(fontsize=8)
    a.set_title("Latency climbs; p99 TTFT explodes at saturation"); logx(a)

    # 3: KV pressure & preemptions
    a = ax[0][2]; a.plot(C, kv, "o-", color="tab:red", label="KV-cache usage max %")
    a.set_ylabel("KV-cache usage (max %)", color="tab:red"); a.tick_params(axis='y', colors="tab:red"); a.set_ylim(0, 105)
    a2 = a.twinx(); a2.bar(C, preempt, width=[c*0.3 for c in C], color="tab:purple", alpha=.4)
    a2.set_ylabel("preemptions", color="tab:purple"); a2.tick_params(axis='y', colors="tab:purple")
    a.set_title("KV fills to 100% → preemptions kick in (C=256)"); logx(a)

    # 4: efficiency — power up, energy/token down
    a = ax[1][0]; a.plot(C, power, "o-", color="tab:green", label="power (W)")
    a.set_ylabel("GPU power (W)", color="tab:green"); a.tick_params(axis='y', colors="tab:green")
    a2 = a.twinx(); a2.plot(C, energy, "s--", color="tab:brown")
    a2.set_ylabel("energy (J / output token)", color="tab:brown"); a2.tick_params(axis='y', colors="tab:brown")
    a.set_title("Batching is more energy-efficient (J/tok ↓ 3.8×)"); logx(a)

    # 5: what buys throughput (batch) & costs latency (queue)
    a = ax[1][1]; a.plot(C, batch, "o-", color="tab:blue", label="batch (tok/step)")
    a.set_yscale("log"); a.set_ylabel("batch size (tokens/step, log)", color="tab:blue"); a.tick_params(axis='y', colors="tab:blue")
    a2 = a.twinx(); a2.plot(C, queue, "s--", color="tab:orange"); a2.set_yscale("log")
    a2.set_ylabel("scheduler queue/req (ms, log)", color="tab:orange"); a2.tick_params(axis='y', colors="tab:orange")
    a.set_title("Bigger batches (throughput) ↔ longer queues (latency)"); logx(a)

    # 6: where the time goes — prefill vs decode per request
    a = ax[1][2]
    a.plot(C, prefill, "o-", label="prefill / req"); a.plot(C, decode, "s-", label="decode / req")
    a.set_yscale("log"); a.set_ylabel("time per request (ms, log)"); a.legend(fontsize=8)
    a.set_title("Decode-time/req dominates & grows with load"); logx(a)

    fig.suptitle("Metric trends across the concurrency sweep — GPT-OSS-120B, 1×H200, ISL≈8k/OSL=1k", fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97])
    p = os.path.join(rdir, "metrics_sweep.png"); fig.savefig(p, dpi=140)
    print("wrote", p)

if __name__ == "__main__":
    main()
