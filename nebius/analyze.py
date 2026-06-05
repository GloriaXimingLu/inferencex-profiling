#!/usr/bin/env python3
"""Turn measured calibration timings into an InferenceX vLLM wall-time estimate.

MEASURED (real, on 1xH200): gpt-oss-120b TP=1, 4 cells.
EXTRAPOLATED: rest of the H200 vLLM matrix, using a concurrency-aware benchmark
model fit from the real data + size-relative multipliers for big MoE models
(clearly flagged; to be replaced by direct 8xH200 measurement).
"""
import csv, statistics, pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
TSV = ROOT / "nebius/results/timings.tsv"
MASTER = ROOT / "InferenceX/.github/configs/nvidia-master.yaml"
DL_RATE_GB_S = 63/815.0  # measured: 63 GB in 815 s

def conc_points(s, e):
    pts, c = [], s
    while c <= e: pts.append(c); c *= 2
    return pts

def load_timings():
    rows = []
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k in ("startup_s","bench_s","conc","isl","osl","tp"):
                try: r[k] = int(r[k])
                except: pass
            rows.append(r)
    return rows

# size on disk (GB) + relative startup/decode-latency factors vs gpt-oss.
# Factors are ESTIMATES for the big models (no direct measurement yet).
MODELS = {
  "gptoss":      dict(gb=63,  f_start=1.0, f_tpot=1.0),   # MEASURED anchor
  "minimaxm2.5": dict(gb=230, f_start=1.9, f_tpot=1.6),
  "kimik2.5":    dict(gb=500, f_start=2.8, f_tpot=3.0),
  "dsv4":        dict(gb=670, f_start=3.3, f_tpot=3.5),
  "dsr1":        dict(gb=670, f_start=3.3, f_tpot=3.5),
  "glm5":        dict(gb=350, f_start=2.3, f_tpot=2.0),
  "qwen3.5":     dict(gb=235, f_start=1.9, f_tpot=1.4),
}

def h200_vllm_configs():
    cfg = yaml.safe_load(open(MASTER))
    out = {}
    for name, v in cfg.items():
        if not isinstance(v, dict): continue
        if v.get("framework")!="vllm" or v.get("runner")!="h200": continue
        if "agentic" in name or name.endswith("-mtp"): continue
        det=[]
        for e in (v.get("scenarios",{}).get("fixed-seq-len") or []):
            for ss in e.get("search-space", []):
                for c in conc_points(ss["conc-start"], ss["conc-end"]):
                    det.append((e["isl"], e["osl"], ss["tp"], c))
        out[name] = dict(cells=det, model=v.get("model",""))
    return out

def main():
    rows = load_timings()
    meas = [r for r in rows if r.get("status")=="ok"]
    base_start = statistics.median(r["startup_s"] for r in meas)
    def fit(isl):
        pts = sorted((r["conc"], r["bench_s"]) for r in meas if r["isl"]==isl)
        (c0,b0),(c1,b1) = pts[0], pts[-1]
        slope = (b1-b0)/(c1-c0); intc = b0 - slope*c0
        return intc, slope
    i1,s1 = fit(1024); i8,s8 = fit(8192)
    def bench(isl, conc):
        intc, slope = (i8,s8) if isl==8192 else (i1,s1)
        return max(20.0, intc + slope*conc)

    print("="*74)
    print("MEASURED — gpt-oss-120b on 1xH200, TP=1 (real wall times)")
    print("="*74)
    print(f"{'cell':22}{'conc':>5}{'ISL/OSL':>10}{'startup_s':>10}{'bench_s':>9}{'total_s':>9}")
    for r in meas:
        islosl = f"{r['isl']}/{r['osl']}"
        print(f"{r['label']:22}{r['conc']:>5}{islosl:>10}{r['startup_s']:>10}{r['bench_s']:>9}{r['total_s']:>9}")
    print(f"\nfit: startup(median)={base_start:.0f}s | "
          f"bench_1k≈{i1:.0f}+{s1:.2f}·conc | bench_8k≈{i8:.0f}+{s8:.2f}·conc")
    print(f"one-time model download measured: 815s for 63GB ({DL_RATE_GB_S*1e3:.0f} MB/s eff.)")

    def cell_time(pfx, isl, conc):
        m = MODELS.get(pfx, MODELS["minimaxm2.5"])
        return base_start*m["f_start"] + bench(isl, conc)*m["f_tpot"]

    cfgs = h200_vllm_configs()
    print("\n"+"="*74)
    print("EXTRAPOLATED — H200 vLLM matrix wall time, single node, sequential")
    print("="*74)
    print(f"{'config':30}{'cells':>6}{'dl_h':>7}{'compute_h':>10}{'total_h':>9}  basis")
    grand=0.0; dl_total=0.0
    by_model=set()
    for name in sorted(cfgs):
        pfx = name.split("-")[0]
        det = cfgs[name]["cells"]
        comp = sum(cell_time(pfx, isl, c) for (isl,osl,tp,c) in det)
        dl = MODELS.get(pfx,{}).get("gb",230)/DL_RATE_GB_S
        if pfx not in by_model: by_model.add(pfx); dl_total += dl
        basis = "MEASURED" if pfx=="gptoss" else "est.±50%"
        grand += comp
        print(f"{name:30}{len(det):>6}{dl/3600:>6.1f}h{comp/3600:>9.1f}h{(comp+dl)/3600:>8.1f}h  {basis}")
    total_cells = sum(len(c["cells"]) for c in cfgs.values())
    print("-"*74)
    print(f"{'compute (all cells)':30}{total_cells:>6}{'':>7}{'':>10}{grand/3600:>8.1f}h")
    print(f"{f'one-time downloads ({len(by_model)} models)':30}{'':>6}{'':>7}{'':>10}{dl_total/3600:>8.1f}h")
    print(f"{'GRAND TOTAL (1 node, sequential)':30}{'':>6}{'':>7}{'':>10}{(grand+dl_total)/3600:>8.1f}h")

    # solid subtotal: gpt-oss only (measured-anchored)
    g = {k:v for k,v in cfgs.items() if k.startswith("gptoss")}
    gcomp = sum(cell_time("gptoss",isl,c) for v in g.values() for (isl,osl,tp,c) in v["cells"])
    gcells = sum(len(v["cells"]) for v in g.values())
    print(f"\n>>> SOLID (measured-anchored): gpt-oss-120b full H200 config = "
          f"{gcells} cells, ~{(gcomp+815)/3600:.1f}h incl. download")
    print(">>> Big-MoE rows are ±~50% estimates pending an 8xH200 calibration.")

if __name__ == "__main__":
    main()
