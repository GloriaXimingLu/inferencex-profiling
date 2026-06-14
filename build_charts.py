#!/usr/bin/env python3
"""Generate boss-friendly charts for the report site -> docs/assets/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2933"
MUTE = "#7b8794"
GREEN = "#0f9d58"
GRAY = "#b8c2cc"
RED = "#d64545"
BLUE = "#2b6cb0"
plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": "#cbd2d9", "axes.linewidth": 1,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "font.family": "DejaVu Sans",
})

# --- Chart 1: serving cost per 1M output tokens (the headline) ---
fig, ax = plt.subplots(figsize=(6.4, 4.4))
bars = ax.bar(["Previous setup", "Auto-tuned config"], [1.83, 1.05],
              color=[GRAY, GREEN], width=0.6, zorder=3)
for b, v in zip(bars, [1.83, 1.05]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.04, f"${v:.2f}", ha="center",
            va="bottom", fontsize=15, fontweight="bold")
ax.annotate("", xy=(1, 1.13), xytext=(0, 1.91),
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text(0.5, 1.62, "−43%", ha="center", color=RED, fontsize=20, fontweight="bold")
ax.set_ylabel("Cost per 1M output tokens (US$)")
ax.set_title("Baseline vs. auto-tuned configuration",
             fontsize=14.5, fontweight="bold", pad=12)
ax.set_ylim(0, 2.15); ax.grid(axis="y", color="#eef1f4", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cost_compare.png"), dpi=150)
plt.close(fig)

# --- Chart 2: tau-bench throughput change vs baseline, by regime ---
fig, ax = plt.subplots(figsize=(9.2, 4.8))
labels = ["Claude", "Gemini", "GPT-5", "Qwen", "Claude", "Gemini", "Qwen"]
vals   = [9, 69, 63, 70, -18, 3, -12]
groups = ["air", "air", "air", "air", "bank", "bank", "bank"]
x = [0, 1, 2, 3, 4.7, 5.7, 6.7]
colors = [GREEN if g == "air" else (GREEN if v > 0 else RED) for g, v in zip(groups, vals)]
colors = [GREEN if v >= 5 else (RED if v <= -5 else MUTE) for v in vals]
bars = ax.bar(x, vals, color=colors, width=0.8, zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + (2 if v >= 0 else -2), f"{v:+d}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=12, fontweight="bold")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Throughput change vs. previous setup")
ax.set_ylim(-34, 112)
ax.set_title("tau-bench replay: throughput change vs. baseline",
             fontsize=14.5, fontweight="bold", pad=12)
ax.text(1.5, 96, "HEAVY LOAD  (many users at once)", ha="center", color=BLUE,
        fontsize=11.5, fontweight="bold")
ax.text(5.7, 96, "LIGHT LOAD  (few users)", ha="center", color=MUTE,
        fontsize=11.5, fontweight="bold")
ax.axvline(3.85, color="#dde2e7", lw=1, ls="--")
ax.grid(axis="y", color="#eef1f4", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tau_throughput.png"), dpi=150)
plt.close(fig)

print("wrote", os.path.join(OUT, "cost_compare.png"))
print("wrote", os.path.join(OUT, "tau_throughput.png"))

# --- Chart 3: clean frontier — per-user speed vs total throughput ---
base = {1:(232,214),2:(190,349),4:(140,520),8:(96,722),16:(63,946),32:(40,1212),64:(24,1455),128:(13.5,1647),256:(8.5,1722)}
ceil = {1:(320,225),2:(217,365),4:(133,465),8:(106,768),16:(79,1166),32:(54,1614),64:(32,1936),128:(19,2335),256:(11,2657)}
fw   = {1:(310,250),2:(270,410),4:(255,800),8:(178,1160),16:(113,1570),32:(68,1960),64:(35,2300),128:(33,2180),256:(33,2300)}
fig, ax = plt.subplots(figsize=(8.6, 5.7))
def _curve(d, color, label, ls="-", mk="o", lw=2.6):
    cs = sorted(d)
    ax.plot([d[c][0] for c in cs], [d[c][1] for c in cs], ls, marker=mk,
            color=color, label=label, lw=lw, markersize=5, zorder=3)
_curve(fw, "#e8833a", "Commercial reference (Fireworks)", ls="--")
_curve(ceil, GREEN, "Auto-tuned config")
_curve(base, GRAY, "Baseline (previous config)")
ax.set_xlabel("Per-user output speed  (tokens/sec)")
ax.set_ylabel("Total throughput  (tokens/sec)")
ax.set_title("Total throughput vs. per-user speed, concurrency 1–256",
             fontsize=14.5, fontweight="bold", pad=12)
ax.legend(loc="center right", fontsize=11, frameon=False)
ax.set_xlim(0, 345); ax.set_ylim(0, 2980)
ax.grid(color="#eef1f4", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "frontier_clean.png"), dpi=150)
plt.close(fig)
print("wrote", os.path.join(OUT, "frontier_clean.png"))
