#!/usr/bin/env python3
"""Plot ONLY the measured dots (avg divergence vs K) from divergence_summary_1000.csv.
No fit line, no asymptote, no annotations — just the points."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_IN   = "divergence_summary_1000.csv"
PLOT_OUT = "divergence_vs_k_dots.png"

K, Y = [], []
with open(CSV_IN) as f:
    for row in csv.DictReader(f):
        K.append(float(row["K"]))
        Y.append(float(row["avg_divergence"]))

INK, MUTED, GRID, DOT = "#111827", "#6b7280", "#eceff3", "#2563eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.scatter(K, Y, s=46, color=DOT, edgecolor="white", linewidth=0.9, zorder=3)

ax.text(0.0, 1.03, "SmolVLA — mean per-joint divergence vs K", transform=ax.transAxes,
        color=INK, fontsize=15, fontweight="bold", va="bottom")
ax.set_xlabel("K   (number of sampled trajectories)", fontsize=11.5, labelpad=8)
ax.set_ylabel("Mean per-joint divergence", fontsize=11.5, labelpad=8)

ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.margins(x=0.02)

fig.subplots_adjust(top=0.9, left=0.075, right=0.97, bottom=0.12)
fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"-> Wrote {PLOT_OUT} ({len(K)} points)")
