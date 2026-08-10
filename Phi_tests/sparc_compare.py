"""Steps 4-5 — windowed SPARC per policy, then compare the distributions.

The speed never returns to zero, so rest-to-rest segmentation is impossible.
Instead lambda_S is computed on sliding windows and the two policies are
compared as distributions: median, IQR, Mann-Whitney U, and U converted to
AUC = P(lambda_S(ACT) > lambda_S(pi0.5)).

Several window/filter configurations are run, because a long window on a
continuous stream measures the task rhythm rather than the dither.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu

from rollout import load_speed
from sparc import sparc

HERE = Path(__file__).resolve().parent
DATA = HERE / "recorded trajectories"
OUT = HERE / "sparc_compare.png"

RUNS = {"ACT": DATA / "record_run1.csv", "pi0.5": DATA / "record_pi05_run1.csv"}
COLOR = {"ACT": "#2a78d6", "pi0.5": "#eb6834"}

INK, MUTED, GRID = "#1c1c1e", "#8a8a8e", "#e3e3e6"


def windows(speed, fs, width_s, hop_s):
    """Start indices and slices for sliding windows."""
    w, h = int(round(width_s * fs)), int(round(hop_s * fs))
    return [(i, speed[i : i + w]) for i in range(0, speed.size - w + 1, h)]


def highpass(speed, fs, fc=2.0, order=4):
    b, a = butter(order, fc, btype="high", fs=fs)
    return filtfilt(b, a, speed)


def lam_series(speed, fs, width_s, hop_s):
    return np.array([sparc(seg, fs) for _, seg in windows(speed, fs, width_s, hop_s)])


def mean_speed_series(speed, fs, width_s, hop_s):
    return np.array([np.abs(seg).mean() for _, seg in windows(speed, fs, width_s, hop_s)])


def residualise(lam, mu):
    """lambda_S with the window's mean speed regressed out (pooled fit)."""
    Z = np.column_stack([np.ones(mu.size), mu])
    return lam - Z @ np.linalg.lstsq(Z, lam, rcond=None)[0]


def compare(a, b):
    """Mann-Whitney U on ACT vs pi0.5, plus U as an AUC."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return u, p, u / (a.size * b.size)


# ------------------------------------------------------------------ load
runs = {k: load_speed(v) for k, v in RUNS.items()}
for k, r in runs.items():
    print(f"{k:6s} {r.speed.size} samples @ {r.fs:.2f} Hz  ({r.t[-1]:.1f} s)")

# ------------------------------------------------- the configurations to try
CONFIGS = [
    ("2 s / 1 s hop", 2.0, 1.0, False),
    ("1 s / 0.5 s hop", 1.0, 0.5, False),
    ("2 s / 1 s hop, HP 2 Hz", 2.0, 1.0, True),
    ("1 s / 0.5 s hop, HP 2 Hz", 1.0, 0.5, True),
    ("2 s / 2 s hop (independent)", 2.0, 2.0, False),
]

signals = {
    (k, False): r.speed for k, r in runs.items()
} | {
    (k, True): highpass(r.speed, r.fs) for k, r in runs.items()
}

results = []
print(f"\n{'configuration':30s} {'n':>7s}  {'median ACT':>11s} {'IQR':>12s}"
      f"  {'median pi0.5':>13s} {'IQR':>12s}   {'p':>8s} {'AUC':>6s} {'AUC|speed':>10s}")
for name, width, hop, hp in CONFIGS:
    lam = {k: lam_series(signals[(k, hp)], runs[k].fs, width, hop) for k in RUNS}
    mu = {k: mean_speed_series(signals[(k, hp)], runs[k].fs, width, hop) for k in RUNS}
    a, b = lam["ACT"], lam["pi0.5"]
    u, p, auc = compare(a, b)

    # the same test on lambda_S with window mean speed regressed out
    res = residualise(np.hstack([a, b]), np.hstack([mu["ACT"], mu["pi0.5"]]))
    _, p_r, auc_r = compare(res[: a.size], res[a.size :])

    q = {k: np.percentile(v, [25, 50, 75]) for k, v in lam.items()}
    results.append(dict(name=name, width=width, hop=hop, hp=hp, lam=lam,
                        u=u, p=p, auc=auc, auc_r=auc_r, p_r=p_r))
    print(f"{name:30s} {a.size:3d}/{b.size:<3d}  {q['ACT'][1]:11.3f} "
          f"[{q['ACT'][0]:5.2f},{q['ACT'][2]:5.2f}]  {q['pi0.5'][1]:13.3f} "
          f"[{q['pi0.5'][0]:5.2f},{q['pi0.5'][2]:5.2f}]   {p:8.4f} {auc:6.3f} "
          f"{auc_r:10.3f}")

print("\nAUC = P(lambda_S(ACT) > lambda_S(pi0.5)); 0.5 is chance, >0.5 means ACT")
print("scores smoother. AUC|speed repeats the test after regressing the window's")
print("mean speed out of lambda_S: the unfiltered rows move toward chance (pi0.5")
print("simply moves faster), while the high-passed rows are unaffected by it.")
print("Overlapping windows share data, so p is anticonservative; the last row uses")
print("non-overlapping windows as the honest check.")

# ------------------------------------------------------------------- plot
primary = results[0]
fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.75, 0.9], hspace=0.52)
ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
fig.patch.set_facecolor("white")

for ax in (ax1, ax2, ax3):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

# 1 — lambda_S over time, both policies
for k in RUNS:
    lam = primary["lam"][k]
    tw = np.arange(lam.size) * primary["hop"] + primary["width"] / 2
    ax1.plot(tw, lam, color=COLOR[k], lw=2.0, marker="o", ms=4.5, label=k)
    ax1.annotate(k, (tw[-1], lam[-1]), textcoords="offset points", xytext=(8, 0),
                 va="center", fontsize=10, color=COLOR[k], fontweight="bold")
ax1.set_title(f"Windowed SPARC over the rollout  ·  {primary['name']}",
              color=INK, fontsize=13, fontweight="600", loc="left", pad=10)
ax1.set_xlabel("window centre (s)", color=MUTED, fontsize=10)
ax1.set_ylabel("$\\lambda_S$", color=MUTED, fontsize=11)
ax1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.005),
           fontsize=9, ncol=2, labelcolor=MUTED)
ax1.set_xlim(0, max(r.t[-1] for r in runs.values()))

# 2 — the two distributions
rng = np.random.default_rng(0)
for row, k in enumerate(RUNS):
    lam = primary["lam"][k]
    y = row + rng.uniform(-0.11, 0.11, lam.size)
    ax2.plot(lam, y, "o", ms=5, color=COLOR[k], alpha=0.55, mec="white", mew=0.6)
    q1, med, q3 = np.percentile(lam, [25, 50, 75])
    ax2.plot([q1, q3], [row - 0.26] * 2, color=COLOR[k], lw=6, solid_capstyle="butt")
    ax2.plot([med], [row - 0.26], "|", ms=16, mew=2.5, color="white")
    ax2.annotate(f"median {med:.2f}   IQR [{q1:.2f}, {q3:.2f}]",
                 (q3, row - 0.26), textcoords="offset points", xytext=(10, 0),
                 va="center", fontsize=9, color=COLOR[k])
ax2.set_yticks(range(len(RUNS)))
ax2.set_yticklabels(list(RUNS), color=INK, fontsize=10)
ax2.set_ylim(-0.55, len(RUNS) - 0.4)
ax2.set_title(
    f"Distribution of $\\lambda_S$ per window   ·   "
    f"AUC {primary['auc']:.3f},  p = {primary['p']:.3f}",
    color=INK, fontsize=13, fontweight="600", loc="left", pad=10)
ax2.set_xlabel("$\\lambda_S$  (less negative = smoother)", color=MUTED, fontsize=10)

# 3 — AUC across configurations
names = [r["name"] for r in results]
aucs = [r["auc"] for r in results]
ypos = np.arange(len(results))[::-1]
ax3.axvline(0.5, color=MUTED, lw=1.2, ls="--", zorder=1)
for y, r in zip(ypos, results):
    c = COLOR["ACT"] if r["auc"] >= 0.5 else COLOR["pi0.5"]
    ax3.plot([0.5, r["auc"]], [y, y], color=c, lw=3, solid_capstyle="round", zorder=2)
    ax3.plot([r["auc"]], [y], "o", ms=9, color=c, zorder=3,
             label="raw" if y == ypos[0] else None)
    ax3.plot([r["auc_r"]], [y], "o", ms=8, mfc="white", mec=c, mew=2, zorder=4,
             label="mean speed removed" if y == ypos[0] else None)
    ax3.annotate(f"{r['auc']:.3f}   p = {r['p']:.3g}",
                 (min(r["auc"], r["auc_r"]), y), textcoords="offset points",
                 xytext=(-12, 0), ha="right", va="center", fontsize=9, color=c)
ax3.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.005),
           fontsize=9, ncol=2, labelcolor=MUTED)
ax3.set_yticks(ypos)
ax3.set_yticklabels(names, color=INK, fontsize=9.5)
ax3.set_xlim(0, 1)
ax3.set_ylim(-0.6, len(results) - 0.4)
ax3.set_title("Discrimination across window / filter settings",
              color=INK, fontsize=13, fontweight="600", loc="left", pad=10)
ax3.set_xlabel("AUC = P($\\lambda_S$ ACT > $\\lambda_S$ pi0.5)   ·   0.5 = chance",
               color=MUTED, fontsize=10)
ax3.annotate("chance", (0.5, len(results) - 0.5), textcoords="offset points",
             xytext=(0, -4), ha="center", va="top", fontsize=9, color=MUTED)
ax3.annotate("hollow = mean speed removed: unfiltered rows drift to chance, high-passed rows do not",
             (0.5, -0.55), textcoords="offset points", xytext=(-10, 0),
             ha="right", va="center", fontsize=9, color=MUTED)

full = {k: sparc(r.speed, r.fs) for k, r in runs.items()}
fig.text(
    0.5, 0.045,
    f"Whole-rollout $\\lambda_S$ runs the other way: ACT {full['ACT']:.2f}  vs  "
    f"pi0.5 {full['pi0.5']:.2f}  —  windowing changes the sign of the effect.",
    ha="center", fontsize=9.5, color=INK,
)

fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nwrote {OUT}")
