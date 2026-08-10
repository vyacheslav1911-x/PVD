"""FFT of the speed-vs-time curve of one recorded movement.

Speed is joint-space speed: ||dq/dt||_2 over the 6 joints, in deg/s.
Positions are logged at irregular intervals, so they are resampled onto a
uniform grid before differentiating and transforming.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "recorded trajectories/record_run1.csv"
OUT = HERE / f"speed_fft_{CSV.stem}.png"

INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e3e3e6"
SERIES = "#3b6ef6"
ACCENT = "#d1495b"

# ---------------------------------------------------------------- load
df = pd.read_csv(CSV)
joints = [c[4:] for c in df.columns if c.startswith("pos_")]
t = df["t_obs"].to_numpy()
t -= t[0]
q = df[[f"pos_{j}" for j in joints]].to_numpy()

# ------------------------------------------------- uniform resample + speed
dt = float(np.median(np.diff(t)))
fs = 1.0 / dt
tu = np.arange(0.0, t[-1], dt)
qu = np.column_stack([np.interp(tu, t, q[:, i]) for i in range(q.shape[1])])

dqdt = np.gradient(qu, dt, axis=0)          # deg/s per joint
speed = np.linalg.norm(dqdt, axis=1)        # deg/s, joint-space norm

# ---------------------------------------------------------------- FFT
# SPARC preprocessing: no window, DC kept, heavy zero-padding, normalised by
# its own max (so the y-axis is a fraction of the DC peak, not deg/s).
FMAX = 10.0
n = speed.size
nfft = 2 ** (int(np.ceil(np.log2(n))) + 4)
freq = np.fft.rfftfreq(nfft, dt)
mag = np.abs(np.fft.rfft(speed, nfft))
mag /= mag.max()

# Zero-padding interpolates; the true resolution is still fs/n, so peaks are
# only reported once per resolution cell, and the DC main lobe is skipped.
res = fs / n
lo = int(np.ceil(2 * res / freq[1]))
sep = int(round(res / freq[1]))
loc = np.where((mag[lo:-1] > mag[lo - 1 : -2]) & (mag[lo:-1] > mag[lo + 1 :]))[0] + lo
peaks = []
for i in loc[np.argsort(mag[loc])[::-1]]:
    if all(abs(i - j) >= sep for j in peaks):
        peaks.append(i)
    if len(peaks) == 5:
        break
peaks = np.array(sorted(peaks))

# SPARC's adaptive band: from DC out to the last bin above 5% of the max
TH = 0.05
band = np.where(mag[freq <= FMAX] >= TH)[0]
f_cut = freq[band[-1]]

print(f"file            : {CSV}")
print(f"samples         : {n} resampled ({len(df)} raw)")
print(f"duration        : {tu[-1]:.2f} s   fs = {fs:.2f} Hz   Nyquist = {fs/2:.2f} Hz")
print(f"nfft            : {nfft} (zero-padded from {n})")
print(f"bin spacing     : {freq[1]:.5f} Hz   true resolution {res:.4f} Hz")
print(f"speed           : mean {speed.mean():.1f}  peak {speed.max():.1f} deg/s")
print(f"5% cutoff       : {f_cut:.3f} Hz")
print("\ntop spectral peaks (normalised magnitude, DC excluded)")
for i in peaks[np.argsort(mag[peaks])[::-1]]:
    print(f"  {freq[i]:6.3f} Hz  ({1/freq[i]:6.2f} s)   mag {mag[i]:6.4f}")

# ---------------------------------------------------------------- plot
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 9.5), gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.42}
)
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

# 1 — speed vs time
ax1.plot(tu, speed, color=SERIES, lw=1.6)
ax1.set_title(
    f"Joint-space speed of one movement  ·  {CSV.stem}",
    color=INK, fontsize=13, fontweight="600", loc="left", pad=10,
)
ax1.set_xlabel("time (s)", color=MUTED, fontsize=10)
ax1.set_ylabel("‖dq/dt‖  (deg/s)", color=MUTED, fontsize=10)
ax1.set_xlim(0, tu[-1])

# 2 — normalised magnitude spectrum, linear
ax2.plot(freq, mag, color=SERIES, lw=1.4)
ax2.set_title(
    f"Normalised magnitude spectrum, 0–{FMAX:.0f} Hz",
    color=INK, fontsize=13, fontweight="600", loc="left", pad=10,
)
ax2.set_xlabel("frequency (Hz)", color=MUTED, fontsize=10)
ax2.set_ylabel("|X(f)| / max", color=MUTED, fontsize=10)
ax2.set_xlim(0, FMAX)

ymax = mag[lo:][freq[lo:] <= FMAX].max()
ax2.set_ylim(0, ymax * 1.55)
ax2.annotate(
    "DC = 1.0, off scale",
    (0, ymax * 1.5), textcoords="offset points", xytext=(6, 0),
    ha="left", va="top", fontsize=9, color=MUTED,
)

# the 5% threshold and the band SPARC integrates over
ax2.axhline(TH, color=MUTED, lw=1.0, ls=":", zorder=1)
ax2.axvspan(0, f_cut, color=SERIES, alpha=0.06, zorder=0)
ax2.annotate(
    f"5% threshold → SPARC cutoff {f_cut:.2f} Hz",
    (f_cut, TH), textcoords="offset points", xytext=(6, 6),
    ha="left", va="bottom", fontsize=9, color=MUTED,
)

# mean action-chunk rate, for reference (chunk spacing is irregular)
f_chunk = 1.0 / np.mean(np.diff(t[df["chunk_start"].to_numpy() == 1]))
ax2.axvline(f_chunk, color=MUTED, lw=1.2, ls="--", zorder=1)
ax2.annotate(
    f"mean chunk rate {f_chunk:.2f} Hz",
    (f_chunk, ymax * 1.28), textcoords="offset points", xytext=(6, 0),
    ha="left", va="top", fontsize=9, color=MUTED,
)

# the peaks now crowd into the first fraction of a Hz, so they are marked on
# the curve and listed off to the side rather than labelled in place
ax2.plot(freq[peaks], mag[peaks], "o", ms=5, color=ACCENT, zorder=3, ls="none")
ranked = peaks[np.argsort(mag[peaks])[::-1]]
ax2.text(
    0.985, 0.94,
    "peaks (DC excluded)\n"
    + "\n".join(f"{freq[i]:5.2f} Hz   {mag[i]:.2f}" for i in ranked),
    transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
    color=ACCENT, linespacing=1.5, family="monospace",
)

# 3 — same spectrum, log magnitude (shows the noise floor / roll-off)
ax3.semilogy(freq, np.maximum(mag, 1e-5), color=SERIES, lw=1.2)
ax3.axhline(TH, color=MUTED, lw=1.0, ls=":", zorder=1)
ax3.set_title(
    f"Same spectrum, log magnitude",
    color=INK, fontsize=13, fontweight="600", loc="left", pad=10,
)
ax3.set_xlabel("frequency (Hz)", color=MUTED, fontsize=10)
ax3.set_ylabel("|X(f)| / max", color=MUTED, fontsize=10)
ax3.set_xlim(0, FMAX)

fig.text(
    0.5, 0.012,
    f"{n} samples resampled to {fs:.2f} Hz  ·  {tu[-1]:.1f} s  ·  no window, "
    f"DC kept  ·  nfft {nfft}  ·  true resolution {res:.3f} Hz",
    ha="center", fontsize=9, color=MUTED,
)

fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nwrote {OUT}")
