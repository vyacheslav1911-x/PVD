#!/usr/bin/env python
"""Validate S_vel / S_acc / S_cont against real tracking error, from a
record_rollout.py CSV (commanded cmd_ + achieved pos_ + chunk_start).

Ground truth: tracking error e_j(t) = cmd_j(t) - pos_j(t+L), where the achieved
position lags the command by L ticks (L chosen to minimise mean |e|, ~4).

S_vel  : does high COMMANDED |velocity| co-occur with tracking error?
         r(|v_cmd|, |e|) per joint; binned trend; onset velocity.
S_acc  : does high COMMANDED |accel| co-occur with error, INDEPENDENT of velocity?
         r(|a_cmd|,|e|) raw AND partial-r controlling for |v_cmd|.
S_cont : at chunk boundaries, does the jump |a_1 - q_0| predict a post-boundary
         tracking-error spike?
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "pvd_logs/record_run1.csv"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = J[:5]

rows = list(csv.DictReader(open(CSV)))
def col(p, j): return np.array([float(r[f"{p}_{j}"]) if r[f"{p}_{j}"] not in ("", "nan") else np.nan for r in rows])
cmd = {j: col("cmd", j) for j in J}
pos = {j: col("pos", j) for j in J}
sent = {j: col("sent", j) for j in J}
dt = np.array([float(r["dt"]) for r in rows])
cs = np.array([int(r["chunk_start"]) for r in rows])
N = len(rows)

# drop warmup: first 25 ticks + any abnormally long dt (finite-diff would blow up)
good = np.ones(N, bool); good[:25] = False; good[dt > 0.06] = False

# ---- best lag L (achieved lags command) ----
def mean_abs_err(L):
    e = []
    for j in BODY:
        a, b = cmd[j][:N - L], pos[j][L:]
        m = good[:N - L]
        e.append(np.nanmean(np.abs(a[m] - b[m])))
    return np.mean(e)
L = int(np.argmin([mean_abs_err(l) for l in range(0, 9)]))
print(f"[validate] {N} rows, best achieved-lag L={L} ticks (~{L*33:.0f} ms)\n")

def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]

# ---- per-joint commanded velocity / accel + tracking error (aligned) ----
err, vcmd, acmd = {}, {}, {}
for j in J:
    e = np.full(N, np.nan)
    e[:N - L] = cmd[j][:N - L] - pos[j][L:]          # tracking error at command time
    v = np.full(N, np.nan); v[1:] = (cmd[j][1:] - cmd[j][:-1]) / np.where(dt[1:] > 0, dt[1:], np.nan)
    a = np.full(N, np.nan); a[1:] = (v[1:] - v[:-1]) / np.where(dt[1:] > 0, dt[1:], np.nan)
    err[j], vcmd[j], acmd[j] = e, v, a

# ================= S_vel =================
print("========== S_vel : |commanded velocity| vs |tracking error| ==========")
print(f"{'joint':<14}{'r(|v|,|e|)':>12}{'|v| range deg/s':>18}{'cap-clip%':>10}")
svel_r = {}
for j in J:
    m = good & np.isfinite(err[j]) & np.isfinite(vcmd[j])
    r = pearson(np.abs(vcmd[j])[m], np.abs(err[j])[m]); svel_r[j] = r
    clip = np.mean(np.abs(cmd[j] - sent[j])[good] > 1e-3) * 100
    print(f"{j:<14}{r:>12.3f}{f'[0,{np.nanmax(np.abs(vcmd[j])[m]):.0f}]':>18}{clip:>10.1f}")
mvel = np.nanmean([svel_r[j] for j in BODY])
print(f"mean r over body joints = {mvel:.3f}\n")

# ================= S_acc =================
print("========== S_acc : |commanded accel| vs |error|, raw AND controlling for |v| ==========")
print(f"{'joint':<14}{'r(|a|,|e|)':>12}{'partial r|v':>13}")
sacc_r, sacc_pr = {}, {}
for j in J:
    m = good & np.isfinite(err[j]) & np.isfinite(vcmd[j]) & np.isfinite(acmd[j])
    ae = np.abs(err[j])[m]; av = np.abs(vcmd[j])[m]; aa = np.abs(acmd[j])[m]
    r_ae = pearson(aa, ae); r_av = pearson(aa, av); r_ev = pearson(av, ae)
    denom = np.sqrt(max(1e-9, (1 - r_av**2) * (1 - r_ev**2)))
    pr = (r_ae - r_av * r_ev) / denom if np.isfinite(denom) and denom > 0 else np.nan
    sacc_r[j], sacc_pr[j] = r_ae, pr
    print(f"{j:<14}{r_ae:>12.3f}{pr:>13.3f}")
macc = np.nanmean([sacc_r[j] for j in BODY]); macc_p = np.nanmean([sacc_pr[j] for j in BODY])
print(f"mean r={macc:.3f}   mean partial-r(controlling velocity)={macc_p:.3f}\n")

# ================= S_cont =================
print("========== S_cont : chunk-boundary jump vs post-boundary error spike ==========")
b_idx = [t for t in range(2, N - L - 6) if cs[t] == 1 and good[t]]
jumps, spikes = [], []
for t in b_idx:
    jump = np.sqrt(sum((cmd[j][t] - pos[j][t]) ** 2 for j in BODY))      # ||a_1 - q_0||
    base = np.mean([np.nanmean([abs(err[j][t - 3]), abs(err[j][t - 2])]) for j in BODY])
    post = np.max([np.nanmean([abs(err[j][t + k]) for j in BODY]) for k in range(0, 5)])
    jumps.append(jump); spikes.append(post - base)
jumps, spikes = np.array(jumps), np.array(spikes)
r_cont = pearson(jumps, spikes)
print(f"boundaries analysed = {len(b_idx)}")
print(f"jump ||a1-q0|| range = [{jumps.min():.1f}, {jumps.max():.1f}] deg")
print(f"r(jump, post-boundary error spike) = {r_cont:.3f}\n")

# ================= verdicts =================
def verdict(r, strong=0.5, weak=0.25):
    a = abs(r)
    return "VALIDATED" if a >= strong else ("PARTIAL" if a >= weak else "UNCLEAR/REFUTED")
print("================= VERDICTS =================")
print(f"S_vel : {verdict(mvel):<16} mean r={mvel:+.3f}")
print(f"S_acc : {verdict(macc_p):<16} raw r={macc:+.3f}, partial r|v={macc_p:+.3f} (velocity-controlled is the honest one)")
print(f"S_cont: {verdict(r_cont):<16} r={r_cont:+.3f} over {len(b_idx)} boundaries")

# ================= plot =================
INK, MUTED, GRID = "#111827", "#6b7280", "#eceff3"
C = {"shoulder_lift": "#2563eb", "elbow_flex": "#f59e0b", "wrist_roll": "#10b981"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)

# S_vel binned trend for the 3 most-moving joints
for j in ("shoulder_lift", "elbow_flex", "wrist_roll"):
    m = good & np.isfinite(err[j]) & np.isfinite(vcmd[j])
    v = np.abs(vcmd[j])[m]; e = np.abs(err[j])[m]
    bins = np.linspace(0, np.percentile(v, 98), 12)
    idx = np.digitize(v, bins)
    bx = [v[idx == k].mean() for k in range(1, len(bins)) if (idx == k).sum() > 3]
    by = [e[idx == k].mean() for k in range(1, len(bins)) if (idx == k).sum() > 3]
    ax[0].plot(bx, by, "-o", color=C[j], ms=4, lw=1.8, label=f"{j} (r={svel_r[j]:.2f})")
ax[0].set_title("S_vel: |cmd velocity| → tracking error", fontweight="bold", fontsize=11)
ax[0].set_xlabel("|commanded velocity| (deg/s)"); ax[0].set_ylabel("mean |tracking error| (deg)")
ax[0].legend(frameon=False, fontsize=8)

for j in ("shoulder_lift", "elbow_flex", "wrist_roll"):
    m = good & np.isfinite(err[j]) & np.isfinite(acmd[j])
    a = np.abs(acmd[j])[m]; e = np.abs(err[j])[m]
    bins = np.linspace(0, np.percentile(a, 98), 12)
    idx = np.digitize(a, bins)
    bx = [a[idx == k].mean() for k in range(1, len(bins)) if (idx == k).sum() > 3]
    by = [e[idx == k].mean() for k in range(1, len(bins)) if (idx == k).sum() > 3]
    ax[1].plot(bx, by, "-o", color=C[j], ms=4, lw=1.8, label=f"{j} (pr={sacc_pr[j]:.2f})")
ax[1].set_title("S_acc: |cmd accel| → tracking error", fontweight="bold", fontsize=11)
ax[1].set_xlabel("|commanded accel| (deg/s²)"); ax[1].set_ylabel("mean |tracking error| (deg)")
ax[1].legend(frameon=False, fontsize=8)

ax[2].scatter(jumps, spikes, s=40, color="#7c3aed", edgecolor="white", linewidth=0.8, zorder=3)
ax[2].set_title(f"S_cont: boundary jump → error spike (r={r_cont:.2f})", fontweight="bold", fontsize=11)
ax[2].set_xlabel("chunk-boundary jump ||a₁−q₀|| (deg)"); ax[2].set_ylabel("post-boundary error spike (deg)")

for a in ax:
    a.grid(color=GRID, lw=0.9, zorder=0); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Kinematic-term validation vs real tracking error (record_run1)",
             fontweight="bold", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("kinematic_terms_validation.png", dpi=140, bbox_inches="tight", facecolor="white")
print("\n[validate] wrote kinematic_terms_validation.png")
