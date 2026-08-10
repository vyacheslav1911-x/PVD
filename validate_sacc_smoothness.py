#!/usr/bin/env python
"""Re-test S_acc against the RIGHT outcome: it's a SMOOTHNESS term, not feasibility.

Question: does |commanded acceleration| predict oscillation in the ACHIEVED motion
(measured jerk = 3rd derivative of pos_, and achieved-acceleration magnitude),
INDEPENDENT of velocity?  If yes -> S_acc is grounded, as a smoothness penalty.

y proxies (from achieved pos_, savgol-differentiated):
  |jerk_ach|  = |d3 pos/dt3|      (oscillation / roughness)
  |acc_ach|   = |d2 pos/dt2|      (achieved acceleration magnitude)
Aligned to the command by the achieved-lag L=4.
"""
import csv
import numpy as np
from scipy.signal import savgol_filter

CSV = "pvd_logs/record_run1.csv"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = J[:5]
rows = list(csv.DictReader(open(CSV)))
def col(p, j): return np.array([float(r[f"{p}_{j}"]) if r[f"{p}_{j}"] not in ("", "nan") else np.nan for r in rows])
cmd = {j: col("cmd", j) for j in J}; pos = {j: col("pos", j) for j in J}
dt = np.array([float(r["dt"]) for r in rows]); N = len(rows)
good = np.ones(N, bool); good[:25] = False; good[dt > 0.06] = False
delta = float(np.median(dt[good]))
L = 4

def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 5 and np.ptp(x[m]) and np.ptp(y[m]) else np.nan

def partial(x, y, z):
    """partial corr of x,y controlling for z."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    r_xy, r_xz, r_yz = pearson(x[m], y[m]), pearson(x[m], z[m]), pearson(y[m], z[m])
    d = np.sqrt(max(1e-9, (1 - r_xz**2) * (1 - r_yz**2)))
    return (r_xy - r_xz * r_yz) / d

# commanded velocity/accel (smooth signal); achieved accel/jerk (measured motion)
acmd, vcmd, jerk_ach, acc_ach = {}, {}, {}, {}
for j in J:
    c = savgol_filter(cmd[j], 9, 4, deriv=0, delta=delta)
    vcmd[j] = savgol_filter(cmd[j], 9, 4, deriv=1, delta=delta)
    acmd[j] = savgol_filter(cmd[j], 9, 4, deriv=2, delta=delta)
    acc_ach[j] = savgol_filter(pos[j], 9, 4, deriv=2, delta=delta)
    jerk_ach[j] = savgol_filter(pos[j], 9, 4, deriv=3, delta=delta)

print(f"[sacc] delta={delta*1e3:.1f}ms, lag L={L}\n")
print("========== S_acc as SMOOTHNESS: |cmd accel| -> achieved oscillation ==========")
print(f"{'joint':<14}{'r(a_cmd,jerk)':>15}{'partial|v':>11}{'r(a_cmd,acc_ach)':>18}")
rj_all, prj_all = [], []
for j in J:
    ac = np.abs(acmd[j]); vc = np.abs(vcmd[j])
    # align achieved (t+L) to command (t)
    jk = np.full(N, np.nan); jk[:N - L] = np.abs(jerk_ach[j])[L:]
    aa = np.full(N, np.nan); aa[:N - L] = np.abs(acc_ach[j])[L:]
    m = good.copy(); m[N - L:] = False
    ac_, vc_, jk_, aa_ = ac[m], vc[m], jk[m], aa[m]
    r_aj = pearson(ac_, jk_)
    pr_aj = partial(ac_, jk_, vc_)
    r_aa = pearson(ac_, aa_)
    if j in BODY:
        rj_all.append(r_aj); prj_all.append(pr_aj)
    print(f"{j:<14}{r_aj:>15.3f}{pr_aj:>11.3f}{r_aa:>18.3f}")
mj = np.nanmean(rj_all); mpj = np.nanmean(prj_all)
print(f"\nmean body-joint r(a_cmd, measured jerk) = {mj:+.3f}")
print(f"mean body-joint PARTIAL r (controlling velocity) = {mpj:+.3f}")

def verdict(r, s=0.5, w=0.25):
    a = abs(r); return "VALIDATED" if a >= s else ("PARTIAL" if a >= w else "UNCLEAR/REFUTED")
print("\n================= VERDICT =================")
print(f"S_acc as SMOOTHNESS term: {verdict(mpj)}  (partial r vs measured jerk = {mpj:+.3f})")
print("Interpretation: if positive & surviving velocity control, commanded acceleration")
print("independently produces achieved-motion jerk -> S_acc is a grounded SMOOTHNESS penalty,")
print("NOT a feasibility penalty (which it already failed).")
