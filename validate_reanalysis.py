#!/usr/bin/env python
"""Free reanalysis (no hardware):
 1. S_vel with CAP-affected ticks masked out -> is it servo-grounded or a cap detector?
 2. S_cont with ALL boundaries + boundary-proximity partial correlation (control = velocity).
 3. Documented boundary count (why 36 vs 16).
"""
import csv
import numpy as np

CSV = "pvd_logs/record_run1.csv"
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = J[:5]
rows = list(csv.DictReader(open(CSV)))
def col(p, j): return np.array([float(r[f"{p}_{j}"]) if r[f"{p}_{j}"] not in ("", "nan") else np.nan for r in rows])
cmd = {j: col("cmd", j) for j in J}; pos = {j: col("pos", j) for j in J}; sent = {j: col("sent", j) for j in J}
dt = np.array([float(r["dt"]) for r in rows]); cs = np.array([int(r["chunk_start"]) for r in rows]); N = len(rows)
good = np.ones(N, bool); good[:25] = False; good[dt > 0.06] = False   # warmup / slow-tick mask (for per-tick vel/acc only)
L = 4

def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 5 and np.ptp(x[m]) and np.ptp(y[m]) else np.nan

# per-tick commanded velocity + tracking error
err, vcmd, capped = {}, {}, {}
for j in J:
    e = np.full(N, np.nan); e[:N - L] = cmd[j][:N - L] - pos[j][L:]
    v = np.full(N, np.nan); v[1:] = (cmd[j][1:] - cmd[j][:-1]) / np.where(dt[1:] > 0, dt[1:], np.nan)
    err[j], vcmd[j] = e, v
    cp = np.abs(cmd[j] - sent[j]) > 0.5                                  # tick was clipped by the cap
    # window-capped: any clip in [t, t+L] contaminates pos(t+L) vs cmd(t)
    wc = np.zeros(N, bool)
    for t in range(N - L):
        wc[t] = cp[t:t + L + 1].any()
    capped[j] = wc

print("========== S_vel : |cmd velocity| vs |tracking error|, ALL vs CAP-MASKED ==========")
print(f"{'joint':<14}{'r_all':>8}{'r_masked':>10}{'n_all':>8}{'n_masked':>10}{'|v|max_masked':>15}")
rall, rmsk = {}, {}
for j in J:
    base = good & np.isfinite(err[j]) & np.isfinite(vcmd[j])
    msk = base & ~capped[j]
    ra = pearson(np.abs(vcmd[j])[base], np.abs(err[j])[base])
    rm = pearson(np.abs(vcmd[j])[msk], np.abs(err[j])[msk])
    rall[j], rmsk[j] = ra, rm
    vmax = np.nanmax(np.abs(vcmd[j])[msk]) if msk.sum() else np.nan
    print(f"{j:<14}{ra:>8.3f}{rm:>10.3f}{base.sum():>8}{msk.sum():>10}{vmax:>15.0f}")
print(f"mean body-joint r:  ALL={np.nanmean([rall[j] for j in BODY]):+.3f}   "
      f"CAP-MASKED={np.nanmean([rmsk[j] for j in BODY]):+.3f}")
print("  -> survives masking = servo-grounded;  collapses = cap detector.\n")

print("========== S_cont : boundary accounting + tests ==========")
tot = int(cs.sum())
in_warm = int((cs[:25] == 1).sum())
in_slow = int(((cs == 1) & (dt > 0.06))[25:].sum())
print(f"total chunk_start markers = {tot}")
print(f"  dropped by old filter: {in_warm} in warmup(<25), {in_slow} on slow/inference ticks (dt>0.06)")
print(f"  -> the boundary tick IS the inference tick (large dt), so the old dt-mask wrongly dropped it.")
b_all = [t for t in range(2, N - L - 6) if cs[t] == 1]
print(f"boundaries usable now (all, only needing room after) = {len(b_all)}\n")

# (a) jump vs post-boundary spike, ALL boundaries
jumps, spikes = [], []
for t in b_all:
    jump = np.sqrt(sum((cmd[j][t] - pos[j][t]) ** 2 for j in BODY))
    base = np.mean([np.nanmean([abs(err[j][t - 3]), abs(err[j][t - 2])]) for j in BODY])
    post = np.max([np.nanmean([abs(err[j][t + k]) for j in BODY]) for k in range(0, 5)])
    jumps.append(jump); spikes.append(post - base)
jumps, spikes = np.array(jumps), np.array(spikes)
r_js = pearson(jumps, spikes)
print(f"(a) jump ||a1-q0|| vs post-boundary error spike:  r={r_js:.3f}  (n={len(b_all)}, "
      f"jump range [{jumps.min():.1f},{jumps.max():.1f}] deg)")

# (b) per-tick boundary-proximity, partial correlation controlling for velocity (pooled over body joints)
prox = np.zeros(N)                                   # 1 for ticks within [b, b+4] of any boundary
for t in b_all:
    prox[min(t, N - 1):min(t + 5, N)] = 1.0
Xp, Xe, Xv = [], [], []
for j in BODY:
    m = good & np.isfinite(err[j]) & np.isfinite(vcmd[j])
    Xp.append(prox[m]); Xe.append(np.abs(err[j])[m]); Xv.append(np.abs(vcmd[j])[m])
Xp, Xe, Xv = np.concatenate(Xp), np.concatenate(Xe), np.concatenate(Xv)
r_pe = pearson(Xp, Xe); r_pv = pearson(Xp, Xv); r_ev = pearson(Xv, Xe)
den = np.sqrt(max(1e-9, (1 - r_pv**2) * (1 - r_ev**2)))
pr = (r_pe - r_pv * r_ev) / den
print(f"(b) boundary-proximity vs |error|:  raw r={r_pe:.3f}   partial r|velocity={pr:.3f}  (pooled n={len(Xp)})")
print(f"    (proximity->error controlling for velocity; ~0 = boundaries add nothing beyond velocity)\n")

def verdict(r, s=0.5, w=0.25):
    a = abs(r); return "VALIDATED" if a >= s else ("PARTIAL" if a >= w else "UNCLEAR/REFUTED")
mv = np.nanmean([rmsk[j] for j in BODY])
print("================= VERDICTS (reanalysis) =================")
print(f"S_vel (cap-masked): {verdict(mv):<16} mean r={mv:+.3f}")
print(f"S_cont jump->spike: {verdict(r_js):<16} r={r_js:+.3f} (n={len(b_all)})")
print(f"S_cont proximity  : {verdict(pr):<16} partial r|v={pr:+.3f}")
