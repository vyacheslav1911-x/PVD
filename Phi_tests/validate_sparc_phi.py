"""Step 6 — is lambda_S worth a term in Phi?

Follows the validate_sacc_smoothness.py / validate_reanalysis.py precedent:

  x  candidate term   lambda_S of the COMMANDED speed profile (Phi scores
                      commands, not achieved motion, so the term has to be
                      computable from a candidate chunk)
  y  outcome          (a) mean |tracking error|, cmd_(t) vs pos_(t+L), L=4
                      (b) mean |measured jerk| of the achieved motion
  z  controls         S_vel, and mean |commanded velocity| (the control the
                      earlier validations used)

Everything is aggregated over sliding windows, because lambda_S is only
defined on a segment, not on a single tick.
"""

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from sparc import sparc

HERE = Path(__file__).resolve().parent
DATA = HERE / "recorded trajectories"
RUNS = {"ACT": DATA / "record_run1.csv", "pi0.5": DATA / "record_pi05_run1.csv"}

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY = J[:5]
L = 4                      # achieved-position lag, ticks (~132 ms)
QDOT_MAX = 3.0 * 180 / np.pi   # scorer's placeholder limit, deg/s (~172)
WINDOWS = [(2.0, 1.0), (1.0, 0.5)]


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 6 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def partial(x, y, *z):
    """Partial correlation of x and y controlling for every z, by residuals."""
    m = np.isfinite(x) & np.isfinite(y)
    for zi in z:
        m &= np.isfinite(zi)
    Z = np.column_stack([np.ones(m.sum())] + [zi[m] for zi in z])
    rx = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]
    ry = y[m] - Z @ np.linalg.lstsq(Z, y[m], rcond=None)[0]
    return pearson(rx, ry), int(m.sum())


def prepare(csv):
    """Per-tick command/achieved derivatives plus a uniform commanded speed."""
    import pandas as pd

    df = pd.read_csv(csv)
    t = df["t_obs"].to_numpy()
    t = t - t[0]
    dtc = df["dt"].to_numpy()

    good = np.ones(len(df), bool)
    good[:25] = False            # startup transient, per the earlier validations
    good[dtc > 0.06] = False     # dropped ticks
    delta = float(np.median(dtc[good]))

    cmd = {j: df[f"cmd_{j}"].to_numpy(float) for j in J}
    pos = {j: df[f"pos_{j}"].to_numpy(float) for j in J}
    vcmd = {j: savgol_filter(cmd[j], 9, 4, deriv=1, delta=delta) for j in J}
    jerk = {j: savgol_filter(pos[j], 9, 4, deriv=3, delta=delta) for j in J}

    n = len(df)
    err = {}
    for j in J:
        e = np.full(n, np.nan)
        e[: n - L] = cmd[j][: n - L] - pos[j][L:]
        err[j] = np.abs(e)

    # commanded speed on a uniform grid, for SPARC
    tu = np.arange(0.0, t[-1], delta)
    cu = np.column_stack([np.interp(tu, t, cmd[j]) for j in J])
    speed_cmd = np.linalg.norm(np.gradient(cu, delta, axis=0), axis=1)

    return dict(t=t, good=good, delta=delta, vcmd=vcmd, jerk=jerk, err=err,
                tu=tu, speed_cmd=speed_cmd, fs=1.0 / delta, n=n)


def window_table(d, width, hop):
    """One row per window: lambda_S, both outcomes, both controls."""
    rows = []
    w = int(round(width * d["fs"]))
    h = int(round(hop * d["fs"]))
    for i in range(0, d["speed_cmd"].size - w + 1, h):
        t0, t1 = d["tu"][i], d["tu"][i] + width
        lam = sparc(d["speed_cmd"][i : i + w], d["fs"])

        m = d["good"] & (d["t"] >= t0) & (d["t"] < t1)
        m[len(d["t"]) - L :] = False
        if m.sum() < 8:
            continue

        e = np.nanmean([np.nanmean(d["err"][j][m]) for j in BODY])
        jk = np.nanmean([np.nanmean(np.abs(d["jerk"][j][m])) for j in BODY])
        v = np.nanmean([np.nanmean(np.abs(d["vcmd"][j][m])) for j in BODY])
        s_vel = float(np.sum([np.maximum(0, np.abs(d["vcmd"][j][m]) - QDOT_MAX).sum()
                              for j in J]))
        rows.append((lam, e, jk, v, s_vel))
    return np.array(rows).T   # lam, err, jerk, vcmd, s_vel


prep = {k: prepare(v) for k, v in RUNS.items()}
for k, d in prep.items():
    print(f"{k:6s} delta={d['delta']*1e3:.1f} ms  n_ticks={d['n']}  "
          f"good={d['good'].sum()}")

print(f"\nS_vel uses the scorer's placeholder qdot_max = {QDOT_MAX:.0f} deg/s "
      "(the notes flag this as un-calibrated)")

for width, hop in WINDOWS:
    print(f"\n{'=' * 78}\nwindows {width:.0f} s / {hop:.1f} s hop")
    tables = {k: window_table(d, width, hop) for k, d in prep.items()}
    for k, T in tables.items():
        lam, err, jerk, v, s_vel = T
        print(f"  {k}: n={lam.size}  lambda_S {np.median(lam):+.2f} "
              f"[{np.percentile(lam,25):+.2f},{np.percentile(lam,75):+.2f}]   "
              f"S_vel fires in {100*np.mean(s_vel>0):.0f}% of windows")

    pooled = np.hstack(list(tables.values()))
    for label, T in list(tables.items()) + [("pooled", pooled)]:
        lam, err, jerk, v, s_vel = T
        print(f"\n  --- {label} (n={lam.size}) ---")
        for yname, y in (("|tracking error|", err), ("|measured jerk|", jerk)):
            r = pearson(lam, y)
            pr_v, _ = partial(lam, y, v)
            pr_s, _ = partial(lam, y, s_vel)
            pr_both, nb = partial(lam, y, v, s_vel)
            print(f"    lambda_S vs {yname:17s} raw r {r:+.3f} | "
                  f"partial|vcmd {pr_v:+.3f} | partial|S_vel {pr_s:+.3f} | "
                  f"partial|both {pr_both:+.3f}")

print("\nSign convention: lambda_S is negative-is-rougher, so a term that predicts")
print("bad outcomes should show a NEGATIVE correlation with error/jerk.")
