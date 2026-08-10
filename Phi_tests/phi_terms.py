"""S_dither and S_cont on the PROPOSED actions, plus a full Phi over both policies.

S_dither (R)
    d1 = a[i+1] - a[i]          per-tick motion        (49 per 50-action chunk)
    d2 = d1[i+1] - d1[i]        change in that motion  (48)
    R  = mean|d2| / mean|d1|
  Computed PER JOINT then averaged across joints. R is scale-invariant per joint,
  so degrees vs radians vs gripper-percent cannot change it -- which is the whole
  point: no limit constant to get wrong. R > 1 means the command reverses faster
  than it advances.

S_cont (seam)
    gap  = |a_1(new chunk) - a_prev(last action of the previous chunk)|
    step = mean|d1| within the new chunk
    S_cont = gap / step        -- "the seam, in units of a typical step"
  Also per joint then averaged. One number per chunk boundary, not an average
  over the chunk.

S_vel / S_pos use the MEASURED limits from ramp_test.py (velocity) and the
calibrated travel range (position), as relu fractions:
    max(0, (|x| - lim) / lim)

AGGREGATION NOTE: score_trajectories.py SUMS its relu fractions over the chunk,
which makes a term scale with chunk length. Here every term is a per-chunk MEAN
so that all four are per-tick quantities and therefore commensurate when added.
Sums are printed alongside for comparison with the existing scorer.
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramp_test import analyse  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REPO = os.path.expanduser("~/Desktop/PVD")
CALIB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")
RAMP = os.path.join(REPO, "pvd_logs", "ramp_{}.csv")
D2R = np.pi / 180
GRIP = (1.74533 + 0.174533) / 100
SCALE = np.array([D2R] * 5 + [GRIP])          # LeRobot units -> URDF radians

RUNS = [
    ("smolvla r1", os.path.join(REPO, "Phi_tests/recorded trajectories/record_run1.csv")),
    ("smolvla r2", os.path.join(REPO, "pvd_logs/record_run2.csv")),
    ("pi0.5   r1", os.path.join(REPO, "Phi_tests/recorded trajectories/record_pi05_run1.csv")),
]
WEIGHTS = {"vel": 1.0, "pos": 1.0, "dither": 1.0, "cont": 1.0}


def measured_qd_max():
    qd = []
    for j in JOINTS:
        rows = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                for r in list(csv.reader(open(RAMP.format(j))))[1:]]
        analyse(rows)
        qd.append(min(analyse.data[k]["plateau"] for k in analyse.data))
    return np.array(qd)


def position_limits(cal):
    """Symmetric |q|max in URDF radians, from the calibrated travel range."""
    out = []
    for j in JOINTS:
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]
        if j == "gripper":
            out.append(100.0 * GRIP)
        else:
            out.append((hi - lo) / 2 * 360 / 4095 * D2R)
    return np.array(out)


def chunk_bounds(chunk_start):
    """Real chunk starts. The recorder flags the boundary tick AND the next one,
    so consecutive flags are collapsed to the first of each run."""
    idx = np.where(chunk_start == 1)[0]
    return [i for k, i in enumerate(idx) if k == 0 or i - idx[k - 1] > 1]


def relu_frac(x, lim):
    return np.maximum(0.0, (np.abs(x) - lim) / lim)


def score_run(path, qd_max, q_max):
    df = pd.read_csv(path)
    a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float)     # PROPOSED actions
    a_rad = a * SCALE
    dt = float(np.median(df["dt"].to_numpy(float)))
    starts = chunk_bounds(df["chunk_start"].to_numpy())

    per_chunk = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(df)
        blk = a_rad[s:e]
        if len(blk) < 4:
            continue
        d1 = np.diff(blk, axis=0)
        d2 = np.diff(d1, axis=0)

        m1 = np.abs(d1).mean(axis=0)                          # per joint
        m2 = np.abs(d2).mean(axis=0)
        ok = m1 > 1e-12
        R = float(np.mean(m2[ok] / m1[ok])) if ok.any() else np.nan

        # seam against the previous chunk's last proposed action
        if s > 0:
            gap = np.abs(blk[0] - a_rad[s - 1])
            cont = float(np.mean(gap[ok] / m1[ok])) if ok.any() else np.nan
        else:
            cont = np.nan

        v = d1 / dt
        s_vel = relu_frac(v, qd_max).mean()
        s_pos = relu_frac(blk, q_max).mean()
        per_chunk.append(dict(n=len(blk), R=R, cont=cont, s_vel=s_vel, s_pos=s_pos,
                              s_vel_sum=relu_frac(v, qd_max).sum(),
                              s_pos_sum=relu_frac(blk, q_max).sum()))
    return pd.DataFrame(per_chunk)


cal = json.load(open(CALIB))
qd_max = measured_qd_max()
q_max = position_limits(cal)

print("LIMITS USED")
print(f"{'joint':<15}{'qdot_max (rad/s)':>18}{'|q|max (rad)':>15}")
for i, j in enumerate(JOINTS):
    print(f"{j:<15}{qd_max[i]:>18.2f}{q_max[i]:>15.2f}")
print("  qdot_max: measured, ramp_test.py (sustained plateau, min over direction)")
print("  |q|max  : calibrated travel half-span")

tables = {}
print("\n\nPER-RUN TERMS ON THE PROPOSED ACTIONS (cmd_), per-chunk mean +- sd")
print(f"{'run':<12}{'chunks':>7}{'R (dither)':>22}{'S_cont (seam)':>20}"
      f"{'S_vel':>10}{'S_pos':>8}")
for lab, path in RUNS:
    if not os.path.exists(path):
        print(f"{lab:<12}  MISSING {path}")
        continue
    t = score_run(path, qd_max, q_max)
    tables[lab] = t
    print(f"{lab:<12}{len(t):>7}"
          f"{t.R.mean():>13.3f} +-{t.R.std():<6.3f}"
          f"{t.cont.mean():>12.3f} +-{t.cont.std():<6.3f}"
          f"{t.s_vel.mean():>10.4f}{t.s_pos.mean():>8.4f}")

print(f"\n{'run':<12}{'R min':>9}{'R max':>9}   {'cont min':>9}{'cont max':>10}")
for lab, t in tables.items():
    print(f"{lab:<12}{t.R.min():>9.3f}{t.R.max():>9.3f}   "
          f"{t.cont.min():>9.3f}{t.cont.max():>10.3f}")

print("\n\nFULL PHI = w_vel*S_vel + w_pos*S_pos + w_dither*R + w_cont*S_cont")
print("  (all weights 1.0; every term a per-chunk mean)")
print(f"\n{'run':<12}{'S_vel':>9}{'S_pos':>9}{'R':>9}{'S_cont':>9}{'PHI':>10}"
      f"{'  dominant term':>18}")
for lab, t in tables.items():
    terms = {"S_vel": WEIGHTS["vel"] * t.s_vel.mean(),
             "S_pos": WEIGHTS["pos"] * t.s_pos.mean(),
             "R": WEIGHTS["dither"] * t.R.mean(),
             "S_cont": WEIGHTS["cont"] * t.cont.mean()}
    phi = sum(terms.values())
    dom = max(terms, key=lambda k: terms[k])
    print(f"{lab:<12}{terms['S_vel']:>9.4f}{terms['S_pos']:>9.4f}{terms['R']:>9.3f}"
          f"{terms['S_cont']:>9.3f}{phi:>10.3f}{dom:>18}  "
          f"({100*terms[dom]/phi:.0f}% of Phi)")

print("\nSeparation check (does the term tell the two policies apart?)")
sm = pd.concat([tables[k] for k in tables if k.startswith("smolvla")])
pi = pd.concat([tables[k] for k in tables if k.startswith("pi0.5")])
for name, col in [("R (dither)", "R"), ("S_cont", "cont"), ("S_vel", "s_vel"), ("S_pos", "s_pos")]:
    a, b = sm[col].dropna(), pi[col].dropna()
    overlap = not (a.min() > b.max() or b.min() > a.max())
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(a, b, alternative="two-sided")
    print(f"  {name:<12} smolvla {a.mean():7.3f} [{a.min():.2f},{a.max():.2f}]   "
          f"pi0.5 {b.mean():7.3f} [{b.min():.2f},{b.max():.2f}]   "
          f"AUC {u.statistic/(len(a)*len(b)):.3f}  p {u.pvalue:.2g}  "
          f"{'RANGES OVERLAP' if overlap else 'DISJOINT'}")
