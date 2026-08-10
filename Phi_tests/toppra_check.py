"""TOPP-RA feasibility check driven by the MEASURED ramp-test limits.

Takes the geometric path a policy actually commanded, re-times it optimally
subject to the per-joint velocity/acceleration limits measured by ramp_test.py,
and reports the headroom: how much faster the same path could legally be
executed, and whether the original execution ever violated a limit.

Limits are parsed from pvd_logs/ramp_<joint>.csv, not hardcoded, so they cannot
drift away from the measurements. Both are CONSERVATIVE per joint:
  qd_max  = min sustained plateau over the two directions
  qdd_max = min of accel and decel over the two directions
Gravity makes shoulder_lift markedly asymmetric, so taking the min is the only
direction-agnostic reading that is safe in both.
"""

import csv
import os
import sys

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramp_test import analyse  # noqa: E402

ta.setup_logging("ERROR")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEG2RAD = np.pi / 180.0
GRIP2RAD = (1.74533 - (-0.174533)) / 100.0
SCALE = np.array([DEG2RAD] * 5 + [GRIP2RAD])

REPO = os.path.expanduser("~/Desktop/PVD")
RAMP = os.path.join(REPO, "pvd_logs", "ramp_{}.csv")
RUNS = {"smolvla": os.path.join(REPO, "Phi_tests/recorded trajectories/record_run1.csv"),
        "pi0.5": os.path.join(REPO, "Phi_tests/recorded trajectories/record_pi05_run1.csv")}


def measured_limits():
    """Per-joint (qd_max, qdd_max) from the ramp logs; conservative over direction."""
    qd, qdd, detail = [], [], {}
    for j in JOINTS:
        rows = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                for r in list(csv.reader(open(RAMP.format(j))))[1:]]
        analyse(rows)
        d = analyse.data
        v = min(d[k]["plateau"] for k in d)
        a = min(min(d[k]["accel"], d[k]["decel"]) for k in d)
        qd.append(v)
        qdd.append(a)
        detail[j] = dict(qd=v, qdd=a,
                         peak=max(d[k]["peak"] for k in d),
                         agree=min(d[k]["agree"] for k in d))
    return np.array(qd), np.array(qdd), detail


def load_path(csv_path, prefix="cmd"):
    """Commanded joint path in URDF radians, plus the wall-clock duration."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    q = df[[f"{prefix}_{j}" for j in JOINTS]].to_numpy(float) * SCALE
    t = df["t_obs"].to_numpy(float)
    keep = np.ones(len(q), bool)
    keep[:25] = False                       # startup transient
    q, t = q[keep], t[keep] - t[keep][0]
    # drop duplicate waypoints; a zero-length segment breaks the spline
    d = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-9]
    return q[d], t[d], t[-1]


qd_max, qdd_max, detail = measured_limits()

print("MEASURED LIMITS (from ramp_test.py logs, conservative over direction)")
print(f"{'joint':<15}{'qd_max':>9}{'qdd_max':>10}{'peak seen':>11}{'agreement':>11}")
for i, j in enumerate(JOINTS):
    d = detail[j]
    print(f"{j:<15}{qd_max[i]:>9.2f}{qdd_max[i]:>10.1f}{d['peak']:>11.2f}"
          f"{100*(1-d['agree']):>10.0f}%")
print(f"{'':<15}{'rad/s':>9}{'rad/s^2':>10}{'rad/s':>11}")

print("\nTOPP-RA RE-TIMING")
print("  cmd_ = the path the policy ASKED for; pos_ = the path the arm ACTUALLY took")
print(f"\n{'run':<10}{'path':>6}{'waypts':>8}{'actual s':>10}{'toppra s':>10}"
      f"{'speedup':>9}{'worst joint':>13}   over limit?")
for (name, path), prefix in [(kv, p) for kv in RUNS.items() for p in ("cmd", "pos")]:
    q, t, dur = load_path(path, prefix)
    ss = np.linspace(0, 1, len(q))
    p = ta.SplineInterpolator(ss, q)

    pc_vel = constraint.JointVelocityConstraint(np.vstack([-qd_max, qd_max]).T)
    pc_acc = constraint.JointAccelerationConstraint(
        qdd_max, discretization_scheme=constraint.DiscretizationType.Interpolation)

    inst = algo.TOPPRA([pc_vel, pc_acc], p, solver_wrapper="seidel")
    traj = inst.compute_trajectory(0, 0)
    if traj is None:
        print(f"{name:<10}{prefix:>6}   TOPP-RA found no feasible parameterisation")
        continue

    # what the ORIGINAL execution actually demanded
    dt = np.median(np.diff(t))
    v_actual = np.abs(np.gradient(q, dt, axis=0)).max(axis=0)
    over = np.where(v_actual > qd_max)[0]
    worst = int(np.argmax(v_actual / qd_max))
    verdict = ("YES: " + ", ".join(JOINTS[i] for i in over)) if len(over) else "no"

    print(f"{name:<10}{prefix:>6}{len(q):>8}{dur:>10.2f}{traj.duration:>10.2f}"
          f"{dur/traj.duration:>8.2f}x{JOINTS[worst]:>13}"
          f"   {np.max(v_actual/qd_max):>4.0%} of limit -> {verdict}")

print("\nspeedup = how much faster the SAME geometric path could legally run at the")
print("measured limits. 'max |qd| used' is the worst joint's fraction of its limit")
print("during the real execution.")
