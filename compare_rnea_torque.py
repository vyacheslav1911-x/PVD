#!/usr/bin/env python
"""Validate RNEA torque against the MEASURED servo torque proxy from a rollout log.

Input : a dynamics CSV from log_dynamics_rollout.py (per-step pos / load / current).
Method:
  1. positions (deg / gripper %) -> URDF radians   (reuse so_arm_viz/common.py)
  2. q_dot, q_ddot via Savitzky-Golay derivatives    (smooth; raw double-diff is noisy)
  3. tau_RNEA = pin.rnea(model, data, q, q_dot, q_ddot)   [N*m, per joint]
     (reuses the SAME Pinocchio model as score_trajectories.py)
  4. Compare tau_RNEA to the measured proxy per joint. The servo has no N*m sensor,
     so we AUTO-FIT a scale by least squares and report correlation:
        signed:     tau_RNEA[:,j]  ~= a*Present_Load[:,j]   + b     (Load is signed)
        magnitude: |tau_RNEA[:,j]| ~= c*Present_Current[:,j] + d     (Current is unsigned)
     A high |r| means the RNEA torque SHAPE matches the real arm -> RNEA is correct
     (up to the per-joint torque constant a/c, which is exactly the calibration).

Outputs: printed per-joint r + fitted scale, rnea_vs_measured.csv, rnea_vs_measured.png

RUN: /home/g/miniconda3/envs/lerobot_v6/bin/python compare_rnea_torque.py <dynamics.csv>
"""

import argparse
import csv
import os
import sys

import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import score_trajectories as SCORE   # reuse load_common / load_model_and_limits


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[compare] empty log: {path}")
    joints = [c[4:] for c in rows[0] if c.startswith("pos_")]
    t = np.array([float(r["wall_time"]) for r in rows])

    def col(prefix, j, default=np.nan):
        out = []
        for r in rows:
            v = r.get(f"{prefix}_{j}", "")
            out.append(float(v) if v not in ("", None) else default)
        return np.array(out)

    pos_deg  = np.stack([col("pos", j) for j in joints], axis=1)        # [N,6] deg/%
    load     = np.stack([col("load", j) for j in joints], axis=1)       # [N,6] signed raw
    current  = np.stack([col("current", j) for j in joints], axis=1)    # [N,6] unsigned raw
    return joints, t, pos_deg, load, current


def positions_to_rad(common, joints, pos_deg):
    """[N,6] lerobot deg/% (log column order) -> [N,6] URDF radians (Pinocchio order)."""
    # reorder log columns into common.LEROBOT_JOINT_ORDER, then convert row by row
    idx = [joints.index(j) for j in common.LEROBOT_JOINT_ORDER]
    q = np.empty((pos_deg.shape[0], 6))
    for n in range(pos_deg.shape[0]):
        row = pos_deg[n, idx]
        obs = {f"{lr}.pos": row[i] for i, lr in enumerate(common.LEROBOT_JOINT_ORDER)}
        _, q_rad = common.lerobot_obs_to_urdf(obs)
        q[n] = q_rad
    return q, idx   # idx also reorders load/current into the same joint order


def savgol_derivs(q, dt, window, poly=3):
    """Return q_dot, q_ddot [N,6] via Savitzky-Golay (clean derivatives)."""
    N = q.shape[0]
    w = min(window, N if N % 2 == 1 else N - 1)
    if w < poly + 2:                      # too few samples -> plain finite diff
        qd = np.gradient(q, dt, axis=0)
        qdd = np.gradient(qd, dt, axis=0)
        return qd, qdd
    if w % 2 == 0:
        w -= 1
    qd = savgol_filter(q, w, poly, deriv=1, delta=dt, axis=0)
    qdd = savgol_filter(q, w, poly, deriv=2, delta=dt, axis=0)
    return qd, qdd


def lstsq_fit(x, y):
    """y ~= a*x + b. Returns a, b, pearson r (nan-safe)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return np.nan, np.nan, np.nan
    A = np.vstack([x[m], np.ones(m.sum())]).T
    (a, b), *_ = np.linalg.lstsq(A, y[m], rcond=None)
    r = np.corrcoef(x[m], y[m])[0, 1]
    return a, b, r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="dynamics CSV from log_dynamics_rollout.py")
    ap.add_argument("--window", type=int, default=11, help="Savitzky-Golay window (odd)")
    ap.add_argument("--out_csv", default="rnea_vs_measured.csv")
    ap.add_argument("--out_png", default="rnea_vs_measured.png")
    args = ap.parse_args()

    common = SCORE.load_common()
    model, data, _, joint_names = SCORE.load_model_and_limits()

    joints, t, pos_deg, load, current = load_log(args.log)
    N = pos_deg.shape[0]
    dt = float(np.median(np.diff(t))) if N > 1 else SCORE.DT
    print(f"[compare] {N} steps, median dt={dt*1e3:.1f} ms ({1/dt:.1f} Hz), joints={joints}")

    q, idx = positions_to_rad(common, joints, pos_deg)
    load, current = load[:, idx], current[:, idx]          # align to Pinocchio joint order
    qd, qdd = savgol_derivs(q, dt, args.window)

    tau = np.array([pin.rnea(model, data, q[n], qd[n], qdd[n]) for n in range(N)])  # [N,6] N*m

    # ---- per-joint fit + correlation --------------------------------------
    print(f"\n{'joint':<13}{'r(load)':>9}{'a[Nm/unit]':>12}{'r(|cur|)':>10}{'c[Nm/unit]':>12}")
    stats = []
    for j, name in enumerate(joint_names):
        a_l, b_l, r_l = lstsq_fit(load[:, j], tau[:, j])               # signed vs signed
        a_c, b_c, r_c = lstsq_fit(current[:, j], np.abs(tau[:, j]))    # magnitude vs unsigned
        stats.append(dict(joint=name, r_load=r_l, a_load=a_l, b_load=b_l,
                          r_cur=r_c, a_cur=a_c, b_cur=b_c))
        print(f"{name:<13}{r_l:>9.3f}{a_l:>12.4g}{r_c:>10.3f}{a_c:>12.4g}")

    finite = [s["r_load"] for s in stats[:5] if np.isfinite(s["r_load"])]
    print(f"\n[compare] mean |r(load)| over 5 body joints = "
          f"{np.mean(np.abs(finite)):.3f}  (closer to 1.0 = RNEA shape matches the real arm)")

    # ---- CSV --------------------------------------------------------------
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_time"]
                   + [f"tau_rnea_{n}" for n in joint_names]
                   + [f"load_{n}" for n in joint_names]
                   + [f"current_{n}" for n in joint_names])
        for n in range(N):
            w.writerow([n, f"{t[n]:.4f}"]
                       + [f"{tau[n, j]:.5f}" for j in range(6)]
                       + [f"{load[n, j]:.1f}" for j in range(6)]
                       + [f"{current[n, j]:.1f}" for j in range(6)])
    print(f"[compare] wrote {args.out_csv}")

    # ---- plot: per joint, tau_RNEA vs fitted-measured (same N*m axis) ------
    INK, MUTED, GRID = "#111827", "#6b7280", "#eceff3"
    C_RNEA, C_MEAS = "#2563eb", "#f59e0b"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    tt = t - t[0]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), dpi=140)
    for j, (ax, name) in enumerate(zip(axes.ravel(), joint_names)):
        s = stats[j]
        ax.plot(tt, tau[:, j], color=C_RNEA, lw=1.8, label="τ RNEA", zorder=3)
        if np.isfinite(s["a_load"]):
            meas_nm = s["a_load"] * load[:, j] + s["b_load"]           # measured, scaled to N*m
            ax.plot(tt, meas_nm, color=C_MEAS, lw=1.6, alpha=0.9,
                    label="measured load (fit)", zorder=2)
        ax.set_title(f"{name}   r={s['r_load']:.2f}", color=INK, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8.5)
        if j >= 3:
            ax.set_xlabel("time (s)", fontsize=9.5)
        if j % 3 == 0:
            ax.set_ylabel("torque (N·m)", fontsize=9.5)

    fig.suptitle("RNEA torque vs measured servo load (auto-scaled) — per joint",
                 color=INK, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[compare] wrote {args.out_png}")


if __name__ == "__main__":
    main()
