"""Execute a TOPP-RA re-timed policy trajectory on the real arm, and log it.

Takes the path a policy COMMANDED, re-times it to be feasible under the
ramp-test limits, and plays it back so the achieved motion can be compared with
the original rollout.

THE HAZARD THIS SCRIPT EXISTS TO HANDLE
  The original rollout was protected by its own infeasibility: smolvla commanded
  positions outside the calibrated range (shoulder_lift -104.7 deg vs a -100.5
  floor), but the arm could not move fast enough to reach them, so the servo
  never got there. TOPP-RA slows the path down until the arm CAN track it --
  which makes those same out-of-range commands reachable. Re-timing therefore
  turns a harmless unreachable command into a real collision with the mechanical
  stop. Every sample is clipped to the calibrated band BEFORE re-timing, and
  asserted in-band after.

RUN:
  python execute_toppra.py --dry-run
  python execute_toppra.py
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramp_test import analyse  # noqa: E402

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ta.setup_logging("ERROR")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533793-if00"
CALIB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")
REPO = os.path.expanduser("~/Desktop/PVD")
RAMP = os.path.join(REPO, "pvd_logs", "ramp_{}.csv")
MAX_RES = 4095.0


def measured_limits():
    qd, qdd = [], []
    for j in JOINTS:
        rows = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                for r in list(csv.reader(open(RAMP.format(j))))[1:]]
        analyse(rows)
        d = analyse.data
        qd.append(min(d[k]["plateau"] for k in d))
        qdd.append(min(min(d[k]["accel"], d[k]["decel"]) for k in d))
    return np.array(qd), np.array(qdd)


def deg_to_raw(cal, j, v):
    lo, hi = cal[j]["range_min"], cal[j]["range_max"]
    mid = (lo + hi) / 2
    if j == "gripper":
        return v / 100.0 * (hi - lo) + lo
    return v * MAX_RES / 360.0 + mid


def safe_band(cal, j, keepout):
    """Calibrated travel limits expressed in the same units as cmd_, minus keep-out."""
    lo, hi = cal[j]["range_min"], cal[j]["range_max"]
    mid = (lo + hi) / 2
    if j == "gripper":
        return keepout, 100.0 - keepout      # scale with keepout, not hardcoded
    k = keepout * MAX_RES / 360.0
    return ((lo + k) - mid) * 360 / MAX_RES, ((hi - k) - mid) * 360 / MAX_RES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(
        REPO, "Phi_tests/recorded trajectories/record_run1.csv"))
    ap.add_argument("--prefix", default="cmd", choices=["cmd", "sent", "pos"])
    ap.add_argument("--no-retime", action="store_true",
                    help="replay the source at its RECORDED timestamps, no TOPP-RA")
    ap.add_argument("--keepout", type=float, default=5.0)
    ap.add_argument("--spline-margin", type=float, default=2.0,
                    help="extra deg shaved off the band when clipping waypoints, to\n                          absorb spline overshoot")
    ap.add_argument("--rate", type=float, default=100.0, help="control/log Hz")
    ap.add_argument("--approach-rate", type=float, default=15.0, help="deg/s to the start pose")
    ap.add_argument("--max-load", type=int, default=700)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 = TOPP-RA optimal; <1 is slower still")
    ap.add_argument("--out", default=os.path.join(REPO, "pvd_logs/exec_toppra.csv"))
    ap.add_argument("--skip", type=int, default=25,
                    help="leading frames to drop (30Hz rollouts have a startup "
                         "transient; a dataset episode does not -- use 0)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cal = json.load(open(CALIB))
    qd_max, qdd_max = measured_limits()
    df = pd.read_csv(args.src)
    q_deg = df[[f"{args.prefix}_{j}" for j in JOINTS]].to_numpy(float)[args.skip:]

    # ---- clip into the calibrated band BEFORE re-timing --------------------
    print(f"{'joint':<15}{'raw min':>9}{'raw max':>9}{'band lo':>9}{'band hi':>9}"
          f"{'clipped':>9}")
    clipped_total = 0
    # Waypoints are clipped to a SHRUNK band: the spline through them overshoots
    # the waypoint envelope, so clipping only to the true band still lets the
    # interpolated trajectory leave it (this fired on elbow_flex).
    for i, j in enumerate(JOINTS):
        lo, hi = safe_band(cal, j, args.keepout)
        n = int(((q_deg[:, i] < lo) | (q_deg[:, i] > hi)).sum())
        clipped_total += n
        print(f"{j:<15}{q_deg[:,i].min():>9.1f}{q_deg[:,i].max():>9.1f}"
              f"{lo:>9.1f}{hi:>9.1f}{n:>9d}")
        q_deg[:, i] = np.clip(q_deg[:, i], lo + args.spline_margin,
                              hi - args.spline_margin)
    print(f"clipped {clipped_total} of {q_deg.size} samples "
          f"({100*clipped_total/q_deg.size:.2f}%)")

    scale = np.array([np.pi / 180] * 5 + [(1.74533 + 0.174533) / 100])
    t_orig_all = df["t_obs"].to_numpy(float)[args.skip:]
    t_orig_all = t_orig_all - t_orig_all[0]

    # ---- plain replay: recorded setpoints at their recorded timestamps -----
    if args.no_retime:
        q_traj_deg = q_deg
        t_grid = t_orig_all
        print(f"\nPLAIN REPLAY of {args.prefix}_ at recorded timing: "
              f"{len(t_grid)} setpoints over {t_grid[-1]:.2f} s "
              f"(median dt {1000*np.median(np.diff(t_grid)):.1f} ms) -- no re-timing")
        for i, j in enumerate(JOINTS):
            lo, hi = safe_band(cal, j, args.keepout)
            assert q_traj_deg[:, i].min() >= lo - 1e-6 and q_traj_deg[:, i].max() <= hi + 1e-6, \
                f"{j} out of band"
        print("all replay samples verified inside the calibrated band")
        raw = np.column_stack([[deg_to_raw(cal, j, v) for v in q_traj_deg[:, i]]
                               for i, j in enumerate(JOINTS)]).astype(int)
        return run_on_arm(args, cal, raw, t_grid)

    # ---- re-time ----------------------------------------------------------
    q_rad = q_deg * scale
    keep = np.r_[True, np.linalg.norm(np.diff(q_rad, axis=0), axis=1) > 1e-9]
    q_rad, q_deg = q_rad[keep], q_deg[keep]

    path = ta.SplineInterpolator(np.linspace(0, 1, len(q_rad)), q_rad)
    inst = algo.TOPPRA(
        [constraint.JointVelocityConstraint(np.vstack([-qd_max, qd_max]).T),
         constraint.JointAccelerationConstraint(
             qdd_max, discretization_scheme=constraint.DiscretizationType.Interpolation)],
        path, solver_wrapper="seidel")
    traj = inst.compute_trajectory(0, 0)
    if traj is None:
        print("TOPP-RA: no feasible parameterisation")
        return 1

    dur = traj.duration / max(args.speed, 1e-6)
    t_grid = np.arange(0, dur, 1.0 / args.rate)
    q_traj_rad = traj(t_grid * args.speed)
    q_traj_deg = q_traj_rad / scale

    t_orig = df["t_obs"].to_numpy(float)[25:]
    print(f"\noriginal duration {t_orig[-1]-t_orig[0]:.2f} s -> TOPP-RA {dur:.2f} s "
          f"({dur/(t_orig[-1]-t_orig[0]):.2f}x)   {len(t_grid)} setpoints @ {args.rate} Hz")

    # measure the spline's overshoot past the waypoints, then hard-clip and assert
    over = []
    for i, j in enumerate(JOINTS):
        lo, hi = safe_band(cal, j, args.keepout)
        o = max(lo - q_traj_deg[:, i].min(), q_traj_deg[:, i].max() - hi, 0.0)
        if o > 0:
            over.append(f"{j} {o:.2f}deg")
        q_traj_deg[:, i] = np.clip(q_traj_deg[:, i], lo, hi)
    print("spline overshoot past band: " + (", ".join(over) if over else "none")
          + " -> hard-clipped")
    for i, j in enumerate(JOINTS):
        lo, hi = safe_band(cal, j, args.keepout)
        assert q_traj_deg[:, i].min() >= lo - 1e-6 and q_traj_deg[:, i].max() <= hi + 1e-6, \
            f"{j} out of band after re-timing"
    print("all re-timed samples verified inside the calibrated band")

    raw = np.column_stack([[deg_to_raw(cal, j, v) for v in q_traj_deg[:, i]]
                           for i, j in enumerate(JOINTS)]).astype(int)
    return run_on_arm(args, cal, raw, t_grid)


def run_on_arm(args, cal, raw, t_grid):
    """Approach the start pose slowly, then play `raw` setpoints on `t_grid`."""
    dur = t_grid[-1]
    if args.dry_run:
        print("\n--dry-run: nothing moved.")
        return 0

    # ---- execute ----------------------------------------------------------
    bus = FeetechMotorsBus(port=PORT, motors={
        n: Motor(i + 1, "sts3215",
                 MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.RANGE_M100_100)
        for i, n in enumerate(JOINTS)})
    bus.connect(handshake=True)
    try:
        cur = {n: int(bus.read("Present_Position", n, normalize=False)) for n in JOINTS}
        for n in JOINTS:
            bus.write("Goal_Position", n, cur[n], normalize=False)
        for n in JOINTS:
            bus.enable_torque(n)
        time.sleep(0.3)

        # slow approach to the trajectory start
        tgt = {j: int(raw[0, i]) for i, j in enumerate(JOINTS)}
        steps = max(1, int(max(abs(tgt[j] - cur[j]) for j in JOINTS) * 360 / 4096
                           / args.approach_rate * 50))
        print(f"approaching start pose over ~{steps/50:.1f}s")
        for k in range(steps):
            for j in JOINTS:
                bus.write("Goal_Position", j,
                          int(round(cur[j] + (tgt[j] - cur[j]) * (k + 1) / steps)),
                          normalize=False)
            time.sleep(0.02)
        time.sleep(0.8)

        print(f"executing {dur:.1f} s trajectory...")
        rows, t0, aborted = [], time.perf_counter(), False
        for k in range(len(t_grid)):
            bus.sync_write("Goal_Position", {j: int(raw[k, i]) for i, j in enumerate(JOINTS)},
                           normalize=False)
            pos = bus.sync_read("Present_Position", JOINTS, normalize=False)
            rows.append([time.perf_counter() - t0] + [int(raw[k, i]) for i in range(6)]
                        + [int(pos[j]) for j in JOINTS])
            if k % 200 == 0:
                ld = {j: int(bus.read("Present_Load", j, normalize=False)) for j in JOINTS}
                if max(abs(v) for v in ld.values()) > args.max_load:
                    print(f"  ABORT at t={rows[-1][0]:.1f}s: load {ld}")
                    aborted = True
                    break
            while (time.perf_counter() - t0) < t_grid[k]:
                time.sleep(0.0005)

        for j in JOINTS:
            bus.write("Goal_Position", j,
                      int(bus.read("Present_Position", j, normalize=False)), normalize=False)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"goal_{j}" for j in JOINTS] + [f"pos_{j}" for j in JOINTS])
            w.writerows(rows)
        print(f"{'ABORTED, ' if aborted else ''}wrote {len(rows)} samples -> {args.out}")
        return 0
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
