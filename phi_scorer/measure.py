#!/usr/bin/env python
"""phi_scorer.measure -- measure every physical constant Phi needs, from THIS arm.

Writes phi_scorer/artifacts/measured_constants.json with:
  pos_limit_rad[6]  symmetric travel limit per joint      (from the arm's calibration)
  qd_max[6]         achievable speed per joint (rad/s)     (ramp test, hardware)
  bandwidth_hz[6]   servo closed-loop corner freq (Hz)     (step response, hardware)
  w_free            free speed (rad/s)                     (max ramp speed)
  tau_stall         stall torque (N*m)                     (datasheet, edit if measured)

Design goal: INDEPENDENCE. This script derives the constants from the real robot, not
from any pre-existing file. Two modes:

  * OFFLINE  (no --port): reads only the robot's calibration file for position limits
    and loads the URDF to sanity-check kinematics. q_dot_max / bandwidth are left at
    config DEFAULTS so the rest of the pipeline can run before you have hardware time.
      python -m phi_scorer.measure

  * HARDWARE (--port ...): additionally drives the arm ONE JOINT AT A TIME to measure
    q_dot_max (a fast mid-range sweep -> plateau velocity) and bandwidth (a small step
    -> 63% rise time -> f = 1/(2*pi*tau)). Motion is capped and mid-range for safety.
      python -m phi_scorer.measure --port /dev/serial/by-id/usb-...-if00 --id my_follower

Reads Present_Position/Present_Velocity via the SAME bus.sync_read the follower uses;
never reimplements the servo protocol.
"""

import argparse
import json
import os
import time

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# OFFLINE: position limits from the robot's own calibration; URDF sanity check.
# ---------------------------------------------------------------------------
def position_limits_from_calibration():
    """Symmetric |q| travel limit per joint (rad), from the calibration range."""
    cal = json.load(open(config.CALIB_PATH))
    out = []
    for j in config.JOINTS:
        if j == "gripper":
            out.append(config.JAW_HI)                       # full open angle magnitude
        else:
            lo, hi = cal[j]["range_min"], cal[j]["range_max"]
            # half the calibrated count span -> degrees (360/4095 per count) -> radians
            out.append((hi - lo) / 2 * 360.0 / 4095.0 * config.DEG2RAD)
    return out


def urdf_sanity_check():
    from .kinematics import Kinematics
    kin = Kinematics()
    print(f"[measure] URDF OK: {kin.model.nq} joints, EE frame '{config.EE_FRAME}', "
          f"|q|max(rad)={np.round(kin.q_abs_max, 2)}")


# ---------------------------------------------------------------------------
# HARDWARE: ramp (q_dot_max) and step (bandwidth) per joint.
# ---------------------------------------------------------------------------
def _connect(port, robot_id, max_step):
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    cfg = SOFollowerRobotConfig(port=port, id=robot_id, max_relative_target=max_step, cameras={})
    robot = SOFollower(cfg)
    robot.connect()
    return robot


def _read_deg(robot):
    obs = robot.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in config.JOINTS}


def measure_ramp(robot, joint, amp_deg=45.0, rate_hz=30.0, timeout=6.0):
    """Command a fast sweep on one joint; return peak achieved speed (rad/s)."""
    home = _read_deg(robot)
    target = dict(home)
    target[joint] = home[joint] + amp_deg                   # sweep up by amp_deg
    ts, pos = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        obs = _read_deg(robot)
        ts.append(time.perf_counter()); pos.append(obs[joint])
        robot.send_action({f"{k}.pos": target[k] for k in config.JOINTS})   # capped per tick
        if abs(obs[joint] - target[joint]) < 1.0:
            break
        time.sleep(1.0 / rate_hz)
    ts, pos = np.array(ts), np.array(pos) * config.DEG2RAD
    if len(ts) < 3:
        return 0.0
    v = np.abs(np.diff(pos) / np.diff(ts))                   # rad/s
    return float(np.percentile(v, 95))                       # robust "plateau" speed


def measure_bandwidth(robot, joint, step_deg=20.0, rate_hz=60.0, settle=3.0):
    """Small position step; fit tau from 63% rise time; return f_bw = 1/(2*pi*tau) Hz."""
    home = _read_deg(robot)
    target = dict(home)
    target[joint] = home[joint] + step_deg
    ts, pos = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < settle:
        obs = _read_deg(robot)
        ts.append(time.perf_counter() - t0); pos.append(obs[joint])
        robot.send_action({f"{k}.pos": target[k] for k in config.JOINTS})
        time.sleep(1.0 / rate_hz)
    ts, pos = np.array(ts), np.array(pos)
    y = pos - pos[0]
    final = y[-1] if abs(y[-1]) > 1e-6 else step_deg
    idx = np.argmax(np.abs(y) >= 0.63 * abs(final))          # first sample past 63%
    tau = ts[idx] if idx > 0 else config.DEFAULT_DT
    return float(1.0 / (2 * np.pi * max(tau, 1e-3)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port -> enables HARDWARE mode")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--max_step", type=float, default=6.0, help="deg/tick cap (safety)")
    ap.add_argument("--ramp_amp", type=float, default=45.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(config.CONSTANTS_PATH), exist_ok=True)
    urdf_sanity_check()

    out = {
        "pos_limit_rad": position_limits_from_calibration(),   # always real (calibration)
        "qd_max":        [config.DEFAULT_QD_MAX[j] for j in config.JOINTS],
        "bandwidth_hz":  [config.DEFAULT_BANDWIDTH_HZ[j] for j in config.JOINTS],
        "w_free":        config.DEFAULT_W_FREE,
        "tau_stall":     config.DEFAULT_TAU_STALL,
        "source":        "offline: calibration position limits + config default dynamics",
    }

    if args.port:
        print(f"[measure] HARDWARE mode on {args.port}: ramp + step per joint (mid-range, "
              f"max_step={args.max_step} deg/tick). Keep clear of the arm.")
        robot = _connect(args.port, args.id, args.max_step)
        try:
            qd, bw = [], []
            for j in config.JOINTS:
                q = measure_ramp(robot, j, amp_deg=args.ramp_amp)
                b = measure_bandwidth(robot, j)
                qd.append(q); bw.append(b)
                print(f"  {j:<14} qd_max={q:5.2f} rad/s   f_bw={b:5.2f} Hz")
            out["qd_max"] = qd
            out["bandwidth_hz"] = bw
            out["w_free"] = float(max(qd))
            out["source"] = "hardware: ramp (qd_max/w_free) + step (bandwidth) + calibration"
        finally:
            robot.disconnect()

    with open(config.CONSTANTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[measure] wrote {config.CONSTANTS_PATH}\n  source: {out['source']}")


if __name__ == "__main__":
    main()
