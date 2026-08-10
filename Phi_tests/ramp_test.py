"""Single-joint ramp test: drive one joint to saturation and log at ~1.5 kHz.

Measures what the earlier log analysis could not: the velocity plateau and the
acceleration ramp of one joint, sampled fast enough that the estimate does not
depend on a smoothing choice.

WHY THIS WORKS WHERE THE 30 Hz ROLLOUT LOGS DID NOT
  * ~1.5 kHz sampling (measured: 3388 Hz position-only, 1694 Hz position+velocity
    for one joint) instead of 30 Hz, so a 60 deg move is ~500 samples not ~15.
  * Two INDEPENDENT velocity signals: differentiated Present_Position and the
    servo's own Present_Velocity register. If they agree, the estimate is real;
    if they disagree, it is filter bandwidth. The rollout logs had only one.
  * A commanded step large enough to saturate, so the plateau is the joint's
    limit rather than whatever the policy happened to ask for.

WHAT IT ACTUALLY MEASURES
  The limit AS CONFIGURED, not the motor's datasheet capability. The STS3215
  runs an internal trapezoidal profile: Acceleration=0 with
  Maximum_Acceleration=254, P=16 / I=0 / D=32, Max_Torque_Limit=1000 on the body
  joints (500 on the gripper). That configuration is what executes every rollout,
  so it is the correct limit for Phi -- but it is a property of the stack, and a
  limit measured here is void if those registers change.

SAFETY
  * Works in RAW TICKS, so it needs no calibration file and cannot be thrown off
    by a bad homing offset.
  * Goal_Position is set to the CURRENT position BEFORE torque is enabled, so
    the arm cannot snap to a stale goal.
  * Torque is enabled on ALL joints and every joint except the one under test is
    commanded to hold, so the arm stays rigid instead of flopping.
  * Targets are clamped to the servo's own Min/Max_Position_Limit and to
    --excursion from the start pose.
  * A watchdog aborts (freezes the joint at its current position) if it travels
    past the clamp plus margin.
  * --dry-run prints the whole plan and exits without energizing anything.

RUN (conda lerobot_v6):
  python ramp_test.py --joint wrist_roll --excursion 30 --dry-run
  python ramp_test.py --joint wrist_roll --excursion 30
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT_DEFAULT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533793-if00"
CALIB_DEFAULT = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")


def load_calib(path):
    """Per-joint calibrated mechanical range. The ONLY trustworthy source of
    where the joint physically stops -- see the clamp comment in main()."""
    return json.load(open(path))
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

TICKS_PER_REV = 4096
TICK2RAD = 2.0 * np.pi / TICKS_PER_REV
TICK2DEG = 360.0 / TICKS_PER_REV
DEG2TICK = TICKS_PER_REV / 360.0


FIT_WINDOW_S = 0.015     # linear-fit baseline for velocity, seconds


def analyse(rows, fit_window=FIT_WINDOW_S):
    """Velocity/acceleration from a ramp log, quantisation-safe.

    Differentiating raw ticks sample-to-sample is INVALID here: one encoder tick
    is 2*pi/4096 rad, so at ~1.7 kHz a single-tick jitter reads as ~2.6 rad/s and
    ~80% of samples show no tick change at all. Velocity is therefore the slope
    of a linear fit over a `fit_window` baseline (~25 samples), which averages
    the quantisation out instead of amplifying it.

    Acceleration comes from the 10->90% rise of that velocity profile, not from
    a second derivative, for the same reason.

    Returns a list of printable lines; the per-ramp dicts are in `.data`.
    """
    a = np.array([r[1:] for r in rows], float)
    tags = np.array([r[0] for r in rows])
    out, data = [], {}

    out.append(f"\n{'ramp':>5}{'n':>7}{'Hz':>7}{'qdot pk':>9}{'qdot reg':>10}"
               f"{'agree':>8}{'plateau':>9}{'accel':>8}{'decel':>8}")
    out.append(f"{'':>5}{'':>7}{'':>7}{'rad/s':>9}{'rad/s':>10}{'':>8}{'rad/s':>9}"
               f"{'rad/s2':>8}{'rad/s2':>8}")
    for tag in ("pos", "neg"):
        m = tags == tag
        if m.sum() < 50:
            continue
        t, p, v = a[m, 0], a[m, 1] * TICK2RAD, np.abs(a[m, 2]) * TICK2RAD
        dt = np.median(np.diff(t))
        rate = len(t) / (t[-1] - t[0])
        w = max(5, int(round(fit_window / dt)))

        tv = np.array([t[i:i + w].mean() for i in range(len(t) - w)])
        vel = np.abs([np.polyfit(t[i:i + w], p[i:i + w], 1)[0] for i in range(len(t) - w)])
        peak = float(vel.max())
        reg_peak = float(v.max())
        agree = abs(peak - reg_peak) / max(peak, 1e-9)
        plateau = float(np.median(vel[vel > 0.5 * peak]))

        def edge(sig, tt, rising):
            """time across the 10%-90% band of sig, either rising or falling."""
            pk = sig.max()
            idx = range(len(sig)) if rising else range(len(sig) - 1, -1, -1)
            i10 = next((i for i in idx if sig[i] > 0.1 * pk), None)
            idx = range(len(sig)) if rising else range(len(sig) - 1, -1, -1)
            i90 = next((i for i in idx if sig[i] > 0.9 * pk), None)
            if i10 is None or i90 is None or tt[i90] == tt[i10]:
                return np.nan
            return 0.8 * pk / abs(tt[i90] - tt[i10])

        accel = edge(vel, tv, True)
        decel = edge(vel, tv, False)
        data[tag] = dict(peak=peak, reg_peak=reg_peak, agree=agree, plateau=plateau,
                         accel=accel, decel=decel, n=int(m.sum()), rate=rate)
        flag = "" if agree < 0.10 else "  <-- MISMATCH"
        out.append(f"{tag:>5}{m.sum():>7}{rate:>7.0f}{peak:>9.2f}{reg_peak:>10.2f}"
                   f"{100*(1-agree):>7.0f}%{plateau:>9.2f}{accel:>8.1f}{decel:>8.1f}{flag}")

    out.append("\n  qdot pk / qdot reg are INDEPENDENT (fitted position vs the servo's")
    out.append("  own register). Agreement >90% means the number is real; a mismatch")
    out.append("  means the fit window is wrong for this joint's speed.")
    analyse.data = data
    return out


def build_bus(port):
    return FeetechMotorsBus(
        port=port,
        motors={n: Motor(i + 1, "sts3215",
                         MotorNormMode.RANGE_0_100 if n == "gripper"
                         else MotorNormMode.RANGE_M100_100)
                for i, n in enumerate(NAMES)},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", default="wrist_roll", choices=NAMES)
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--excursion", type=float, default=30.0,
                    help="max deg from the start pose in either direction")
    ap.add_argument("--margin", type=float, default=10.0,
                    help="extra deg beyond the clamp before the watchdog aborts")
    ap.add_argument("--duration", type=float, default=1.2,
                    help="seconds to log per ramp")
    ap.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--approach-rate", type=float, default=40.0,
                    help="deg/s for the SLOW moves into position")
    ap.add_argument("--out", default="pvd_logs/ramp_<joint>.csv")
    ap.add_argument("--calib", default=CALIB_DEFAULT)
    ap.add_argument("--keepout", type=float, default=15.0,
                    help="deg to stay clear of each calibrated mechanical end")
    ap.add_argument("--max-offset", type=float, default=25.0,
                    help="refuse to start more than this far from calibrated mid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = args.out.replace("<joint>", args.joint)
    bus = build_bus(args.port)
    bus.connect(handshake=True)
    print(f"[ramp] connected to {args.port}, joint under test = {args.joint}")

    try:
        start = {n: int(bus.read("Present_Position", n, normalize=False)) for n in NAMES}
        p0 = start[args.joint]
        torque_now = {n: int(bus.read("Torque_Enable", n, normalize=False)) for n in NAMES}

        # Mechanical range comes from the CALIBRATION FILE. The servo's own
        # Min/Max_Position_Limit registers read [0,4095] on this arm -- the full
        # encoder range -- so clamping against them is no clamp at all and will
        # happily drive a joint through its mechanical floor into the table.
        cal = load_calib(args.calib)[args.joint]
        keepout = int(round(args.keepout * DEG2TICK))
        lo_reg, hi_reg = cal["range_min"] + keepout, cal["range_max"] - keepout
        mid = (cal["range_min"] + cal["range_max"]) // 2

        if not (lo_reg <= p0 <= hi_reg):
            print(f"[ramp] ABORT: start {p0} is outside the safe band "
                  f"[{lo_reg},{hi_reg}] (calibrated range "
                  f"[{cal['range_min']},{cal['range_max']}] minus {args.keepout}deg "
                  f"keep-out). The joint is at/near a mechanical end -- run "
                  f"goto_mid.py first.")
            return 1
        if abs(p0 - mid) * TICK2DEG > args.max_offset:
            print(f"[ramp] ABORT: start is {abs(p0-mid)*TICK2DEG:.1f}deg from "
                  f"calibrated mid ({mid}); limit is {args.max_offset}deg. "
                  f"Run goto_mid.py first.")
            return 1

        exc = int(round(args.excursion * DEG2TICK))
        lo = max(p0 - exc, lo_reg)
        hi = min(p0 + exc, hi_reg)
        abort_lo = lo - int(round(args.margin * DEG2TICK))
        abort_hi = hi + int(round(args.margin * DEG2TICK))

        print(f"[ramp] start pose (raw ticks): {start}")
        print(f"[ramp] torque currently: {torque_now}")
        print(f"[ramp] {args.joint}: p0={p0}  servo limits=[{lo_reg},{hi_reg}]")
        print(f"[ramp] travel clamp = [{lo},{hi}] ticks "
              f"= [{(lo-p0)*TICK2DEG:+.1f},{(hi-p0)*TICK2DEG:+.1f}] deg from start")
        print(f"[ramp] watchdog aborts outside [{abort_lo},{abort_hi}]")
        print(f"[ramp] plan: slow to {lo}, STEP to {hi} (+{(hi-lo)*TICK2DEG:.1f} deg), "
              f"settle, STEP to {lo} (-{(hi-lo)*TICK2DEG:.1f} deg), slow back to {p0}")
        print(f"[ramp] log -> {out}")

        if args.dry_run:
            print("[ramp] --dry-run: nothing energized. Remove --dry-run to execute.")
            return 0
        if hi - lo < 5 * DEG2TICK:
            print("[ramp] ABORT: clamped travel is under 5 deg; servo limits too tight.")
            return 1

        # ---- energize safely: goal := present BEFORE torque on -------------
        for n in NAMES:
            bus.write("Goal_Position", n, start[n], normalize=False)
        for n in NAMES:
            bus.enable_torque(n)
        print("[ramp] torque enabled on all joints, holding start pose")
        time.sleep(0.4)

        def slow_to(target):
            """Ramp the GOAL gradually so the joint never gets a step it can slam."""
            cur = int(bus.read("Present_Position", args.joint, normalize=False))
            steps = max(1, int(abs(target - cur) * TICK2DEG / args.approach_rate * 50))
            for g in np.linspace(cur, target, steps):
                bus.write("Goal_Position", args.joint, int(round(g)), normalize=False)
                time.sleep(0.02)
            time.sleep(args.settle)

        def ramp(target, tag):
            """Command a saturating step and log as fast as the bus allows."""
            rows = []
            bus.write("Goal_Position", args.joint, int(target), normalize=False)
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < args.duration:
                p = int(bus.sync_read("Present_Position", [args.joint], normalize=False)[args.joint])
                v = int(bus.sync_read("Present_Velocity", [args.joint], normalize=False)[args.joint])
                rows.append((tag, t, p, v, int(target)))
                if p < abort_lo or p > abort_hi:
                    bus.write("Goal_Position", args.joint, p, normalize=False)
                    print(f"[ramp] WATCHDOG: {p} outside [{abort_lo},{abort_hi}] -- frozen")
                    break
            return rows

        slow_to(lo)
        rows = ramp(hi, "pos")
        time.sleep(args.settle)
        rows += ramp(lo, "neg")
        time.sleep(args.settle)
        slow_to(p0)
        print("[ramp] returned to start pose; leaving torque ON to hold")

        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ramp", "t", "pos_raw", "vel_raw", "goal_raw"])
            w.writerows(rows)
        print(f"[ramp] wrote {len(rows)} samples -> {out}")

        # ---------------------------- analysis ------------------------------
        for line in analyse(rows):
            print(line)
        return 0

    finally:
        bus.disconnect(disable_torque=False)   # KEEP the arm held; default True drops it
        print("[ramp] disconnected")


if __name__ == "__main__":
    sys.exit(main())
