"""Move the arm slowly to the CALIBRATED mid-range pose, with load monitoring.

Why this exists: "mid-range" is not raw tick 2048 and it is definitely not
"wherever the arm happens to be sitting". It is the middle of each joint's
calibrated range_min..range_max, which is what the calibration JSON records.
Reading that file is the only way to know where the mechanical floor is --
the servo's own Min/Max_Position_Limit registers read [0,4095] on this arm,
i.e. the full encoder range, which is no constraint at all.

Moves at --rate deg/s by ramping Goal_Position, monitors Present_Load every
step, and aborts if a joint strains (it is pressing into something).

RUN:
  python goto_mid.py --dry-run
  python goto_mid.py
  python goto_mid.py --joints shoulder_lift,elbow_flex
"""

import argparse
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
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICK2DEG = 360.0 / 4096
DEG2TICK = 4096 / 360.0


def load_calib(path):
    """{joint: (range_min, range_max, mid)} from the lerobot calibration file."""
    c = json.load(open(path))
    return {k: (v["range_min"], v["range_max"], (v["range_min"] + v["range_max"]) // 2)
            for k, v in c.items()}


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
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--calib", default=CALIB_DEFAULT)
    ap.add_argument("--joints", default=",".join(NAMES))
    ap.add_argument("--rate", type=float, default=15.0, help="deg/s, deliberately slow")
    ap.add_argument("--max-load", type=int, default=600,
                    help="abort a joint if |Present_Load| exceeds this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    calib = load_calib(args.calib)
    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    bus = build_bus(args.port)
    bus.connect(handshake=True)
    try:
        cur = {n: int(bus.read("Present_Position", n, normalize=False)) for n in NAMES}
        print(f"{'joint':<15}{'now':>7}{'mid':>7}{'move':>9}{'range':>16}{'status':>12}")
        plan = {}
        for n in NAMES:
            lo, hi, mid = calib[n]
            d = (mid - cur[n]) * TICK2DEG
            at_floor = cur[n] <= lo + 20
            at_ceil = cur[n] >= hi - 20
            status = "AT FLOOR" if at_floor else ("AT CEILING" if at_ceil else "ok")
            if n in joints:
                plan[n] = mid
            print(f"{n:<15}{cur[n]:>7}{mid:>7}{d:>+8.1f}d  [{lo:>4},{hi:>4}]{status:>12}")

        if args.dry_run:
            print("\n--dry-run: nothing moved.")
            return 0

        # ramp every selected joint together, slowly, watching load
        steps = max(1, int(max(abs(plan[n] - cur[n]) for n in plan) * TICK2DEG
                           / args.rate * 50))
        print(f"\nmoving {len(plan)} joints to calibrated mid over ~{steps/50:.1f}s "
              f"at {args.rate} deg/s")
        paths = {n: np.linspace(cur[n], plan[n], steps) for n in plan}
        aborted = set()
        for k in range(steps):
            for n in plan:
                if n in aborted:
                    continue
                bus.write("Goal_Position", n, int(round(paths[n][k])), normalize=False)
            if k % 10 == 0:
                for n in plan:
                    if n in aborted:
                        continue
                    ld = int(bus.read("Present_Load", n, normalize=False))
                    if abs(ld) > args.max_load:
                        p = int(bus.read("Present_Position", n, normalize=False))
                        bus.write("Goal_Position", n, p, normalize=False)
                        aborted.add(n)
                        print(f"  ABORT {n}: load {ld} > {args.max_load}, frozen at {p}")
            time.sleep(0.02)

        time.sleep(0.5)
        print(f"\n{'joint':<15}{'pos':>7}{'goal':>7}{'load':>7}{'err_deg':>9}")
        for n in NAMES:
            p = int(bus.read("Present_Position", n, normalize=False))
            g = int(bus.read("Goal_Position", n, normalize=False))
            ld = int(bus.read("Present_Load", n, normalize=False))
            print(f"{n:<15}{p:>7}{g:>7}{ld:>7}{(p-calib[n][2])*TICK2DEG:>+8.1f}")
        print("\ntorque left ON, arm held at mid-range" +
              (f"; ABORTED: {sorted(aborted)}" if aborted else ""))
        return 0
    finally:
        bus.disconnect(disable_torque=False)   # KEEP the arm held; default True drops it


if __name__ == "__main__":
    sys.exit(main())
