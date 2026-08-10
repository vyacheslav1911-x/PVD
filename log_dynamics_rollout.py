#!/usr/bin/env python
"""Run INSTEAD of `lerobot-rollout` (same args) to LOG per-step arm dynamics to CSV.

It changes NOTHING about the rollout except that, inside SOFollower.get_observation
(called once per control tick), it also reads a few extra Feetech registers and
appends one CSV row per step:

  step, wall_time, dt,
  pos_<joint>      measured position   (deg for the 5 body joints, % for gripper)
  qdot_<joint>     velocity            (deg/s, finite-difference of pos, online)
  qddot_<joint>    acceleration        (deg/s^2, finite-difference of qdot, online)
  velraw_<joint>   servo Present_Velocity (raw, signed) -- cross-check
  load_<joint>     servo Present_Load     (raw, signed; 0..1000 = 0..100% max torque)
  current_<joint>  servo Present_Current  (raw; ~mA) -- torque proxy for RNEA check

q / qdot / qddot are what you feed RNEA; load & current are the MEASURED torque
proxies to validate the RNEA torque against (see compare_rnea_torque.py).

Custom flag (stripped before draccus/lerobot sees argv):
  --log.path=/path/to/dynamics.csv   (default: <repo>/pvd_logs/dynamics_<ts>.csv)

This is LOG-ONLY: no ROS / RViz. Use your normal rollout args otherwise.
"""

import csv
import datetime
import os
import sys
import time


def _extract_log_args(argv):
    """Pull `--log.*` (both `--log.k=v` and `--log.k v`) out of argv."""
    log, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--log."):
            if "=" in a:
                key, val = a[len("--log."):].split("=", 1)
            else:
                key = a[len("--log."):]
                val = argv[i + 1] if i + 1 < len(argv) else ""
                i += 1
            log[key] = val
        else:
            rest.append(a)
        i += 1
    return log, rest


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    log_args, rest = _extract_log_args(sys.argv[1:])

    log_path = log_args.get("log_path") or log_args.get("path")
    if not log_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(repo, "pvd_logs")
        os.makedirs(d, exist_ok=True)
        log_path = os.path.join(d, f"dynamics_{ts}.csv")
    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # ---- state shared with the patched get_observation --------------------
    st = {"fh": None, "writer": None, "joints": None,
          "prev_pos": None, "prev_vel": None, "prev_t": None, "step": 0}

    from lerobot.robots.so_follower import SOFollower
    _orig_get_obs = SOFollower.get_observation

    # Registers we ALSO read each tick. If a read fails once we stop trying it,
    # so a missing/slow register never repeatedly stalls the 30 Hz control loop.
    extra_regs = {"velraw": "Present_Velocity", "load": "Present_Load",
                  "current": "Present_Current"}
    reg_ok = {k: True for k in extra_regs}

    def _read_extra(bus, reg):
        try:
            return bus.sync_read(reg, normalize=False)  # dict{motor: raw int}
        except Exception:  # noqa: BLE001 -- never break inference for logging
            return None

    def _patched_get_obs(self, *a, **k):
        obs = _orig_get_obs(self, *a, **k)
        now = time.perf_counter()
        try:
            joints = st["joints"]
            if joints is None:
                joints = [key[:-4] for key in obs if key.endswith(".pos")]
                st["joints"] = joints
                cols = (["step", "wall_time", "dt"]
                        + [f"pos_{j}" for j in joints]
                        + [f"qdot_{j}" for j in joints]
                        + [f"qddot_{j}" for j in joints]
                        + [f"velraw_{j}" for j in joints]
                        + [f"load_{j}" for j in joints]
                        + [f"current_{j}" for j in joints])
                st["fh"] = open(log_path, "w", newline="", buffering=1)
                st["writer"] = csv.writer(st["fh"])
                st["writer"].writerow(cols)
                print(f"[dyn-log] writing per-step dynamics -> {log_path}", file=sys.stderr)

            pos = [float(obs[f"{j}.pos"]) for j in joints]

            # online finite-difference velocity/acceleration (deg/s, deg/s^2)
            if st["prev_t"] is None:
                dt = 0.0
                qdot = [0.0] * len(joints)
                qddot = [0.0] * len(joints)
            else:
                dt = now - st["prev_t"]
                inv = 1.0 / dt if dt > 1e-6 else 0.0
                qdot = [(pos[i] - st["prev_pos"][i]) * inv for i in range(len(joints))]
                qddot = [(qdot[i] - st["prev_vel"][i]) * inv for i in range(len(joints))]

            # extra measured registers (servo velocity, load, current)
            extra_vals = {}
            for tag, reg in extra_regs.items():
                d = _read_extra(self.bus, reg) if reg_ok[tag] else None
                if d is None and reg_ok[tag]:
                    reg_ok[tag] = False
                    print(f"[dyn-log] register {reg} unreadable; column left blank.", file=sys.stderr)
                extra_vals[tag] = ([d.get(j, "") for j in joints] if d else [""] * len(joints))

            row = ([st["step"], round(time.time(), 4), round(dt, 5)]
                   + [f"{v:.4f}" for v in pos]
                   + [f"{v:.4f}" for v in qdot]
                   + [f"{v:.3f}" for v in qddot]
                   + list(extra_vals["velraw"])
                   + list(extra_vals["load"])
                   + list(extra_vals["current"]))
            st["writer"].writerow(row)

            st["prev_pos"], st["prev_vel"], st["prev_t"] = pos, qdot, now
            st["step"] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[dyn-log] row skipped: {exc}", file=sys.stderr)
        return obs

    SOFollower.get_observation = _patched_get_obs
    print(f"[dyn-log] installed. Logging q, qdot, qddot, load, current per step.", file=sys.stderr)

    # hand the untouched args to the stock rollout
    sys.argv = [sys.argv[0]] + rest
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    try:
        rollout_main()
    finally:
        if st["fh"] is not None:
            st["fh"].flush()
            st["fh"].close()
            print(f"[dyn-log] wrote {st['step']} rows -> {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
