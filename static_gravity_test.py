#!/usr/bin/env python
"""STEP 4 — static gravity-hold test (the decisive RNEA check).

Slowly moves the SO-101 to a grid of STATIC poses spanning shoulder_lift & elbow,
pauses at each, and logs Present_Load while stationary (q̇≈0, q̈≈0 → no motion
friction, no inertia — pure gravity). Then compares the measured holding load to
pin.computeGeneralizedGravity(model, data, q) per pose.

If static gravity torque correlates with static load  -> physics/masses/sign are FINE,
   and the dynamic-run failure is timing / motion-friction.
If it does NOT correlate even in shape           -> sign, gravity direction, or wrong
   inertial parameters (mass/COM).

SAFETY: moves at most --max_step deg per tick (uses SOFollower.max_relative_target),
so it always ramps slowly and never stalls. Reads Present_Load with the SAME
bus.sync_read pattern the follower already uses for Present_Position. --dry_run
prints the pose plan and exits without moving.

RUN (conda lerobot_v6):
  python static_gravity_test.py --port /dev/serial/by-id/usb-...-if00 --id my_follower
  # widen/narrow the sweep with --lift_deltas / --elbow_deltas (deg, relative to the
  # pose the arm is in at start); --dry_run to preview first.
"""

import argparse
import csv
import sys
import time

import numpy as np
import pinocchio as pin

import score_trajectories as S

LIFT, ELBOW = "shoulder_lift", "elbow_flex"


def parse_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--max_step", type=float, default=4.0, help="deg/tick cap (slow & safe)")
    ap.add_argument("--rate", type=float, default=25.0, help="command Hz")
    ap.add_argument("--settle", type=float, default=1.5, help="s to wait after reaching a pose")
    ap.add_argument("--samples", type=int, default=40, help="Present_Load samples to average per pose")
    ap.add_argument("--lift_deltas", type=parse_list, default="-30,-15,0,15,30")
    ap.add_argument("--elbow_deltas", type=parse_list, default="-20,0,20,40")
    ap.add_argument("--clamp", type=float, default=60.0, help="hard |delta| cap from start pose (deg)")
    ap.add_argument("--out", default="pvd_logs/static_gravity.csv")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(port=args.port, id=args.id,
                                max_relative_target=args.max_step, cameras={})
    robot = SOFollower(cfg)
    print(f"[static] connecting to {args.port} (id={args.id}); torque will be ON to hold poses.")
    robot.connect()
    try:
        joints = [k[:-4] for k in robot.get_observation() if k.endswith(".pos")]
        home = {j: float(robot.get_observation()[f"{j}.pos"]) for j in joints}
        print(f"[static] start pose (deg/%): "
              + ", ".join(f"{j}={home[j]:.1f}" for j in joints))

        # ---- build the pose grid (only shoulder_lift & elbow move) ----------
        poses = []
        for dl in args.lift_deltas:
            for de in args.elbow_deltas:
                dl_c = float(np.clip(dl, -args.clamp, args.clamp))
                de_c = float(np.clip(de, -args.clamp, args.clamp))
                tgt = dict(home)
                tgt[LIFT] = home[LIFT] + dl_c
                tgt[ELBOW] = home[ELBOW] + de_c
                poses.append(tgt)
        print(f"[static] {len(poses)} poses; {LIFT} deltas={args.lift_deltas}, "
              f"{ELBOW} deltas={args.elbow_deltas} (clamped to ±{args.clamp}°)")
        if args.dry_run:
            for i, p in enumerate(poses):
                print(f"  pose {i:2d}: {LIFT}={p[LIFT]:7.1f}  {ELBOW}={p[ELBOW]:7.1f}")
            print("[static] --dry_run: not moving. Remove --dry_run to execute.")
            return

        def move_to(target, tol=1.5, timeout=25.0):
            t0 = time.perf_counter()
            while True:
                obs = robot.get_observation()
                robot.send_action({f"{j}.pos": target[j] for j in joints})  # capped per tick
                err = max(abs(float(obs[f"{j}.pos"]) - target[j]) for j in (LIFT, ELBOW))
                if err < tol or (time.perf_counter() - t0) > timeout:
                    return err
                time.sleep(1.0 / args.rate)

        def sample_static():
            pos_acc = {j: [] for j in joints}
            load_acc = {j: [] for j in joints}
            for _ in range(args.samples):
                obs = robot.get_observation()
                try:
                    load = robot.bus.sync_read("Present_Load", normalize=False)  # same pattern as pos
                except Exception:  # noqa: BLE001
                    load = {}
                for j in joints:
                    pos_acc[j].append(float(obs[f"{j}.pos"]))
                    load_acc[j].append(float(load.get(j, np.nan)))
                time.sleep(1.0 / args.rate)
            pos = {j: float(np.mean(pos_acc[j])) for j in joints}
            load = {j: float(np.nanmean(load_acc[j])) for j in joints}
            return pos, load

        # ---- execute the sweep ---------------------------------------------
        records = []
        for i, tgt in enumerate(poses):
            err = move_to(tgt)
            time.sleep(args.settle)                       # let it fully settle (q̇→0)
            pos, load = sample_static()
            records.append((pos, load))
            print(f"[static] pose {i:2d}/{len(poses)-1}  reach_err={err:.1f}°  "
                  f"{LIFT}={pos[LIFT]:6.1f} {ELBOW}={pos[ELBOW]:6.1f}  "
                  f"load[{LIFT}]={load[LIFT]:7.1f} load[{ELBOW}]={load[ELBOW]:7.1f}")

        print("[static] returning to start pose...")
        move_to(home)

        # ---- save CSV -------------------------------------------------------
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"pos_{j}" for j in joints] + [f"load_{j}" for j in joints])
            for pos, load in records:
                w.writerow([f"{pos[j]:.4f}" for j in joints] + [f"{load[j]:.2f}" for j in joints])
        print(f"[static] wrote {args.out}")
    finally:
        robot.disconnect()
        print("[static] disconnected (torque released).")

    # ---- analysis: static gravity torque vs static load --------------------
    common = S.load_common()
    model, data, _, jn = S.load_model_and_limits()
    JN = common.LEROBOT_JOINT_ORDER
    P = np.array([[rec[0][j] for j in JN] for rec in records])
    L = np.array([[rec[1][j] for j in JN] for rec in records])
    q = np.array([common.lerobot_obs_to_urdf({f"{lr}.pos": P[n, k] for k, lr in enumerate(JN)})[1]
                  for n in range(len(records))])
    taug = np.array([pin.computeGeneralizedGravity(model, data, q[n]) for n in range(len(records))])

    print("\n===== STATIC GRAVITY vs MEASURED LOAD (per joint) =====")
    print(f"{'joint':<13}{'r':>8}{'|r|':>7}{'tau_g range[Nm]':>20}")
    for i, name in enumerate(jn):
        a, b = taug[:, i], L[:, i]
        m = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 and np.ptp(a[m]) > 0 else np.nan
        print(f"{name:<13}{r:>8.3f}{abs(r):>7.3f}{f'[{a.min():.3f},{a.max():.3f}]':>20}")
    body = [i for i, n in enumerate(jn) if n in ("Pitch", "Elbow")]
    rr = [np.corrcoef(taug[:, i], L[:, i])[0, 1] for i in body if np.ptp(taug[:, i]) > 0]
    print(f"\nmean |r| over gravity joints (Pitch,Elbow) = {np.mean(np.abs(rr)):.3f}")
    print("HIGH -> masses/sign/gravity are fine, dynamic issue is timing/friction.")
    print("LOW  -> sign, gravity direction, or wrong inertial parameters.")


if __name__ == "__main__":
    main()
