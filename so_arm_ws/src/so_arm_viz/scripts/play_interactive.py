#!/usr/bin/env python
"""Interactive candidate browser: step through trajectories, SEE feasibility, play one.

One session (policy + scorer loaded ONCE) that:
  * pre-scores every candidate with score_trajectories.py (the SAME scorer -- single
    source of truth for Φ and the per-term violations),
  * lets you iterate: type an index, or 'n'/'p' for next/previous, 't' for the table,
  * on each selection PRINTS that candidate's Φ + S_pos/S_vel/S_acc/S_torque/S_cont
    and feasibility, and publishes it to the existing RViz ghost (which loops it until
    you pick another).

Kinematic ghost only -- never opens /dev/ttyACM* or moves a motor.

RUN (conda lerobot_v6 + ROS Jazzy + workspace sourced; needs torch + pinocchio +
lerobot + rclpy). Start the RViz side first:
    ros2 launch so_arm_viz rollout_viz.launch.py                 # terminal 1
    ./src/so_arm_viz/scripts/run_interactive.sh                  # terminal 2  (this)
"""

import argparse
import importlib.util
import os

import numpy as np
import torch

SCORE_PATH = os.path.expanduser("~/Desktop/PVD/score_trajectories.py")
DEFAULT_PT = os.path.expanduser("~/Desktop/PVD/k_trajectories.pt")


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", default=DEFAULT_PT)
    p.add_argument("--policy-path", default=None, help="override the scorer's default")
    p.add_argument("--threshold", type=float, default=None,
                   help="feasibility threshold Φ ≤ ? (default = scorer's)")
    p.add_argument("--chunk-topic", default="/planned_chunk")
    p.add_argument("--no-reference", action="store_true",
                   help="don't pose the solid arm at the selected candidate's start")
    return p.parse_args()


def main():
    args = parse_args()
    S = load_module("score_trajectories", SCORE_PATH)  # reuse the exact scorer
    policy = args.policy_path or S.POLICY_PATH
    threshold = args.threshold if args.threshold is not None else S.FEASIBILITY_THRESHOLD

    common = S.load_common()
    model, data, q_max, _ = S.load_model_and_limits()

    traj = torch.load(args.pt, map_location="cpu", weights_only=False)
    traj = torch.as_tensor(traj, dtype=torch.float32)
    if traj.ndim == 2:
        traj = traj.unsqueeze(0)
    K, H, _ = traj.shape
    print(f"[interactive] {args.pt}: K={K} candidates × H={H} steps")

    print(f"[interactive] scoring all candidates (unnormalize via {policy}) ...")
    post = S.load_unnormalizer(policy)
    q0_rad = np.array(common.lerobot_chunk_row_to_urdf(list(S.Q0_LEROBOT)))

    cands = []
    for i in range(K):
        real_deg = post(traj[i:i + 1].clone()).reshape(H, 6).detach().cpu().numpy()  # stage1
        A_rad = S.chunk_to_radians(common, real_deg)                                  # stage2
        terms = S.score_candidate(A_rad, q0_rad, model, data, q_max)
        phi = S.phi(terms)
        cands.append({"idx": i, "terms": terms, "phi": phi,
                      "feasible": phi <= threshold, "real_deg": real_deg})

    order = sorted(range(K), key=lambda i: cands[i]["phi"])  # ascending Φ
    rank_of = {idx: r for r, idx in enumerate(order)}

    # ---- ROS ghost publisher (reused, no reinvention) ----------------------
    from so_arm_viz.chunk_publisher import RolloutBridge
    from so_arm_viz.common import LEROBOT_JOINT_ORDER
    import rclpy

    bridge = RolloutBridge(chunk_topic=args.chunk_topic)
    print(f"[interactive] waiting for ghost node on {args.chunk_topic} ...")
    end = __import__("time").monotonic() + 10.0
    while rclpy.ok() and __import__("time").monotonic() < end:
        if bridge.chunk_pub.get_subscription_count() >= 1:
            break
        rclpy.spin_once(bridge, timeout_sec=0.05)
    else:
        print("[interactive] (no ghost subscriber yet -- is rollout_viz.launch.py up? "
              "continuing anyway)")

    def flush():
        for _ in range(3):
            rclpy.spin_once(bridge, timeout_sec=0.02)

    def print_table():
        print(f"\n{'rank':>4} {'cand':>4} {'S_pos':>8} {'S_vel':>8} {'S_acc':>9} "
              f"{'S_torque':>8} {'S_cont':>7} {'Phi':>10} {'feasible':>9}")
        for i in order:  # best (lowest Φ) first
            t = cands[i]["terms"]
            print(f"{rank_of[i]:>4} {i:>4} {t['pos']:>8.3f} {t['vel']:>8.3f} "
                  f"{t['acc']:>9.3f} {t['torque']:>8.3f} {t['cont']:>7.3f} "
                  f"{cands[i]['phi']:>10.3f} {str(cands[i]['feasible']):>9}")
        print(f"(lower Φ = more feasible; threshold Φ ≤ {threshold})\n")

    def show_and_play(i):
        c = cands[i]
        t = c["terms"]
        dom = max(S.TERMS, key=lambda k: S.WEIGHTS[k] * t[k])
        print(f"\n──  candidate {i}   rank {rank_of[i]}/{K - 1}   "
              f"feasible={c['feasible']}  ──")
        for k in S.TERMS:
            mark = "  <-- dominant" if k == dom else ""
            print(f"     S_{k:<7} = {t[k]:>10.3f}{mark}")
        print(f"     {'Φ':<9} = {c['phi']:>10.3f}   (threshold {threshold})")
        # publish to the ghost (LeRobot units [H,6]); ghost loops it in RViz
        if not args.no_reference:
            ref = c["real_deg"][0].tolist()
            bridge.publish_state({f"{n}.pos": ref[j]
                                  for j, n in enumerate(LEROBOT_JOINT_ORDER)})
        bridge.publish_chunk(c["real_deg"])
        flush()
        print(f"     → playing candidate {i} as ghost in RViz (loops until next pick)")

    print_table()
    print("commands:  <index> play it | n next | p prev | t table | <Enter> replay | q quit")
    cur = order[0]  # start on the most feasible
    show_and_play(cur)

    try:
        while rclpy.ok():
            try:
                cmd = input(f"[cand {cur}] > ").strip().lower()
            except EOFError:
                break
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "":
                show_and_play(cur)
            elif cmd in ("n", "next"):
                cur = (cur + 1) % K
                show_and_play(cur)
            elif cmd in ("p", "prev", "previous"):
                cur = (cur - 1) % K
                show_and_play(cur)
            elif cmd in ("t", "table", "l", "list"):
                print_table()
            elif cmd.lstrip("-").isdigit():
                i = int(cmd)
                if 0 <= i < K:
                    cur = i
                    show_and_play(cur)
                else:
                    print(f"     index out of range 0..{K - 1}")
            else:
                print("     commands: <index> | n | p | t | <Enter> | q")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[interactive] bye.")
        bridge.close()


if __name__ == "__main__":
    main()
