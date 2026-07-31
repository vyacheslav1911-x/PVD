#!/usr/bin/env python
"""Play back K saved SmolVLA candidate trajectories as a kinematic GHOST in RViz.

NO robot, NO cameras, NO serial port -- this only loads a tensor from disk and
publishes ROS topics. It never touches /dev/ttyACM* or any motor.

INPUT: a PyTorch tensor of shape [K, T, 6]  (candidate, timestep, joint), joints
in LEROBOT order [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper], in *normalized* action space (~[-1,1]) -- i.e. the raw output of
SmolVLAPolicy._get_action_chunk, exactly what generate_k_trajectories.py saves.
(K and T are read from the tensor; K need not be 50.)

THE CONVERSION (this is the part that makes the ghost move right) has TWO stages,
and only the second one lives in common.py:

  1. normalized  -> LeRobot real units (deg for the 5 body joints, 0..100 for the
     gripper).  This is the POLICY POSTPROCESSOR (UnnormalizerProcessorStep,
     ACTION norm mode = MEAN_STD -> x*std+mean).  It is NOT in common.py; we load
     it here from the same checkpoint that generated the trajectories.
  2. LeRobot real units -> URDF radians.  This IS common.py
     (lerobot_chunk_row_to_urdf), applied INSIDE ghost_replay_node.

So this script does stage 1, hands the real-unit chunk to RolloutBridge, which
publishes trajectory_msgs/JointTrajectory on /planned_chunk; ghost_replay_node
does stage 2 and animates /ghost/joint_states.  If we skipped stage 1 and pushed
the normalized values straight out, the ghost node would read "1.4" as 1.4
DEGREES (~0.02 rad) and the ghost would barely twitch -- the classic "moves
wrong" failure.

PER CANDIDATE: publish its [T,6] chunk, wait one sweep (T / playback_hz seconds,
matching the ghost node's rate), optionally freeze on the last pose for --pause
seconds, then advance.  --loop repeats the whole set.

RUN under conda lerobot_v6 with ROS Jazzy + the workspace sourced (needs torch +
lerobot for stage 1, and rclpy + so_arm_viz for publishing).  See run_playback.sh.
"""

import argparse
import os
import time

import torch

DEFAULT_PT = os.path.expanduser("~/Desktop/PVD/k_trajectories.pt")
# The checkpoint that GENERATED k_trajectories.pt (see generate_k_trajectories.py).
# Its action mean/std are what correctly inverts the normalization. Override with
# --policy-path if your tensor came from a different checkpoint.
DEFAULT_POLICY = "qualia-robotics/smolvla-so101-candy-33c62cfe"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", default=DEFAULT_PT, help="path to the [K,T,6] tensor")
    p.add_argument("--index", type=int, default=None,
                   help="play ONLY this candidate index (0..K-1); default plays all")
    p.add_argument("--policy-path", default=DEFAULT_POLICY,
                   help="checkpoint whose UNnormalizer inverts the trajectories")
    p.add_argument("--pause", type=float, default=0.8,
                   help="seconds to freeze on the last pose between candidates")
    p.add_argument("--loop", action="store_true",
                   help="repeat the whole set of candidates forever")
    p.add_argument("--playback-hz", type=float, default=25.0,
                   help="MUST match ghost_replay_node's playback_hz (default 25)")
    p.add_argument("--margin", type=float, default=0.3,
                   help="extra seconds added to each sweep wait for render latency")
    p.add_argument("--chunk-topic", default="/planned_chunk",
                   help="must match ghost_replay_node's chunk_topic")
    p.add_argument("--no-reference", action="store_true",
                   help="do NOT pose the solid arm at the candidates' start pose")
    return p.parse_args()


def load_trajectories(path: str) -> torch.Tensor:
    """Load the tensor and coerce to [K, T, 6] (float, cpu)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):  # be forgiving if someone saved {"trajectories": ...}
        for key in ("trajectories", "chunks", "actions"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(f"{path} is a dict without a trajectories tensor")
    t = torch.as_tensor(obj, dtype=torch.float32)
    if t.ndim == 2:            # a single [T,6] candidate
        t = t.unsqueeze(0)
    if t.ndim != 3 or t.shape[-1] != 6:
        raise ValueError(f"expected [K,T,6] (or [T,6]); got {tuple(t.shape)}")
    return t


def load_unnormalizer(policy_path: str):
    """Return the policy postprocessor (stage 1: normalized -> deg/percent).

    Loads config + processor pipelines ONLY (no VLA weights), on CPU. Importing
    the smolvla modeling module registers the 'smolvla' choice for the config
    parser; without it PreTrainedConfig.from_pretrained can't decode the config.
    """
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401  (registers choice)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    _, post = make_pre_post_processors(
        cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return post


def sleep_spin(node, seconds: float) -> None:
    """Sleep while keeping the rclpy node responsive (and Ctrl-C-able)."""
    import rclpy
    end = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def wait_for_ghost(bridge, topic: str, timeout: float = 10.0) -> bool:
    """Block until ghost_replay_node is subscribed to the chunk topic."""
    import rclpy
    end = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < end:
        if bridge.chunk_pub.get_subscription_count() >= 1:
            return True
        rclpy.spin_once(bridge, timeout_sec=0.05)
    return False


def main():
    args = parse_args()

    traj = load_trajectories(args.pt)          # [K, T, 6] normalized
    K, T, _ = traj.shape
    print(f"[playback] loaded {args.pt}: K={K} candidates x T={T} steps x 6 joints "
          f"(normalized action space)")

    # --index N -> play only that candidate; otherwise play all in order.
    if args.index is not None:
        if not (0 <= args.index < K):
            raise SystemExit(f"--index {args.index} out of range 0..{K - 1}")
        play_indices = [args.index]
        print(f"[playback] single-candidate mode: index {args.index}")
    else:
        play_indices = list(range(K))

    print(f"[playback] loading unnormalizer from {args.policy_path} ...")
    post = load_unnormalizer(args.policy_path)
    # Stage 1 up front for every candidate, so any error surfaces before we start
    # animating. Each real chunk is [1, T, 6] in LeRobot units (deg / 0..100).
    real_chunks = [post(traj[i : i + 1].clone()) for i in range(K)]
    r0 = real_chunks[0].reshape(-1, 6)[0].tolist()
    print("[playback] stage 1 OK. candidate 0 @ t=0 (deg/deg/deg/deg/deg/pct): "
          + ", ".join(f"{v:.1f}" for v in r0))

    # --- ROS side: reuse RolloutBridge's publish path (no reinvention) ---------
    from so_arm_viz.chunk_publisher import RolloutBridge
    from so_arm_viz.common import LEROBOT_JOINT_ORDER

    bridge = RolloutBridge(chunk_topic=args.chunk_topic)

    print(f"[playback] waiting for ghost node on {args.chunk_topic} ...")
    if not wait_for_ghost(bridge, args.chunk_topic):
        print("[playback] WARNING: no subscriber on the chunk topic yet -- is "
              "rollout_viz.launch.py running? Publishing anyway.")

    # Pose the SOLID (live) RobotModel at the pose the candidates start from, so
    # it renders as a static reference (and RViz doesn't spam 'No transform').
    # Pure topic publish -- still no hardware. Disable with --no-reference.
    if not args.no_reference:
        ref = real_chunks[play_indices[0]].reshape(-1, 6)[0].tolist()
        bridge.publish_state({f"{n}.pos": ref[j] for j, n in enumerate(LEROBOT_JOINT_ORDER)})
        print("[playback] posed solid arm at start pose (reference).")

    sweep_s = T / args.playback_hz            # one full pass at the ghost's rate
    print(f"[playback] each sweep ~{sweep_s:.1f}s @ {args.playback_hz:.0f} Hz; "
          f"pause {args.pause:.1f}s; loop={args.loop}")

    import rclpy
    try:
        first = True
        while rclpy.ok() and (first or args.loop):
            first = False
            for n, i in enumerate(play_indices):
                if not rclpy.ok():
                    break
                print(f"[playback] playing candidate {i}  ({n + 1}/{len(play_indices)})")
                bridge.publish_chunk(real_chunks[i])       # ghost sweeps it
                sleep_spin(bridge, sweep_s + args.margin)  # let the pass render
                if args.pause > 0:
                    # Freeze on the final pose (single-waypoint chunk) for a clean
                    # beat between candidates.
                    bridge.publish_chunk(real_chunks[i][:, -1:, :])
                    sleep_spin(bridge, args.pause)
            if args.loop:
                print("[playback] looping ...")
        print("[playback] done.")
    except KeyboardInterrupt:
        print("\n[playback] interrupted.")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
