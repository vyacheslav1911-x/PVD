#!/usr/bin/env python
"""Run this INSTEAD of `lerobot-rollout`, with your EXACT same args.

It transparently visualizes your SmolVLA rollout in RViz -- WITHOUT you editing
any lerobot code or your rollout -- by publishing two ROS topics:

  * /planned_chunk  <- SmolVLA's 50-step action chunk, UNNORMALIZED (the ghost)
  * /joint_states   <- the live arm, from the SAME encoder reads the policy uses

It does this with three tiny monkeypatches applied before the stock rollout runs:
  1. grab the inference engine's postprocessor (to unnormalize the chunk),
  2. wrap SmolVLAPolicy._get_action_chunk to publish each new plan,
  3. wrap SOFollower.get_observation to publish the live state.

Bus ownership (Topology A): the rollout OWNS /dev/ttyACM0 for inference, so it is
the sole port owner. Do NOT run encoder_node at the same time.

Usage (conda lerobot_v6 + ROS Jazzy + workspace sourced) -- or use
run_ghost_rollout.sh which sets all that up:

    python ghost_rollout.py <your usual lerobot-rollout args> --display_data=true

First launch the RViz side (no demo publisher needed):
    ros2 launch so_arm_viz ghost.launch.py   # then Ctrl-C the demo_chunk_pub, OR
    ros2 launch so_arm_viz bringup.launch.py  # (Phase 5, once available)
"""

import sys

from so_arm_viz.chunk_publisher import RolloutBridge

# One bridge for the whole process. rclpy.init() happens inside.
_bridge = RolloutBridge()
_engine_post = {"pipeline": None}  # filled in when the inference engine is built


def _install_patches() -> None:
    # 1) Capture the engine's postprocessor (both sync and rtc engines store it).
    for module_name, cls_name in (
        ("lerobot.rollout.inference.sync", "SyncInferenceEngine"),
        ("lerobot.rollout.inference.rtc", "RTCInferenceEngine"),
    ):
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:  # noqa: BLE001 -- rtc may be unavailable; that's fine
            continue

        def _wrap_init(orig_init):
            def patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                _engine_post["pipeline"] = getattr(self, "_postprocessor", None)
            return patched_init

        cls.__init__ = _wrap_init(cls.__init__)

    # 2) Publish the UNNORMALIZED chunk whenever the policy computes a new plan.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    _orig_get_chunk = SmolVLAPolicy._get_action_chunk

    def _patched_get_chunk(self, *args, **kwargs):
        chunk_norm = _orig_get_chunk(self, *args, **kwargs)  # [1,50,6], normalized
        post = _engine_post["pipeline"]
        if post is not None:
            try:
                chunk_real = post(chunk_norm.clone())  # MEAN_STD -> deg/pct
                _bridge.publish_chunk(chunk_real)
            except Exception as exc:  # noqa: BLE001 -- never break inference for viz
                print(f"[ghost] chunk publish skipped: {exc}", file=sys.stderr)
        return chunk_norm

    SmolVLAPolicy._get_action_chunk = _patched_get_chunk

    # 3) Publish the live arm from the same observation the policy consumes.
    from lerobot.robots.so_follower import SOFollower

    _orig_get_obs = SOFollower.get_observation

    def _patched_get_obs(self, *args, **kwargs):
        obs = _orig_get_obs(self, *args, **kwargs)
        try:
            _bridge.publish_state(obs)
        except Exception as exc:  # noqa: BLE001
            print(f"[ghost] state publish skipped: {exc}", file=sys.stderr)
        return obs

    SOFollower.get_observation = _patched_get_obs

    print("[ghost] patches installed: publishing /planned_chunk + /joint_states", file=sys.stderr)


def main():
    _install_patches()
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    rollout_main()  # draccus parses sys.argv -> your normal rollout args


if __name__ == "__main__":
    main()
