#!/usr/bin/env python
"""Phase 4 plumbing test -- publish a SYNTHETIC 'real-units' chunk via RolloutBridge.

Runs the exact Phase-4 publish path (conda python -> rclpy -> /planned_chunk) with
a hand-made chunk in LeRobot units, so you can confirm the ghost animates from the
rollout side BEFORE plugging in the actual SmolVLA policy. No policy, no hardware.

Launch the ghost first:  ros2 launch so_arm_viz ghost.launch.py
Then, under conda lerobot_v6 with ROS + workspace sourced:
    conda activate lerobot_v6
    source /opt/ros/jazzy/setup.bash
    source ~/Desktop/PVD/so_arm_ws/install/setup.bash
    python ~/Desktop/PVD/so_arm_ws/src/so_arm_viz/scripts/test_chunk_bridge.py
Ctrl-C to stop.
"""

import math

import rclpy

from so_arm_viz.chunk_publisher import RolloutBridge

CHUNK_LEN = 50


def make_fake_chunk():
    """A plausible 'reach' plan in LeRobot units (deg for body, 0..100 gripper)."""
    rows = []
    for t in range(CHUNK_LEN):
        u = t / (CHUNK_LEN - 1)
        rows.append([
            40.0 * math.sin(math.pi * u),   # shoulder_pan
            -45.0 * u,                      # shoulder_lift  ramp down
            60.0 * u,                       # elbow_flex     ramp up
            30.0 * math.sin(math.pi * u),   # wrist_flex
            0.0,                            # wrist_roll     hold
            100.0 * u,                      # gripper        open
        ])
    return rows  # [50,6]


def main():
    bridge = RolloutBridge()
    chunk = make_fake_chunk()
    bridge.get_logger().info("Publishing synthetic chunk to /planned_chunk every 3s (Ctrl-C to stop).")
    bridge.create_timer(3.0, lambda: bridge.publish_chunk(chunk))
    bridge.publish_chunk(chunk)  # immediate first publish
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
