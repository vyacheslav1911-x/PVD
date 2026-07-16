#!/usr/bin/env python
"""Phase 3 -- planned-chunk ghost animator.

Subscribes to a trajectory_msgs/JointTrajectory carrying a [T,6] joint chunk in
LeRobot units (degrees for the 5 body joints, 0..100 percent for the gripper --
same units a live observation is in). Converts each waypoint to URDF radians via
the shared common.py mapping (identical to the live arm), then steps through the
waypoints on a timer, publishing sensor_msgs/JointState to /ghost/joint_states so
a second (translucent) RobotModel animates the plan. Loops by default.

Needs only rclpy + common.py (NO lerobot), so it can run as a normal ros2 entry
point under system python.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from so_arm_viz.common import (
    LEROBOT_JOINT_ORDER,
    URDF_JOINT_ORDER,
    lerobot_chunk_row_to_urdf,
)

GHOST_PREFIX = "ghost_"


class GhostReplayNode(Node):
    def __init__(self):
        super().__init__("so_arm_ghost_replay")

        self.declare_parameter("playback_hz", 25.0)
        self.declare_parameter("chunk_topic", "/planned_chunk")
        self.declare_parameter("loop", True)

        playback_hz = float(self.get_parameter("playback_hz").value)
        self._loop = bool(self.get_parameter("loop").value)
        chunk_topic = self.get_parameter("chunk_topic").value

        self._ghost_names = [GHOST_PREFIX + n for n in URDF_JOINT_ORDER]
        self._waypoints: list[list[float]] | None = None  # rows of 6 URDF radians
        self._idx = 0

        self.sub = self.create_subscription(
            JointTrajectory, chunk_topic, self._on_chunk, 10
        )
        self.pub = self.create_publisher(JointState, "/ghost/joint_states", 10)
        self.timer = self.create_timer(1.0 / playback_hz, self._on_timer)

        # Publish a neutral pose immediately so the ghost renders (overlapping the
        # live arm) even before any plan arrives.
        self._publish(lerobot_chunk_row_to_urdf([0.0] * 6))
        self.get_logger().info(
            f"Ghost replay ready @ {playback_hz:.0f} Hz. Waiting for "
            f"JointTrajectory on {chunk_topic}."
        )

    def _on_chunk(self, msg: JointTrajectory):
        names = list(msg.joint_names)
        idx_of = {n: i for i, n in enumerate(names)}
        rows: list[list[float]] = []
        for pt in msg.points:
            if names:
                # reorder incoming joints into LEROBOT_JOINT_ORDER
                lr_row = [pt.positions[idx_of[n]] for n in LEROBOT_JOINT_ORDER]
            else:
                lr_row = list(pt.positions[:6])
            rows.append(lerobot_chunk_row_to_urdf(lr_row))
        self._waypoints = rows
        self._idx = 0
        self.get_logger().info(f"Received plan: {len(rows)} waypoints; animating.")

    def _on_timer(self):
        if not self._waypoints:
            return
        self._publish(self._waypoints[self._idx])
        self._idx += 1
        if self._idx >= len(self._waypoints):
            self._idx = 0 if self._loop else len(self._waypoints) - 1

    def _publish(self, positions_rad):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._ghost_names
        msg.position = list(positions_rad)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GhostReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
