#!/usr/bin/env python
"""Phase 3 test stimulus -- publish a HARDCODED smooth 50x6 joint chunk.

Emits a trajectory_msgs/JointTrajectory in LeRobot units (degrees for body,
0..100 percent for gripper) on /planned_chunk, so the ghost node animates a
visible sweep with no policy/inference involved. Republishes periodically so a
late RViz/ghost still receives it.

This mimics EXACTLY the message the real SmolVLA hook will publish in Phase 4.
"""

import math

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from so_arm_viz.common import LEROBOT_JOINT_ORDER

CHUNK_LEN = 50
CONTROL_DT = 0.05  # s per step (informational; ghost plays at its own rate)


class DemoChunkPub(Node):
    def __init__(self):
        super().__init__("so_arm_demo_chunk_pub")
        self.declare_parameter("chunk_topic", "/planned_chunk")
        self.declare_parameter("period_s", 5.0)
        topic = self.get_parameter("chunk_topic").value
        period_s = float(self.get_parameter("period_s").value)

        self.pub = self.create_publisher(JointTrajectory, topic, 10)
        self.timer = self.create_timer(period_s, self._publish)
        # one quick initial publish (~1s) once subscribers have connected
        self._init_timer = self.create_timer(1.0, self._first_publish)
        self.get_logger().info(f"Demo chunk publisher -> {topic} (every {period_s:.0f}s)")

    def _make_chunk(self) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = list(LEROBOT_JOINT_ORDER)
        for t in range(CHUNK_LEN):
            u = t / (CHUNK_LEN - 1)          # 0..1
            s = math.sin(2.0 * math.pi * u)  # smooth -1..1..-1
            c = 0.5 * (1.0 - math.cos(2.0 * math.pi * u))  # smooth 0..1..0
            positions = [
                60.0 * s,                      # shoulder_pan  +-60 deg
                -35.0 * c,                     # shoulder_lift  dip to -35 deg
                50.0 * c,                      # elbow_flex     rise to 50 deg
                40.0 * math.sin(math.pi * u),  # wrist_flex     0..40..0 deg
                90.0 * s,                      # wrist_roll     +-90 deg
                100.0 * c,                     # gripper        0..100..0 percent
            ]
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in positions]
            tsec = t * CONTROL_DT
            pt.time_from_start = Duration(
                sec=int(tsec), nanosec=int((tsec - int(tsec)) * 1e9)
            )
            traj.points.append(pt)
        return traj

    def _publish(self):
        self.pub.publish(self._make_chunk())

    def _first_publish(self):
        self._init_timer.cancel()
        self._publish()
        self.get_logger().info("Published initial demo chunk.")


def main(args=None):
    rclpy.init(args=args)
    node = DemoChunkPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
