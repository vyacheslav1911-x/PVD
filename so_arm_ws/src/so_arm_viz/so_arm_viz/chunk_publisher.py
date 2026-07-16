#!/usr/bin/env python
"""Phase 4 -- bridge a real SmolVLA action chunk (and optionally live state) to ROS.

Import this INSIDE your LeRobot rollout process (conda lerobot_v6, ROS Jazzy +
workspace sourced). It is a thin rclpy publisher; the ONLY coupling between
LeRobot and ROS is topics -- the two processes are never merged:

  * /planned_chunk  trajectory_msgs/JointTrajectory  -- the ghost plan (this is
                    byte-for-byte the message demo_chunk_pub sent in Phase 3)
  * /joint_states   sensor_msgs/JointState            -- OPTIONAL live arm, only
                    for "Topology A" where the rollout owns the port (see below)

IMPORTANT -- units: publish_chunk() expects the chunk in LeRobot units (degrees
for the 5 body joints, 0..100 for the gripper), i.e. AFTER the policy
postprocessor has UNNORMALIZED it. The 32->6 slice is already done inside
SmolVLAPolicy._get_action_chunk. See the Phase-4 hook for exactly where.

Bus ownership (reconciling Phase 0): only ONE process may open /dev/ttyACM*.
  * Topology A (recommended for live inference): the rollout already opens the
    follower for inference, so it is the sole owner. Do NOT run encoder_node at
    the same time; instead call publish_state(obs) here so RViz still shows the
    live arm, and publish_chunk(chunk) for the ghost.
  * Topology B: encoder_node owns the port; the rollout must NOT open it -- it
    reads proprioception from the /joint_states topic and cameras directly. Then
    only call publish_chunk() here (never publish_state()).
"""

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from so_arm_viz.common import LEROBOT_JOINT_ORDER, lerobot_obs_to_urdf


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


class RolloutBridge(Node):
    """Publishes SmolVLA chunks (and optionally live joint state) to RViz."""

    def __init__(
        self,
        chunk_topic: str = "/planned_chunk",
        control_dt: float = 0.05,
        node_name: str = "lerobot_rollout_bridge",
    ):
        if not rclpy.ok():
            rclpy.init()
        super().__init__(node_name)
        self._control_dt = control_dt
        self.chunk_pub = self.create_publisher(JointTrajectory, chunk_topic, 10)
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)

    def publish_chunk(self, chunk) -> None:
        """Publish a planned chunk to /planned_chunk.

        Args:
            chunk: [T,6] or [1,T,6], torch/np, in LeRobot units (deg for body,
                   0..100 for gripper), columns in LEROBOT_JOINT_ORDER.
        """
        arr = _to_numpy(chunk)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"expected [T,6] (or [1,T,6]); got shape {arr.shape}")

        traj = JointTrajectory()
        traj.joint_names = list(LEROBOT_JOINT_ORDER)
        for t in range(arr.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in arr[t]]
            ts = t * self._control_dt
            pt.time_from_start = Duration(sec=int(ts), nanosec=int((ts - int(ts)) * 1e9))
            traj.points.append(pt)
        self.chunk_pub.publish(traj)

    def publish_state(self, obs: dict) -> None:
        """Publish live joint state to /joint_states (Topology A only).

        Args:
            obs: {"shoulder_pan.pos": deg, ..., "gripper.pos": pct} exactly as
                 SO101Follower.get_observation() returns (LeRobot units).
        """
        names, positions = lerobot_obs_to_urdf(obs)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.state_pub.publish(msg)

    def close(self) -> None:
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
