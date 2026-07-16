#!/usr/bin/env python
"""Phase 2 -- live SO-101 encoder bridge.

SOLE OWNER of /dev/ttyACM1. Reads the 6 joint positions via LeRobot's existing
read path (bus.sync_read("Present_Position")) -- we do NOT reimplement the
Feetech protocol -- converts LeRobot units to URDF radians (see common.py), and
publishes sensor_msgs/JointState on /joint_states at ~30 Hz.

Torque is disabled on connect so the arm can be backdriven by hand for the
Phase-2 verification (the hard gate). Nothing here commands or stiffens the arm.

MUST run under the conda `lerobot_v6` interpreter (imports rclpy + lerobot in one
process), with ROS Jazzy + the workspace install/ sourced. Do NOT `ros2 run`
this (that uses system python without lerobot); launch via conda python, e.g.:

    python -m so_arm_viz.encoder_node
"""

import sys

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError as _exc:  # almost always: running under the wrong Python
    sys.stderr.write(
        "\n[so_arm_encoder] Could not import rclpy.\n"
        "This node MUST run under the conda 'lerobot_v6' env (Python 3.12, ABI-\n"
        "compatible with ROS Jazzy's rclpy) -- NOT conda 'base' (Python 3.13).\n\n"
        "Fix:\n"
        "  conda activate lerobot_v6\n"
        "  source /opt/ros/jazzy/setup.bash\n"
        "  source ~/Desktop/PVD/so_arm_ws/install/setup.bash\n"
        "  python -m so_arm_viz.encoder_node\n\n"
        f"Current interpreter: Python {sys.version.split()[0]} at {sys.executable}\n"
        f"Underlying error: {_exc}\n"
    )
    raise SystemExit(1)

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from so_arm_viz.common import URDF_JOINT_ORDER, lerobot_obs_to_urdf


class EncoderNode(Node):
    def __init__(self):
        super().__init__("so_arm_encoder")

        # Stable by-id path for the follower (serial 5B41533793) -- immune to
        # /dev/ttyACMx renumbering on replug. Override with -p port:=... if needed.
        self.declare_parameter(
            "port",
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533793-if00",
        )
        self.declare_parameter("robot_id", "my_follower")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("log_period_s", 1.0)  # throttled value logging; 0 disables

        port = self.get_parameter("port").value
        robot_id = self.get_parameter("robot_id").value
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._log_period = float(self.get_parameter("log_period_s").value)

        # --- Connect to the arm via LeRobot, read-only, torque OFF -----------
        self.get_logger().info(f"Opening SO-101 follower on {port} (id={robot_id})...")
        cfg = SO101FollowerConfig(port=port, id=robot_id)
        self.robot = SO101Follower(cfg)
        if not self.robot.calibration:
            raise RuntimeError(
                f"No calibration loaded for id={robot_id} at {self.robot.calibration_fpath}"
            )
        self.robot.bus.connect()          # opens port + pings motors; does NOT touch torque
        self.robot.bus.disable_torque()   # ensure arm is limp so it can be moved by hand
        self.get_logger().info("Connected. Torque DISABLED (arm is free to move by hand).")

        # --- ROS publisher + timer ------------------------------------------
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self._last_log = self.get_clock().now()
        self.get_logger().info(
            f"Publishing /joint_states for {URDF_JOINT_ORDER} at {rate_hz:.0f} Hz."
        )

    def _on_timer(self):
        try:
            raw = self.robot.bus.sync_read("Present_Position")  # {motor: deg|pct}
        except Exception as exc:  # noqa: BLE001 -- skip a bad read, don't crash the node
            self.get_logger().warn(f"sync_read failed, skipping cycle: {exc}")
            return

        obs = {f"{motor}.pos": val for motor, val in raw.items()}
        names, positions = lerobot_obs_to_urdf(obs)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.pub.publish(msg)

        self._maybe_log(raw, names, positions)

    def _maybe_log(self, raw, names, positions):
        if self._log_period <= 0.0:
            return
        now = self.get_clock().now()
        if (now - self._last_log).nanoseconds < self._log_period * 1e9:
            return
        self._last_log = now
        # Show LeRobot value (deg/pct) -> published URDF radians, per joint.
        parts = []
        for lr_name, urdf_name, pos in zip(raw.keys(), names, positions):
            parts.append(f"{urdf_name}={raw[lr_name]:+.1f}->{pos:+.3f}rad")
        self.get_logger().info("  ".join(parts))

    def destroy_node(self):
        try:
            if self.robot.bus.is_connected:
                self.robot.bus.disconnect()  # closes port, disables torque
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
