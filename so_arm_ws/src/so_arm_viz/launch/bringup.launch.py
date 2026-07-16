"""Phase 5 -- one-command bringup (non-inference: live encoder + ghost).

Single launch that starts everything for viewing the LIVE arm (from real encoders)
plus a ghost of a planned 50-step chunk:

  * live  robot_state_publisher   <- /joint_states        (from the encoder node)
  * ghost robot_state_publisher   <- /ghost/joint_states
  * static tf  base -> ghost_base
  * encoder_node (conda lerobot_v6, sole owner of the serial port) -> /joint_states
  * ghost_replay_node             <- /planned_chunk -> /ghost/joint_states
  * demo_chunk_pub (optional, on by default) -> a demo plan on /planned_chunk
  * rviz2 with live (solid) + ghost (translucent) RobotModels

Because the encoder owns the port, this is the NON-inference view: hand-move the
(torque-disabled) arm and the live model follows; the ghost sweeps whatever is on
/planned_chunk. For a live SmolVLA rollout instead, use rollout_viz.launch.py +
run_ghost_rollout.sh (Topology A -- do NOT run this at the same time).

Launch args:
  demo:=false            -> don't publish the demo plan (a real chunk source will)
  port:=<path>           -> follower serial port (default: stable by-id path)
  encoder_python:=<path> -> interpreter for the encoder (default: conda lerobot_v6)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from so_arm_viz.ghost_urdf import prefix_urdf

GHOST_RGBA = "0.1 0.8 1.0 1.0"
DEFAULT_ENCODER_PY = "/home/v1/miniconda3/envs/lerobot_v6/bin/python"
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533793-if00"


def generate_launch_description():
    desc_share = get_package_share_directory("so_arm_description")
    viz_share = get_package_share_directory("so_arm_viz")
    urdf_path = os.path.join(desc_share, "urdf", "so101_new_calib.urdf")
    rviz_config = os.path.join(viz_share, "rviz", "so_arm_ghost.rviz")

    with open(urdf_path) as f:
        base_urdf = f.read()
    ghost_urdf = prefix_urdf(base_urdf, prefix="ghost_", ghost_rgba=GHOST_RGBA)

    demo_arg = DeclareLaunchArgument("demo", default_value="true")
    port_arg = DeclareLaunchArgument("port", default_value=DEFAULT_PORT)
    encpy_arg = DeclareLaunchArgument("encoder_python", default_value=DEFAULT_ENCODER_PY)

    live_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": base_urdf}],
    )
    ghost_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ghost_state_publisher",
        output="screen",
        parameters=[{"robot_description": ghost_urdf}],
        remappings=[
            ("robot_description", "/ghost/robot_description"),
            ("joint_states", "/ghost/joint_states"),
        ],
    )
    ghost_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ghost_base_link",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "base", "--child-frame-id", "ghost_base",
        ],
    )

    # Encoder runs under the conda lerobot_v6 interpreter (imports lerobot). We
    # invoke that binary directly; rclpy + so_arm_viz come from the inherited
    # PYTHONPATH (ROS + workspace), so no conda activation is needed here.
    encoder = ExecuteProcess(
        cmd=[
            LaunchConfiguration("encoder_python"),
            "-m", "so_arm_viz.encoder_node",
            "--ros-args", "-p", ["port:=", LaunchConfiguration("port")],
        ],
        output="screen",
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    ghost_replay = Node(
        package="so_arm_viz",
        executable="ghost_replay_node",
        name="so_arm_ghost_replay",
        output="screen",
    )
    demo_pub = Node(
        package="so_arm_viz",
        executable="demo_chunk_pub",
        name="so_arm_demo_chunk_pub",
        output="screen",
        condition=IfCondition(LaunchConfiguration("demo")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([
        demo_arg, port_arg, encpy_arg,
        live_rsp, ghost_rsp, ghost_static_tf,
        encoder, ghost_replay, demo_pub, rviz,
    ])
