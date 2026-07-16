"""Phase 3 -- planned-chunk ghost demo (dummy data, no hardware).

Brings up, all in one RViz:
  * live robot_state_publisher (/robot_description) + joint_state_publisher
    (publishes zeros -> the live arm stands still at neutral),
  * ghost robot_state_publisher: a name-prefixed, translucent-cyan copy driven by
    /ghost/joint_states, its root tied to `base` by a static transform,
  * ghost_replay_node: animates an incoming JointTrajectory as /ghost/joint_states,
  * demo_chunk_pub: hardcoded smooth 50x6 sweep on /planned_chunk,
  * rviz2 with two RobotModel displays (live solid + ghost translucent).

Phase 5 swaps the zeros publisher for the real encoder node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from so_arm_viz.ghost_urdf import prefix_urdf

GHOST_RGBA = "0.1 0.8 1.0 1.0"  # cyan; RViz Alpha makes it translucent


def generate_launch_description():
    desc_share = get_package_share_directory("so_arm_description")
    viz_share = get_package_share_directory("so_arm_viz")

    urdf_path = os.path.join(desc_share, "urdf", "so101_new_calib.urdf")
    rviz_config = os.path.join(viz_share, "rviz", "so_arm_ghost.rviz")

    with open(urdf_path) as f:
        base_urdf = f.read()
    ghost_urdf = prefix_urdf(base_urdf, prefix="ghost_", ghost_rgba=GHOST_RGBA)

    # --- LIVE model (held still at neutral by joint_state_publisher zeros) ----
    live_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": base_urdf}],
    )
    live_jsp = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"robot_description": base_urdf}],
    )

    # --- GHOST model (prefixed URDF; publish/subscribe remapped to /ghost/*) --
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
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription(
        [live_rsp, live_jsp, ghost_rsp, ghost_static_tf, ghost_replay, demo_pub, rviz]
    )
