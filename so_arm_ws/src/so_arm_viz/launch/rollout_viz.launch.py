"""Phase 4 -- RViz side for a REAL SmolVLA rollout (Topology A).

Starts ONLY the RViz consumers; the live arm and the ghost plan are both fed by
the rollout process (run via scripts/run_ghost_rollout.sh):
  * live robot_state_publisher   <- /joint_states   (published by the rollout)
  * ghost robot_state_publisher  <- /ghost/joint_states
  * ghost_replay_node            <- /planned_chunk  (published by the rollout)
  * static tf base -> ghost_base
  * rviz2 (live solid + ghost translucent)

No joint_state_publisher (live comes from the rollout) and no demo_chunk_pub
(the real chunk comes from the rollout). Because the rollout owns /dev/ttyACM0,
do NOT run encoder_node with this.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from so_arm_viz.ghost_urdf import prefix_urdf

GHOST_RGBA = "0.1 0.8 1.0 1.0"


def generate_launch_description():
    desc_share = get_package_share_directory("so_arm_description")
    viz_share = get_package_share_directory("so_arm_viz")
    urdf_path = os.path.join(desc_share, "urdf", "so101_new_calib.urdf")
    rviz_config = os.path.join(viz_share, "rviz", "so_arm_ghost.rviz")

    with open(urdf_path) as f:
        base_urdf = f.read()
    ghost_urdf = prefix_urdf(base_urdf, prefix="ghost_", ghost_rgba=GHOST_RGBA)

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
    ghost_replay = Node(
        package="so_arm_viz",
        executable="ghost_replay_node",
        name="so_arm_ghost_replay",
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([live_rsp, ghost_rsp, ghost_static_tf, ghost_replay, rviz])
