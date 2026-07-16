"""Phase 2 -- live arm view (no slider GUI, no encoder here).

Starts robot_state_publisher + rviz2 only. The encoder node is run separately
under conda python (it needs lerobot) and publishes /joint_states, which drives
the RViz model live:

    python -m so_arm_viz.encoder_node --ros-args -p port:=/dev/ttyACM1

Kept separate from the encoder on purpose so its per-joint deg->rad logs are
visible during the Phase-2 hand-move gate. Phase 5 unifies everything.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    desc_share = get_package_share_directory("so_arm_description")
    viz_share = get_package_share_directory("so_arm_viz")

    urdf_path = os.path.join(desc_share, "urdf", "so101_new_calib.urdf")
    rviz_config = os.path.join(viz_share, "rviz", "so_arm.rviz")

    robot_description = ParameterValue(Command(["xacro ", urdf_path]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([robot_state_publisher, rviz])
