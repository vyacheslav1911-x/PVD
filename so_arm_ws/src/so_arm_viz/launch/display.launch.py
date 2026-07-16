"""Phase 1 — static model in RViz.

Brings up:
  * robot_state_publisher  (publishes /robot_description + TF from /joint_states)
  * joint_state_publisher_gui  (sliders to move each joint)
  * rviz2  (RobotModel display, meshes)

No hardware, no lerobot. This is purely to confirm the URDF + meshes render and
the kinematic chain is correct before we drive it from encoders in Phase 2.
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

    # Plain URDF (not xacro); `xacro` still parses it fine and keeps us future-proof.
    robot_description = ParameterValue(
        Command(["xacro ", urdf_path]), value_type=str
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
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
        [robot_state_publisher, joint_state_publisher_gui, rviz]
    )
