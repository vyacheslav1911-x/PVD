from glob import glob

from setuptools import find_packages, setup

package_name = "so_arm_viz"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="v1",
    maintainer_email="prog74194@gmail.com",
    description="Live encoder + planned VLA-chunk ghost visualization for the SO-101 arm in RViz.",
    license="BSD",
    entry_points={
        "console_scripts": [
            # NOTE: encoder_node is intentionally NOT here -- it needs lerobot and
            # must run under conda lerobot_v6 (see scripts/run_encoder.sh). These
            # two need only rclpy + common.py, so they run under system python.
            "ghost_replay_node = so_arm_viz.ghost_replay_node:main",
            "demo_chunk_pub = so_arm_viz.demo_chunk_pub:main",
        ],
    },
)
