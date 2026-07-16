# PVD

SO-101 arm workspace: ROS 2 visualization, LeRobot bridge, and calibration data.

## Layout

| Path | Contents |
| --- | --- |
| `so_arm_ws/` | ROS 2 workspace — `so_arm_viz` and `so_arm_description` packages |
| `SO-ARM_ROS2_URDF/` | URDF + meshes, vendored from [MuammerBay/SO-ARM_ROS2_URDF](https://github.com/MuammerBay/SO-ARM_ROS2_URDF) |
| `calibration_backup/`, `calibration_backup_v051/` | Saved arm calibrations |
| `outputs/` | Captured images |
| `lerobot_v6_requirements.txt` | Python deps for the `lerobot_v6` conda env |

Build artifacts (`build/`, `install/`, `log/`) are not tracked — rebuild after cloning.

## Setup on a new machine

```bash
git clone git@github.com:vyacheslav1911-x/PVD.git
cd PVD/so_arm_ws

# colcon needs system python on PATH, not conda's
export PATH=/usr/bin:$PATH
source /opt/ros/jazzy/setup.bash
colcon build
```

The bridge nodes run under the `lerobot_v6` conda env with ROS Jazzy sourced into the
same interpreter. The follower arm is addressed via its stable
`/dev/serial/by-id/` path, since `ttyACMx` renumbers between plug-ins.
