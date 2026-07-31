#!/usr/bin/env python3
"""
visualize_trajectories_3d.py
============================
Project the K joint-space trajectories into 3D CARTESIAN space via forward
kinematics ("rzutowanie"), and plot all K end-effector paths in one 3D figure.

WHY:
    A trajectory is 6 joint numbers over time — you cannot "see" the motion from
    the numbers. Forward kinematics (FK) maps each joint configuration to the
    3D position of the end-effector (the gripper tip). Doing that for every
    timestep of every candidate turns each trajectory into a CURVE IN SPACE.
    Plotting all K curves shows, at a glance:
      - where the candidates agree (curves bundle together)
      - where they diverge (curves fan out)
    That fan is exactly the room PVD's feasibility selection has to work in.

PIPELINE per candidate, per timestep:
    6 joint values (NORMALIZED)  --unnormalize-->  real joint angles (rad)
        --Pinocchio FK-->  end-effector (x, y, z)
    => a [T, 3] path in space. Stack K of them, plot.

REQUIREMENTS:
    pip install pin matplotlib          # 'pin' is the Pinocchio pip package
    A URDF for the SO-101 (the same one your RViz setup uses).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pinocchio as pin

# ----------------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------------
IN_FILE   = "k_trajectories.pt"
URDF_PATH = "SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf"   # <-- point at your actual URDF
EE_FRAME  = "gripper"                            # <-- end-effector frame name in the URDF
                                                 #     (check with the print below if unsure)

# The 6 joints are NORMALIZED (~[-1,1]) in the saved tensor. To do FK we need
# real radians. If you have the true per-joint (min,max) from calibration, put
# them here to unnormalize properly. Otherwise the ANGLE_SCALE fallback just
# maps [-1,1] -> [-pi, pi] so you still see relative SHAPE (not absolute pose).
USE_CALIB = False
# joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
CALIB_MIN_RAD = np.array([-1.92, -1.75, -1.69, -1.66, -2.79, 0.0])   # example — replace
CALIB_MAX_RAD = np.array([ 1.92,  1.75,  1.69,  1.66,  2.79, 1.75])  # example — replace
ANGLE_SCALE   = np.pi   # fallback: normalized [-1,1] -> [-pi, pi]

# ----------------------------------------------------------------------------
# 1. Load trajectories
# ----------------------------------------------------------------------------
traj = torch.load(IN_FILE).numpy()      # [K, T, 6]
K, T, J = traj.shape
print(f"loaded {K} candidates x {T} timesteps x {J} joints")

# ----------------------------------------------------------------------------
# 2. Build the Pinocchio kinematic model from the URDF
# ----------------------------------------------------------------------------
# buildModelFromUrdf parses links/joints/lengths. data holds computed transforms.
model = pin.buildModelFromUrdf(URDF_PATH)
data  = model.createData()
print(f"model built: {model.nq} DOF")
print("available frames:", [f.name for f in model.frames])   # find your EE frame here

try:
    ee_id = model.getFrameId(EE_FRAME)
except Exception:
    raise SystemExit(f"Frame '{EE_FRAME}' not in URDF. Pick one from the list above.")

# ----------------------------------------------------------------------------
# 3. Unnormalize -> radians
# ----------------------------------------------------------------------------
def to_radians(norm_row: np.ndarray) -> np.ndarray:
    """One [6] normalized row -> [6] radians."""
    x = np.clip(norm_row, -1.0, 1.0)
    if USE_CALIB:
        # map [-1,1] -> [min,max] per joint
        return CALIB_MIN_RAD + (x + 1.0) * 0.5 * (CALIB_MAX_RAD - CALIB_MIN_RAD)
    return x * ANGLE_SCALE

# ----------------------------------------------------------------------------
# 4. Forward kinematics: joints -> end-effector (x,y,z)
# ----------------------------------------------------------------------------
def ee_position(q_rad: np.ndarray) -> np.ndarray:
    """One [nq] joint config (rad) -> [3] end-effector position."""
    q = np.zeros(model.nq)
    q[:min(len(q_rad), model.nq)] = q_rad[:model.nq]   # map our 6 into model DOF
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.array(data.oMf[ee_id].translation)       # [x, y, z]

paths = np.zeros((K, T, 3))
for k in range(K):
    for t in range(T):
        paths[k, t] = ee_position(to_radians(traj[k, t]))

# ----------------------------------------------------------------------------
# 5. 3D plot — all K end-effector paths
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for k in range(K):
    ax.plot(paths[k, :, 0], paths[k, :, 1], paths[k, :, 2],
            alpha=0.35, linewidth=1)
# mark all start points (should coincide — same observation) and end points
ax.scatter(paths[:, 0, 0],  paths[:, 0, 1],  paths[:, 0, 2],
           c="green", s=30, label="start")
ax.scatter(paths[:, -1, 0], paths[:, -1, 1], paths[:, -1, 2],
           c="red",   s=30, label="end (K different)")

ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_title(f"{K} candidate end-effector trajectories (FK projection)")
ax.legend()
plt.tight_layout()
plt.savefig("trajectories_3d.png", dpi=150)
print("saved -> trajectories_3d.png")
plt.show()
