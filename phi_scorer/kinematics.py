"""phi_scorer.kinematics -- URDF-based kinematics/dynamics for min_invk and S_env.

Loads the SO-101 URDF once with Pinocchio and exposes:
  * manipulability(q)  -> inverse condition number of the EE Jacobian (feeds min_invk)
  * rnea_torque(q,v,a) -> joint torques for the S_env motor-envelope gate

FRAMES & UNITS (read before trusting a number)
----------------------------------------------
Positions q are in URDF RADIANS, in Pinocchio joint order
[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]. Convert LeRobot deg/percent
with config.SCALE + the common map BEFORE calling anything here.

Jacobian: the SO-101 has 5 joints that move the end-effector plus a Jaw that only
opens the gripper (its Jacobian column is identically zero). So the 6x6 pose Jacobian
has rank <= 5 and Yoshikawa's m = sqrt(det(J J^T)) is IDENTICALLY ZERO -- unusable.
For a joint-deficient arm the correct manipulability is sqrt(det(J^T J)) over the 5
non-trivial columns. We report the inverse condition number sigma_min/sigma_max of
that 6x5 arm Jacobian: 1.0 = perfectly conditioned, 0.0 = singular. min_invk takes the
MINIMUM over a chunk (the worst, most-singular pose the chunk passes through).
"""

import numpy as np
import pinocchio as pin

from . import config


class Kinematics:
    """Thin, cached wrapper around the Pinocchio model built from the URDF."""

    def __init__(self, urdf_path=config.URDF_PATH, ee_frame=config.EE_FRAME,
                 arm_cols=config.ARM_COLS):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame)
        self.arm_cols = arm_cols
        # per-joint symmetric position magnitude limit (rad), straight from the URDF,
        # kept here so callers can cross-check S_pos against the model as well.
        self.q_abs_max = np.maximum(np.abs(self.model.lowerPositionLimit),
                                    np.abs(self.model.upperPositionLimit))

    # -- manipulability / conditioning at one configuration --------------------
    def inv_condition(self, q):
        """sigma_min/sigma_max of the 6x5 arm Jacobian at q. 1=well-conditioned, 0=singular."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id,
                                     pin.LOCAL_WORLD_ALIGNED)
        Ja = J[:, :self.arm_cols]                     # drop the zero Jaw column
        s = np.linalg.svd(Ja, compute_uv=False)
        return float(s.min() / s.max()) if s.max() > 0 else 0.0

    def min_inv_condition(self, chunk_rad):
        """min over a [T,6] radian chunk of the inverse condition number = min_invk."""
        return min(self.inv_condition(q) for q in chunk_rad)

    # -- inverse dynamics for the torque-envelope gate -------------------------
    def rnea_torque(self, q, v, a):
        """Joint torques (N*m) via recursive Newton-Euler at (q, v, a), all in URDF rad."""
        return np.asarray(pin.rnea(self.model, self.data, q, v, a), dtype=float)
