"""Shared LeRobot<->URDF joint maps and unit conversions for the SO-101 arm.

This module is the SINGLE source of truth for:
  * the LeRobot-motor-name -> URDF-joint-name remap,
  * the publishing order (URDF base->tip),
  * the per-joint sign/offset that align LeRobot's zero with the URDF's zero,
  * the gripper percent -> Jaw-angle mapping.

It is used identically by the live encoder node (Phase 2) and the planned-chunk
ghost node (Phase 3/4), so a joint tuned once is correct everywhere.

Units coming IN from LeRobot's `bus.sync_read("Present_Position")`
(with the SO-101 follower's default norm modes):
  * 5 body joints  -> DEGREES, zeroed at the middle of each calibrated range.
  * gripper        -> 0..100  (percent open), NOT an angle.

Units going OUT (what we publish in sensor_msgs/JointState):
  * radians, in URDF joint order, under URDF joint names.
"""

import math

DEG2RAD = math.pi / 180.0

# --- Name remap: LeRobot motor name -> URDF <joint name> -------------------
LEROBOT_TO_URDF = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
    "gripper": "Jaw",
}

# Publishing order = URDF kinematic order (base -> tip). We iterate the LeRobot
# names in this matching order so positions line up with URDF_JOINT_ORDER.
LEROBOT_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
URDF_JOINT_ORDER = [LEROBOT_TO_URDF[n] for n in LEROBOT_JOINT_ORDER]
# -> ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# --- Per-joint alignment (TUNED BY THE PHASE-2 HAND-MOVE TEST) --------------
# urdf_rad = SIGN * (deg * DEG2RAD) + OFFSET_RAD
# Start neutral: +1 sign, 0 offset. Flip a SIGN if a joint moves BACKWARDS in
# RViz vs the real arm; set an OFFSET if a joint is shifted by a constant.
SIGN = {
    "shoulder_pan": +1.0,
    "shoulder_lift": +1.0,
    "elbow_flex": +1.0,
    "wrist_flex": +1.0,
    "wrist_roll": +1.0,
}
OFFSET_RAD = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}

# --- Gripper mapping: LeRobot 0..100 (percent open) -> URDF Jaw angle -------
# URDF <joint name="Jaw"> limits: lower=-0.174533, upper=1.74533 rad.
JAW_LOWER_RAD = -0.174533
JAW_UPPER_RAD = 1.74533
# Set True if 0%/100% animate to the wrong end (closed<->open swapped).
GRIPPER_INVERT = False


def gripper_pct_to_rad(pct: float) -> float:
    """Map LeRobot gripper percent (0..100) to the URDF Jaw angle in radians."""
    pct = max(0.0, min(100.0, float(pct)))
    if GRIPPER_INVERT:
        pct = 100.0 - pct
    return JAW_LOWER_RAD + (pct / 100.0) * (JAW_UPPER_RAD - JAW_LOWER_RAD)


def body_deg_to_rad(lerobot_name: str, deg: float) -> float:
    """Convert one body joint's LeRobot degrees to a URDF-aligned radian."""
    return SIGN[lerobot_name] * (float(deg) * DEG2RAD) + OFFSET_RAD[lerobot_name]


def lerobot_obs_to_urdf(obs: dict) -> tuple[list[str], list[float]]:
    """Convert a LeRobot observation dict to (urdf_names, positions_rad).

    Args:
        obs: dict like {"shoulder_pan.pos": deg, ..., "gripper.pos": pct}
             (exactly what SO101Follower.get_observation() / a ".pos"-suffixed
             sync_read returns).

    Returns:
        (names, positions) both in URDF base->tip order, positions in radians.
    """
    positions: list[float] = []
    for lr_name in LEROBOT_JOINT_ORDER:
        val = float(obs[f"{lr_name}.pos"])
        if lr_name == "gripper":
            positions.append(gripper_pct_to_rad(val))
        else:
            positions.append(body_deg_to_rad(lr_name, val))
    return list(URDF_JOINT_ORDER), positions


def lerobot_chunk_row_to_urdf(row: dict | list) -> list[float]:
    """Convert one row of a [T,6] action chunk to URDF radians (Phase 3/4).

    Accepts either a dict keyed by LeRobot names (with or without ".pos") or a
    plain length-6 sequence already in LEROBOT_JOINT_ORDER. Body values are
    degrees, gripper value is 0..100 -- same units as a live observation.
    """
    if isinstance(row, dict):
        obs = {}
        for lr_name in LEROBOT_JOINT_ORDER:
            key = f"{lr_name}.pos" if f"{lr_name}.pos" in row else lr_name
            obs[f"{lr_name}.pos"] = row[key]
        _, positions = lerobot_obs_to_urdf(obs)
        return positions
    # sequence in LEROBOT_JOINT_ORDER
    obs = {f"{lr_name}.pos": row[i] for i, lr_name in enumerate(LEROBOT_JOINT_ORDER)}
    _, positions = lerobot_obs_to_urdf(obs)
    return positions
