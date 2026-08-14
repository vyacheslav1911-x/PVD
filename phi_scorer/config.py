"""phi_scorer.config -- the ONE source of truth for conventions and defaults.

WHAT LIVES HERE vs WHAT DOESN'T
------------------------------
This file holds only things that are *chosen*, never *measured*:
  * file paths, joint order, unit conventions,
  * the list of feature names and gate names,
  * the gate weight W_GATE and the D_demo pass percentile,
  * PHYSICS-BASED DEFAULT limits, used only as a fallback before you have run
    measure.py on the real arm.

Everything that is *measured* (per-joint velocity limit q_dot_max, per-joint servo
bandwidth, the demo mean mu / covariance Sigma, the pass threshold) is written by
measure.py / build_reference.py into GENERATED ARTIFACTS (measured_constants.json,
reference.npz). Nothing measured is hardcoded here, so the scorer stays regenerable
and independent of any stale numbers.

Load order at score time (phi.py):
  measured_constants.json  overrides  the DEFAULT_* values below,
  reference.npz            provides    mu, Sigma^-1, kept features, threshold.
If an artifact is missing, the DEFAULT_* fallback is used and a warning is printed.
"""

import os

# ---------------------------------------------------------------------------
# Paths (conventions -- edit here if the repo layout changes)
# ---------------------------------------------------------------------------
REPO = os.path.expanduser("~/Desktop/PVD")
HERE = os.path.dirname(os.path.abspath(__file__))

URDF_PATH = os.path.join(REPO, "SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")

# The robot's own calibration file defines the mechanical travel range per joint;
# S_pos reads its limits straight from here (this is robot config, not a "measurement").
CALIB_PATH = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")

# Human teleoperation dataset that DEFINES the demo reference for D_demo.
# (D_demo is demo-referenced by definition, so this is the one unavoidable data input.)
DEMO_DATASET_DIR = ("/home/g/.cache/huggingface/hub/datasets--polrolnik2--so101_candy/"
                    "snapshots/6fc8a48159bb7416c962cd3c3ea9ce8f6e23c8fb")

# Generated artifacts (self-contained; this folder produces and consumes them).
CONSTANTS_PATH = os.path.join(HERE, "artifacts", "measured_constants.json")
REFERENCE_PATH = os.path.join(HERE, "artifacts", "reference.npz")
REFERENCE_CSV  = os.path.join(HERE, "artifacts", "reference_features.csv")   # LOG 1
CHUNK_LOG_PATH = os.path.join(HERE, "artifacts", "phi_chunks.jsonl")         # LOG 2

# ---------------------------------------------------------------------------
# Joint / unit conventions
# ---------------------------------------------------------------------------
# LeRobot SO-101 motor order (the order cmd_/pos_ columns and the dataset action use).
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = len(JOINTS)

# Matching URDF joint order (base -> tip). One-to-one with JOINTS above.
URDF_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# LeRobot units -> URDF radians. 5 body joints are degrees; the gripper is a 0..100
# percent that maps linearly onto the URDF Jaw angle range [-0.174533, 1.74533] rad.
import numpy as np  # noqa: E402
DEG2RAD = np.pi / 180.0
JAW_LO, JAW_HI = -0.174533, 1.74533
GRIPPER_PCT_TO_RAD = (JAW_HI - JAW_LO) / 100.0
SCALE = np.array([DEG2RAD] * 5 + [GRIPPER_PCT_TO_RAD])   # multiply cmd_/action by this

# End-effector for the manipulability (min_invk) feature. The Jaw joint only opens the
# gripper and contributes a zero column to the EE Jacobian, so we keep 5 arm columns.
EE_FRAME = "gripper"
ARM_COLS = 5

# Control period (s). Overridden per-run by the measured median dt from the recording;
# this is only the nominal 30 Hz fallback.
DEFAULT_DT = 1.0 / 30.0

# Chunk windowing for the demo reference (50 frames = one SmolVLA/pi0.5 action chunk).
WINDOW = 50

# ---------------------------------------------------------------------------
# The feature set and the aggregate structure
# ---------------------------------------------------------------------------
# Graded features -> D_demo (Mahalanobis distance to the demo cloud). Order matters:
# it is the column order of the demo feature matrix and reference.npz.
FEATURES = ["R", "fracE_bw", "p99_v", "p99_a", "min_invk", "seam_v"]

# Hard gates -> n_gate (a single hard violation should outrank any graded roughness).
GATES = ["S_pos", "S_vel", "S_env"]

# Phi = W_GATE * (sum of gate violation counts) + D_demo.
W_GATE = 10.0

# Two nearly-collinear features make the demo covariance ill-conditioned; drop the
# later of any demo-correlated pair above this |r| before fitting Sigma.
CORR_DROP = 0.90

# Pass threshold = this percentile of the demo D_demo distribution (95th = "as unlike
# a demo as the most-unusual 5% of real human teleop").
PASS_PERCENTILE = 95.0

# ---------------------------------------------------------------------------
# PHYSICS-BASED DEFAULTS (fallback only -- measure.py overrides these per joint)
# ---------------------------------------------------------------------------
# Servo closed-loop corner frequency (Hz), used by fracE_bw to split "in-band" vs
# "above what the servo can track". Default ~1.27 Hz = 1/(2*pi*tau), tau~125 ms, the
# first-order corner of a kp~8 rad/s position loop. measure.py replaces this with the
# per-joint value fitted from a real step response.
DEFAULT_BANDWIDTH_HZ = {j: 1.273 for j in JOINTS}

# Per-joint achievable speed (rad/s), used by S_vel and seam_v. Conservative default;
# measure.py replaces it with the ramp-test plateau/peak per joint.
DEFAULT_QD_MAX = {j: 3.0 for j in JOINTS}

# Torque envelope for S_env: tau_avail(w) = tau_stall * (1 - |w|/w_free). Defaults from
# the STS3215 datasheet; measure.py refines w_free from the ramp peak.
DEFAULT_TAU_STALL = 1.9     # N*m
DEFAULT_W_FREE = 5.4        # rad/s (measured ramp peak exceeded the 4.71 datasheet no-load)
