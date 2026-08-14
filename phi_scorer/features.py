"""phi_scorer.features -- the six graded features that feed D_demo.

All operate on ONE commanded action chunk. Input is the raw LeRobot chunk
`chunk_lr` of shape [T, 6] (5 body joints in degrees, gripper in 0..100 percent) plus
the sample period dt; `prev_lr` is the last action of the PREVIOUS chunk (length 6,
same LeRobot units) or None for the first chunk.

Everything is converted to URDF radians first (to_urdf_rad). Derivative-based features
(p99_v, p99_a, seam_v) are unaffected by the gripper's constant offset; min_invk drops
the gripper column entirely; only absolute-position uses (S_pos, gates) need the offset,
which to_urdf_rad supplies.

The six features:
  R         mean|2nd difference| / mean|1st difference|, per joint then averaged.
            Scale-invariant "dither"/reversal rate. >1 => command reverses faster than
            it advances. No physical constant -- pure shape.
  fracE_bw  fraction of commanded AC energy above each joint's servo bandwidth
            (see spectral.py). Needs the measured per-joint bandwidth.
  p99_v     99th percentile of |commanded velocity| (rad/s).
  p99_a     99th percentile of |commanded acceleration| (rad/s^2).
  min_invk  worst (minimum) EE-Jacobian inverse condition number over the chunk
            (see kinematics.py). Needs the URDF.
  seam_v    velocity of the jump INTO this chunk (|a_1 - q_prev|/dt) as a fraction of
            the joint's measured q_dot_max; max over joints. Measures chunk-seam jerk.
"""

import numpy as np

from . import config


def to_urdf_rad(chunk_lr):
    """[T,6] LeRobot units (deg x5, gripper %) -> [T,6] URDF radians (absolute angles)."""
    out = np.asarray(chunk_lr, dtype=float) * config.SCALE
    out[:, 5] = config.JAW_LO + np.asarray(chunk_lr, dtype=float)[:, 5] * config.GRIPPER_PCT_TO_RAD
    return out


def vec_to_urdf_rad(vec_lr):
    """length-6 LeRobot vector -> URDF radians (for prev-action seam)."""
    return to_urdf_rad(np.asarray(vec_lr, dtype=float)[None, :])[0]


# --- individual features -----------------------------------------------------
def feat_R(chunk_rad):
    d1 = np.diff(chunk_rad, axis=0)                        # [T-1,6] per-tick motion
    d2 = np.diff(d1, axis=0)                               # [T-2,6] change in motion
    m1 = np.abs(d1).mean(axis=0)
    m2 = np.abs(d2).mean(axis=0)
    ok = m1 > 1e-12                                        # ignore joints that never move
    return float(np.mean(m2[ok] / m1[ok])) if ok.any() else np.nan


def feat_p99_v(chunk_rad, dt):
    v = np.diff(chunk_rad, axis=0) / dt
    return float(np.percentile(np.abs(v), 99))


def feat_p99_a(chunk_rad, dt):
    v = np.diff(chunk_rad, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    return float(np.percentile(np.abs(a), 99))


def feat_seam_v(chunk_rad, prev_rad, dt, qd_max):
    """Implied velocity of the jump into the chunk, as a fraction of q_dot_max, max joint."""
    if prev_rad is None:
        step = np.abs(chunk_rad[1] - chunk_rad[0])        # no previous chunk: first real step
    else:
        step = np.abs(chunk_rad[0] - prev_rad)            # the seam jump
    return float(np.max(step / dt / np.asarray(qd_max)))


# --- bundle ------------------------------------------------------------------
def compute_features(chunk_lr, dt, prev_lr, kin, constants):
    """Return {name: value} for all six graded features.

    kin        : a kinematics.Kinematics instance (URDF loaded once).
    constants  : dict with 'bandwidth_hz' (len-6) and 'qd_max' (len-6), from
                 measured_constants.json or config defaults.
    """
    from .spectral import frac_E_bw
    chunk_rad = to_urdf_rad(chunk_lr)
    prev_rad = vec_to_urdf_rad(prev_lr) if prev_lr is not None else None
    return {
        "R":        feat_R(chunk_rad),
        "fracE_bw": frac_E_bw(chunk_rad, dt, constants["bandwidth_hz"]),
        "p99_v":    feat_p99_v(chunk_rad, dt),
        "p99_a":    feat_p99_a(chunk_rad, dt),
        "min_invk": kin.min_inv_condition(chunk_rad),
        "seam_v":   feat_seam_v(chunk_rad, prev_rad, dt, constants["qd_max"]),
    }
