"""phi_scorer.gates -- the three HARD feasibility gates.

A gate counts physically-impossible waypoints. Any single violation should make a
chunk infeasible regardless of how "human-like" it otherwise looks, which is why Phi
weights the gate sum by a large W_GATE. Gates are ABSOLUTE (no demos needed):

  S_pos  waypoints commanded outside the joint's mechanical travel range.
  S_vel  |commanded velocity| beyond the joint's measured q_dot_max.
  S_env  |RNEA torque| beyond what the motor can produce AT THAT SPEED. A DC motor's
         available torque falls with speed: tau_avail(w) = tau_stall*(1 - |w|/w_free).
         This catches (torque, speed) pairs that a static torque box would pass but
         the real motor cannot deliver.

All inputs are URDF radians (features.to_urdf_rad). `constants` supplies the measured
limits; see config for the fallback defaults.
"""

import numpy as np


def gate_S_pos(chunk_rad, pos_limit):
    """Count waypoints (joint x timestep) outside +/- pos_limit (rad)."""
    return int((np.abs(chunk_rad) > np.asarray(pos_limit)).sum())


def gate_S_vel(chunk_rad, dt, qd_max):
    """Count per-tick velocities exceeding the measured q_dot_max (rad/s)."""
    v = np.diff(chunk_rad, axis=0) / dt
    return int((np.abs(v) > np.asarray(qd_max)).sum())


def gate_S_env(chunk_rad, dt, kin, tau_stall, w_free):
    """Count timesteps where any joint's RNEA torque exceeds the speed-derated envelope."""
    v = np.diff(chunk_rad, axis=0) / dt                 # [T-1,6]
    a = np.diff(v, axis=0) / dt                         # [T-2,6]
    # pad velocity/acceleration so every position waypoint has an aligned (v, a)
    vv = np.vstack([v, v[-1]]) if len(v) else np.zeros_like(chunk_rad)
    aa = np.vstack([a, a[-1], a[-1]]) if len(a) else np.zeros_like(chunk_rad)
    n_env = 0
    for t in range(len(chunk_rad)):
        tau = np.abs(kin.rnea_torque(chunk_rad[t], vv[t], aa[t]))
        # available torque shrinks with speed; never negative
        tau_avail = tau_stall * np.clip(1.0 - np.abs(vv[t]) / w_free, 0.0, 1.0)
        n_env += int((tau > tau_avail).any())
    return n_env


def compute_gates(chunk_lr, dt, kin, constants):
    """Return {S_pos, S_vel, S_env} violation counts for one chunk."""
    from .features import to_urdf_rad
    chunk_rad = to_urdf_rad(chunk_lr)
    return {
        "S_pos": gate_S_pos(chunk_rad, constants["pos_limit_rad"]),
        "S_vel": gate_S_vel(chunk_rad, dt, constants["qd_max"]),
        "S_env": gate_S_env(chunk_rad, dt, kin, constants["tau_stall"], constants["w_free"]),
    }
