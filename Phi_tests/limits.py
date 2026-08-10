"""Resolve per-joint dynamic limits (qd_max, qdd_max, tau_max) from explicit
sources, and say which source each one came from.

RESOLUTION ORDER
----------------
For every joint and every limit, in order, first hit wins:

  1. "user"       an explicit value passed in user_config. A measured number
                  from a ramp test belongs here. Always wins.
  2. "urdf"       a limit carried by the Pinocchio model, but ONLY if it looks
                  like real per-joint data. so101_new_calib.urdf carries
                  effort=10 N.m and velocity=10 rad/s on all six joints --
                  identical values on every joint are a URDF-authoring
                  placeholder, not a measurement, so they are REJECTED here (see
                  _urdf_is_informative). URDF has no acceleration field at all.
  3. "empirical"  the 99th percentile of |qdot| / |qddot| over recorded
                  rollouts, on the masked uniform-time grid (see _empirical).
                  Accepted only if the estimate is stable across smoothing
                  windows; an estimate that moves more than `spread_tol` between
                  savgol windows is rejected, because then it is reporting the
                  filter's bandwidth rather than the arm's capability.
  4. "unresolved" nothing credible was available.

WHY "unresolved" EXISTS INSTEAD OF A FALLBACK CONSTANT
------------------------------------------------------
A feasibility score is a comparison against a limit, so a wrong limit does not
degrade the score, it invalidates it -- silently. Two failure modes in this repo
motivate the choice:

  * S_torque is identically zero across every candidate and every recorded
    rollout, because TAU_MAX = 3.0 N.m (score_trajectories.py:68) sits above any
    torque a 0.485 kg arm can generate: peak RNEA on the real logs is 1.00 N.m.
    The term looked like it was passing; it was never able to fire.
  * S_vel uses a placeholder qdot_max = 3.0 rad/s while the arm's measured p99 is
    ~1.3-1.6 rad/s, so that penalty cannot fire on real motion either.

In both cases a plausible-looking constant produced a confidently wrong answer.
Returning "unresolved" makes the gap a caller-visible decision: a caller may
refuse to score that term, widen its uncertainty, or ask for calibration -- but
it cannot mistake a guess for a measurement. Callers MUST handle value=None.

EMPIRICAL ESTIMATES ARE A FLOOR, NOT A LIMIT
--------------------------------------------
A rollout only shows what a policy asked for. If a policy never drove a joint
hard, its p99 is a lower bound on capability, not the capability. With a single
rollout per policy there is also no held-out set: a limit fitted on these logs
and then used to score these same logs is circular. Estimates carry n_rollouts
and per-rollout values so a caller can see how thin the evidence is.

Units are URDF radians throughout, matching Pinocchio's configuration order.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError as exc:  # pragma: no cover
    raise ImportError("limits.py needs scipy for the derivative estimates") from exc


# LeRobot CSV column stems, in Pinocchio joint order
# [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw].
LEROBOT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"]

# LeRobot units -> URDF radians. Body joints are degrees; the gripper is 0..100
# percent open mapped onto the Jaw angle. Slope mirrors common.py's
# gripper_pct_to_rad (JAW_LOWER_RAD=-0.174533, JAW_UPPER_RAD=1.74533).
_DEG2RAD = np.pi / 180.0
_GRIP2RAD = (1.74533 - (-0.174533)) / 100.0
UNIT_SCALE = np.array([_DEG2RAD] * 5 + [_GRIP2RAD])

# Masking conventions, matched to validate_*.py so estimates stay comparable.
SKIP_TICKS = 25          # startup transient
DT_GAP = 0.06            # a raw interval longer than this is a dropped tick
SG_WINDOW, SG_POLY = 9, 4
SG_WINDOW_SWEEP = (5, 7, 9, 13, 17)   # used only to test estimate stability
PERCENTILE = 99.0
SPREAD_TOL = 0.20        # reject an estimate that moves more than this


def _entry(value, source, **extra):
    d = {"value": value, "source": source}
    d.update(extra)
    return d


def _unresolved(reason):
    return _entry(None, "unresolved", reason=reason)


def _urdf_is_informative(vec):
    """True if a URDF limit vector looks per-joint rather than boilerplate.

    A vector with the same number on every joint (10/10/10/10/10/10) is a
    template default, and a vector of zeros or non-finite entries is missing
    data. Neither is evidence about this arm.
    """
    v = np.asarray(vec, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)) or np.any(v <= 0):
        return False
    return not np.allclose(v, v[0])


def _load_rollout(source):
    """Accept a path, or anything already exposing the rollout.py fields."""
    if hasattr(source, "speed") and hasattr(source, "t_raw"):
        return source
    from rollout import load_speed          # same-directory loader, reused as-is
    return load_speed(source)


def _good_mask(r, halfwidth):
    """Uniform-grid samples usable for differentiation.

    Drops the startup transient and any sample sitting inside a raw gap longer
    than DT_GAP -- interpolating across a dropped tick invents motion that then
    shows up as a spurious velocity spike. Dilated by the derivative stencil's
    half-width so no stencil straddles a discarded region.
    """
    t, t_raw = r.t, r.t_raw
    bad = t < t_raw[min(SKIP_TICKS, len(t_raw) - 1)]
    dtc = r.df["dt"].to_numpy(float)
    for i in np.where(dtc > DT_GAP)[0]:
        lo = t_raw[max(0, i - 1)]
        hi = t_raw[min(i, len(t_raw) - 1)]
        bad |= (t >= lo) & (t <= hi)
    if halfwidth > 0:
        bad = np.convolve(bad, np.ones(2 * halfwidth + 1, bool), mode="same") > 0
    return ~bad


def _uniform_q(r):
    """Achieved positions on the uniform grid, in URDF radians."""
    cols = [r.df[f"pos_{j}"].to_numpy(float) * UNIT_SCALE[i]
            for i, j in enumerate(LEROBOT_ORDER)]
    return np.column_stack([np.interp(r.t, r.t_raw, c) for c in cols])


def _empirical(rollouts, deriv, spread_tol=SPREAD_TOL):
    """Per-joint p99 of |d^deriv q / dt^deriv|, plus a stability verdict.

    Returns (values, stable, detail) where `values` is per joint (max over
    rollouts, since each rollout is only a floor), `stable` is a per-joint bool
    from the smoothing-window sweep, and `detail` carries the per-rollout
    numbers so a caller can see how thin the evidence is.
    """
    per_run, per_run_sweep = [], []
    for src in rollouts:
        r = _load_rollout(src)
        q = _uniform_q(r)
        good = _good_mask(r, SG_WINDOW_SWEEP[-1] // 2)
        if good.sum() < 3 * SG_WINDOW_SWEEP[-1]:
            continue
        d = savgol_filter(q, SG_WINDOW, SG_POLY, deriv=deriv, delta=r.dt, axis=0)
        per_run.append(np.percentile(np.abs(d[good]), PERCENTILE, axis=0))

        sweep = []
        for w in SG_WINDOW_SWEEP:
            dd = savgol_filter(q, w, min(SG_POLY, w - 1), deriv=deriv,
                               delta=r.dt, axis=0)
            sweep.append(np.percentile(np.abs(dd[good]), PERCENTILE, axis=0))
        per_run_sweep.append(np.array(sweep))

    if not per_run:
        return None, None, {"n_rollouts": 0}

    per_run = np.array(per_run)
    values = per_run.max(axis=0)

    # an estimate is stable only if it is stable in every rollout
    stable = np.ones(values.size, bool)
    spreads = []
    for sweep in per_run_sweep:
        spread = (sweep.max(axis=0) - sweep.min(axis=0)) / np.median(sweep, axis=0)
        spreads.append(spread)
        stable &= spread <= spread_tol

    return values, stable, {
        "n_rollouts": len(per_run),
        "per_rollout": per_run,
        "window_spread": np.array(spreads).max(axis=0),
        "percentile": PERCENTILE,
    }


def resolve_limits(model, rollouts=None, user_config=None, spread_tol=SPREAD_TOL):
    """Resolve qd_max, qdd_max and tau_max for every joint of `model`.

    Args:
        model:       a Pinocchio model (joint order defines the output order).
        rollouts:    optional iterable of recorded-CSV paths, or objects already
                     carrying the rollout.py fields. Only these enable the
                     "empirical" source.
        user_config: optional {joint_name: {"qd_max": v, ...}} of measured
                     values. Highest priority, never overridden.
        spread_tol:  reject an empirical estimate whose p99 moves by more than
                     this fraction across smoothing windows.

    Returns:
        {joint_name: {"qd_max": entry, "qdd_max": entry, "tau_max": entry}}
        where each entry is {"value": float | None, "source": str, ...}.
        A value of None always comes with source == "unresolved" and a reason;
        callers must handle it rather than assuming a default.
    """
    names = [model.names[i] for i in range(1, model.njoints)]
    user_config = user_config or {}

    vel_ok = _urdf_is_informative(model.velocityLimit)
    eff_ok = _urdf_is_informative(model.effortLimit)

    if rollouts:
        qd_emp, qd_stable, qd_detail = _empirical(rollouts, 1, spread_tol)
        qdd_emp, qdd_stable, qdd_detail = _empirical(rollouts, 2, spread_tol)
    else:
        qd_emp = qdd_emp = qd_stable = qdd_stable = None
        qd_detail = qdd_detail = {"n_rollouts": 0}

    out = {}
    for i, name in enumerate(names):
        u = user_config.get(name, {})
        joint = {}

        # ---- velocity ----
        if "qd_max" in u:
            joint["qd_max"] = _entry(float(u["qd_max"]), "user")
        elif vel_ok:
            joint["qd_max"] = _entry(float(model.velocityLimit[i]), "urdf")
        elif qd_emp is not None and qd_stable[i]:
            joint["qd_max"] = _entry(
                float(qd_emp[i]), "empirical",
                floor_only=True, n_rollouts=qd_detail["n_rollouts"],
                per_rollout=qd_detail["per_rollout"][:, i].tolist(),
                window_spread=float(qd_detail["window_spread"][i]),
                percentile=PERCENTILE)
        elif qd_emp is not None:
            joint["qd_max"] = _unresolved(
                f"empirical p99 unstable across smoothing windows "
                f"(spread {qd_detail['window_spread'][i]:.0%} > {spread_tol:.0%}); "
                "URDF velocityLimit is a uniform placeholder")
        else:
            joint["qd_max"] = _unresolved(
                "no rollouts supplied and URDF velocityLimit is a uniform "
                "placeholder")

        # ---- acceleration (no URDF field exists for this) ----
        if "qdd_max" in u:
            joint["qdd_max"] = _entry(float(u["qdd_max"]), "user")
        elif qdd_emp is not None and qdd_stable[i]:
            joint["qdd_max"] = _entry(
                float(qdd_emp[i]), "empirical",
                floor_only=True, n_rollouts=qdd_detail["n_rollouts"],
                per_rollout=qdd_detail["per_rollout"][:, i].tolist(),
                window_spread=float(qdd_detail["window_spread"][i]),
                percentile=PERCENTILE)
        elif qdd_emp is not None:
            joint["qdd_max"] = _unresolved(
                f"double-differentiated p99 unstable across smoothing windows "
                f"(spread {qdd_detail['window_spread'][i]:.0%} > {spread_tol:.0%}) "
                "-- the estimate tracks filter bandwidth, not the arm; URDF has "
                "no acceleration field")
        else:
            joint["qdd_max"] = _unresolved(
                "URDF carries no acceleration limit and no rollouts were supplied")

        # ---- torque ----
        if "tau_max" in u:
            joint["tau_max"] = _entry(float(u["tau_max"]), "user")
        elif eff_ok:
            joint["tau_max"] = _entry(float(model.effortLimit[i]), "urdf")
        else:
            joint["tau_max"] = _unresolved(
                "URDF effortLimit is a uniform placeholder and the logs carry no "
                "torque measurement (servo load is friction-dominated, r~0.26), "
                "so there is no empirical route")

        out[name] = joint
    return out


if __name__ == "__main__":
    import os
    import pinocchio as pin

    URDF = os.path.expanduser(
        "~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
    DATA = os.path.expanduser("~/Desktop/PVD/Phi_tests/recorded trajectories")
    model = pin.buildModelFromUrdf(URDF)
    runs = [os.path.join(DATA, "record_run1.csv"),
            os.path.join(DATA, "record_pi05_run1.csv")]

    res = resolve_limits(model, rollouts=runs)
    print(f"{'joint':<14}{'qd_max':>22}{'qdd_max':>16}{'tau_max':>16}")
    for name, lims in res.items():
        cells = []
        for key in ("qd_max", "qdd_max", "tau_max"):
            e = lims[key]
            cells.append("unresolved" if e["value"] is None
                         else f"{e['value']:.3f} ({e['source']})")
        print(f"{name:<14}{cells[0]:>22}{cells[1]:>16}{cells[2]:>16}")

    print("\nreasons for every unresolved entry:")
    for name, lims in res.items():
        for key, e in lims.items():
            if e["value"] is None:
                print(f"  {name}.{key}: {e['reason']}")
