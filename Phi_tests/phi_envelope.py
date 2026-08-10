"""S_env -- torque-speed (back-EMF) envelope.

S_torque and S_vel each check one limit in isolation. A DC servo cannot deliver
stall torque and free speed at the same time: available torque falls roughly
linearly with speed,

    tau_avail(w) = tau_stall * (1 - |w| / w_free)

so a trajectory can pass BOTH box checks while demanding a (tau, w) pair the
motor cannot physically produce. This is the class of infeasibility nothing in
the current Phi sees.

CONSTANTS. Datasheet STS3215 @12 V: stall 30 kg.cm = 2.942 N.m, no-load
~45 RPM = 4.712 rad/s. The no-load figure is independently corroborated by our
own ramp test: measured peak joint speed was 4.5-5.3 rad/s at light load, which
brackets 4.712 -- so the envelope's right-hand intercept is measured, not just
quoted. The gripper carries Max_Torque_Limit=500 (half), so its tau_stall is
halved; the five body joints run at 1000 (full).
"""

import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds  # noqa: E402

URDF = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
KGCM_TO_NM = 0.0980665
TAU_STALL = np.array([30.0] * 5 + [15.0]) * KGCM_TO_NM   # gripper derated 50%
W_FREE = 4.712                                            # rad/s, 45 RPM
TAU_BOX = 3.0                                             # the current S_torque limit


def tau_avail(w):
    """Available torque at speed w, floored at 0 (past free speed: none)."""
    return np.maximum(TAU_STALL * (1.0 - np.abs(w) / W_FREE), 0.0)


def main():
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()

    print("TORQUE-SPEED ENVELOPE")
    print(f"  tau_stall {TAU_STALL[0]:.3f} N.m (body), {TAU_STALL[5]:.3f} (gripper); "
          f"w_free {W_FREE:.3f} rad/s")
    print(f"  box limit currently used by S_torque: {TAU_BOX} N.m")
    print("  available torque at speed:")
    for w in (0, 1, 2, 3, 3.5, 4):
        print(f"    w={w:>4.1f} rad/s -> tau_avail {tau_avail(w)[0]:.3f} N.m")

    print(f"\n{'run':<12}{'chunks':>7}{'max|tau|':>10}{'S_torque fires':>16}"
          f"{'S_env fires':>13}{'worst excess Nm':>17}{'chunks w/ viol':>16}")
    store = {}
    for lab, path in RUNS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
        dt = float(np.median(df["dt"].to_numpy(float)))
        bounds = chunk_bounds(df["chunk_start"].to_numpy())
        rows, tau_peak, box_hits, env_hits, worst = [], 0.0, 0, 0, 0.0
        for k, s in enumerate(bounds):
            e = bounds[k + 1] if k + 1 < len(bounds) else len(df)
            blk = a[s:e]
            if len(blk) < 4:
                continue
            v = np.gradient(blk, dt, axis=0)
            acc = np.gradient(v, dt, axis=0)
            # Severity in TORQUE UNITS, not as a ratio: past w_free the available
            # torque is 0 and any ratio diverges (observed 3e5), which is a
            # normalisation artefact, not a physical severity. The N.m shortfall
            # is bounded by tau_stall and stays interpretable.
            excess = []
            for t in range(len(blk)):
                tau = np.abs(pin.rnea(model, data, blk[t], v[t], acc[t]))
                av = tau_avail(v[t])
                ex = np.maximum(tau - av, 0.0)
                excess.append(ex)
                tau_peak = max(tau_peak, tau.max())
                box_hits += int((tau > TAU_BOX).any())
                env_hits += int((ex > 0).any())
            E = np.array(excess)
            worst = max(worst, E.max())
            rows.append(E.mean())
        store[lab] = np.array(rows)
        nviol = int((store[lab] > 0).sum())
        print(f"{lab:<12}{len(rows):>7}{tau_peak:>10.3f}{box_hits:>16d}"
              f"{env_hits:>13d}{worst:>17.2f}{nviol:>16d}")

    from scipy.stats import mannwhitneyu
    sm = np.concatenate([store[k] for k in store if k.startswith("smolvla")])
    pi = np.concatenate([store[k] for k in store if k.startswith("pi0.5")])
    u = mannwhitneyu(sm, pi, alternative="two-sided")
    print(f"\nSEPARATION S_env: smolvla {sm.mean():.4f}  pi0.5 {pi.mean():.4f}  "
          f"AUC {u.statistic/(len(sm)*len(pi)):.3f}  p {u.pvalue:.3g}")
    print("\n'S_torque fires' counts waypoints where |tau| > 3.0 N.m (the box).")
    print("'S_env fires' counts waypoints where |tau| exceeds what the motor can")
    print("produce AT THAT SPEED. The gap between the two columns is the class of")
    print("infeasibility the box check structurally cannot see.")


if __name__ == "__main__":
    main()
