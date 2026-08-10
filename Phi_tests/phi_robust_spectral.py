"""Sections 5 and 6 -- dynamics-ensemble margin, and command spectral power.

Section 5: perturb the inertial parameters (mass, CoM, payload) by +-X% and
recompute RNEA torque feasibility across the ensemble; score worst case and
fraction-feasible.

Section 6: fraction of each joint's commanded-position spectral energy above the
servo's closed-loop bandwidth. The bandwidth is NOT guessed -- it comes from the
rate-limited servo model identified in servo_model.py (kp with delay Td), whose
first-order corner is f_bw = kp / (2 pi).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds  # noqa: E402

URDF = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
REPO = os.path.expanduser("~/Desktop/PVD")
TAU_MAX = 3.0
N_ENSEMBLE = 40
PERTURB = 0.20
PAYLOAD_KG = 0.05          # a candy-sized payload in the gripper


def chunk_iter(path):
    df = pd.read_csv(path)
    a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
    dt = float(np.median(df["dt"].to_numpy(float)))
    for k, s in enumerate(bounds := chunk_bounds(df["chunk_start"].to_numpy())):
        e = bounds[k + 1] if k + 1 < len(bounds) else len(df)
        if e - s >= 4:
            yield a[s:e], dt


# ============================ Section 5 ====================================
def section5():
    base = pin.buildModelFromUrdf(URDF)
    rng = np.random.default_rng(0)

    models = []
    for i in range(N_ENSEMBLE):
        m = base.copy()
        for k in range(1, m.njoints):
            I = m.inertias[k]
            f = 1 + rng.uniform(-PERTURB, PERTURB)
            lever = I.lever * (1 + rng.uniform(-PERTURB, PERTURB, 3))
            m.inertias[k] = pin.Inertia(I.mass * f, lever, I.inertia * f)
        # payload in the last link
        I = m.inertias[m.njoints - 1]
        m.inertias[m.njoints - 1] = pin.Inertia(I.mass + PAYLOAD_KG, I.lever, I.inertia)
        models.append((m, m.createData()))
    nominal = (base, base.createData())

    print("SECTION 5 -- dynamics-ensemble feasibility margin")
    print(f"  ensemble of {N_ENSEMBLE}: inertias +-{100*PERTURB:.0f}%, "
          f"payload +{PAYLOAD_KG*1000:.0f} g in the gripper link, TAU_MAX={TAU_MAX} N.m")
    print(f"\n{'run':<12}{'chunks':>7}{'nominal max|tau|':>18}{'ensemble worst':>16}"
          f"{'ens/nominal':>13}{'chunks infeasible':>19}")
    for lab, path in RUNS:
        if not os.path.exists(path):
            continue
        nom_max, ens_max, infeas = 0.0, 0.0, 0
        n = 0
        for blk, dt in chunk_iter(path):
            n += 1
            v = np.gradient(blk, dt, axis=0)
            a_ = np.gradient(v, dt, axis=0)
            tn = max(np.abs(pin.rnea(nominal[0], nominal[1], blk[t], v[t], a_[t])).max()
                     for t in range(len(blk)))
            te = 0.0
            for m, d in models:
                te = max(te, max(np.abs(pin.rnea(m, d, blk[t], v[t], a_[t])).max()
                                 for t in range(0, len(blk), 5)))
            nom_max = max(nom_max, tn); ens_max = max(ens_max, te)
            if te > TAU_MAX:
                infeas += 1
        print(f"{lab:<12}{n:>7}{nom_max:>18.4f}{ens_max:>16.4f}"
              f"{ens_max/max(nom_max,1e-9):>13.2f}{infeas:>19}")
    print("  -> if the ensemble worst case is still far below TAU_MAX, the term is")
    print("     vacuous for this arm, exactly as S_torque already is.")


# ============================ Section 6 ====================================
def section6():
    mp = os.path.join(REPO, "pvd_logs/servo_models.json")
    if not os.path.exists(mp):
        print("\nSECTION 6 -- need servo_models.json; run servo_model.py first")
        return
    M = json.load(open(mp))
    fbw = {j: M[j]["rate"]["kp"] / (2 * np.pi) for j in JOINTS}

    print("\n\nSECTION 6 -- commanded spectral energy above the servo bandwidth")
    print("  f_bw from the IDENTIFIED rate model (kp/2pi), not a guess:")
    print("   " + "  ".join(f"{j.split('_')[0][:5]} {fbw[j]:.2f}Hz" for j in JOINTS))

    print(f"\n{'run':<12}{'chunks':>7}{'frac E > f_bw':>15}{'frac E > 5 Hz':>15}"
          f"{'frac E > 10 Hz':>16}")
    store = {}
    for lab, path in RUNS:
        if not os.path.exists(path):
            continue
        rows = []
        for blk, dt in chunk_iter(path):
            fs = 1 / dt
            n = len(blk)
            f = np.fft.rfftfreq(n, dt)
            per_joint = []
            for i, j in enumerate(JOINTS):
                x = blk[:, i] - blk[:, i].mean()
                P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
                tot = P.sum()
                if tot <= 0:
                    continue
                per_joint.append([P[f > fbw[j]].sum() / tot,
                                  P[f > 5].sum() / tot,
                                  P[f > 10].sum() / tot])
            if per_joint:
                rows.append(np.mean(per_joint, axis=0))
        r = np.array(rows)
        store[lab] = r
        print(f"{lab:<12}{len(r):>7}{r[:,0].mean():>15.4f}{r[:,1].mean():>15.4f}"
              f"{r[:,2].mean():>16.4f}")

    from scipy.stats import mannwhitneyu
    sm = np.vstack([store[k] for k in store if k.startswith("smolvla")])
    pi = np.vstack([store[k] for k in store if k.startswith("pi0.5")])
    print("\nSEPARATION")
    for c, nm in enumerate(("E>f_bw", "E>5Hz", "E>10Hz")):
        u = mannwhitneyu(sm[:, c], pi[:, c], alternative="two-sided")
        print(f"  {nm:<8} smolvla {sm[:,c].mean():.4f}  pi0.5 {pi[:,c].mean():.4f}  "
              f"AUC {u.statistic/(len(sm)*len(pi)):.3f}  p {u.pvalue:.3g}")


if __name__ == "__main__":
    section5()
    section6()
