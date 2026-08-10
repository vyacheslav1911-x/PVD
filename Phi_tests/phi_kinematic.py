"""Section 3a -- manipulability / Jacobian conditioning terms.

IMPORTANT STRUCTURAL POINT. The SO-101 has 5 joints that move the end-effector
(Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll) plus a Jaw that only opens the
gripper. The Jaw's Jacobian column is identically zero, so the pose Jacobian is
effectively 6x5.

Yoshikawa's m = sqrt(det(J J^T)) is therefore IDENTICALLY ZERO for this arm --
J J^T is 6x6 with rank <= 5. The survey's formula as written cannot be used
here. The correct form for a joint-deficient arm (m < 6 task dims) is

    m = sqrt(det(J^T J)) = prod(sigma_i)

over the 5 singular values. That is what this module computes, and it is a real
correction to the survey's Stage-1 recommendation, not a detail.

Units are mixed in a 6-row Jacobian (m/rad on top, rad/rad below), so the
translational and rotational blocks are also reported separately, as the survey
itself advises.
"""

import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds  # noqa: E402

URDF = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
EE_FRAME = "gripper"          # after Wrist_Roll; 'jaw' hangs off the Jaw joint
ARM_COLS = 5                  # Jaw contributes nothing to EE pose


def conditioning(model, data, frame_id, q):
    """Return manipulability and conditioning measures at one configuration."""
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
    Ja = J[:, :ARM_COLS]                      # drop the Jaw column

    out = {}
    for tag, M in (("full", Ja), ("trans", Ja[:3, :]), ("rot", Ja[3:, :])):
        s = np.linalg.svd(M, compute_uv=False)
        s = s[s > 0] if tag != "full" else s
        out[f"m_{tag}"] = float(np.prod(s)) if len(s) else 0.0
        out[f"invk_{tag}"] = float(s.min() / s.max()) if len(s) and s.max() > 0 else 0.0
        out[f"smin_{tag}"] = float(s.min()) if len(s) else 0.0
    return out


def main():
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)

    # sanity: prove the JJ^T form degenerates, so the correction is on the record
    q0 = pin.neutral(model)
    pin.computeJointJacobians(model, data, q0)
    pin.updateFramePlacements(model, data)
    J0 = pin.computeFrameJacobian(model, data, q0, fid, pin.LOCAL_WORLD_ALIGNED)
    print(f"J at neutral: {J0.shape}, rank {np.linalg.matrix_rank(J0)}, "
          f"Jaw column norm {np.linalg.norm(J0[:,5]):.3e}")
    print(f"  det(J J^T)     = {np.linalg.det(J0 @ J0.T):.3e}   <- survey formula, degenerate")
    Ja = J0[:, :ARM_COLS]
    print(f"  sqrt(det(Ja^T Ja)) = {np.sqrt(max(np.linalg.det(Ja.T @ Ja),0)):.6f}   <- used here")

    print("\nPER-CHUNK KINEMATIC CONDITIONING ON THE PROPOSED ACTIONS")
    print(f"{'run':<12}{'chunks':>7}{'min m_full':>12}{'min 1/k full':>14}"
          f"{'min 1/k trans':>15}{'min sigma_min':>15}")
    store = {}
    for lab, path in RUNS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
        starts = chunk_bounds(df["chunk_start"].to_numpy())
        rows = []
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(df)
            blk = a[s:e]
            if len(blk) < 4:
                continue
            c = [conditioning(model, data, fid, q) for q in blk]
            rows.append({key: min(ci[key] for ci in c) for key in c[0]})
        t = pd.DataFrame(rows)
        store[lab] = t
        print(f"{lab:<12}{len(t):>7}{t.m_full.min():>12.5f}{t.invk_full.min():>14.5f}"
              f"{t.invk_trans.min():>15.5f}{t.smin_full.min():>15.5f}")

    print("\nSEPARATION (does conditioning tell the two policies apart?)")
    from scipy.stats import mannwhitneyu
    sm = pd.concat([store[k] for k in store if k.startswith("smolvla")])
    pi = pd.concat([store[k] for k in store if k.startswith("pi0.5")])
    for col in ("m_full", "invk_full", "invk_trans", "smin_full"):
        u = mannwhitneyu(sm[col], pi[col], alternative="two-sided")
        print(f"  {col:<12} smolvla {sm[col].mean():9.5f}   pi0.5 {pi[col].mean():9.5f}   "
              f"AUC {u.statistic/(len(sm)*len(pi)):.3f}  p {u.pvalue:.3g}")

    # how close to singular does the workspace ever get, for context
    rng = np.random.default_rng(0)
    Q = rng.uniform(model.lowerPositionLimit, model.upperPositionLimit, size=(4000, model.nq))
    vals = np.array([conditioning(model, data, fid, q)["invk_full"] for q in Q])
    mv = np.array([conditioning(model, data, fid, q)["m_full"] for q in Q])
    print(f"\nworkspace reference (4000 random poses): 1/kappa p1 {np.percentile(vals,1):.5f} "
          f"median {np.median(vals):.5f} max {vals.max():.5f}")
    print(f"                                        m      p1 {np.percentile(mv,1):.5f} "
          f"median {np.median(mv):.5f} max {mv.max():.5f}")


if __name__ == "__main__":
    main()
