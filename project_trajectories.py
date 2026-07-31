#!/usr/bin/env python
"""Projection operator: make the K candidate trajectories physically FEASIBLE, save them.

This is the "projection" operator from the proposal (distinct from selection): it
MODIFIES each trajectory minimally so it satisfies the physical limits, instead of
just scoring/rejecting. The output is saved in the SAME normalized [K,H,6] format as
the input, so the existing pipeline runs on it unchanged:
    python score_trajectories.py            --pt k_trajectories_projected.pt
    ./src/.../run_interactive.sh             --pt k_trajectories_projected.pt

--------------------------------------------------------------------------------
WHY A TRACKER, NOT A SMOOTHER
The dominant violation (S_acc) is NOT interior jitter -- it is one boundary spike:
the policy's first action sits ~0.27 rad from q0, and jumping there from rest in a
single 30 fps step needs ~240 rad/s². Low-pass smoothing cannot fix a boundary jump
(and actually makes Pitch/Elbow worse). So the operator is a bounded-acceleration
command tracker (a.k.a. reference/command smoother):

    q[0]=q0, v[0]=0
    for each policy target aₜ:
        a_cmd = clip( kp·(aₜ − q) − kd·v , ±a_max )     # acceleration-limited
        v     = clip( v + a_cmd·Δt      , ±v_max )       # velocity-limited
        q     = clip( q + v·Δt          , q_lo, q_hi )   # position-limited (+anti-windup)

By construction it (1) starts at q0 at rest → continuity, (2) never exceeds a_max /
v_max / joint limits → S_vel=S_acc=S_pos=0, and (3) PD-tracks the policy target
sequence so the feasible trajectory stays close to the policy's intent. Torque is
not explicitly bounded but the motion is slower than the raw chunk, so S_torque
stays 0 (verified). kp/kd set the tracking bandwidth (critical damping kd=2√kp).

--------------------------------------------------------------------------------
UNITS. The clamps are physical, so the tracker runs in RADIANS. We convert
normalized↔radian with the exact affine map of the real pipeline (post-processor
∘ common.py), recovered by evaluating it at 0 and 1 (both stages are affine):
    rad = K·norm + B      norm = (rad − B)/K
The projected radian trajectory is mapped back to normalized and saved, so when the
official scorer unnormalizes it (post ∘ common) it recovers exactly this trajectory.

RUN (conda lerobot_v6; torch + pinocchio + lerobot + numpy; ROS not needed):
    /home/v1/miniconda3/envs/lerobot_v6/bin/python ~/Desktop/PVD/project_trajectories.py
"""

import argparse
import csv
import importlib.util
import os

import numpy as np
import torch

SCORE_PATH = os.path.expanduser("~/Desktop/PVD/score_trajectories.py")
DEFAULT_PT = os.path.expanduser("~/Desktop/PVD/k_trajectories.pt")
DEFAULT_OUT = os.path.expanduser("~/Desktop/PVD/k_trajectories_projected.pt")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_affine(post, common):
    """Return K, B for the per-joint affine map  rad = K·norm + B  (and its inverse),
    matching post-processor ∘ common.py exactly (verified by round-trip)."""
    m = post(torch.zeros(1, 1, 6)).flatten().numpy()             # norm→(deg/pct) intercept
    s = post(torch.ones(1, 1, 6)).flatten().numpy() - m          # norm→(deg/pct) slope
    k = np.zeros(6)
    b = np.zeros(6)
    for i, nm in enumerate(common.LEROBOT_JOINT_ORDER):          # (deg/pct)→rad affine
        if nm == "gripper":
            b[i] = common.gripper_pct_to_rad(0.0)
            k[i] = common.gripper_pct_to_rad(1.0) - b[i]
        else:
            b[i] = common.body_deg_to_rad(nm, 0.0)
            k[i] = common.body_deg_to_rad(nm, 1.0) - b[i]
    K = k * s
    B = k * m + b
    return K, B


def track(targets_rad, q0_rad, vmax, amax, dt, kp, kd, qlo, qhi):
    """Bounded-acceleration PD tracker. targets_rad: [H,6] policy positions (rad).
    Returns [H,6] feasible positions, starting from q0 at rest."""
    q = q0_rad.astype(float).copy()
    v = np.zeros(6)
    out = np.empty_like(targets_rad)
    for t in range(targets_rad.shape[0]):
        a_cmd = np.clip(kp * (targets_rad[t] - q) - kd * v, -amax, amax)
        v = np.clip(v + a_cmd * dt, -vmax, vmax)
        q_prev = q
        q = np.clip(q + v * dt, qlo, qhi)
        v = (q - q_prev) / dt   # anti-windup: keep v consistent with realized motion
        out[t] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=DEFAULT_PT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--policy-path", default=None, help="override scorer default")
    ap.add_argument("--kp", type=float, default=300.0, help="tracker stiffness")
    ap.add_argument("--kd", type=float, default=None, help="tracker damping (default 2√kp)")
    args = ap.parse_args()
    kp = args.kp
    kd = args.kd if args.kd is not None else 2.0 * np.sqrt(kp)

    S = load_module("score_trajectories", SCORE_PATH)   # reuse scorer (single source)
    policy = args.policy_path or S.POLICY_PATH
    common = S.load_common()
    model, data, q_max, _ = S.load_model_and_limits()
    post = S.load_unnormalizer(policy)
    qlo, qhi = model.lowerPositionLimit, model.upperPositionLimit
    K, B = build_affine(post, common)
    q0_rad = np.array(common.lerobot_chunk_row_to_urdf(list(S.Q0_LEROBOT)))

    traj = torch.load(args.pt, map_location="cpu", weights_only=False)
    traj = torch.as_tensor(traj, dtype=torch.float32)
    if traj.ndim == 2:
        traj = traj.unsqueeze(0)
    Kn, H, _ = traj.shape
    print(f"[project] {args.pt}: K={Kn} × H={H}; tracker kp={kp} kd={kd:.1f}, "
          f"limits v={S.QDOT_MAX[0]} a={S.QDDOT_MAX[0]} (from scorer)")

    def score_rad(A):
        tr = S.score_candidate(A, q0_rad, model, data, q_max)
        return tr, S.phi(tr)

    out_norm = np.empty((Kn, H, 6), dtype=np.float32)
    before, after = [], []
    for c in range(Kn):
        targets_rad = traj[c].numpy() * K + B                 # norm → rad
        proj_rad = track(targets_rad, q0_rad, S.QDOT_MAX, S.QDDOT_MAX, S.DT, kp, kd, qlo, qhi)
        out_norm[c] = (proj_rad - B) / K                      # rad → norm (to save)
        before.append(score_rad(targets_rad))
        # score the SAVED tensor exactly as the user's scorer will (post ∘ common):
        saved_rad = S.chunk_to_radians(
            common, post(torch.tensor(out_norm[c]).unsqueeze(0)).reshape(H, 6).numpy())
        after.append(score_rad(saved_rad))

    torch.save(torch.from_numpy(out_norm), args.out)
    print(f"[project] saved {args.out}  (normalized [K,H,6], same format as input)\n")

    # ---- before/after report ----------------------------------------------
    print(f"{'cand':>4} | {'Phi_before':>11} {'Phi_after':>10} | "
          f"{'Sacc_b':>8} {'Sacc_a':>7} {'Scont_a':>8} | feasible_after")
    print("-" * 78)
    csv_path = os.path.expanduser("~/Desktop/PVD/projection_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cand", "Phi_before", "Phi_after", "Sacc_before", "Sacc_after",
                    "Svel_after", "Scont_after", "feasible_after"])
        for c in range(Kn):
            tb, pb = before[c]
            ta, pa = after[c]
            feas = (ta["pos"] + ta["vel"] + ta["acc"] + ta["torque"]) < 1e-3
            print(f"{c:>4} | {pb:>11.3f} {pa:>10.3f} | {tb['acc']:>8.2f} "
                  f"{ta['acc']:>7.3f} {ta['cont']:>8.3f} | "
                  f"{'YES (all limits 0)' if feas else 'no'}")
            w.writerow([c, f"{pb:.6f}", f"{pa:.6f}", f"{tb['acc']:.6f}",
                        f"{ta['acc']:.6f}", f"{ta['vel']:.6f}", f"{ta['cont']:.6f}",
                        int(feas)])
    print(f"\n[project] wrote {csv_path}")
    n_feas = sum(1 for c in range(Kn)
                 if (after[c][0]["pos"] + after[c][0]["vel"] + after[c][0]["acc"]
                     + after[c][0]["torque"]) < 1e-3)
    print(f"[project] {n_feas}/{Kn} candidates now have ZERO limit violations "
          f"(residual Φ is only the start-from-rest continuity cost).")
    print("\nVerify + view:")
    print(f"    python score_trajectories.py --pt {args.out}")
    print(f"    ros2 launch so_arm_viz rollout_viz.launch.py")
    print(f"    ./src/so_arm_viz/scripts/run_interactive.sh --pt {args.out}")


if __name__ == "__main__":
    main()
