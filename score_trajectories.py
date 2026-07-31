#!/usr/bin/env python
"""Feasibility scorer + ranker for K SmolVLA candidate action chunks (SELECTION only).

This is the "selection" operator from the research proposal, Phase 1. It is
ANALYTIC and READ-ONLY: it scores and ranks the candidates by physical
feasibility and logs the result. It does NOT modify any trajectory (that would be
projection, not selection) and it does NOT execute or auto-play anything. You pick
a candidate by hand and replay it with playback_trajectories.py --index N.

INPUT: ~/Desktop/PVD/k_trajectories.pt, a tensor [K, H, 6] = (candidate, timestep,
joint), joints in LEROBOT order [shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper], in NORMALIZED action space (~[-1,1]). K and H are
read from the tensor.

================================ THE UNNORMALIZATION ============================
The scorer needs REAL units (radians / rad·s⁻¹) for finite differences and RNEA;
the tensor is normalized. This is done in TWO stages, and only stage 2 lives in
common.py (the user note said "reuse common.py" -- but common.py alone is NOT
enough; it does deg→rad, not normalized→deg):

  STAGE 1  normalized → LeRobot real units (deg for the 5 body joints, 0..100 for
           the gripper).  = the POLICY POSTPROCESSOR (UnnormalizerProcessorStep,
           ACTION norm mode = MEAN_STD → x·std+mean). Loaded here from the SAME
           checkpoint that generated the tensor (identical to what
           playback_trajectories.py does before publishing to the ghost).
  STAGE 2  LeRobot real units → URDF radians (and gripper % → Jaw rad).
           = common.py's lerobot_chunk_row_to_urdf (reused verbatim, by file path).

After stage 2 the positions are in radians, in URDF joint order
[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw], which is EXACTLY
Pinocchio's configuration order (verified) -- so q feeds RNEA directly.
Scoring the normalized numbers instead would give garbage torques; that is why
both stages are applied before any score is computed.

OUTPUT: a per-term table, a ranking, the filter-then-prefer selection, summary
stats, and ~/Desktop/PVD/trajectory_scores.csv.

RUN (conda lerobot_v6; needs torch + pinocchio + lerobot + numpy; ROS not needed):
    /home/v1/miniconda3/envs/lerobot_v6/bin/python ~/Desktop/PVD/score_trajectories.py
"""

import argparse
import csv
import importlib.util
import os

import numpy as np
import pinocchio as pin
import torch

# ============================================================================
# CONFIG  --  the physical limits below are the only "tunable" knobs.
# ============================================================================
DT = 1.0 / 30.0  # 30 fps control period (seconds)

# Position limits q_max come from the URDF (read from the Pinocchio model), NOT
# from here -- see load_position_limits().

# ---- PLACEHOLDER LIMITS (STS3215 rough defaults) ---------------------------
# The proposal calls for MEASURED limits, not datasheet numbers. These are rough
# stand-ins so the pipeline runs end-to-end; CALIBRATE THEM EMPIRICALLY later
# (drive the real servos, log peak velocity/accel/torque, replace these).
# For reference the URDF itself only carries uniform placeholders
# (effort=10 N·m, velocity=10 rad/s on every joint), which is why we don't trust
# it for the dynamic limits.
QDOT_MAX = np.full(6, 3.0)   # rad/s    (STS3215 ~60 rpm no-load ≈ 6.3; ~half loaded)
QDDOT_MAX = np.full(6, 20.0)  # rad/s²   (pure guess -- MUST be measured)
TAU_MAX = np.full(6, 3.0)    # N·m      (STS3215 stall ≈ 30 kg·cm ≈ 3 N·m @12V; cont. lower)

# Aggregate weights. All 1.0 because every term is already normalized by its own
# limit (dimensionless), so the weights are unit-consistent, not task-tuned.
WEIGHTS = {"pos": 1.0, "vel": 1.0, "acc": 1.0, "torque": 1.0, "cont": 1.0}

# A candidate is feasible iff Φ ≤ this. Tunable; see the printed Φ distribution to
# choose. With the placeholder dynamic limits above this mostly reflects velocity/
# accel/torque overshoot, so treat it as provisional until the limits are measured.
FEASIBILITY_THRESHOLD = 5.0

# Initial state q0 (the observation the chunks were sampled from), in LeRobot real
# units [deg×5, gripper %], from generate_k_trajectories.py's JOINT_STATE. Ideally
# read from the actual logged observation; hardcoded here because the .pt doesn't
# carry it. q̇0 = 0 (assumed, per the proposal).
Q0_LEROBOT = [30.68, 10.95, -18.64, 87.47, -22.81, 2.05]

# Checkpoint whose action mean/std invert the normalization (the generator).
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"

URDF_PATH = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
COMMON_PATH = os.path.expanduser(
    "~/Desktop/PVD/so_arm_ws/src/so_arm_viz/so_arm_viz/common.py")
DEFAULT_PT = os.path.expanduser("~/Desktop/PVD/k_trajectories.pt")
DEFAULT_CSV = os.path.expanduser("~/Desktop/PVD/trajectory_scores.csv")

TERMS = ["pos", "vel", "acc", "torque", "cont"]


# ============================================================================
# Reused conversions + model
# ============================================================================
def load_common():
    """Import so_arm_viz/common.py by file path (it only needs stdlib `math`),
    so the scorer reuses the EXACT ghost/live conversions without sourcing ROS."""
    spec = importlib.util.spec_from_file_location("so_arm_common", COMMON_PATH)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common


def load_unnormalizer(policy_path: str):
    """Stage-1 unnormalizer (normalized → deg/percent). Config + processors only,
    no VLA weights, on CPU. Identical to playback_trajectories.load_unnormalizer."""
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401  registers choice
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    _, post = make_pre_post_processors(
        cfg, pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return post


def load_model_and_limits():
    """Build the Pinocchio model and per-joint q_max from the URDF.

    q_max[j] = max(|lower_j|, |upper_j|) to match the proposal's symmetric
    S_pos = Σ max(0, |a| − q_max). Two joints are slightly asymmetric (Elbow
    −100/+90°, Jaw −10/+100°); using the larger magnitude is the loosest faithful
    reading of that formula -- a per-side check is a future refinement.
    """
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    joint_names = [model.names[i] for i in range(1, model.njoints)]
    expected = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert joint_names == expected, f"Pinocchio joint order {joint_names} != {expected}"
    q_max = np.maximum(np.abs(model.lowerPositionLimit), np.abs(model.upperPositionLimit))
    return model, data, q_max, joint_names


# ============================================================================
# Scoring
# ============================================================================
def chunk_to_radians(common, chunk_deg: np.ndarray) -> np.ndarray:
    """[H,6] LeRobot real units (deg/percent) → [H,6] URDF radians (Pinocchio order)."""
    return np.array([common.lerobot_chunk_row_to_urdf(list(row)) for row in chunk_deg])


def score_candidate(A_rad, q0_rad, model, data, q_max):
    """Score one candidate. A_rad: [H,6] radian positions (a₁…a_H), q0_rad: [6].

    Returns (terms_dict, per-timestep arrays for optional inspection). Every term
    is a DIMENSIONLESS sum of fractional violations (violation / that joint's own
    limit), so the aggregate weights are unit-consistent:

        S_pos    = Σ_t Σ_j max(0, (|aₜⱼ|    − q_maxⱼ)  / q_maxⱼ)
        S_vel    = Σ_t Σ_j max(0, (|q̇ₜⱼ|   − q̇_maxⱼ) / q̇_maxⱼ)
        S_acc    = Σ_t Σ_j max(0, (|q̈ₜⱼ|   − q̈_maxⱼ) / q̈_maxⱼ)
        S_torque = Σ_t Σ_j max(0, (|τₜⱼ|    − τ_maxⱼ)  / τ_maxⱼ)   τ = RNEA(qₜ,q̇ₜ,q̈ₜ)
        S_cont   = ‖(a₁ − q₀)/q_max‖ + ‖(q̇₁ − q̇₀)/q̇_max‖         (q̇₀ = 0)
    """
    H = A_rad.shape[0]
    # positions prepended with q0: pos[0]=q0, pos[1..H]=a₁..a_H  → finite diffs
    pos = np.vstack([q0_rad, A_rad])           # [H+1, 6]
    vel = np.zeros_like(pos)                    # vel[0] = q̇₀ = 0 (assumed)
    vel[1:] = (pos[1:] - pos[:-1]) / DT         # q̇ₜ = (aₜ − aₜ₋₁)/Δt
    acc = np.zeros_like(pos)                     # acc[0] unused
    acc[1:] = (vel[1:] - vel[:-1]) / DT          # q̈ₜ = (q̇ₜ − q̇ₜ₋₁)/Δt

    a = pos[1:]      # [H,6]  positions aₜ, t=1..H
    v = vel[1:]
    ac = acc[1:]

    def relu_frac(mag, lim):
        return np.maximum(0.0, (mag - lim) / lim)

    s_pos = relu_frac(np.abs(a), q_max).sum()
    s_vel = relu_frac(np.abs(v), QDOT_MAX).sum()
    s_acc = relu_frac(np.abs(ac), QDDOT_MAX).sum()

    # torque: RNEA per timestep
    tau = np.array([pin.rnea(model, data, a[t], v[t], ac[t]) for t in range(H)])  # [H,6]
    s_tau = relu_frac(np.abs(tau), TAU_MAX).sum()

    # continuity at the chunk boundary (relative norms → dimensionless)
    cont_pos = np.linalg.norm((a[0] - q0_rad) / q_max)
    cont_vel = np.linalg.norm((v[0] - 0.0) / QDOT_MAX)
    s_cont = cont_pos + cont_vel

    return {"pos": s_pos, "vel": s_vel, "acc": s_acc, "torque": s_tau, "cont": s_cont}


def phi(terms):
    return sum(WEIGHTS[k] * terms[k] for k in TERMS)


# ============================================================================
# Report
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", default=DEFAULT_PT)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--policy-path", default=POLICY_PATH)
    ap.add_argument("--threshold", type=float, default=FEASIBILITY_THRESHOLD)
    args = ap.parse_args()

    common = load_common()
    model, data, q_max, joint_names = load_model_and_limits()
    print(f"[score] Pinocchio model: joints={joint_names}")
    print(f"[score] q_max (rad, from URDF)= {np.round(q_max,3)}")
    print(f"[score] PLACEHOLDER limits  q̇_max={QDOT_MAX[0]} rad/s  "
          f"q̈_max={QDDOT_MAX[0]} rad/s²  τ_max={TAU_MAX[0]} N·m  (CALIBRATE)")

    traj = torch.load(args.pt, map_location="cpu", weights_only=False)
    traj = torch.as_tensor(traj, dtype=torch.float32)
    if traj.ndim == 2:
        traj = traj.unsqueeze(0)
    K, H, _ = traj.shape
    print(f"[score] loaded {args.pt}: K={K} candidates × H={H} steps × 6 joints (normalized)")

    print(f"[score] STAGE 1 unnormalize via {args.policy_path} ...")
    post = load_unnormalizer(args.policy_path)

    # q0 (radians) from the observation state, reusing common.py (stage 2 only).
    q0_rad = np.array(common.lerobot_chunk_row_to_urdf(list(Q0_LEROBOT)))
    print(f"[score] q0 (rad) = {np.round(q0_rad,3)}   q̇0 = 0 (assumed)")

    rows = []
    for i in range(K):
        real_deg = post(traj[i:i + 1].clone()).reshape(H, 6).detach().cpu().numpy()  # stage1
        A_rad = chunk_to_radians(common, real_deg)                                    # stage2
        terms = score_candidate(A_rad, q0_rad, model, data, q_max)
        val = phi(terms)
        rows.append({"idx": i, **terms, "phi": val,
                     "feasible": bool(val <= args.threshold)})

    # ---- CSV (log only) ----------------------------------------------------
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "S_pos", "S_vel", "S_acc", "S_torque", "S_cont",
                    "Phi", "feasible"])
        for r in rows:
            w.writerow([r["idx"]] + [f"{r[k]:.6f}" for k in TERMS] +
                       [f"{r['phi']:.6f}", int(r["feasible"])])
    print(f"[score] wrote {args.csv}")

    # ---- Table (natural candidate order) -----------------------------------
    hdr = f"{'cand':>4} {'S_pos':>9} {'S_vel':>9} {'S_acc':>9} {'S_torque':>9} " \
          f"{'S_cont':>9} {'Phi':>10} {'feasible':>9}"
    print("\n=== PER-CANDIDATE SCORES (each S = Σ fractional violations, dimensionless) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['idx']:>4} {r['pos']:>9.3f} {r['vel']:>9.3f} {r['acc']:>9.3f} "
              f"{r['torque']:>9.3f} {r['cont']:>9.3f} {r['phi']:>10.3f} "
              f"{str(r['feasible']):>9}")

    # ---- Ranking (by Φ) ----------------------------------------------------
    ranked = sorted(rows, key=lambda r: r["phi"])
    print("\n=== RANKING (ascending Φ = most feasible first) ===")
    print(f"{'rank':>4} {'cand':>4} {'Phi':>10} {'feasible':>9}  dominant-term")
    for rank, r in enumerate(ranked):
        dom = max(TERMS, key=lambda k: WEIGHTS[k] * r[k])
        print(f"{rank:>4} {r['idx']:>4} {r['phi']:>10.3f} {str(r['feasible']):>9}  {dom}")

    # ---- Filter-then-prefer selection --------------------------------------
    survivors = [r for r in rows if r["feasible"]]  # feasibility filter
    print(f"\n=== SELECTION (filter-then-prefer, threshold Φ ≤ {args.threshold}) ===")
    print(f"survivors (feasible): {[r['idx'] for r in survivors]}")
    print("PREFERENCE ASSUMPTION: 'policy rank' = original candidate index "
          "(lower index = earlier sample); the scorer imposes NO preference beyond "
          "feasibility.")
    if survivors:
        chosen = min(survivors, key=lambda r: r["idx"])  # keep policy's first survivor
        print(f"CHOSEN candidate = {chosen['idx']}  (Φ={chosen['phi']:.3f})  "
              f"-- lowest index among feasible")
        print(f"\nTo replay it as a ghost (NOT auto-executed):")
        print(f"    ros2 launch so_arm_viz rollout_viz.launch.py        # terminal 1")
        print(f"    ./src/so_arm_viz/scripts/run_playback.sh --index {chosen['idx']}  # terminal 2")
    else:
        chosen = None
        print("CHOSEN candidate = NONE  (no candidate passes the feasibility "
              "threshold -- nothing selected; loosen --threshold or calibrate limits).")

    # ---- Summary / characterization ----------------------------------------
    n_feas = len(survivors)
    term_totals = {k: sum(r[k] for r in rows) for k in TERMS}
    dominant = max(TERMS, key=lambda k: term_totals[k])
    print("\n=== SUMMARY (characterization byproduct) ===")
    print(f"feasible: {n_feas}/{K}   ({100.0*n_feas/K:.0f}%)")
    print(f"Φ  min/median/max: {ranked[0]['phi']:.3f} / "
          f"{np.median([r['phi'] for r in rows]):.3f} / {ranked[-1]['phi']:.3f}")
    tot = sum(term_totals.values()) or 1.0
    print("violation mass by term (share of total Φ across all candidates):")
    for k in TERMS:
        print(f"    S_{k:<7} {term_totals[k]:>10.3f}   ({100.0*term_totals[k]/tot:>4.0f}%)")
    print(f"DOMINANT violation term overall: S_{dominant}")


if __name__ == "__main__":
    main()
