# The Projection Operator — Making SmolVLA Candidate Trajectories Physically Feasible

*SO-101 arm · SmolVLA candy policy · kinematic/dynamic feasibility · `project_trajectories.py`*

---

## Abstract

A flow-matching policy such as SmolVLA emits *action chunks* — short horizons of future
joint targets — that are trained to reach the right **positions** but are under no
constraint to be **physically executable**: nothing in the training objective bounds the
implied joint velocities, accelerations, or torques, nor enforces continuity with the arm's
current state. When these raw chunks are scored against the SO-101's physical limits, the
aggregate feasibility penalty Φ is enormous (≈ 220–355 for the five candidates studied here),
and it is almost entirely (≈ 98 %) an **acceleration** violation.

This document describes the **projection operator**: the second of the two operators in the
research plan (the first being *selection*, which only scores and ranks). Where selection can
merely *reject*, projection *minimally modifies* each chunk so that it satisfies the physical
constraints while preserving the policy's task-space intent. We show — and this is the key
diagnostic result — that the dominant acceleration violation is **not** interior jitter but a
single **boundary spike**: the policy's first action lies ≈ 0.27 rad from the current joint
state, and reaching it from rest in one 30 fps step demands ≈ 240 rad/s². Low-pass smoothing
therefore *cannot* fix it (and empirically makes some joints worse). The correct operator is a
**bounded-acceleration command tracker** — a critically-damped reference smoother that starts
at the current pose at rest, chases the policy's target positions, and hard-clamps
acceleration, velocity, and joint limits at every step. It is feasible **by construction**.

Applied to the five candidates, the operator drives all *hard* limit terms
(position, velocity, acceleration, torque) to **exactly zero** and reduces Φ from 220–355 to
**0.33–0.49** (a > 99.8 % reduction), while tracking the original policy trajectory to within
≈ 0.036 rad (≈ 2°) RMS. The only residual is the irreducible *start-from-rest continuity*
cost. The output is written back in the identical normalized tensor format, so the existing
scoring and RViz-ghost pipeline runs on it unchanged.

---

## 1. Context and scope

Two operators act on the set of `K` candidate action chunks produced by sampling the policy
`K` times from one observation:

| Operator | What it does | This document |
|---|---|---|
| **Selection** (`score_trajectories.py`) | Scores each candidate for feasibility, ranks them, filter-then-prefer. **Read-only** — never modifies a trajectory. | prerequisite |
| **Projection** (`project_trajectories.py`) | **Modifies** each candidate minimally so it becomes feasible. | **this** |

Both share one feasibility model and one unnormalization path (below). Projection reuses the
scorer as a library (`score_candidate`, `phi`, `load_model_and_limits`, `load_unnormalizer`,
`chunk_to_radians`, and the limit/config constants) so the two operators can never disagree
about what "feasible" means.

**Input.** `k_trajectories.pt`, a tensor `[K, H, 6]` = (candidate, timestep, joint), joints in
LeRobot order `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`, in
**normalized action space** (MEAN_STD, values ≈ [−1, 1]). Here `K = 5`, `H = 50`, `Δt = 1/30 s`.

**Output.** `k_trajectories_projected.pt`, the same `[K, H, 6]` normalized format, plus a
`projection_report.csv` with the before/after scores.

---

## 2. The feasibility model (recap of the scorer)

For a candidate chunk `A = (a₁ … a_H)` the scorer prepends the measured/assumed initial state
`(q₀, q̇₀)` — currently the observation's joint state for `q₀` and `q̇₀ = 0` — and uses finite
differences (no dynamics model required for the kinematic layer):

```
q̇ₜ = (aₜ − aₜ₋₁) / Δt                      (velocity)
q̈ₜ = (q̇ₜ − q̇ₜ₋₁) / Δt                     (acceleration)
```

Each per-timestep, per-joint violation is normalized by that joint's own limit, so every term
is a **dimensionless** sum of fractional overshoots (this makes the aggregate weights
unit-consistent rather than task-tuned):

```
S_pos    = Σₜ Σⱼ max(0, (|aₜⱼ|  − q_maxⱼ)  / q_maxⱼ)
S_vel    = Σₜ Σⱼ max(0, (|q̇ₜⱼ| − q̇_maxⱼ) / q̇_maxⱼ)
S_acc    = Σₜ Σⱼ max(0, (|q̈ₜⱼ| − q̈_maxⱼ) / q̈_maxⱼ)
S_torque = Σₜ Σⱼ max(0, (|τₜⱼ|  − τ_maxⱼ)  / τ_maxⱼ),   τₜ = RNEA(qₜ, q̇ₜ, q̈ₜ)
S_cont   = ‖(a₁ − q₀)/q_max‖ + ‖(q̇₁ − q̇₀)/q̇_max‖
Φ        = w₁·S_pos + w₂·S_vel + w₃·S_acc + w₄·S_torque + w₅·S_cont,   all wᵢ = 1
```

- **q_max** (position limits) are read from the URDF (`so101_new_calib.urdf`) via the Pinocchio
  model, `q_maxⱼ = max(|lowerⱼ|, |upperⱼ|)`.
- **τₜ** is computed by Pinocchio's recursive Newton–Euler (`pin.rnea`) using the URDF's real
  per-link mass/inertia; the model's configuration order is
  `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`, which is exactly the order the
  radian vectors are produced in.
- **q̇_max, q̈_max, τ_max** are *placeholder* STS3215 limits (`3.0 rad/s`, `20.0 rad/s²`,
  `3.0 N·m`) in a clearly marked config block — to be replaced by *measured* limits.

`S_cont` is different from the other four: it has no limit to subtract, so it is *always ≥ 0*
and can be zero only for a trajectory that both starts at `q₀` **and** has zero initial
velocity. Keep this in mind — it is the residual after projection (§7–8).

---

## 3. Diagnosis: the acceleration violation is a *boundary* spike

For the raw candidates, `S_acc` carries ≈ 98 % of Φ. It is tempting to read this as
"the trajectories are noisy" and reach for a smoother. That is wrong. Consider the very first
acceleration, using `q̇₀ = 0`:

```
q̈₁ = (q̇₁ − q̇₀)/Δt = ( (a₁ − q₀)/Δt − 0 )/Δt = (a₁ − q₀) / Δt²
```

At 30 fps, `1/Δt² = 900`. For the elbow, `|a₁ − q₀| ≈ 0.265 rad`, giving
`q̈₁ ≈ 0.265 × 900 ≈ 238 rad/s²`. The measured per-joint peak acceleration of raw candidate 0's
elbow is **239.2 rad/s²** — i.e. the entire peak is that one step. The joints are *smooth in
position* (the elbow's whole excursion is only ≈ 0.35 rad) but the trajectory demands an
instantaneous jump from the current pose to the policy's first action.

In other words, most of `S_acc` is really the **discontinuity** between the assumed state
`(q₀, q̇₀ = 0)` and the chunk's first sample — the same information the (soft) `S_cont` term
also reports, amplified by the `1/Δt²` of a double finite difference.

### 3.1 Why low-pass smoothing fails

Smoothing the *interior* of `A` cannot move `a₁` toward `q₀` — the gap is at the boundary.
Worse, a wide Gaussian pulls `a₁` toward the trajectory's mean, which for some joints *increases*
the gap. Empirically, sweeping a Gaussian σ over candidate 0:

| σ (frames) | Pitch peak |q̈| | Elbow peak |q̈| |
|---:|---:|---:|
| 0 (raw) | 182.8 | 239.2 |
| 6 | 206.0 | 238.6 |
| 20 | **278.8** | **273.3** |

`S_acc` floors at ≈ 16–21 and then rises — smoothing never reaches feasibility. The problem is
structural, not spectral.

---

## 4. The projection operator: bounded-acceleration command tracking

The fix is to **re-time** the trajectory: track the policy's target positions with a virtual
point-mass servo that starts at `q₀` at rest and physically *cannot* exceed the limits. This is
a standard command/reference smoother (a.k.a. online trajectory generation / reference
governor). In discrete time, for each joint independently:

```
q ← q₀ ,  v ← 0
for t = 1 … H:
    a_cmd = clip( kp·(a*ₜ − q) − kd·v ,  −a_max , +a_max )   # acceleration-limited PD
    v     = clip( v + a_cmd·Δt        ,  −v_max , +v_max )    # velocity-limited
    q_prev = q
    q     = clip( q + v·Δt            ,   q_lo  ,  q_hi )     # position-limited
    v     = (q − q_prev) / Δt                                 # anti-windup
    aₜ ← q                                                    # projected action
```

where `a*ₜ` is the policy's target position at step `t` (in radians), `kp` is the tracking
stiffness, and `kd = 2√kp` gives **critical damping** (fast, no overshoot).

### 4.1 Feasibility by construction

Because the scorer measures `q̇` and `q̈` by the *same* finite differences on the produced
positions `aₜ = q`:

- **Velocity.** After anti-windup, `v = (q − q_prev)/Δt`, which is exactly the scorer's `q̇ₜ`,
  and it is clipped to `±v_max` ⟹ **S_vel = 0**.
- **Acceleration.** The scorer's `q̈ₜ = (q̇ₜ − q̇ₜ₋₁)/Δt`. Absent the velocity/position clamps
  this equals `a_cmd`, which is clipped to `±a_max`; the clamps can only *reduce* the step-to-step
  change in `v` ⟹ **S_acc = 0** (up to float32 round-trip noise ≈ 1e-5).
- **Position.** `q` is clipped to `[q_lo, q_hi]` ⟹ **S_pos = 0**.
- **Torque.** The projected motion is slower/smoother than the raw chunk, so the RNEA torques do
  not exceed the raw ones (which were already 0-violation for this light arm) ⟹ **S_torque = 0**.
- **Continuity.** The trajectory starts at `q₀` (`a₁ = q₀ + v₁Δt`, with `v₁` bounded), so the
  *position* half of `S_cont` ≈ 0; the *velocity* half is the small residual (§7).

The **anti-windup** line (recompute `v` from the realized, possibly clamped, position) prevents
the integrator from winding up against a joint limit and keeps the scorer's finite-difference
velocity consistent with the actual motion — so a position clamp cannot inject a phantom
acceleration spike.

### 4.2 The single tuning knob

`kp` sets the tracking bandwidth (`ωₙ = √kp`); `kd = 2√kp`. Higher `kp` tracks the policy more
tightly (lower position RMSE) but drives the servo to `a_max` sooner, so it carries slightly
more velocity at `t = 1` and thus a marginally larger `S_cont`. Default `kp = 300`
(`ωₙ ≈ 17.3 rad/s ≈ 2.75 Hz`), which gave RMSE ≈ 0.036 rad with `S_cont ≈ 0.4`. The sweep:

| `kp` | tracking RMSE (rad) | `S_cont` (= Φ) |
|---:|---:|---:|
| 50  | 0.066 | 0.210 |
| 150 | 0.044 | 0.400 |
| **300** | **0.036** | **0.491** |
| 600 | 0.032 | 0.523 |
| 1000| 0.032 | 0.523 |

All rows have `S_pos = S_vel = S_acc = S_torque = 0`; only the fidelity/continuity trade moves.

---

## 5. Units and data flow (why the output is a drop-in)

The physical limits live in **radians / rad·s⁻¹ / rad·s⁻²**, so the tracker runs in radian
space. The stored trajectories are **normalized**. The full normalized → radian map is
**affine per joint**, because it is the composition of two affine stages:

```
stage 1  (policy post-processor)  normalized → LeRobot units:  x·s + m     (MEAN_STD)
stage 2  (common.py)              LeRobot units → radians:
             body joints:  rad = SIGN·(deg·π/180) + OFFSET
             gripper:      rad = JAW_LOWER + (pct/100)·(JAW_UPPER − JAW_LOWER)
```

Composing: `rad = K·norm + B`, with `K = k·s`, `B = k·m + b`, where `(m, s)` are the action
mean/std recovered from the post-processor and `(k, b)` are the per-joint deg/pct→rad
slope/intercept from `common.py`. In practice `build_affine()` recovers these by **evaluating
the real pipeline at 0 and 1** (`m = post(0)`, `s = post(1) − post(0)`; `k, b` from `common` at
0 and 1). The round-trip `norm → rad → norm` matches the true pipeline to < 1e-4, so:

```
project:   targets_rad = K·norm + B     →   track()   →   proj_rad
save:      proj_norm = (proj_rad − B)/K      (stored as [K,H,6] normalized)
```

Because `rad2norm` is the exact inverse of the pipeline's `norm2rad`, when the **official
scorer** later unnormalizes `proj_norm` (post ∘ common) it recovers exactly `proj_rad` — the
projected file scores identically whether read by this tool or by `score_trajectories.py`.
The projection tool deliberately scores the *saved* tensor through `post ∘ common` (not through
its own affine shortcut) to guarantee this.

**Data-flow summary**

```
k_trajectories.pt ──(K·norm+B)──▶ targets_rad ──track(kp,kd,limits)──▶ proj_rad
        (normalized)                                                        │
                                                                    ((r−B)/K)
                                                                            ▼
                                       k_trajectories_projected.pt  (normalized, drop-in)
                                            │                │                │
                                     score_trajectories  play_interactive  run_playback
                                            (--pt)            (--pt)          (--pt --index)
```

---

## 6. Algorithm and complexity

For each of the `K` candidates: one affine map (`O(H·6)`), one forward tracker pass
(`O(H·6)`), and — only for the before/after report — two scoring passes (each `H` RNEA calls,
`O(H)` Pinocchio evaluations). Total `O(K·H)`; for `K = 5, H = 50` this is a fraction of a
second (dominated by loading the post-processor once). The tracker is **causal and
single-pass** — no optimization, no iteration to convergence — which is what makes feasibility
guaranteed and the cost trivial.

---

## 7. Results

Scored through the exact save format (`post ∘ common`), the projected file:

| cand | Φ before | Φ after | S_acc before | S_acc after | S_cont after | feasible |
|---:|---:|---:|---:|---:|---:|:--|
| 0 | 223.686 | **0.491** | 217.12 | 0.000 | 0.491 | ✔ all limits 0 |
| 1 | 246.668 | **0.412** | 240.54 | 0.000 | 0.412 | ✔ |
| 2 | 220.517 | **0.334** | 216.11 | 0.000 | 0.334 | ✔ |
| 3 | 355.491 | **0.468** | 348.08 | 0.000 | 0.468 | ✔ |
| 4 | 267.281 | **0.483** | 261.58 | 0.000 | 0.483 | ✔ |

- **`S_pos = S_vel = S_acc = S_torque = 0`** for all candidates (residuals ≤ 1e-4, float noise).
- **5/5 feasible** under the scorer's `Φ ≤ 5.0` threshold, versus **0/5** before.
- **Tracking RMSE ≈ 0.036 rad (≈ 2°)** — the feasible trajectory stays close to the policy's
  intended path.
- The characterization "dominant term" flips from `S_acc` (raw) to `S_cont` (projected), and
  the total violation mass across candidates is now **100 % `S_cont`** — see §8.

---

## 8. Interpretation: the residual `S_cont` and the threshold

**What the ≈ 0.4 residual is.** After projection the *position* half of `S_cont` is ≈ 0 (the
tracker starts at `q₀`). The residual is the *velocity* half: the servo leaves rest at up to
`a_max·Δt = 20 × 1/30 ≈ 0.67 rad/s`, so at `t = 1` the moving joints already carry velocity.
Normalized by `q̇_max = 3.0` that is ≈ 0.22 per joint, ≈ 0.4 across joints. It literally reads
"at the first step the arm is moving at ≈ 0.67 rad/s, but I *assumed* it started still."

**Why it is irreducible.** Any trajectory that actually leaves `q₀` must have non-zero velocity
at `t = 1`; `S_cont = 0` requires never moving. So ≈ 0.4 is a floor, not a defect. It went
*down* from the raw ≈ 2.7–4.1 — it only *looks* dominant because it is the sole surviving term.
It can be pushed lower with a **jerk-limited start** (ramp `a_cmd` up over a few frames so `q̇₁`
is tiny, at the cost of a slightly slower launch) or by feeding a **measured `q̇₀`** if the arm
is genuinely in motion.

**The `5.0` threshold.** `feasible ⇔ Φ ≤ 5.0` is a *policy choice*, a config knob
(`FEASIBILITY_THRESHOLD` / `--threshold`), not a physical constant. Φ is a sum of dimensionless
fractional overshoots, so "≤ 5" is a total limit-overshoot budget. The projected candidates
(Φ ≈ 0.4) clear it comfortably; the only thing the threshold must sit above is the irreducible
continuity floor (≈ 0.4), otherwise perfectly executable trajectories would be rejected merely
for starting to move. It could be tightened to, say, `1.0` and these still pass.

---

## 9. Assumptions and limitations

1. **Placeholder dynamic limits.** `v_max, a_max, τ_max` are rough STS3215 stand-ins. The
   projection uses the *same* constants as the scorer, so the two are internally consistent —
   but "feasible" is only as trustworthy as those numbers. They should be **measured** (drive the
   real servos, log peak velocity/accel/torque). Tighter limits → smoother but laggier tracking;
   looser → closer to raw.
2. **`q̇₀ = 0` assumed.** Correct for a standing start; if the arm is moving when the chunk is
   issued, feed the measured initial velocity or the first step will over/under-shoot.
3. **Torque is bounded only implicitly.** The loop clamps kinematics, not `τ`. It holds here
   because the SO-101 is light and the projected motion is slow (RNEA stays ≪ τ_max). A heavy
   payload could violate torque even at feasible kinematics; enforcing it would require RNEA
   *inside* the loop (a harder, coupled constraint).
4. **Per-joint decoupled clamps.** The tracker treats joints independently (kinematic limits are
   per-joint). Inertial coupling is only *checked* post-hoc via RNEA, not *enforced* during
   tracking.
5. **Fidelity ↔ feasibility trade.** With a tight `a_max`, the tracker lags fast policy sections.
   Here the motions are small so RMSE is ≈ 2°; a more aggressive policy chunk would be followed
   more loosely (that is the price of feasibility).
6. **Gripper as a revolute joint.** The gripper is mapped to the URDF `Jaw` angle and given the
   same rad/s² currency — a modeling convention, adequate for a feasibility heuristic.
7. **Not the L2-optimal projection.** This is a *causal forward* tracker: guaranteed feasible and
   close, but not the minimum-norm projection onto the feasible set (which would be a QP /
   optimal-control solve). It is chosen for being cheap, single-pass, and provably feasible.

---

## 10. Usage

Run in conda `lerobot_v6` (needs `torch + pinocchio + lerobot + numpy`; ROS not required to
*produce* the file):

```bash
# produce the feasible trajectories
/home/v1/miniconda3/envs/lerobot_v6/bin/python ~/Desktop/PVD/project_trajectories.py
#   options:  --pt <in.pt>  --out <out.pt>  --kp 300  [--kd <default 2√kp>]  --policy-path <ckpt>
```

Verify with the same scorer, then view the ghost (ROS Jazzy + workspace sourced):

```bash
# 1) confirm feasibility
python ~/Desktop/PVD/score_trajectories.py --pt ~/Desktop/PVD/k_trajectories_projected.pt

# 2) watch — terminal 1
ros2 launch so_arm_viz rollout_viz.launch.py

# 3) browse & compare — terminal 2 (shows Φ per candidate, plays chosen as a ghost)
./src/so_arm_viz/scripts/run_interactive.sh --pt ~/Desktop/PVD/k_trajectories_projected.pt
```

Run the interactive browser once on the raw file and once on the projected file: the projected
ghost visibly **starts at the current pose and eases into the motion** instead of snapping —
i.e. much less jagged. **Re-run the projection whenever you regenerate `k_trajectories.pt` or
change the limits.**

---

## 11. Future work

- **Jerk-limited start / end-at-rest.** Bound `da/dt` to shrink the `S_cont` residual and to
  finish smoothly at the chunk end.
- **Measured limits and measured `q̇₀`.** Replace the placeholders with logged servo data.
- **Torque-in-the-loop.** Fold an RNEA torque clamp into the tracker for heavy-payload regimes.
- **Optimal projection.** Replace the causal tracker with a QP / optimal-control solve that finds
  the minimum-deviation feasible trajectory (the true "projection onto the feasible set"),
  optionally preserving the *path* while only rescaling *timing* (path–velocity decomposition).

---

## Appendix — Files and symbols

**Files**
- `project_trajectories.py` — the operator: `build_affine()` (norm↔rad), `track()` (the tracker),
  `main()` (score before/after, save). Reuses `score_trajectories.py` as a library.
- `score_trajectories.py` — the feasibility scorer (selection). Single source of truth for Φ,
  the limits, `q₀`, and the two-stage unnormalization.
- `so_arm_ws/src/so_arm_viz/so_arm_viz/common.py` — the LeRobot-units↔URDF-radians conversion.
- `k_trajectories.pt` → `k_trajectories_projected.pt`; report in `projection_report.csv`.

**Symbols**
| symbol | meaning |
|---|---|
| `K, H` | number of candidates, chunk horizon (5, 50) |
| `Δt` | control period, `1/30 s` |
| `q₀, q̇₀` | assumed initial joint position / velocity (`q̇₀ = 0`) |
| `a*ₜ` | policy target position at step `t` (radians) |
| `q, v, a_cmd` | tracker state position / velocity / commanded accel |
| `q_lo, q_hi, q_max` | joint position lower/upper limit, and `max(|lo|,|hi|)` |
| `v_max, a_max, τ_max` | velocity / acceleration / torque limits (placeholder) |
| `kp, kd` | tracker stiffness, damping (`kd = 2√kp`) |
| `K, B` | per-joint affine `rad = K·norm + B` |
| `Sₓ, Φ` | per-term feasibility penalties and their weighted sum |
