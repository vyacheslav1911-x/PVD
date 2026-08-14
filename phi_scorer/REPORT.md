# phi_scorer — implementation report

A clean, library-style rebuild of the Φ feasibility + quality scorer for SO-101 action
chunks, replacing the scattered `Phi_tests/` prototype. This report captures the design,
the implementation, the verification, and the portability discussion.

---

## 1. What Φ is

```
Φ(chunk) = W_GATE · (S_pos + S_vel + S_env)   +   D_demo
           └────── hard gates (absolute) ──────┘     └ distance to human demos ┘
```

- A **hard-gate** violation makes a chunk physically infeasible; `W_GATE` (=10) is large
  so a single violation outranks any amount of graded roughness.
- Among gate-clean chunks, **`D_demo`** ranks by how *unlike human teleoperation* the
  motion is — the Mahalanobis distance of the chunk's feature vector to the distribution
  of the same features over 200 human demo episodes.

### The six graded features (→ D_demo)
| feature | meaning | needs |
|---|---|---|
| `R` | dither: mean\|Δ²a\|/mean\|Δa\| per joint (reversal rate), scale-invariant | — |
| `fracE_bw` | commanded AC energy above each joint's servo bandwidth | measured bandwidth |
| `p99_v` | 99th-pct commanded velocity (rad/s) | dt |
| `p99_a` | 99th-pct commanded acceleration (rad/s²) | dt |
| `min_invk` | worst (min) EE-Jacobian inverse condition number over the chunk | **URDF** |
| `seam_v` | chunk-boundary jump velocity ÷ measured q̇_max (max over joints) | measured q̇_max |

### The three hard gates (→ n_gate)
- `S_pos` — waypoint outside the calibrated mechanical travel.
- `S_vel` — \|commanded velocity\| beyond the measured per-joint q̇_max.
- `S_env` — \|RNEA torque\| beyond the **speed-derated** motor envelope
  `τ_avail(ω) = τ_stall·(1 − |ω|/ω_free)` (catches (torque, speed) pairs a static box misses). **URDF.**

### D_demo reference: "1515 demo windows"
The reference dataset (`polrolnik2/so101_candy`) is 200 human teleop episodes (84,865
frames @ 30 fps). Each episode is sliced into consecutive **non-overlapping 50-frame
windows** (≈1.67 s, the length of one policy chunk) → 1,593 windows; 78 near-motionless
ones (where `R = 0/0` is undefined) are dropped → **1,515**. Each becomes one point in the
6-D feature cloud; `D_demo` is the Mahalanobis distance to that cloud.

---

## 2. File structure (`phi_scorer/`)

| file | role |
|---|---|
| `config.py` | ONE source of truth: paths, joint order, unit scales, feature/gate names, `W_GATE`, physics-based **default** limits. Nothing *measured* is hardcoded here. |
| `spectral.py` | standalone FFT for `fracE_bw` (numpy-only, unit-testable): detrend + optional window + per-joint bandwidth split. |
| `kinematics.py` | loads the **URDF** (pinocchio); EE Jacobian → `min_invk`, RNEA torque for `S_env`. |
| `collision.py` | URDF self-collision model (excludes always-touching adjacent pairs) → `is_collision(q)`. |
| `features.py` | the 6 graded features + LeRobot-units→URDF-rad conversion. |
| `gates.py` | the 3 hard gates. |
| `reference.py` | D_demo: constants loader (measured-overrides-defaults) + `build_reference` (fit μ, Σ⁻¹, threshold) + `mahalanobis`. |
| `phi.py` | top-level `PhiScorer.score_chunk(chunk, dt, prev)` → result dict + LOG 2. |
| `measure.py` | measurement script: position limits from calibration (offline) + ramp/step on hardware for q̇_max/bandwidth → `measured_constants.json`. |
| `build_reference.py` | runs the demo-reference build. |
| `test_spectral.py` | unit tests pinning the `fracE_bw` conditioning (trend-leakage + Hann-smearing). |
| `README.md` | quick usage. |

**Generated artifacts** (`artifacts/`): `measured_constants.json`, `reference.npz`,
`reference_features.csv` (LOG 1), `phi_chunks.jsonl` (LOG 2).

---

## 3. Measurements Φ needs, and where they come from

| constant | feeds | source | measured how |
|---|---|---|---|
| position limits | `S_pos` | robot calibration file | offline (`measure.py`) |
| q̇_max per joint | `S_vel`, `seam_v` | ramp on the arm | hardware (`measure.py --port`) |
| servo bandwidth | `fracE_bw` | step response on the arm | hardware (`measure.py --port`) |
| ω_free, τ_stall | `S_env` | ramp peak + datasheet | hardware / datasheet |
| URDF (+ meshes) | `min_invk`, `S_env`, collision | model file | — |
| demo reference μ, Σ | `D_demo` | 200 teleop episodes | offline (`build_reference.py`) |

**Independence:** the folder generates its **own** artifacts and never reads the old
`Phi_tests/` files. `config.py` holds conventions + physics defaults; measured values live
in generated artifacts. Demo data is used because `D_demo` is demo-referenced *by
definition*; calibration is used because it's the robot's own config.

---

## 4. Two logs per run (as requested)

- **`reference_features.csv`** (LOG 1) — every demo window's 6 features (the reference
  distribution), 1,515 rows.
- **`phi_chunks.jsonl`** (LOG 2) — one JSON line per scored chunk with **every** feature
  *and* gate: `step, wall_time, dt, phi, n_gate, D_demo, threshold, feasible, pass_demo,
  S_pos, S_vel, S_env, R, fracE_bw, p99_v, p99_a, min_invk, seam_v`. Extract anything from
  this file.

---

## 5. Verification (offline, no hardware)

- `test_spectral.py` → **all pass** (trend-leakage: mean 0.24 → linear 0.00; Hann-smearing:
  none 0.04 → hann 0.13; motionless joint → 0).
- `measure.py` (offline) → real position limits from calibration; URDF loads (6 joints,
  EE `gripper`).
- `build_reference.py` → **1,515** demo windows, all 6 features kept, D_demo threshold
  **3.91**; wrote `reference.npz` + `reference_features.csv`.
- Scored 3 real SmolVLA chunks end-to-end → each got Φ / n_gate / D_demo / 3 gates / 6
  features, appended to `phi_chunks.jsonl`. (With *default* dynamics the SmolVLA chunks
  trip `S_vel` heavily — those counts will change once real q̇_max is measured on hardware.)

Pipeline: `python -m phi_scorer.measure` → `python -m phi_scorer.build_reference` →
`from phi_scorer.phi import PhiScorer; PhiScorer().score_chunk(chunk, dt, prev)`.
Run all commands from `~/Desktop/PVD` (so the package imports), not from inside the folder.

---

## 6. Portability — running Φ on a different arm (e.g. Waveshare)

**Arm-specific inputs:** URDF+meshes, calibration file, joint names/order, gripper map,
EE-frame name, measured q̇_max/bandwidth, and a human-demo dataset for `D_demo`.

- **Waveshare SO-101 kit (same hardware, same trained policy):** mechanically identical —
  URDF/joints/gripper/EE unchanged. Only: point `CALIB_PATH` at your calibration, run
  `measure.py --port …` for your servos, `build_reference.py` (reuse or your own demos).
  **No code changes.**
- **Genuinely different arm:** additionally supply its URDF+meshes, set `EE_FRAME`/joint
  names, and collect demos on that arm (`D_demo` is meaningless without them).

**User-friendly plan (not yet built):** refactor `config` globals into a `RobotProfile`
dataclass passed to `PhiScorer(profile)`, plus a `python -m phi_scorer.setup` wizard that
loads the URDF, checks calibration, runs the collision-safe `measure`, builds the
reference, and prints a ready/not-ready checklist.

---

## 7. Known limitations / open items

1. **`measure.py` sweep is not yet collision-safe.** The current ramp/step hardcode a
   blind `+45°` / `+20°` sweep in one direction (arbitrary defaults — *this is what
   collided the arm*). `collision.py` is done and verified (correctly flags "wrist into
   base"); the sweeps still need wiring to it (limit-aware direction toward mid-range +
   self-collision check before every step). **Do not run hardware `measure.py` until this
   is finished.**
2. **Gripper units are hardcoded to LeRobot 0–100 %.** A rad/deg/mm gripper would be
   mis-converted, corrupting `S_pos`/`S_vel`/`seam_v` on the gripper channel (`min_invk`,
   `R`, `fracE_bw` are immune). Fix: per-joint unit specs in the `RobotProfile`. See
   `docs/gripper_units.md`.
3. **Default dynamics are placeholders.** Until hardware `measure.py` runs, `S_vel`/`S_env`
   counts reflect defaults (q̇_max = 3 rad/s, bandwidth ≈ 1.27 Hz), not this arm.
4. **`D_demo` is not portable** — it encodes *this task on this arm*; any new arm/task needs
   its own demo set.

---

## 8. Next steps (proposed)

- [ ] Finish collision-safe, limit-aware `measure.py` (wire to `collision.py`).
- [ ] Refactor `config` → `RobotProfile` (incl. per-joint unit specs) + `setup` wizard.
- [ ] Run hardware `measure.py` on the real arm → real q̇_max/bandwidth → rebuild reference.
- [ ] Re-evaluate SmolVLA vs pi0.5 chunk Φ with the measured constants.
