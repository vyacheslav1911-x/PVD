# phi_scorer

A clean, self-contained rebuild of the Φ feasibility + quality scorer for SO-101 action chunks.

```
Φ(chunk) = W_GATE · (S_pos + S_vel + S_env)   +   D_demo
           └── hard gates (absolute) ──┘        └ Mahalanobis distance to human demos ┘
```

A single hard-gate violation makes a chunk infeasible (`W_GATE` large); among gate-clean
chunks, `D_demo` ranks by how *unlike human teleoperation* the motion is.

## The six graded features (→ D_demo)
| feature | meaning | needs |
|---|---|---|
| `R` | dither: mean\|Δ²a\|/mean\|Δa\| per joint (reversal rate), scale-invariant | — |
| `fracE_bw` | commanded energy above the servo bandwidth (`spectral.py`) | measured bandwidth |
| `p99_v` | 99th-pct commanded velocity | dt |
| `p99_a` | 99th-pct commanded acceleration | dt |
| `min_invk` | worst EE-Jacobian inverse condition number over the chunk | **URDF** |
| `seam_v` | chunk-boundary jump velocity ÷ q̇_max | measured q̇_max |

## The three hard gates (→ n_gate)
`S_pos` (calibrated travel), `S_vel` (measured q̇_max), `S_env` (RNEA torque vs the
speed-derated motor envelope τ_avail(ω) = τ_stall·(1−|ω|/ω_free), **URDF**).

## Independence
The folder generates its **own** artifacts and never reads the old `Phi_tests/` files.
Config holds only conventions + physics-based defaults; measured values live in
`artifacts/measured_constants.json`, the demo reference in `artifacts/reference.npz`.

## Pipeline
```bash
# 1. measure the arm's constants (offline = calibration limits + defaults; add --port for hardware)
python -m phi_scorer.measure                       # or: --port /dev/serial/by-id/... --id my_follower

# 2. build the demo reference (μ, Σ, threshold) from the 200 teleop episodes
python -m phi_scorer.build_reference

# 3. score chunks
python - <<'PY'
from phi_scorer.phi import PhiScorer
import numpy as np
s = PhiScorer()
chunk = np.random.randn(50, 6)      # [T,6] commanded, LeRobot deg/% units
print(s.score_chunk(chunk, dt=1/30))
PY
```

## Two logs per run (artifacts/)
- **`reference_features.csv`** — every demo window's 6 features (the reference distribution).
- **`phi_chunks.jsonl`** — one line per scored chunk with **every** feature *and* gate
  (`S_pos`,`S_vel`,`S_env`), `D_demo`, threshold, feasibility, and `Φ`. Extract anything from here.

## Files
`config.py` conventions/defaults · `spectral.py` FFT (unit-tested) · `kinematics.py` URDF/Jacobian/RNEA ·
`features.py` 6 features · `gates.py` 3 gates · `reference.py` D_demo + constants loader ·
`phi.py` top-level scorer · `measure.py` measurement script · `build_reference.py` reference builder ·
`test_spectral.py` spectral unit tests.
