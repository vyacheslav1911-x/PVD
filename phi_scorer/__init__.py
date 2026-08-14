"""phi_scorer -- a clean, self-contained implementation of the Phi feasibility+quality
scorer for SO-101 action chunks.

    Phi = W_GATE * (S_pos + S_vel + S_env)  +  D_demo

Pipeline:
    1. python -m phi_scorer.measure           # -> artifacts/measured_constants.json
    2. python -m phi_scorer.build_reference    # -> artifacts/reference.npz + reference_features.csv
    3. from phi_scorer.phi import PhiScorer; PhiScorer().score_chunk(chunk, dt, prev)
"""
from .phi import PhiScorer  # noqa: F401
