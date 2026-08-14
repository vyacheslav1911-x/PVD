#!/usr/bin/env python
"""phi_scorer.build_reference -- (re)build the demo reference for D_demo.

Runs offline over the 200 teleop episodes: computes the 6 features per 50-frame window
with the SAME features.compute_features used to score candidates, prunes near-collinear
features, fits mu + Sigma^-1, sets the pass threshold, and writes:
    artifacts/reference.npz            (mu, Sigma^-1, kept features, threshold)
    artifacts/reference_features.csv   (LOG 1: every demo window's features)

    python -m phi_scorer.build_reference
"""
from .reference import build_reference

if __name__ == "__main__":
    build_reference(verbose=True)
