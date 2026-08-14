"""phi_scorer.phi -- the top-level scorer.

    Phi(chunk) = W_GATE * (S_pos + S_vel + S_env)  +  D_demo

A hard gate violation (W_GATE large) outranks any amount of graded roughness; among
gate-clean chunks, D_demo ranks by how unlike human teleop the motion is.

Public API
----------
    scorer = PhiScorer()                       # loads URDF, constants, demo reference once
    result = scorer.score_chunk(chunk_lr, dt, prev_lr)   # -> dict, and appends LOG 2

`result` contains EVERY sub-quantity so nothing is hidden: the 6 features, the 3 gate
counts, D_demo, the pass threshold, feasibility, and Phi. The same dict is written as
one JSON line to artifacts/phi_chunks.jsonl (LOG 2) -- extract any S_pos / S_vel / R /
... straight from that file.
"""

import json
import os
import time

from . import config
from .kinematics import Kinematics
from .features import compute_features
from .gates import compute_gates
from .reference import load_constants, load_reference, mahalanobis


class PhiScorer:
    def __init__(self, log_path=config.CHUNK_LOG_PATH):
        self.kin = Kinematics()                    # URDF loaded once
        self.constants = load_constants()          # measured (or default) physical limits
        self.reference = load_reference()          # demo mu / Sigma^-1 / kept / threshold
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.step = 0

    def score_chunk(self, chunk_lr, dt=None, prev_lr=None):
        """Score one commanded chunk [T,6] (LeRobot deg/%); append a row to LOG 2."""
        dt = float(dt) if dt else config.DEFAULT_DT

        features = compute_features(chunk_lr, dt, prev_lr, self.kin, self.constants)
        gates = compute_gates(chunk_lr, dt, self.kin, self.constants)

        n_gate = int(gates["S_pos"] + gates["S_vel"] + gates["S_env"])
        d_demo = mahalanobis(features, self.reference)
        phi = config.W_GATE * n_gate + d_demo

        result = {
            "step": self.step,
            "wall_time": round(time.time(), 3),
            "dt": dt,
            "phi": phi,
            "n_gate": n_gate,
            "D_demo": d_demo,
            "threshold": self.reference["threshold"],
            "feasible": bool(n_gate == 0),                       # no hard violation
            "pass_demo": bool(d_demo <= self.reference["threshold"]),
            **{k: gates[k] for k in config.GATES},               # S_pos, S_vel, S_env
            **{k: features[k] for k in config.FEATURES},         # R, fracE_bw, ...
        }
        # LOG 2: one JSON line per scored chunk (line-buffered so it survives a crash).
        with open(self.log_path, "a", buffering=1) as fh:
            fh.write(json.dumps(result) + "\n")
        self.step += 1
        return result
