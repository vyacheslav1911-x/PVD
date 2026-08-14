"""phi_scorer.reference -- the D_demo reference and the measured-constants loader.

D_demo answers "how unlike a human demonstration is this chunk?" as the Mahalanobis
distance of the chunk's feature vector to the distribution of the SAME features over
200 human teleop episodes. Mahalanobis (not Euclidean) because the features are
correlated: whitening by the demo covariance stops redundant features double-counting.

This module:
  load_constants()  -> the measured physical constants (or config defaults).
  build_reference() -> computes demo features, prunes near-collinear ones, fits
                       mu + Sigma^-1, sets the pass threshold, writes reference.npz
                       and reference_features.csv (LOG 1).
  load_reference()  -> reads reference.npz.
  mahalanobis()     -> D_demo for a feature dict.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Measured constants: measured_constants.json overrides config defaults.
# ---------------------------------------------------------------------------
def load_constants():
    """Return the physical constants dict, preferring measured values, else defaults.

    keys: bandwidth_hz[6], qd_max[6], pos_limit_rad[6], tau_stall, w_free  (+ meta)
    """
    defaults = {
        "bandwidth_hz": [config.DEFAULT_BANDWIDTH_HZ[j] for j in config.JOINTS],
        "qd_max":       [config.DEFAULT_QD_MAX[j] for j in config.JOINTS],
        # symmetric position magnitude limit per joint (rad). Default = URDF limits;
        # measure.py replaces this with the arm's own calibrated travel half-span.
        "pos_limit_rad": _urdf_pos_limits(),
        "tau_stall":    config.DEFAULT_TAU_STALL,
        "w_free":       config.DEFAULT_W_FREE,
        "source":       "config defaults (run measure.py for real values)",
    }
    if os.path.exists(config.CONSTANTS_PATH):
        with open(config.CONSTANTS_PATH) as f:
            measured = json.load(f)
        defaults.update(measured)
        defaults["source"] = measured.get("source", "measured_constants.json")
    else:
        print(f"[reference] {config.CONSTANTS_PATH} not found -> using config DEFAULTS. "
              f"Run measure.py for arm-specific values.")
    # numpy-ify the per-joint arrays
    for k in ("bandwidth_hz", "qd_max", "pos_limit_rad"):
        defaults[k] = np.asarray(defaults[k], dtype=float)
    return defaults


def _urdf_pos_limits():
    """Fallback position limits (rad) straight from the URDF, symmetric magnitude."""
    from .kinematics import Kinematics
    return list(Kinematics().q_abs_max)


# ---------------------------------------------------------------------------
# Build the demo reference from the raw teleop dataset.
# ---------------------------------------------------------------------------
def _demo_feature_matrix():
    """Compute the 6 features for every 50-frame window of every teleop episode.

    Returns a DataFrame with the 6 feature columns (rows = demo windows). Uses the
    SAME features.compute_features that scores candidates -> reference and candidates
    are measured on an identical footing.
    """
    from .kinematics import Kinematics
    from .features import compute_features
    kin = Kinematics()
    constants = load_constants()
    dt = config.DEFAULT_DT                                  # demos are the 30 Hz recordings

    rows = []
    parquets = sorted(glob.glob(os.path.join(config.DEMO_DATASET_DIR, "data", "**", "*.parquet"),
                                recursive=True))
    if not parquets:
        raise FileNotFoundError(f"No demo parquet files under {config.DEMO_DATASET_DIR}")
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, g in df.groupby("episode_index"):
            a = np.stack(g["action"].to_numpy()).astype(float)   # [T,6] LeRobot deg/%
            for s in range(0, len(a) - config.WINDOW + 1, config.WINDOW):
                prev = a[s - 1] if s > 0 else None
                feats = compute_features(a[s:s + config.WINDOW], dt, prev, kin, constants)
                rows.append(feats)
    return pd.DataFrame(rows, columns=config.FEATURES)


def build_reference(verbose=True):
    """Fit the demo reference and write reference.npz + reference_features.csv (LOG 1)."""
    os.makedirs(os.path.dirname(config.REFERENCE_PATH), exist_ok=True)
    demo = _demo_feature_matrix().dropna().reset_index(drop=True)
    if verbose:
        print(f"[reference] {len(demo)} demo windows x {config.N_JOINTS} joints")

    # LOG 1: the full per-window feature table, for inspection.
    demo.to_csv(config.REFERENCE_CSV, index=False)

    # --- prune near-collinear features (keeps Sigma well-conditioned) ---
    corr = demo[config.FEATURES].corr()
    kept = []
    for c in config.FEATURES:
        if any(abs(corr.loc[c, k]) > config.CORR_DROP for k in kept):
            partner = next(k for k in kept if abs(corr.loc[c, k]) > config.CORR_DROP)
            if verbose:
                print(f"[reference] DROP {c}: |r|={abs(corr.loc[c, partner]):.2f} with {partner}")
        else:
            kept.append(c)
    if verbose:
        print(f"[reference] kept features: {kept}")

    X = demo[kept].to_numpy()
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    inv = np.linalg.inv(cov + 1e-12 * np.eye(len(kept)))    # jitter for safety

    # D for every demo window -> pass threshold at the chosen percentile
    d = np.array([_maha(x, mu, inv) for x in X])
    threshold = float(np.percentile(d, config.PASS_PERCENTILE))
    if verbose:
        print(f"[reference] demo D: median {np.median(d):.2f}  "
              f"p{config.PASS_PERCENTILE:.0f} = {threshold:.2f}  <- PASS THRESHOLD")

    np.savez(config.REFERENCE_PATH, kept=np.array(kept), mu=mu, inv=inv,
             threshold=threshold, features=np.array(config.FEATURES))
    if verbose:
        print(f"[reference] wrote {config.REFERENCE_PATH} and {config.REFERENCE_CSV}")
    return dict(kept=kept, mu=mu, inv=inv, threshold=threshold)


def load_reference():
    if not os.path.exists(config.REFERENCE_PATH):
        raise FileNotFoundError(f"{config.REFERENCE_PATH} missing -- run build_reference.py first")
    z = np.load(config.REFERENCE_PATH, allow_pickle=True)
    return dict(kept=list(z["kept"]), mu=z["mu"], inv=z["inv"], threshold=float(z["threshold"]))


def _maha(x, mu, inv):
    d = np.asarray(x) - mu
    return float(np.sqrt(max(d @ inv @ d, 0.0)))


def mahalanobis(features_dict, ref):
    """D_demo for a feature dict, using only the kept features (in reference order)."""
    x = np.array([features_dict[k] for k in ref["kept"]], dtype=float)
    return _maha(x, ref["mu"], ref["inv"])
