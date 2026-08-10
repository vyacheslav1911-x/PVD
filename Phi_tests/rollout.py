"""Shared loader: recorded CSV -> uniformly sampled joint-space speed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Rollout:
    t: np.ndarray        # uniform time base, s
    speed: np.ndarray    # ||dq/dt||_2 over the joints, deg/s
    fs: float            # sampling rate of the uniform grid, Hz
    dt: float
    t_raw: np.ndarray    # original (irregular) timestamps, s from start
    df: pd.DataFrame     # the raw table, for chunk_start etc.
    n_raw: int


def load_speed(csv):
    """Resample logged positions onto a uniform grid and differentiate."""
    df = pd.read_csv(csv)
    t = df["t_obs"].to_numpy()
    t = t - t[0]
    q = df[[c for c in df.columns if c.startswith("pos_")]].to_numpy()

    dt = float(np.median(np.diff(t)))
    tu = np.arange(0.0, t[-1], dt)
    qu = np.column_stack([np.interp(tu, t, q[:, i]) for i in range(q.shape[1])])
    speed = np.linalg.norm(np.gradient(qu, dt, axis=0), axis=1)

    return Rollout(t=tu, speed=speed, fs=1.0 / dt, dt=dt, t_raw=t, df=df, n_raw=len(df))
