"""Section 8 -- distributional / OOD scoring against the demonstrations.

The demonstrations are human teleoperation on this exact robot, so they are the
ground truth for "a trajectory this arm executes well". Every term measured so
far can be given a REFERENCE VALUE by evaluating it on the demos, which converts
each term from a relative comparison between two policies into an absolute
judgement: is this chunk inside or outside the envelope of things that demonstrably
work on the real hardware?

That is exactly the stated application -- score a policy's trajectories without
running them -- so this is the term that makes the others actionable.

Features per 50-frame window (matching the policies' chunk length):
    R                 dither ratio
    frac E > f_bw     command energy above the measured servo bandwidth
    p99 |qdot|        peak commanded speed
    p99 |qddot|       peak commanded acceleration
    min 1/kappa       closest approach to a singularity
Mahalanobis distance of each policy chunk to the demo distribution follows.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds  # noqa: E402
from phi_kinematic import ARM_COLS, EE_FRAME, URDF  # noqa: E402

DSET = ("/home/g/.cache/huggingface/hub/datasets--polrolnik2--so101_candy/"
        "snapshots/6fc8a48159bb7416c962cd3c3ea9ce8f6e23c8fb")
WIN = 50
FS = 30.0
FEATS = ["R", "fracE_bw", "p99_v", "p99_a", "min_invk"]

model = pin.buildModelFromUrdf(URDF)
mdata = model.createData()
FID = model.getFrameId(EE_FRAME)
FBW = json.load(open(os.path.expanduser("~/Desktop/PVD/pvd_logs/servo_models.json")))
FBW = {j: FBW[j]["rate"]["kp"] / (2 * np.pi) for j in JOINTS}


def features(blk, dt):
    """blk: [T,6] in URDF radians."""
    d1 = np.diff(blk, axis=0)
    d2 = np.diff(d1, axis=0)
    m1 = np.abs(d1).mean(axis=0)
    m2 = np.abs(d2).mean(axis=0)
    ok = m1 > 1e-12
    R = float(np.mean(m2[ok] / m1[ok])) if ok.any() else np.nan

    n = len(blk)
    f = np.fft.rfftfreq(n, dt)
    fr = []
    for i, j in enumerate(JOINTS):
        x = blk[:, i] - blk[:, i].mean()
        P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        if P.sum() > 0:
            fr.append(P[f > FBW[j]].sum() / P.sum())
    fracE = float(np.mean(fr)) if fr else np.nan

    v = d1 / dt
    a = np.diff(v, axis=0) / dt
    invk = []
    for q in blk[:: max(1, len(blk) // 12)]:
        pin.computeJointJacobians(model, mdata, q)
        pin.updateFramePlacements(model, mdata)
        J = pin.computeFrameJacobian(model, mdata, q, FID, pin.LOCAL_WORLD_ALIGNED)[:, :ARM_COLS]
        s = np.linalg.svd(J, compute_uv=False)
        invk.append(s.min() / s.max() if s.max() > 0 else 0.0)
    return [R, fracE, np.percentile(np.abs(v), 99), np.percentile(np.abs(a), 99), min(invk)]


# ------------------------------------------------------------ demonstrations
rows = []
for pq in sorted(glob.glob(os.path.join(DSET, "data", "**", "*.parquet"), recursive=True)):
    df = pd.read_parquet(pq)
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(float) * SCALE
        for s in range(0, len(a) - WIN + 1, WIN):
            rows.append(features(a[s:s + WIN], 1 / FS))
demo = pd.DataFrame(rows, columns=FEATS).dropna()
print(f"DEMONSTRATIONS: {len(demo)} windows of {WIN} frames from "
      f"{os.path.basename(DSET)[:12]}... ({FS:.0f} fps)")
print(f"\n{'feature':<12}{'demo mean':>12}{'demo sd':>10}{'p5':>10}{'p50':>10}{'p95':>10}")
for c in FEATS:
    q = np.percentile(demo[c], [5, 50, 95])
    print(f"{c:<12}{demo[c].mean():>12.4f}{demo[c].std():>10.4f}"
          f"{q[0]:>10.4f}{q[1]:>10.4f}{q[2]:>10.4f}")

mu = demo[FEATS].mean().to_numpy()
cov = np.cov(demo[FEATS].to_numpy().T) + 1e-9 * np.eye(len(FEATS))
inv = np.linalg.inv(cov)


def maha(x):
    d = x - mu
    return float(np.sqrt(max(d @ inv @ d, 0)))


# --------------------------------------------------------------- policies
print(f"\n\nPOLICY CHUNKS vs THE DEMONSTRATION ENVELOPE")
print(f"{'run':<12}{'chunks':>7}" + "".join(f"{c:>12}" for c in FEATS) + f"{'Mahalanobis':>13}")
demo_maha = np.array([maha(x) for x in demo[FEATS].to_numpy()])
print(f"{'DEMOS':<12}{len(demo):>7}" + "".join(f"{demo[c].mean():>12.4f}" for c in FEATS)
      + f"{demo_maha.mean():>13.2f}")
store = {}
for lab, path in RUNS:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
    dt = float(np.median(df["dt"].to_numpy(float)))
    bounds = chunk_bounds(df["chunk_start"].to_numpy())
    fs = []
    for k, s in enumerate(bounds):
        e = bounds[k + 1] if k + 1 < len(bounds) else len(df)
        if e - s >= 8:
            fs.append(features(a[s:e], dt))
    t = pd.DataFrame(fs, columns=FEATS).dropna()
    t["maha"] = [maha(x) for x in t[FEATS].to_numpy()]
    store[lab] = t
    print(f"{lab:<12}{len(t):>7}" + "".join(f"{t[c].mean():>12.4f}" for c in FEATS)
          + f"{t['maha'].mean():>13.2f}")

print(f"\ndemo Mahalanobis p95 = {np.percentile(demo_maha,95):.2f}  "
      f"(the 'inside the envelope' threshold)")
for lab, t in store.items():
    frac = 100 * np.mean(t["maha"] > np.percentile(demo_maha, 95))
    print(f"  {lab:<12} {frac:5.0f}% of chunks outside the demo 95th-percentile envelope"
          f"   (mean D = {t['maha'].mean():.1f})")

print("\nPER-FEATURE: how many demo sd's is each policy from the demo mean?")
print(f"{'run':<12}" + "".join(f"{c:>12}" for c in FEATS))
for lab, t in store.items():
    print(f"{lab:<12}" + "".join(
        f"{(t[c].mean()-demo[c].mean())/demo[c].std():>+12.1f}" for c in FEATS))
