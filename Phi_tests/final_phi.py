"""FINAL Phi -- an analytical feasibility + quality scorer for action chunks.

STRUCTURE

    Phi(chunk) = W_GATE * n_violations   +   D_demo

  GATES (hard, physical, absolute -- no demonstrations needed)
      S_pos   waypoint outside the calibrated mechanical range
      S_vel   |qdot| beyond the MEASURED per-joint limit (ramp_test.py)
      S_env   |tau| beyond what the motor can produce AT THAT SPEED,
              tau_avail = tau_stall (1 - |w|/w_free)   <- catches (tau,w) pairs
              that both box checks pass
    Any gate violation makes the chunk infeasible; W_GATE is large so that a
    single hard violation outranks any amount of graded roughness.

  D_demo (graded quality, in units of "how unlike a human demonstration")
      Mahalanobis distance of the chunk's feature vector to the distribution of
      the 200 teleop episodes. Mahalanobis is used specifically BECAUSE the
      candidate features are correlated: it whitens by the demo covariance, so
      redundant features stop double-counting on their own. Features whose
      demo correlation exceeds CORR_DROP are still pruned first, because near
      collinearity makes the covariance ill-conditioned.

WHY DEMO-REFERENCED. Every term measured in isolation only ranks one policy
against another. The demonstrations are human teleoperation on this exact arm,
so they define what the hardware demonstrably executes well, and they turn each
term into an absolute judgement. Pass threshold is the demo 95th percentile.

Terms deliberately EXCLUDED and why:
  S_torque box  never fires (peak 1.4 N.m vs a 3.0 limit) -- subsumed by S_env
  S_robust      ensemble worst case 1.9 N.m, still under the box: vacuous
  lambda_S      collapses under velocity control; anti-correlated with jerk
  S_cont/step   normalising the seam by the policy's own step size rewards
                large steps and ranked the two policies backwards
  self-collision convex-hull inflation produces false positives here
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds, measured_qd_max, position_limits  # noqa
from phi_kinematic import ARM_COLS, EE_FRAME, URDF  # noqa: E402
import phi_envelope as _env  # noqa: E402
_env.W_FREE = 5.40          # measured: ramp peak reached 5.31 rad/s, above the
                            # 4.712 datasheet no-load figure
tau_avail = _env.tau_avail

DSET = ("/home/g/.cache/huggingface/hub/datasets--polrolnik2--so101_candy/"
        "snapshots/6fc8a48159bb7416c962cd3c3ea9ce8f6e23c8fb")
CACHE = os.path.expanduser("~/Desktop/PVD/pvd_logs/demo_features_peak.csv")
CALIB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")
WIN, FS = 50, 30.0
W_GATE = 10.0
CORR_DROP = 0.90

CAND = ["R", "fracE_bw", "p99_v", "p99_a", "min_invk", "seam_v"]

model = pin.buildModelFromUrdf(URDF)
mdata = model.createData()
FID = model.getFrameId(EE_FRAME)
def measured_qd_peak():
    """PEAK measured speed per joint, from ramp_test.py.

    A gate asks "can the motor ever produce this", so it must use the peak, not
    the sustained plateau measured_qd_max() returns. Using the plateau made the
    gate fire on 1473 per-1000 demo windows -- i.e. it rejected human teleop that
    demonstrably executed -- which is a miscalibrated gate, not a bad demo.
    """
    import csv as _csv
    from ramp_test import analyse as _an
    out = []
    for j in JOINTS:
        rows = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]))
                for r in list(_csv.reader(open(
                    os.path.expanduser(f"~/Desktop/PVD/pvd_logs/ramp_{j}.csv"))))[1:]]
        _an(rows)
        out.append(max(_an.data[k]["peak"] for k in _an.data))
    return np.array(out)


QD = measured_qd_peak()
QMAX = position_limits(json.load(open(CALIB)))
_sm = json.load(open(os.path.expanduser("~/Desktop/PVD/pvd_logs/servo_models.json")))
FBW = {j: _sm[j]["rate"]["kp"] / (2 * np.pi) for j in JOINTS}


def chunk_features(blk, dt, prev=None):
    """Graded features + hard-gate violation counts for one chunk."""
    d1 = np.diff(blk, axis=0)
    m1 = np.abs(d1).mean(axis=0)
    m2 = np.abs(np.diff(d1, axis=0)).mean(axis=0)
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
    acc = np.diff(v, axis=0) / dt
    invk = []
    for q in blk[:: max(1, len(blk) // 12)]:
        pin.computeJointJacobians(model, mdata, q)
        pin.updateFramePlacements(model, mdata)
        J = pin.computeFrameJacobian(model, mdata, q, FID, pin.LOCAL_WORLD_ALIGNED)[:, :ARM_COLS]
        s = np.linalg.svd(J, compute_uv=False)
        invk.append(s.min() / s.max() if s.max() > 0 else 0.0)

    # seam: implied velocity of the jump into this chunk, as a fraction of limit
    if prev is None:
        seam = float(np.max(np.abs(d1[0]) / dt / QD))     # first real step
    else:
        seam = float(np.max(np.abs(blk[0] - prev) / dt / QD))

    feats = [R, fracE, float(np.percentile(np.abs(v), 99)),
             float(np.percentile(np.abs(acc), 99)), float(min(invk)), seam]

    # ---- hard gates ----
    n_pos = int((np.abs(blk) > QMAX).sum())
    n_vel = int((np.abs(v) > QD).sum())
    n_env = 0
    vv = np.vstack([v, v[-1]])
    aa = np.vstack([acc, acc[-1], acc[-1]])
    for t in range(len(blk)):
        tau = np.abs(pin.rnea(model, mdata, blk[t], vv[t], aa[t]))
        n_env += int((tau > tau_avail(vv[t])).any())
    return feats, dict(S_pos=n_pos, S_vel=n_vel, S_env=n_env)


def demo_features():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    rows = []
    for pq in sorted(glob.glob(os.path.join(DSET, "data", "**", "*.parquet"), recursive=True)):
        df = pd.read_parquet(pq)
        for ep, g in df.groupby("episode_index"):
            a = np.stack(g["action"].to_numpy()).astype(float) * SCALE
            for s in range(0, len(a) - WIN + 1, WIN):
                prev = a[s - 1] if s > 0 else None
                fe, gt = chunk_features(a[s:s + WIN], 1 / FS, prev)
                rows.append(fe + [gt["S_pos"], gt["S_vel"], gt["S_env"]])
    d = pd.DataFrame(rows, columns=CAND + ["S_pos", "S_vel", "S_env"]).dropna()
    d.to_csv(CACHE, index=False)
    return d


demo = demo_features()
print(f"DEMO REFERENCE: {len(demo)} windows x {WIN} frames from 200 teleop episodes\n")

# ---------------- redundancy pruning on the demo distribution ---------------
C = demo[CAND].corr()
print("feature correlation on the demos (this is what decides the term set):")
print("        " + "".join(f"{c[:8]:>10}" for c in CAND))
for a_ in CAND:
    print(f"{a_[:8]:<8}" + "".join(f"{C.loc[a_,b]:>10.2f}" for b in CAND))

keep = []
for c in CAND:
    if any(abs(C.loc[c, k]) > CORR_DROP for k in keep):
        drop_partner = [k for k in keep if abs(C.loc[c, k]) > CORR_DROP][0]
        print(f"\n  DROP {c}: |r| = {abs(C.loc[c,drop_partner]):.2f} with {drop_partner}")
    else:
        keep.append(c)
print(f"\nfeature set kept: {keep}")

mu = demo[keep].mean().to_numpy()
cov = np.cov(demo[keep].to_numpy().T)
inv = np.linalg.inv(cov + 1e-12 * np.eye(len(keep)))


def D(x):
    d = np.asarray(x) - mu
    return float(np.sqrt(max(d @ inv @ d, 0)))


demo_D = np.array([D(x) for x in demo[keep].to_numpy()])
THRESH = float(np.percentile(demo_D, 95))
print(f"demo D: median {np.median(demo_D):.2f}, p95 = {THRESH:.2f}  <- PASS THRESHOLD")
print(f"demo gate violations per window: S_pos {demo.S_pos.mean():.3f}  "
      f"S_vel {demo.S_vel.mean():.3f}  S_env {demo.S_env.mean():.3f}")

# --------------------------------- evaluate --------------------------------
print(f"\n\nFINAL PHI = {W_GATE}*n_gate_violations + D_demo\n")
print(f"{'run':<12}{'chunks':>7}{'S_pos':>7}{'S_vel':>7}{'S_env':>7}"
      f"{'D_demo':>9}{'PHI':>9}{'% feasible':>12}{'% pass D':>10}")

results = {}
demo_phi = W_GATE * (demo.S_pos + demo.S_vel + demo.S_env).to_numpy() + demo_D
print(f"{'DEMOS':<12}{len(demo):>7}{demo.S_pos.sum():>7.0f}{demo.S_vel.sum():>7.0f}"
      f"{demo.S_env.sum():>7.0f}{np.median(demo_D):>9.2f}{np.median(demo_phi):>9.2f}"
      f"{100*np.mean((demo.S_pos+demo.S_vel+demo.S_env)==0):>11.0f}%"
      f"{100*np.mean(demo_D<=THRESH):>9.0f}%")

for lab, path in RUNS:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
    dt = float(np.median(df["dt"].to_numpy(float)))
    bounds = chunk_bounds(df["chunk_start"].to_numpy())
    rows = []
    for k, s in enumerate(bounds):
        e = bounds[k + 1] if k + 1 < len(bounds) else len(df)
        if e - s < 8:
            continue
        fe, gt = chunk_features(a[s:e], dt, a[s - 1] if s > 0 else None)
        d_ = D([fe[CAND.index(c)] for c in keep])
        nv = gt["S_pos"] + gt["S_vel"] + gt["S_env"]
        rows.append(dict(**gt, D=d_, phi=W_GATE * nv + d_, nv=nv))
    t = pd.DataFrame(rows)
    results[lab] = t
    print(f"{lab:<12}{len(t):>7}{t.S_pos.sum():>7.0f}{t.S_vel.sum():>7.0f}{t.S_env.sum():>7.0f}"
          f"{t.D.median():>9.2f}{t.phi.median():>9.2f}"
          f"{100*np.mean(t.nv==0):>11.0f}%{100*np.mean(t.D<=THRESH):>9.0f}%")

print("\nPER-CHUNK PHI DISTRIBUTION")
print(f"{'run':<12}{'min':>9}{'p25':>9}{'median':>9}{'p75':>9}{'max':>9}")
for lab, t in results.items():
    q = np.percentile(t.phi, [0, 25, 50, 75, 100])
    print(f"{lab:<12}" + "".join(f"{x:>9.2f}" for x in q))
print(f"{'DEMOS':<12}" + "".join(f"{x:>9.2f}" for x in
                                 np.percentile(demo_phi, [0, 25, 50, 75, 100])))

from scipy.stats import mannwhitneyu  # noqa: E402
sm = pd.concat([results[k] for k in results if k.startswith("smolvla")])
pi = pd.concat([results[k] for k in results if k.startswith("pi0.5")])
u = mannwhitneyu(sm.phi, pi.phi, alternative="two-sided")
print(f"\nSEPARATION Phi: smolvla {sm.phi.median():.2f}  pi0.5 {pi.phi.median():.2f}  "
      f"AUC {u.statistic/(len(sm)*len(pi)):.3f}  p {u.pvalue:.3g}")
print(f"vs the demo envelope: smolvla {100*np.mean(sm.D>THRESH):.0f}% of chunks outside, "
      f"pi0.5 {100*np.mean(pi.D>THRESH):.0f}%")
