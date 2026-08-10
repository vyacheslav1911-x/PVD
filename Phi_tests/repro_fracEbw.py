"""Reproduce the published frac_E_bw figures (demos 7.7 / pi0.5 6.2 / SmolVLA 30-36)
using the EXACT formula from final_phi.chunk_features (lines 104-111), extracted
verbatim -- no second FFT. Proven faithful by reproducing the demo cache column.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
D2R = np.pi / 180
GRIP = (1.74533 + 0.174533) / 100
SCALE = np.array([D2R] * 5 + [GRIP])
WIN, FS = 50, 30.0
REPO = os.path.expanduser("~/Desktop/PVD")
DSET = ("/home/g/.cache/huggingface/hub/datasets--polrolnik2--so101_candy/"
        "snapshots/6fc8a48159bb7416c962cd3c3ea9ce8f6e23c8fb")
CACHE = os.path.join(REPO, "pvd_logs/demo_features_peak.csv")
_sm = json.load(open(os.path.join(REPO, "pvd_logs/servo_models.json")))
FBW = np.array([_sm[j]["rate"]["kp"] / (2 * np.pi) for j in JOINTS])
RUNS = [
    ("smolvla r1", os.path.join(REPO, "Phi_tests/recorded trajectories/record_run1.csv")),
    ("smolvla r2", os.path.join(REPO, "pvd_logs/record_run2.csv")),
    ("pi0.5   r1", os.path.join(REPO, "Phi_tests/recorded trajectories/record_pi05_run1.csv")),
]


def fracE_bw_perwindow(blk, dt):
    """VERBATIM from final_phi.chunk_features: mean-detrend + Hann, hard per-joint
    cutoff, plain mean across joints."""
    n = len(blk)
    f = np.fft.rfftfreq(n, dt)
    fr = []
    for i in range(len(JOINTS)):
        x = blk[:, i] - blk[:, i].mean()
        P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        if P.sum() > 0:
            fr.append(P[f > FBW[i]].sum() / P.sum())
    return float(np.mean(fr)) if fr else np.nan


def chunk_bounds(chunk_start):
    idx = np.where(chunk_start == 1)[0]
    return [i for k, i in enumerate(idx) if k == 0 or i - idx[k - 1] > 1]


# ---- 1. recompute demo windows, prove we match the cache exactly --------------
vals = []
for pq in sorted(glob.glob(os.path.join(DSET, "data", "**", "*.parquet"), recursive=True)):
    df = pd.read_parquet(pq)
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(float) * SCALE
        for s in range(0, len(a) - WIN + 1, WIN):
            vals.append(fracE_bw_perwindow(a[s:s + WIN], 1 / FS))
vals = np.array(vals)
cache = pd.read_csv(CACHE)["fracE_bw"].to_numpy()
print(f"FIDELITY vs cache: recomputed {len(vals)} windows, cache {len(cache)}")
if len(vals) == len(cache):
    print(f"  max abs diff per-window = {np.max(np.abs(vals - cache)):.2e}  "
          f"(0 => identical FFT)  mean {vals.mean():.4f} vs cache {cache.mean():.4f}")

print("\n=== REPRODUCTION (current conditioning: mean-detrend + Hann, hard per-joint cutoff) ===")
print(f"{'source':<14}{'windows/chunks':>16}{'mean frac_E_bw':>16}{'median':>10}{'published':>12}")
print(f"{'demos':<14}{len(vals):>16}{vals.mean()*100:>15.1f}%{np.median(vals)*100:>9.1f}%{'7.7%':>12}")

# ---- 2. policy runs, per-run and per-source -----------------------------------
by_src = {"smolvla": [], "pi0.5": []}
for lab, path in RUNS:
    df = pd.read_csv(path)
    a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
    dt = float(np.median(df["dt"].to_numpy(float)))
    b = chunk_bounds(df["chunk_start"].to_numpy())
    fe = []
    for k, s in enumerate(b):
        e = b[k + 1] if k + 1 < len(b) else len(df)
        if e - s < 8:
            continue
        fe.append(fracE_bw_perwindow(a[s:e], dt))
    fe = np.array(fe)
    src = "smolvla" if lab.startswith("smolvla") else "pi0.5"
    by_src[src].append(fe)
    pub = "30-36%" if src == "smolvla" else "6.2%"
    print(f"{lab:<14}{len(fe):>16}{fe.mean()*100:>15.1f}%{np.median(fe)*100:>9.1f}%{pub:>12}")
for src in ("smolvla", "pi0.5"):
    allf = np.concatenate(by_src[src])
    print(f"{'  '+src+' (all)':<14}{len(allf):>16}{allf.mean()*100:>15.1f}%{np.median(allf)*100:>9.1f}%")
