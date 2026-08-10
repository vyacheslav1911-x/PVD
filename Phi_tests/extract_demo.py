"""Pull one episode out of the so101_candy dataset and save it replayable.

Writes a CSV in the same column layout as record_rollout.py's logs, so it can be
played back by execute_toppra.py --no-retime without any special-casing:

    t_obs, dt, cmd_<joint>, pos_<joint>, sent_<joint>, chunk_start

  cmd_ = the dataset's `action`            (leader arm / teleop command)
  pos_ = the dataset's `observation.state` (what the follower achieved)
  sent_ = cmd_ clipped to the calibrated range

The clip matters: the leader can command past the follower's mechanical range
(episode 160 does, on shoulder_lift, by 0.92 deg over 42 frames), and the real
follower simply never got there. Replaying the raw leader command would drive
the joint into its hard stop, so sent_ is clipped to the calibrated range minus
a small keep-out and is the column meant for playback.

RUN:
  python extract_demo.py --episode 160
  python extract_demo.py --episode 160 --out /path/to/file.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DSET = ("/home/g/.cache/huggingface/hub/datasets--polrolnik2--so101_candy/"
        "snapshots/6fc8a48159bb7416c962cd3c3ea9ce8f6e23c8fb")
CALIB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30.0


def band(cal, j, keepout):
    lo, hi = cal[j]["range_min"], cal[j]["range_max"]
    mid = (lo + hi) / 2
    if j == "gripper":
        return 1.0, 99.0
    k = keepout * 4095 / 360
    return ((lo + k) - mid) * 360 / 4095, ((hi - k) - mid) * 360 / 4095


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=160)
    ap.add_argument("--keepout", type=float, default=1.0, help="deg clear of each hard stop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cal = json.load(open(CALIB))
    g = None
    for pq in sorted(glob.glob(os.path.join(DSET, "data", "**", "*.parquet"), recursive=True)):
        d = pd.read_parquet(pq)
        if args.episode in d["episode_index"].unique():
            g = d[d["episode_index"] == args.episode].sort_values("frame_index")
            src = os.path.basename(pq)
            break
    if g is None:
        raise SystemExit(f"episode {args.episode} not found")

    act = np.stack(g["action"].to_numpy()).astype(float)
    st = np.stack(g["observation.state"].to_numpy()).astype(float)
    t = g["timestamp"].to_numpy().astype(float)
    t = t - t[0]

    sent = act.copy()
    clipped = {}
    for i, j in enumerate(JOINTS):
        lo, hi = band(cal, j, args.keepout)
        n = int(((sent[:, i] < lo) | (sent[:, i] > hi)).sum())
        if n:
            clipped[j] = n
        sent[:, i] = np.clip(sent[:, i], lo, hi)

    out = pd.DataFrame({"t_obs": t, "dt": np.r_[1 / FPS, np.diff(t)]})
    for i, j in enumerate(JOINTS):
        out[f"cmd_{j}"] = act[:, i]
        out[f"pos_{j}"] = st[:, i]
        out[f"sent_{j}"] = sent[:, i]
    out["chunk_start"] = 0
    out.loc[0, "chunk_start"] = 1

    path = args.out or os.path.expanduser(
        f"~/Desktop/PVD/pvd_logs/demo_ep{args.episode:03d}.csv")
    out.to_csv(path, index=False)

    print(f"episode {args.episode} from {src}")
    print(f"  {len(out)} frames, {t[-1]:.2f} s at {FPS:.0f} fps")
    print(f"  clipped to calibrated range minus {args.keepout} deg keep-out: "
          f"{clipped if clipped else 'nothing needed'}")
    print(f"  start pose (deg): {np.round(sent[0], 2)}")
    print(f"  wrote {path}")
    print(f"\nreplay with:\n  python Phi_tests/execute_toppra.py --src {path} "
          f"--prefix sent --no-retime --skip 0 --out ~/Desktop/PVD/pvd_logs/"
          f"exec_demo{args.episode:03d}.csv")


if __name__ == "__main__":
    main()
