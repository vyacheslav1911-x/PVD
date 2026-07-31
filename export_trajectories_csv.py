#!/usr/bin/env python3
"""
export_trajectories_csv.py
==========================
Load the K trajectories saved by generate_k_trajectories.py and write them to a
human-readable CSV you can open in Excel / pandas.

The saved tensor is [K, T, 6]  (K candidates, T timesteps, 6 joints).
We flatten it to a long table: one row per (candidate, timestep), with the 6
joint values as columns. Long format is the friendliest for filtering/plotting.
"""

import torch
import csv

IN_FILE  = "k_trajectories.pt"
OUT_FILE = "k_trajectories.csv"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

traj = torch.load(IN_FILE)          # [K, T, 6]
K, T, J = traj.shape
print(f"loaded {K} candidates x {T} timesteps x {J} joints")

with open(OUT_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate", "timestep"] + JOINT_NAMES)   # header
    for k in range(K):
        for t in range(T):
            row = [k, t] + [f"{traj[k, t, j].item():.5f}" for j in range(J)]
            w.writerow(row)

print(f"wrote {K*T} rows -> {OUT_FILE}")
print("open with: libreoffice, Excel, or  pandas.read_csv('k_trajectories.csv')")
