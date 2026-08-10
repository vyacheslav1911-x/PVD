#!/usr/bin/env python3
import torch
import csv

# 1. Import the Policy (this avoids the Draccus strict-parsing bug)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# 2. PolicyAction is just a Tensor alias — the postprocessor takes the action tensor directly
from lerobot.processor import PolicyAction  # noqa: F401  (kept for reference)

# --- CONFIG ---
IN_FILE = "k_trajectories.pt"
OUT_FILE = "k_trajectories.csv"
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Load normalized trajectories
traj = torch.load(IN_FILE)          # [K, T, 6]
K, T, J = traj.shape
print(f"Loaded {K} candidates x {T} timesteps x {J} joints (Normalized)")

# Load the policy to get the config and postprocessor safely
print("Loading policy and postprocessor (ignoring VLM log messages)...")
policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
_, postprocessor = make_pre_post_processors(policy.config, pretrained_path=POLICY_PATH)

# Un-normalize: the postprocessor takes the action tensor directly and returns a tensor
# (same call convention as rollout/inference/sync.py: `action = postprocessor(action)`).
print("Un-normalizing actions...")
physical_traj = postprocessor(traj)

# Write to CSV
print("Writing physical units to CSV...")
with open(OUT_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate", "timestep"] + JOINT_NAMES)   # header
    for k in range(K):
        for t in range(T):
            row = [k, t] + [f"{physical_traj[k, t, j].item():.5f}" for j in range(J)]
            w.writerow(row)

print(f"Wrote {K*T} rows -> {OUT_FILE}")
print("Open with: libreoffice, Excel, or pandas.read_csv('k_trajectories.csv')")

# --- Per-joint spread: max - min across ALL candidates & timesteps ---------
# "How far apart do the executed values get for each joint" (degrees for the arm
# joints; gripper in its own closure unit).
flat = physical_traj.reshape(-1, J)          # [K*T, 6]
jmin = flat.min(dim=0).values
jmax = flat.max(dim=0).values
spread = jmax - jmin                          # [6]

print("\nPer-joint spread (max - min over all candidates x timesteps):")
print(f"{'joint':<14}{'min':>10}{'max':>10}{'spread':>10}")
for j, name in enumerate(JOINT_NAMES):
    print(f"{name:<14}{jmin[j].item():>10.3f}{jmax[j].item():>10.3f}{spread[j].item():>10.3f}")

SPREAD_FILE = "k_trajectories_spread.csv"
with open(SPREAD_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["joint", "min", "max", "spread"])
    for j, name in enumerate(JOINT_NAMES):
        w.writerow([name, f"{jmin[j].item():.5f}", f"{jmax[j].item():.5f}", f"{spread[j].item():.5f}"])
print(f"\nWrote per-joint spread -> {SPREAD_FILE}")