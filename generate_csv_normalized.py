#!/usr/bin/env python3
import torch
import csv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# --- CONFIG ---
IN_FILE = "k_trajectories.pt"
OUT_FILE = "k_trajectories.csv"
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# 1. Load normalized trajectories
traj = torch.load(IN_FILE)          # [K, T, 6]
K, T, J = traj.shape
print(f"Loaded {K} candidates x {T} timesteps x {J} joints (Normalized)")

# 2. Load policy config to get the postprocessor
print("Loading postprocessor to un-normalize actions...")
policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
_, postprocessor = make_pre_post_processors(policy.config, pretrained_path=POLICY_PATH)

# 3. Un-normalize back to physical degrees!
action_dict = {"action": traj}
unnormalized_dict = postprocessor(action_dict)
physical_traj = unnormalized_dict["action"]

# 4. Write to CSV
print("Writing physical units to CSV...")
with open(OUT_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate", "timestep"] + JOINT_NAMES)   # header
    for k in range(K):
        for t in range(T):
            # Using the new physical_traj instead of the raw traj
            row = [k, t] + [f"{physical_traj[k, t, j].item():.5f}" for j in range(J)]
            w.writerow(row)

print(f"Wrote {K*T} rows -> {OUT_FILE}")
print("open with: libreoffice, Excel, or pandas.read_csv('k_trajectories.csv')")
