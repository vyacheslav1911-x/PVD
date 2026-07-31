#!/usr/bin/env python3
"""
generate_k_trajectories.py
==========================
Generate K candidate action-chunk trajectories from a SmolVLA policy, from a
SINGLE static observation: two images on disk + a fixed joint state + the task.
No robot, no cameras. Pure offline inference.

WHY K TRAJECTORIES?
    SmolVLA is flow-matching: it denoises a chunk of random noise into a smooth
    trajectory. The ONLY randomness is the initial noise. Same observation +
    different noise => different (valid) trajectory. So for K candidates we draw
    K independent noises and denoise each against the same frozen observation.
    (Scoring/selection is a later step; this script only generates.)

OUTPUT: tensor [K, chunk_size, 6], still in NORMALIZED action space.
"""

import torch
import numpy as np
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"
DEVICE      = "cpu"          # "cuda" on the RTX box; "cpu" on the MX110 laptop
K           = 15

TOP_IMG   = "observation/top.png"
WRIST_IMG = "observation/wrist.png"
TASK      = "Put the black candy in the box"

# observation.state — current joint angles (deg for 5 body joints, 0-100 gripper)
JOINT_STATE = [
    30.68,   # shoulder_pan
    10.95,   # shoulder_lift
    -18.64,  # elbow_flex
    87.47,   # wrist_flex
    -22.81,  # wrist_roll
    2.05,    # gripper
]

# ----------------------------------------------------------------------------
# 1. Policy + preprocessor
# ----------------------------------------------------------------------------
policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
policy.eval().to(DEVICE)

# The preprocessor turns a raw obs dict into the fully-tokenized batch the model
# consumes — crucially it creates observation.language.tokens from the task
# string. Without it, _get_action_chunk KeyErrors on missing tokens.
preprocessor, _ = make_pre_post_processors(
    policy.config,
    pretrained_path=POLICY_PATH,
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)

CHUNK = policy.config.chunk_size       # 50 timesteps
ADIM  = policy.config.max_action_dim   # 32 padded width (draw noise at 32)
print(f"policy loaded: chunk_size={CHUNK}, padded action_dim={ADIM}")

# ----------------------------------------------------------------------------
# 2. Build one observation from the two images + joint state
# ----------------------------------------------------------------------------
def load_image(path: str) -> torch.Tensor:
    """PNG -> [1,3,480,640] float in [0,1] on DEVICE (channels-first)."""
    img = Image.open(path).convert("RGB").resize((640, 480))   # (W,H) for PIL
    arr = np.asarray(img, dtype=np.float32) / 255.0            # [480,640,3]
    t = torch.from_numpy(arr).permute(2, 0, 1)                 # [3,480,640]
    return t.unsqueeze(0).to(DEVICE)                           # [1,3,480,640]

state = torch.tensor(JOINT_STATE, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,6]

obs = {
    "observation.images.top":   load_image(TOP_IMG),
    "observation.images.wrist": load_image(WRIST_IMG),
    "observation.state":        state,
    "task": TASK,
}
print("Preprocessor...")
# ----------------------------------------------------------------------------
# 3. Preprocess (tokenizes task, fixes dtype/device)
# ----------------------------------------------------------------------------
batch = preprocessor(obs)

# ----------------------------------------------------------------------------
# 4. Generate K trajectories
# ----------------------------------------------------------------------------
trajectories = []
print("Calculating trajectories...")
with torch.no_grad():
    for k in range(K):
        noise = torch.randn(1, CHUNK, ADIM, device=DEVICE)      # [1,50,32] independent
        chunk = policy._get_action_chunk(batch, noise=noise)    # [1,50,6] unpadded
        trajectories.append(chunk.squeeze(0).cpu())             # [50,6]

trajectories = torch.stack(trajectories)   # [K,50,6]

# ----------------------------------------------------------------------------
# 5. Report + save
# ----------------------------------------------------------------------------
print("trajectories shape:", tuple(trajectories.shape))
print("spread across K at t=0 (per joint):",
      trajectories[:, 0, :].std(dim=0).numpy())   # nonzero => the K really differ
torch.save(trajectories, "k_trajectories.pt")
print("saved -> k_trajectories.pt  (normalized action space)")