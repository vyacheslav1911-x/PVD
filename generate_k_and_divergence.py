#!/usr/bin/env python3
"""
generate_k_and_divergence.py
============================
Sweep K = number of candidate action-chunk trajectories sampled from a SmolVLA
policy under ONE static observation, and for each K measure the per-joint
DIVERGENCE (spread = max - min over all candidates x timesteps, in physical
units).

For every K the per-joint block is appended to `Research_notes_divergence.txt`
in the same format as `Research_notes.txt`, and finally a scatter plot of the
average divergence across the 6 joints vs K is written to `divergence_vs_k.png`.

The heavy VLM prefix is extracted ONCE and reused for every trajectory (the
cheap-sampling property), so the whole sweep is a few minutes on a GPU.
"""

import csv
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless — write a PNG, no display needed
import matplotlib.pyplot as plt
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import (
    make_att_2d_masks,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    SmolVLAPolicy,
)
from lerobot.policies.factory import make_pre_post_processors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"
DEVICE      = "cuda"

# K sweep: 5,10,15,...,80 (step 5), then 100,120,140,...,300 (step 20)
K_LIST = list(range(5, 81, 5)) + list(range(100, 301, 20))

SEED       = 0                              # reproducible sweep
NOTES_FILE = "Research_notes_divergence.txt"
PLOT_FILE  = "divergence_vs_k.png"

TOP_IMG   = "observation/top.png"
WRIST_IMG = "observation/wrist.png"
TASK      = "Put the black candy in the box"

# observation.state — current joint angles (deg for 5 body joints, 0-100 gripper)
JOINT_STATE = [30.68, 10.95, -18.64, 87.47, -22.81, 2.05]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
J = len(JOINT_NAMES)

# ----------------------------------------------------------------------------
# 1. Load Policy & BOTH Processors (ONCE)
# ----------------------------------------------------------------------------
print("Loading Policy and Processors...")
policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
policy.eval().to(DEVICE)

preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=POLICY_PATH)

CHUNK = policy.config.chunk_size
ADIM  = policy.config.max_action_dim
print(f"Policy loaded: chunk_size={CHUNK}, padded action_dim={ADIM}\n")

# ----------------------------------------------------------------------------
# 2. Build and Preprocess the (single, static) Observation
# ----------------------------------------------------------------------------
def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((640, 480))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

state = torch.tensor(JOINT_STATE, dtype=torch.float32, device=DEVICE).unsqueeze(0)
obs = {
    "observation.images.top":   load_image(TOP_IMG),
    "observation.images.wrist": load_image(WRIST_IMG),
    "observation.state":        state,
    "task": TASK,
}

print("Running Preprocessor...")
batch = preprocessor(obs)

# ----------------------------------------------------------------------------
# 3. Extract the static VLM prefix ONCE (cached KV reused for every trajectory)
# ----------------------------------------------------------------------------
print("Extracting static VLM embeddings...")
with torch.no_grad():
    vlm_start = time.perf_counter()

    images, img_masks = policy.prepare_images(batch)
    batch_state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=batch_state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    _, past_key_values = policy.model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.config.use_cache,
        fill_kv_cache=True,
    )
    print(f"-> VLM extraction took: {time.perf_counter() - vlm_start:.3f} seconds\n")

num_steps          = policy.config.num_steps
dt                 = -1.0 / num_steps
original_action_dim = policy.config.action_feature.shape[0]


def generate_physical(K: int) -> torch.Tensor:
    """Sample K trajectories from the cached prefix and un-normalize to physical units.

    Returns a [K, CHUNK, J] tensor (5 arm joints in degrees, gripper in its own
    closure unit).
    """
    trajectories = []
    with torch.no_grad():
        for _ in range(K):
            x_t = torch.randn(1, CHUNK, ADIM, device=DEVICE)
            for step in range(num_steps):
                time_val = 1.0 + step * dt
                time_tensor = torch.tensor(time_val, dtype=torch.float32, device=DEVICE).expand(1)
                v_t = policy.model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t

            chunk = x_t[:, :, :original_action_dim]
            if policy.config.adapt_to_pi_aloha:
                chunk = policy._pi_aloha_encode_actions(chunk)
            trajectories.append(chunk.squeeze(0).cpu())

    trajectories = torch.stack(trajectories)            # [K, CHUNK, J] normalized
    physical = postprocessor(trajectories)              # -> physical units (degrees / closure)
    return physical


def joint_spread(physical: torch.Tensor):
    """Per-joint (min, max, spread=max-min) over all candidates x timesteps."""
    flat = physical.reshape(-1, J)
    jmin = flat.min(dim=0).values
    jmax = flat.max(dim=0).values
    return jmin, jmax, (jmax - jmin)


# ----------------------------------------------------------------------------
# 4. Sweep K -> write divergence blocks (same format as Research_notes.txt)
# ----------------------------------------------------------------------------
torch.manual_seed(SEED)

k_values, avg_divergence, per_joint_rows = [], [], []

with open(NOTES_FILE, "w") as nf:
    nf.write("SMOLVLA divergence — K sweep\n")
    nf.write("Per-joint spread (max - min over all candidates x timesteps):\n")
    nf.write("K trajectories | vector of differences in each joint\n")
    nf.write("(5 arm joints in degrees; gripper in its own closure unit)\n\n")
    nf.flush()

    print(f"Sweeping K over {K_LIST} ...\n")
    sweep_start = time.perf_counter()

    for K in K_LIST:
        t0 = time.perf_counter()
        physical = generate_physical(K)
        jmin, jmax, spread = joint_spread(physical)
        avg = float(spread.mean())

        # ---- write the block ----
        block = [f"K = {K}", f"{'joint':<14}{'min':>10}{'max':>10}{'spread':>10}"]
        for j, name in enumerate(JOINT_NAMES):
            block.append(f"{name:<14}{jmin[j].item():>10.3f}{jmax[j].item():>10.3f}{spread[j].item():>10.3f}")
        nf.write("\n".join(block) + "\n\n")
        nf.flush()

        k_values.append(K)
        avg_divergence.append(avg)
        per_joint_rows.append([K, round(avg, 5)] + [round(spread[j].item(), 5) for j in range(J)])
        print(f"K={K:>3}  avg_divergence={avg:7.3f}  ({time.perf_counter() - t0:5.1f}s)")

    print(f"\n-> Sweep of {len(K_LIST)} K-values took {time.perf_counter() - sweep_start:.1f}s")
    print(f"-> Wrote divergence blocks to {NOTES_FILE}")

# machine-readable summary (also lets you re-plot without re-running the sweep)
SUMMARY_CSV = "divergence_summary.csv"
with open(SUMMARY_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["K", "avg_divergence"] + [f"{n}_spread" for n in JOINT_NAMES])
    w.writerows(per_joint_rows)
print(f"-> Wrote summary to {SUMMARY_CSV}")

# ----------------------------------------------------------------------------
# 5. Scatter: average divergence across the 6 joints vs K
# ----------------------------------------------------------------------------
INK, MUTED, ACCENT = "#1f2933", "#6b7280", "#2563eb"   # ink / muted / accessible blue

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(k_values, avg_divergence, s=60, color=ACCENT,
           edgecolor="white", linewidth=1.0, zorder=3)

ax.set_title("SmolVLA: average per-joint divergence vs K", color=INK, fontsize=13, pad=12)
ax.set_xlabel("K  (number of sampled trajectories)", color=INK)
ax.set_ylabel("Average divergence across 6 joints\n(spread = max − min)", color=INK)

ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.margins(x=0.03)

fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
print(f"-> Wrote plot to {PLOT_FILE}")
