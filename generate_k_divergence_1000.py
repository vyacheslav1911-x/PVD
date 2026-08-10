#!/usr/bin/env python3
"""
generate_k_divergence_1000.py
=============================
Divergence-vs-K sweep for SmolVLA under ONE static observation, up to K=1000.

Computational optimization
--------------------------
Instead of regenerating a fresh set for every K (≈21k trajectories total), we
generate N_MAX=1000 trajectories ONCE — in GPU mini-batches denoised in parallel
via `sample_actions` — then read each K off the front (nested subset: the
divergence at K is the spread of the first K of the 1000). One un-normalization
for all 1000. This is ~50× fewer forward passes and yields a clean monotone
curve.

  K sweep : 5,10,…,100 (step 5), then 150,200,…,1000 (step 50)   [38 values]
  divergence(K, joint) = max - min over the first K candidates × all timesteps
  fit     : saturating exponential  y = A - B·exp(-K/τ)   (asymptote A, + R²)

Outputs:
  Research_notes_divergence_1000.txt   per-K blocks (Research_notes.txt format)
  divergence_summary_1000.csv          machine-readable (K, avg, per-joint)
  divergence_vs_k_1000.png             styled scatter + fitted curve
"""

import csv
import time
import torch
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (ensures default fonts are indexed)
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import (
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    SmolVLAPolicy,
    make_att_2d_masks,
)
from lerobot.policies.factory import make_pre_post_processors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"
DEVICE      = "cuda"

N_MAX  = 1000                 # generate this many trajectories ONCE
BATCH  = 100                  # GPU mini-batch for parallel denoising (tune for VRAM)
# Dense at the bend (free with nested subset), coarser in the tail — same as the pi05 run:
K_LIST = sorted(set(
    list(range(2, 20, 1)) + list(range(20, 100, 5)) + list(range(100, 1001, 50))
))
SEED   = 0

NOTES_FILE  = "Research_notes_divergence_1000.txt"
SUMMARY_CSV = "divergence_summary_1000.csv"
PLOT_FILE   = "divergence_vs_k_1000.png"

TOP_IMG, WRIST_IMG = "observation/top.png", "observation/wrist.png"
TASK        = "Put the black candy in the box"
JOINT_STATE = [30.68, 10.95, -18.64, 87.47, -22.81, 2.05]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
J = len(JOINT_NAMES)

# ----------------------------------------------------------------------------
# 1. Load policy + processors ONCE
# ----------------------------------------------------------------------------
print("Loading Policy and Processors...")
policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
policy.eval().to(DEVICE)
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=POLICY_PATH)

CHUNK = policy.config.chunk_size
ADIM  = policy.config.max_action_dim
ORIG_ADIM = policy.config.action_feature.shape[0]
print(f"Policy loaded: chunk_size={CHUNK}, padded action_dim={ADIM}\n")

# ----------------------------------------------------------------------------
# 2. Preprocess the single static observation ONCE (batch-1 tensors)
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

images, img_masks = policy.prepare_images(batch)          # list of [1,C,H,W], list of [1]
batch_state = policy.prepare_state(batch)                 # [1, max_state_dim]
lang_tokens = batch[OBS_LANGUAGE_TOKENS]                  # [1, L]
lang_masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]          # [1, L]

# ----------------------------------------------------------------------------
# 2b. Encode the heavy VLM prefix ONCE (batch-1) and cache its KV.
#     For every mini-batch we only EXPAND this cache — the images/language are
#     never re-encoded, so the SigLIP/VLM forward runs exactly once for all 1000.
# ----------------------------------------------------------------------------
NUM_STEPS = policy.config.num_steps
DT = -1.0 / NUM_STEPS
print("Encoding static VLM prefix once (cached KV reused for all trajectories)...")
with torch.no_grad():
    t_pref = time.perf_counter()
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=batch_state
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, PKV = policy.model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d, position_ids=prefix_pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=policy.config.use_cache, fill_kv_cache=True,
    )
    print(f"-> VLM prefix encoded in {time.perf_counter() - t_pref:.2f}s "
          f"(prefix_len={prefix_pad_masks.shape[1]})\n")


def _expand_cache(pkv, B):
    """Broadcast the batch-1 KV cache to batch B (dict{layer: {key_states,value_states}})."""
    return {li: {k: (v.expand(B, *v.shape[1:]).contiguous()
                     if torch.is_tensor(v) and v.shape[0] == 1 else v)
                 for k, v in layer.items()}
            for li, layer in pkv.items()}


# ----------------------------------------------------------------------------
# 3. Batched generation from the CACHED prefix (denoise only — no re-encoding)
# ----------------------------------------------------------------------------
def generate_batch(B: int) -> torch.Tensor:
    """Return [B, CHUNK, ORIG_ADIM] normalized action chunks (on CPU)."""
    with torch.no_grad():
        pkv_B = _expand_cache(PKV, B)
        ppm_B = prefix_pad_masks.expand(B, -1).contiguous()
        x_t = policy.model.sample_noise((B, CHUNK, ADIM), DEVICE)
        for step in range(NUM_STEPS):
            t_val = 1.0 + step * DT
            t_tensor = torch.tensor(t_val, dtype=torch.float32, device=DEVICE).expand(B)
            v_t = policy.model.denoise_step(
                prefix_pad_masks=ppm_B, past_key_values=pkv_B, x_t=x_t, timestep=t_tensor,
            )
            x_t = x_t + DT * v_t
        chunk = x_t[:, :, :ORIG_ADIM]
        if policy.config.adapt_to_pi_aloha:
            chunk = policy._pi_aloha_encode_actions(chunk)
        return chunk.cpu()


print(f"Generating {N_MAX} trajectories in mini-batches of {BATCH} ...")
torch.manual_seed(SEED)
gen_start = time.perf_counter()
parts = []
made = 0
while made < N_MAX:
    b = min(BATCH, N_MAX - made)
    parts.append(generate_batch(b))
    made += b
    print(f"  {made}/{N_MAX}", end="\r")
all_norm = torch.cat(parts, dim=0)                        # [N_MAX, CHUNK, ORIG_ADIM]
gen_time = time.perf_counter() - gen_start
print(f"\n-> Generated {N_MAX} trajectories in {gen_time:.1f}s "
      f"({gen_time / N_MAX * 1e3:.1f} ms/traj)\n")

# One un-normalization for all 1000 -> physical units (deg / gripper closure)
physical = postprocessor(all_norm)                        # [N_MAX, CHUNK, J]

# ----------------------------------------------------------------------------
# 4. Nested-subset divergence per K -> notes blocks + summary
# ----------------------------------------------------------------------------
k_values, avg_divergence, per_joint_rows = [], [], []

with open(NOTES_FILE, "w") as nf:
    nf.write("SMOLVLA divergence — K sweep (nested subset of 1000)\n")
    nf.write("Per-joint spread (max - min over first-K candidates x timesteps):\n")
    nf.write("K trajectories | vector of differences in each joint\n")
    nf.write("(5 arm joints in degrees; gripper in its own closure unit)\n\n")

    for K in K_LIST:
        sub = physical[:K].reshape(-1, J)
        jmin = sub.min(dim=0).values
        jmax = sub.max(dim=0).values
        spread = jmax - jmin
        avg = float(spread.mean())

        block = [f"K = {K}", f"{'joint':<14}{'min':>10}{'max':>10}{'spread':>10}"]
        for j, name in enumerate(JOINT_NAMES):
            block.append(f"{name:<14}{jmin[j].item():>10.3f}{jmax[j].item():>10.3f}{spread[j].item():>10.3f}")
        nf.write("\n".join(block) + "\n\n")

        k_values.append(K)
        avg_divergence.append(avg)
        per_joint_rows.append([K, round(avg, 5)] + [round(spread[j].item(), 5) for j in range(J)])

print(f"-> Wrote {len(K_LIST)} divergence blocks to {NOTES_FILE}")

with open(SUMMARY_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["K", "avg_divergence"] + [f"{n}_spread" for n in JOINT_NAMES])
    w.writerows(per_joint_rows)
print(f"-> Wrote summary to {SUMMARY_CSV}")

# ----------------------------------------------------------------------------
# 5. Fit saturating exponential  y = A - B·exp(-K/tau)
# ----------------------------------------------------------------------------
k_arr = np.asarray(k_values, dtype=float)
y_arr = np.asarray(avg_divergence, dtype=float)

def sat_exp(K, A, B, tau):
    return A - B * np.exp(-K / tau)

p0 = [y_arr.max(), y_arr.max() - y_arr.min(), 80.0]
popt, _ = curve_fit(sat_exp, k_arr, y_arr, p0=p0, maxfev=20000)
A, B, tau = popt
yhat = sat_exp(k_arr, *popt)
ss_res = float(((y_arr - yhat) ** 2).sum())
ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
r2 = 1.0 - ss_res / ss_tot
print(f"-> Fit: y = {A:.2f} - {B:.2f}·exp(-K/{tau:.1f})   (asymptote={A:.2f}, R²={r2:.4f})")

# ----------------------------------------------------------------------------
# 6. Styled plot — light, clean, expressive
# ----------------------------------------------------------------------------
INK, MUTED, GRID = "#111827", "#6b7280", "#eceff3"
DOT, FIT         = "#2563eb", "#f59e0b"      # blue dots, amber fit (CVD-safe pair)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xscale("log")                                   # LOG-K: stretches the low-K bend

# fitted curve (drawn under the dots), on a log-spaced K grid
kx = np.logspace(np.log10(k_arr.min()), np.log10(k_arr.max()), 500)
ax.plot(kx, sat_exp(kx, *popt), color=FIT, lw=2.4, zorder=2,
        solid_capstyle="round", label="saturating fit")
ax.axhline(A, color=FIT, lw=1.0, ls=(0, (5, 5)), alpha=0.55, zorder=1)
ax.text(k_arr.max(), A, f"  asymptote ≈ {A:.1f}", va="center", ha="left",
        color=FIT, fontsize=9.5, fontweight="bold")

ax.scatter(k_arr, y_arr, s=44, color=DOT, edgecolor="white", linewidth=0.9,
           zorder=3, label="measured")

txt = (f"$y = A - B\\,e^{{-K/\\tau}}$\n"
       f"A = {A:.2f}   B = {B:.2f}   τ = {tau:.1f}\n"
       f"R² = {r2:.3f}")
ax.text(0.975, 0.06, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec="#e2e8f0", lw=1))

ax.text(0.0, 1.11, "SmolVLA — action divergence grows with sample count K",
        transform=ax.transAxes, color=INK, fontsize=15, fontweight="bold", va="bottom")
ax.text(0.0, 1.03, "Mean per-joint spread (max − min) across 6 joints  ·  single observation  ·  "
        "log-scale K to expose the bend", transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")

ax.set_xlabel("K   (number of sampled trajectories, log scale)", fontsize=11.5, labelpad=8)
ax.set_ylabel("Mean per-joint divergence", fontsize=11.5, labelpad=8)

ax.grid(axis="both", which="major", color=GRID, linewidth=1.0, zorder=0)
ax.grid(axis="x", which="minor", color=GRID, linewidth=0.6, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])

leg = ax.legend(loc="lower right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.30))
for t in leg.get_texts():
    t.set_color(INK)

fig.subplots_adjust(top=0.83, left=0.075, right=0.965, bottom=0.12)
fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
print(f"-> Wrote plot to {PLOT_FILE}")
