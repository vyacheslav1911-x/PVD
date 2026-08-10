#!/usr/bin/env python3
"""
generate_k_divergence_pi05.py
=============================
Same divergence-vs-K study as generate_k_divergence_1000.py, but for the LARGER
pi0.5 (pi05) flow-matching policy (~4B, bf16) instead of SmolVLA.

Identical logic: one static observation, K candidate action chunks sampled with
independent random noise, divergence = per-joint spread (max-min) over the first
K candidates × all timesteps, saturating-exponential fit.

Same computational optimization, re-derived for pi05's PaliGemma-with-expert:
  * encode the heavy prefix (2 cams + language) ONCE, cache its KV,
  * expand that batch-1 KV cache to a GPU mini-batch and denoise in parallel,
  * generate N_MAX=1000 once, read each K off the front (nested subset).
pi05's KV cache is a list of (keys, values, sliding_window) tuples and
denoise_step clones it internally, so expanding with broadcast views is safe.

Plot: LOG-scale X so the low-K "bend" (knee) is stretched out and clearly visible;
K is sampled densely at low K (free with nested subset).

Outputs:
  Research_notes_divergence_pi05.txt   per-K blocks (Research_notes.txt format)
  divergence_summary_pi05.csv          machine-readable (K, avg, per-joint)
  divergence_vs_k_pi05.png             log-x scatter + fitted curve
"""

import csv
import time
import torch
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.factory import make_pre_post_processors

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
POLICY_PATH = "qualia-robotics/pi05-so101-candy-ba90eda5"   # cached locally
DEVICE      = "cuda"

N_MAX  = 1000
BATCH  = 50                    # GPU mini-batch (auto-reduced on OOM — 4B model)
# Dense at the bend (cheap: extra K just read off the same 1000), coarser in the tail:
K_LIST = sorted(set(
    list(range(2, 20, 1)) + list(range(20, 100, 5)) + list(range(100, 1001, 50))
))
SEED   = 0

NOTES_FILE  = "Research_notes_divergence_pi05.txt"
SUMMARY_CSV = "divergence_summary_pi05.csv"
PLOT_FILE   = "divergence_vs_k_pi05.png"

TOP_IMG, WRIST_IMG = "observation/top.png", "observation/wrist.png"
TASK        = "Put the black candy in the box"
JOINT_STATE = [30.68, 10.95, -18.64, 87.47, -22.81, 2.05]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
J = len(JOINT_NAMES)

# ----------------------------------------------------------------------------
# 1. Load pi05 policy + processors ONCE (bf16 from the checkpoint config)
# ----------------------------------------------------------------------------
print("Loading pi05 policy (bf16) and processors...")
cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)   # gets dtype=bfloat16 + input/output features
cfg.pretrained_path = POLICY_PATH
cfg.device = DEVICE
policy = PI05Policy.from_pretrained(POLICY_PATH, config=cfg).to(DEVICE).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=POLICY_PATH)

CHUNK     = policy.config.chunk_size
ADIM      = policy.config.max_action_dim
NUM_STEPS = policy.config.num_inference_steps
DT        = -1.0 / NUM_STEPS
try:
    ORIG_ADIM = policy.config.action_feature.shape[0]
except Exception:
    ORIG_ADIM = J
print(f"pi05 loaded: chunk={CHUNK}, padded adim={ADIM}, num_inference_steps={NUM_STEPS}, "
      f"GPU={torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

# ----------------------------------------------------------------------------
# 2. Preprocess the single static observation ONCE
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

images, img_masks = policy._preprocess_images(batch)     # list of [1,C,H,W], list of [1]
tokens = batch[OBS_LANGUAGE_TOKENS]
masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

# ----------------------------------------------------------------------------
# 2b. Encode the PaliGemma prefix ONCE (batch-1) and cache its KV.
# ----------------------------------------------------------------------------
print("Encoding static PaliGemma prefix once (cached KV reused for all trajectories)...")
with torch.no_grad():
    t_pref = time.perf_counter()
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d)
    policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, PKV = policy.model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_4d, position_ids=prefix_pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    print(f"-> prefix encoded in {time.perf_counter() - t_pref:.2f}s "
          f"(prefix_len={prefix_pad_masks.shape[1]})\n")


def _expand_cache(pkv, B):
    """Broadcast the batch-1 KV cache (list of (keys, values, sliding_window)) to batch B.
    denoise_step clones internally, which materializes these views."""
    out = []
    for keys, values, sw in pkv:
        kB = keys.expand(B, *keys.shape[1:]) if keys.shape[0] == 1 else keys
        vB = values.expand(B, *values.shape[1:]) if values.shape[0] == 1 else values
        out.append((kB, vB, sw))
    return out


# ----------------------------------------------------------------------------
# 3. Batched generation from the CACHED prefix (denoise only)
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
        return x_t[:, :, :ORIG_ADIM].cpu()


print(f"Generating {N_MAX} trajectories (mini-batch {BATCH}, auto-reduce on OOM)...")
torch.manual_seed(SEED)
gen_start = time.perf_counter()
parts, made = [], 0
while made < N_MAX:
    b = min(BATCH, N_MAX - made)
    try:
        parts.append(generate_batch(b))
        made += b
        print(f"  {made}/{N_MAX}", end="\r")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if BATCH <= 2:
            raise
        BATCH = max(2, BATCH // 2)
        print(f"\n  OOM -> reducing BATCH to {BATCH}")
all_norm = torch.cat(parts, dim=0)                        # [N_MAX, CHUNK, ORIG_ADIM]
gen_time = time.perf_counter() - gen_start
print(f"\n-> Generated {N_MAX} pi05 trajectories in {gen_time:.1f}s "
      f"({gen_time / N_MAX * 1e3:.1f} ms/traj), peak GPU={torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

physical = postprocessor(all_norm)                        # [N_MAX, CHUNK, J] physical units

# ----------------------------------------------------------------------------
# 4. Nested-subset divergence per K -> notes blocks + summary
# ----------------------------------------------------------------------------
k_values, avg_divergence, per_joint_rows = [], [], []
with open(NOTES_FILE, "w") as nf:
    nf.write("PI0.5 (pi05) divergence — K sweep (nested subset of 1000)\n")
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

popt, _ = curve_fit(sat_exp, k_arr, y_arr,
                    p0=[y_arr.max(), y_arr.max() - y_arr.min(), 80.0], maxfev=20000)
A, B, tau = popt
yhat = sat_exp(k_arr, *popt)
r2 = 1.0 - float(((y_arr - yhat) ** 2).sum()) / float(((y_arr - y_arr.mean()) ** 2).sum())
print(f"-> Fit: y = {A:.2f} - {B:.2f}·exp(-K/{tau:.1f})  (asymptote={A:.2f}, R²={r2:.4f})")

# ----------------------------------------------------------------------------
# 6. Styled LOG-X plot — the bend is stretched out and clearly visible
# ----------------------------------------------------------------------------
INK, MUTED, GRID = "#111827", "#6b7280", "#eceff3"
DOT, FIT         = "#7c3aed", "#f59e0b"      # violet dots (pi05), amber fit

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xscale("log")

kx = np.logspace(np.log10(k_arr.min()), np.log10(k_arr.max()), 500)
ax.plot(kx, sat_exp(kx, *popt), color=FIT, lw=2.4, zorder=2, solid_capstyle="round",
        label="saturating fit")
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

ax.text(0.0, 1.11, "pi0.5 — action divergence grows with sample count K",
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
