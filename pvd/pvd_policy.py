#!/usr/bin/env python
"""PVDSmolVLAPolicy — SmolVLA + Physics-Verified selection, drop-in for the rollout.

This is the "selection" operator from the proposal, wired into inference:

  When PVD is enabled, `_get_action_chunk` samples `num_samples` (K) candidate action
  chunks from ONE observation — by drawing K independent flow-matching noises and
  running them as a SINGLE batched forward pass (the cheap-sampling property the
  proposal relies on) — scores each candidate with the analytic feasibility scorer
  (Pinocchio RNEA + finite-difference kinematics, reused verbatim from
  score_trajectories.py), and picks one by FILTER-THEN-PREFER: hard-reject any
  candidate whose Φ exceeds `threshold`; among survivors return the one the policy
  ranked first (lowest sample index); if none pass, fall back to the least-infeasible
  and flag it. It logs every step (all K candidates' Φ + per-term breakdown) and
  returns the chosen chunk. In `selection` mode the chunk is returned UNMODIFIED. In
  `projection` mode the chosen chunk is additionally REPAIRED to feasibility by a
  bounded-acceleration tracker (reused from project_trajectories.py) that starts at
  the real current pose q0 and hard-clamps velocity/acceleration/joint limits — the
  executed chunk is MODIFIED. Both modes log every candidate's Φ + per-term breakdown;
  projection also logs the executed chunk's post-projection Φ.

  When PVD is disabled, `_get_action_chunk` is a pure pass-through to stock SmolVLA
  (K=1, no scoring) → byte-identical baseline.

Device split: generation runs on the policy's device (e.g. cuda); scoring runs on
CPU (Pinocchio is CPU-only), so candidates are moved to cpu/numpy before scoring.

Config is held in the process-global `PVD_RUNTIME`, populated by pvd_rollout.py from
the `--pvd.*` CLI flags before the rollout starts.
"""

import datetime
import json
import os
import sys
import threading
import time

import numpy as np
import torch

# Make the repo root importable so we can reuse the scorer / affine map by name.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# IMPORTANT (GPU memory): nothing heavy is imported at module load.
#   * The base policy stack (SmolVLA / pi0.5) is imported ONLY for the type actually
#     being run, inside make_pvd_policy_class() — so a pi0.5 run never pulls in the
#     SmolVLA/SmolVLM backbone (and vice versa).
#   * The scorer (pinocchio + a second processor build) is imported lazily in
#     PVDRuntime.ensure_ready(), which runs ONLY when PVD is enabled and only at the
#     first inference step.
# Net: with --pvd.enabled=false this module adds no model and no extra heavy import
# beyond the single policy the plain rollout already loads — same GPU footprint.
SCORE = None      # -> score_trajectories, bound lazily in ensure_ready()
PROJ = None       # -> project_trajectories, bound lazily in ensure_ready()

PVD_SUPPORTED = ("smolvla", "pi05")
DEFAULT_POLICY_PATH = "qualia-robotics/smolvla-so101-candy-33c62cfe"


class PVDRuntime:
    """Process-global PVD configuration + lazily-built scoring resources + log handle."""

    def __init__(self):
        # --- config (set from --pvd.* / --policy.path by the wrapper) ---
        self.enabled = False
        self.num_samples = 1
        self.mode = "selection"          # selection | projection(stub)
        self.threshold = 5.0             # scorer default; wrapper may override via --pvd.threshold
        self.policy_path = DEFAULT_POLICY_PATH   # wrapper overrides from --policy.path
        self.explicit_dtype = False      # True iff the user passed --policy.dtype (respect it)
        self.log_path = None             # auto-timestamped if None
        self.kp = 300.0                  # projection tracker stiffness
        self.kd = None                   # projection tracker damping (None -> 2*sqrt(kp))

        # --- runtime state ---
        self.last_obs = None             # latest REAL observation dict (for q0)
        self.step = 0
        self._ready = False
        self._lock = threading.Lock()
        self._log_fh = None
        # scoring resources (built once)
        self.common = self.model = self.data = self.q_max = None
        self.qlo = self.qhi = None       # joint position limits (for projection clamp)
        self.K_aff = self.B_aff = None

    @property
    def kd_eff(self):
        return self.kd if self.kd is not None else 2.0 * float(np.sqrt(self.kp))

    # -- lazy setup: Pinocchio model, limits, and the affine norm→rad map --------
    def ensure_ready(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            global SCORE, PROJ                      # lazily pull in pinocchio + scorer
            if SCORE is None:
                import score_trajectories as SCORE  # noqa: PLW0603
                import project_trajectories as PROJ
            self.common = SCORE.load_common()
            self.model, self.data, self.q_max, _ = SCORE.load_model_and_limits()
            self.qlo = np.asarray(self.model.lowerPositionLimit, dtype=float)
            self.qhi = np.asarray(self.model.upperPositionLimit, dtype=float)
            post = SCORE.load_unnormalizer(self.policy_path)      # norm→deg/pct
            self.K_aff, self.B_aff = PROJ.build_affine(post, self.common)  # norm→rad
            if self.log_path is None:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                d = os.path.join(_REPO, "pvd_logs")
                os.makedirs(d, exist_ok=True)
                self.log_path = os.path.join(d, f"pvd_run_{ts}.jsonl")
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            self._log_fh = open(self.log_path, "a", buffering=1)  # line-buffered
            self._log_fh.write(json.dumps({"_meta": {
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "policy_path": self.policy_path,
                "num_samples": self.num_samples,
                "mode": self.mode,
                "threshold": self.threshold,
                "dt": SCORE.DT,
                "weights": SCORE.WEIGHTS,
                "limits": {"qdot_max": float(SCORE.QDOT_MAX[0]),
                           "qddot_max": float(SCORE.QDDOT_MAX[0]),
                           "tau_max": float(SCORE.TAU_MAX[0])},
                "projection": {"kp": self.kp, "kd": self.kd_eff} if self.mode == "projection" else None,
                "note": "each S-term = Σ fractional-violation (dimensionless); Φ=Σ wᵢ·Sᵢ; "
                        "limits are PLACEHOLDERS — calibrate. selection=execute UNMODIFIED "
                        "winner; projection=execute the winner REPAIRED by the bounded-accel "
                        "tracker (starts at q0).",
            }}) + "\n")
            self._ready = True

    # -- q0: the real current joint state (radians), from the live observation ---
    def current_q0(self):
        obs = self.last_obs
        if obs is not None:
            try:
                _, pos = self.common.lerobot_obs_to_urdf(obs)
                return np.asarray(pos, dtype=float), "robot_obs"
            except Exception:  # noqa: BLE001 — fall back rather than crash inference
                pass
        return (np.asarray(self.common.lerobot_chunk_row_to_urdf(list(SCORE.Q0_LEROBOT)),
                           dtype=float), "config_fallback")

    # -- score all candidates and pick by filter-then-prefer ---------------------
    def select(self, cand_norm):
        """cand_norm: [K,H,6] normalized (numpy, on CPU). Returns selection + records."""
        rad = cand_norm * self.K_aff + self.B_aff        # [K,H,6] radians (unnormalize∘deg→rad)
        q0, q0_src = self.current_q0()
        results = []
        for k in range(cand_norm.shape[0]):
            terms = SCORE.score_candidate(rad[k], q0, self.model, self.data, self.q_max)
            results.append((terms, float(SCORE.phi(terms))))
        feasible = [i for i, (_, p) in enumerate(results) if p <= self.threshold]
        if feasible:
            chosen, reason, fallback = min(feasible), "filter+policy_preferred", False
        else:
            chosen = min(range(len(results)), key=lambda i: results[i][1])
            reason, fallback = "fallback_least_infeasible", True
        return chosen, results, reason, fallback, q0, q0_src

    def log_step(self, chunk_shape, results, chosen, reason, fallback, q0, q0_src,
                 projected=None):
        rec = {
            "step": self.step,
            "wall_time": round(time.time(), 3),
            "mode": self.mode,
            "executed": "projected" if projected is not None else "selected",
            "num_candidates": len(results),
            "chunk_shape": list(chunk_shape),
            "chosen_index": int(chosen),
            "reason": reason,
            "fallback": bool(fallback),
            "threshold": self.threshold,
            "q0_source": q0_src,
            "q0_rad": [round(float(x), 6) for x in q0],
            "candidates": [
                {"index": i, "phi": round(p, 6), "feasible": bool(p <= self.threshold),
                 "S_pos": round(t["pos"], 6), "S_vel": round(t["vel"], 6),
                 "S_acc": round(t["acc"], 6), "S_torque": round(t["torque"], 6),
                 "S_cont": round(t["cont"], 6)}
                for i, (t, p) in enumerate(results)
            ],
        }
        if projected is not None:
            phi_before, pterms, phi_after = projected
            rec["phi_before"] = round(phi_before, 6)   # chosen candidate, pre-projection
            rec["phi_after"] = round(phi_after, 6)      # executed chunk, post-projection
            rec["projected_terms"] = {"S_pos": round(pterms["pos"], 6),
                                      "S_vel": round(pterms["vel"], 6),
                                      "S_acc": round(pterms["acc"], 6),
                                      "S_torque": round(pterms["torque"], 6),
                                      "S_cont": round(pterms["cont"], 6)}
        self._log_fh.write(json.dumps(rec) + "\n")


PVD_RUNTIME = PVDRuntime()


class _PVDMixin:
    """Policy-agnostic PVD: batch K samples, score, select/project, log.

    A concrete policy subclass overrides its own chunk-generation method, samples the
    K candidates by calling the stock generator with a batched noise, and hands the
    [K, H, A] tensor of NORMALIZED candidate chunks to `_pvd_process`. Everything after
    sampling (scoring, filter-then-prefer, projection, logging) is identical across
    policies — only the sampling hook differs (SmolVLA: `_get_action_chunk`;
    pi0.5: `predict_action_chunk`).
    """

    def _pvd_batch(self, batch, K):
        """Repeat a B=1 observation batch to K and draw K independent noises."""
        device = next(self.parameters()).device
        batch_K = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1:
                batch_K[k] = v.repeat(*([K] + [1] * (v.ndim - 1)))
            else:
                batch_K[k] = v
        noise_K = self.model.sample_noise(
            (K, self.config.chunk_size, self.config.max_action_dim), device)
        return batch_K, noise_K

    def _pvd_process(self, candidates, K, t0):
        """candidates: [K,H,A] normalized. Returns the [1,H,A] chunk to execute."""
        rt = PVD_RUNTIME
        cand_np = candidates.detach().to("cpu", dtype=torch.float32).numpy()  # [K,H,6] norm
        chosen, results, reason, fallback, q0, q0_src = rt.select(cand_np)

        # ---- SELECTION: execute the chosen candidate UNMODIFIED ----
        if rt.mode == "selection":
            rt.log_step(candidates.shape, results, chosen, reason, fallback, q0, q0_src)
            print(f"[PVD] step {rt.step}: selection K={K} in "
                  f"{(time.perf_counter() - t0) * 1e3:.0f}ms → cand {chosen} "
                  f"({reason}, Φ={results[chosen][1]:.2f})", file=sys.stderr)
            rt.step += 1
            return candidates[chosen:chosen + 1].contiguous()

        # ---- PROJECTION: repair the chosen candidate to feasibility, execute it ----
        # Project in radian space (limits are physical), starting at the real q0, then
        # invert the affine so the downstream postprocessor still recovers robot units.
        chosen_rad = cand_np[chosen] * rt.K_aff + rt.B_aff                      # [H,6] rad
        proj_rad = PROJ.track(chosen_rad, q0, SCORE.QDOT_MAX, SCORE.QDDOT_MAX,
                              SCORE.DT, rt.kp, rt.kd_eff, rt.qlo, rt.qhi)        # feasible by constr.
        proj_norm = (proj_rad - rt.B_aff) / rt.K_aff                            # [H,6] norm
        pterms = SCORE.score_candidate(proj_rad, q0, rt.model, rt.data, rt.q_max)
        phi_after = float(SCORE.phi(pterms))
        rt.log_step(candidates.shape, results, chosen, reason, fallback, q0, q0_src,
                    projected=(results[chosen][1], pterms, phi_after))
        print(f"[PVD] step {rt.step}: projection K={K} in "
              f"{(time.perf_counter() - t0) * 1e3:.0f}ms → cand {chosen} "
              f"Φ {results[chosen][1]:.1f} → repaired Φ {phi_after:.2f}", file=sys.stderr)
        rt.step += 1
        out = torch.from_numpy(proj_norm.astype(np.float32)).unsqueeze(0)
        return out.to(candidates.device, dtype=candidates.dtype).contiguous()


def _check_mode(rt):
    if rt.mode not in ("selection", "projection"):
        raise ValueError(f"PVD mode must be selection|projection, got {rt.mode!r}")


def _ckpt_stored_dtype(path):
    """Return the ``dtype`` string stored in the checkpoint's config.json, or None.

    Offline-friendly: reads a local dir directly, else the HF cache (never downloads).
    """
    if not path:
        return None
    path = str(path)  # may arrive as a pathlib.Path; hf_hub_download needs a str repo_id
    try:
        if os.path.isdir(path):
            cfg_file = os.path.join(path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            cfg_file = hf_hub_download(path, "config.json", local_files_only=True)
        with open(cfg_file) as f:
            return json.load(f).get("dtype")
    except Exception:  # noqa: BLE001 — best-effort; fall back to the parsed config's dtype
        return None


def _align_config_dtype(config):
    """Align a pi0/pi05 config's dtype to the checkpoint's stored dtype.

    Why: ``PI05Config.dtype`` (and pi0) default to ``"float32"``. When the policy is
    selected by ``--policy.type=pi05 --policy.pretrained_path=…`` (rather than
    ``--policy.path=…``), draccus builds the config from that default and does NOT read
    the checkpoint's config.json, so a ~4B model is constructed in fp32 (~16.6 GB) and
    OOMs a 16 GB GPU at ``self.model.to(cuda)`` — before any weights load. The weights
    are actually bf16 (+fp32 vision) ≈ 9.35 GB. Constructing in the checkpoint's dtype
    removes that redundant 2× allocation and matches the working ``--policy.path`` run.
    A user-supplied ``--policy.dtype`` is always respected.
    """
    if PVD_RUNTIME.explicit_dtype or not hasattr(config, "dtype"):
        return
    stored = _ckpt_stored_dtype(getattr(config, "pretrained_path", None) or PVD_RUNTIME.policy_path)
    if stored and stored != config.dtype:
        print(f"[PVD] aligning policy dtype {config.dtype!r} -> {stored!r} to match the "
              f"checkpoint (avoids fp32 double-VRAM OOM; pass --policy.dtype to override).",
              file=sys.stderr)
        config.dtype = stored


_pvd_cls_cache: dict = {}


def make_pvd_policy_class(policy_type: str):
    """Return the PVD subclass for `policy_type`, importing ONLY that policy's stack.

    Called by the wrapper's patched get_policy_class, so the heavy backbone
    (SmolVLA/SmolVLM or pi0.5/PaliGemma) is imported exactly once, only for the policy
    the rollout is actually loading — never both. Returns None for unsupported types
    (caller falls back to the stock class). The stock generator is called EXPLICITLY
    (Base.<method>(self, ...)) rather than via super(), so these dynamically-built
    subclasses don't rely on a `__class__` cell.
    """
    if policy_type in _pvd_cls_cache:
        return _pvd_cls_cache[policy_type]

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Base

        def _hook(self, batch, noise=None, **kwargs):   # overrides _get_action_chunk
            rt = PVD_RUNTIME
            if not rt.enabled:                          # exact stock baseline
                return Base._get_action_chunk(self, batch, noise=noise, **kwargs)
            _check_mode(rt)
            rt.ensure_ready()
            K = int(rt.num_samples)
            t0 = time.perf_counter()
            if K > 1:
                batch_K, noise_K = self._pvd_batch(batch, K)
                candidates = Base._get_action_chunk(self, batch_K, noise=noise_K, **kwargs)
            else:
                candidates = Base._get_action_chunk(self, batch, noise=noise, **kwargs)
            return self._pvd_process(candidates, K, t0)

        cls = type("PVDSmolVLAPolicy", (_PVDMixin, Base), {"_get_action_chunk": _hook})

    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Base

        @torch.no_grad()
        def _hook(self, batch, **kwargs):               # overrides predict_action_chunk
            rt = PVD_RUNTIME
            if not rt.enabled:                          # exact stock baseline
                return Base.predict_action_chunk(self, batch, **kwargs)
            _check_mode(rt)
            rt.ensure_ready()
            K = int(rt.num_samples)
            t0 = time.perf_counter()
            if K > 1:
                batch_K, noise_K = self._pvd_batch(batch, K)
                candidates = Base.predict_action_chunk(self, batch_K, noise=noise_K, **kwargs)
            else:
                candidates = Base.predict_action_chunk(self, batch, **kwargs)
            return self._pvd_process(candidates, K, t0)

        # pi05's config.dtype defaults to fp32; align it to the checkpoint's dtype BEFORE
        # construction so the 4B model isn't built in fp32 (~16.6 GB) and OOMs a 16 GB GPU.
        # `.__func__(cls_, …)` keeps cls_ = the PVD subclass so `cls(config)` inside the
        # stock loader still builds the PVD-hooked policy (not a bare PI05Policy).
        def _from_pretrained(cls_, pretrained_name_or_path, *, config=None, **kw):
            if config is not None:
                _align_config_dtype(config)
            return Base.from_pretrained.__func__(cls_, pretrained_name_or_path, config=config, **kw)

        cls = type("PVDPi05Policy", (_PVDMixin, Base),
                   {"predict_action_chunk": _hook,
                    "from_pretrained": classmethod(_from_pretrained)})

    else:
        return None

    _pvd_cls_cache[policy_type] = cls
    return cls
