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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

import score_trajectories as SCORE      # noqa: E402  scorer core (single source of truth)
import project_trajectories as PROJ     # noqa: E402  build_affine (norm↔rad)


class PVDRuntime:
    """Process-global PVD configuration + lazily-built scoring resources + log handle."""

    def __init__(self):
        # --- config (set from --pvd.* / --policy.path by the wrapper) ---
        self.enabled = False
        self.num_samples = 1
        self.mode = "selection"          # selection | projection(stub)
        self.threshold = float(SCORE.FEASIBILITY_THRESHOLD)
        self.policy_path = SCORE.POLICY_PATH
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


class PVDSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA whose chunk generation optionally runs PVD selection over K samples."""

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        rt = PVD_RUNTIME

        # ---- disabled → exact stock behaviour (clean baseline) ----
        if not rt.enabled:
            return super()._get_action_chunk(batch, noise=noise, **kwargs)

        if rt.mode not in ("selection", "projection"):
            raise ValueError(f"PVD mode must be selection|projection, got {rt.mode!r}")

        rt.ensure_ready()
        K = int(rt.num_samples)
        t0 = time.perf_counter()

        # ---- sample K candidates from ONE observation, in a single batched pass ----
        if K > 1:
            device = next(self.parameters()).device
            batch_K = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1:
                    batch_K[k] = v.repeat(*([K] + [1] * (v.ndim - 1)))
                else:
                    batch_K[k] = v
            noise_K = self.model.sample_noise(
                (K, self.config.chunk_size, self.config.max_action_dim), device)
            candidates = super()._get_action_chunk(batch_K, noise=noise_K, **kwargs)
        else:
            candidates = super()._get_action_chunk(batch, noise=noise, **kwargs)

        # ---- score on CPU (Pinocchio is CPU-only) and select ----
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
        # Project in radian space (the limits are physical), starting at the real q0,
        # then invert the affine so the downstream postprocessor still recovers correct
        # robot units. This MODIFIES the chunk (distinct from selection).
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
