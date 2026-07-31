#!/usr/bin/env python
"""Run this INSTEAD of `lerobot-rollout`, with your usual args PLUS `--pvd.*` flags.

It inserts PVD physics-verified SELECTION between chunk generation and execution,
without editing any lerobot code, by:
  1. stripping the `--pvd.*` flags out of argv (draccus never sees them) and using
     them to populate the process-global PVD_RUNTIME,
  2. patching `lerobot.policies.factory.get_policy_class` so "smolvla" resolves to
     PVDSmolVLAPolicy (make_policy then loads the real weights into our subclass),
  3. patching `SOFollower.get_observation` to stash the live REAL observation so the
     scorer's continuity term uses the true current joint state as q0,
  4. calling the stock rollout main() with the remaining (untouched) args.

`--robot.max_relative_target` still applies underneath PVD as the last-resort cap.

CLI (added on top of your normal lerobot-rollout flags):
    --pvd.enabled=true|false     (default false → identical to stock)
    --pvd.num_samples=K          (default 1)
    --pvd.mode=selection|projection   (selection=execute winner UNMODIFIED;
                                       projection=execute winner REPAIRED to feasibility)
    --pvd.kp=300  --pvd.kd=<2√kp>     (projection tracker gains)
    --pvd.threshold=T            (feasibility hard-reject on Φ)
    --pvd.log_path=/path.jsonl   (default auto-timestamped under <repo>/pvd_logs/)
"""

import os
import sys


def _extract_pvd_args(argv):
    """Pull `--pvd.*` (both `--pvd.k=v` and `--pvd.k v`) out of argv."""
    pvd, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--pvd."):
            if "=" in a:
                key, val = a[len("--pvd."):].split("=", 1)
            else:
                key = a[len("--pvd."):]
                val = argv[i + 1] if i + 1 < len(argv) else ""
                i += 1
            pvd[key] = val
        else:
            rest.append(a)
        i += 1
    return pvd, rest


def _find_value(rest, flag):
    for a in rest:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    pvd, rest = _extract_pvd_args(sys.argv[1:])

    from pvd.pvd_policy import PVD_RUNTIME, PVDSmolVLAPolicy

    if "enabled" in pvd:
        PVD_RUNTIME.enabled = _as_bool(pvd["enabled"])
    if "num_samples" in pvd:
        PVD_RUNTIME.num_samples = int(pvd["num_samples"])
    if "mode" in pvd:
        PVD_RUNTIME.mode = pvd["mode"].strip()
    if "threshold" in pvd:
        PVD_RUNTIME.threshold = float(pvd["threshold"])
    if "log_path" in pvd:
        PVD_RUNTIME.log_path = os.path.expanduser(pvd["log_path"])
    if "kp" in pvd:
        PVD_RUNTIME.kp = float(pvd["kp"])
    if "kd" in pvd:
        PVD_RUNTIME.kd = float(pvd["kd"])

    # policy path is needed to build the (un)normalizer for scoring
    pp = _find_value(rest, "--policy.path")
    if pp:
        PVD_RUNTIME.policy_path = pp

    # ---- validation / friendly early guards (before touching hardware) ----
    if PVD_RUNTIME.enabled:
        if PVD_RUNTIME.mode not in ("selection", "projection"):
            sys.exit(f"[PVD] --pvd.mode must be selection|projection (got {PVD_RUNTIME.mode!r})")
        if PVD_RUNTIME.num_samples < 1:
            sys.exit(f"[PVD] --pvd.num_samples must be ≥ 1 (got {PVD_RUNTIME.num_samples})")

    # ---- patch: make the rollout build PVDSmolVLAPolicy ----
    # CRITICAL: the rollout's context.py did `from lerobot.policies import
    # get_policy_class`, so it holds its OWN reference to the function. Patching only
    # `factory.get_policy_class` is a NO-OP for the rollout (that was the bug where PVD
    # silently ran stock SmolVLA). Patch the binding context.py actually calls, plus the
    # re-export and factory for good measure.
    import lerobot.policies.factory as factory
    import lerobot.policies as policies_pkg
    import lerobot.rollout.context as rollout_context
    _orig_get_cls = factory.get_policy_class

    def _patched_get_cls(name):
        if name == "smolvla":
            return PVDSmolVLAPolicy
        return _orig_get_cls(name)

    rollout_context.get_policy_class = _patched_get_cls   # the one that matters
    policies_pkg.get_policy_class = _patched_get_cls
    factory.get_policy_class = _patched_get_cls
    if PVD_RUNTIME.enabled:
        resolved = rollout_context.get_policy_class("smolvla").__name__
        print(f"[PVD] installed → rollout will build {resolved} for 'smolvla'", file=sys.stderr)

    # ---- patch: capture the live REAL observation for q0 (reuse ghost pattern) ----
    try:
        from lerobot.robots.so_follower import SOFollower
        _orig_get_obs = SOFollower.get_observation

        def _patched_get_obs(self, *a, **k):
            obs = _orig_get_obs(self, *a, **k)
            try:
                PVD_RUNTIME.last_obs = obs
            except Exception:  # noqa: BLE001
                pass
            return obs

        SOFollower.get_observation = _patched_get_obs
    except Exception as exc:  # noqa: BLE001
        print(f"[PVD] warn: couldn't patch get_observation for q0 ({exc}); "
              f"scorer will use the config q0 fallback.", file=sys.stderr)

    print(f"[PVD] enabled={PVD_RUNTIME.enabled} K={PVD_RUNTIME.num_samples} "
          f"mode={PVD_RUNTIME.mode} threshold={PVD_RUNTIME.threshold} "
          f"log={PVD_RUNTIME.log_path or '(auto)'}", file=sys.stderr)

    # ---- hand the remaining args to the stock rollout (draccus reads sys.argv) ----
    sys.argv = [sys.argv[0]] + rest
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    rollout_main()


if __name__ == "__main__":
    main()
