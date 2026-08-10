#!/usr/bin/env python
"""Run the (qualia) SmolVLA policy rollout and RECORD it the correct way for
tracking-error / feasibility-scorer validation.

The previous dynamics logger captured only ACHIEVED positions, so tracking error
(commanded - achieved) was impossible. This records BOTH sides, per control tick,
time-synced, plus chunk boundaries:

  step, t_obs, t_cmd, dt
  pos_<joint>    ACHIEVED position this tick   (Present_Position; deg / gripper %)
  cmd_<joint>    COMMANDED by the policy        (action into send_action, pre-cap)
  sent_<joint>   COMMANDED after safety cap      (send_action return, post max_relative_target)
  velraw_<joint> measured velocity              (Present_Velocity, raw signed)
  chunk_start    1 on the tick a NEW policy chunk begins executing (for S_cont)

Why three position signals:
  * cmd_  = what the scorer Φ actually scores (the policy's intended goal).
  * sent_ = what the servo actually received (cmd_ clipped by --robot.max_relative_target).
  * pos_  = what the arm achieved.
Tracking error vs cmd_ folds in BOTH the servo limits AND the safety cap; vs sent_
isolates the servo. The analysis aligns achieved to commanded by a small lag.

Three monkeypatches (no lerobot edits), same pattern as ghost_rollout.py:
  1. SOFollower.get_observation -> record achieved pos + measured velocity
  2. SmolVLAPolicy._get_action_chunk -> mark the tick a new chunk starts
  3. SOFollower.send_action -> record commanded (in) + sent (out), then flush the row

Custom flag (stripped before draccus sees argv):
  --rec.path=/path/to/record.csv   (default: <repo>/pvd_logs/record_<ts>.csv)

LOG-ONLY: no ROS/RViz. Use your normal lerobot-rollout args otherwise.
"""

import csv
import datetime
import os
import sys
import time


def _extract_rec_args(argv):
    rec, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--rec."):
            if "=" in a:
                key, val = a[len("--rec."):].split("=", 1)
            else:
                key = a[len("--rec."):]
                val = argv[i + 1] if i + 1 < len(argv) else ""
                i += 1
            rec[key] = val
        else:
            rest.append(a)
        i += 1
    return rec, rest


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    rec_args, rest = _extract_rec_args(sys.argv[1:])

    rec_path = rec_args.get("path")
    if not rec_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(repo, "pvd_logs")
        os.makedirs(d, exist_ok=True)
        rec_path = os.path.join(d, f"record_{ts}.csv")
    rec_path = os.path.expanduser(rec_path)
    os.makedirs(os.path.dirname(os.path.abspath(rec_path)), exist_ok=True)

    st = {"fh": None, "writer": None, "joints": None, "cur": None,
          "prev_t": None, "step": 0, "boundary": False, "vel_ok": True}

    from lerobot.robots.so_follower import SOFollower

    _orig_get_obs = SOFollower.get_observation
    _orig_send = SOFollower.send_action

    def _read_vel(bus, joints):
        if not st["vel_ok"]:
            return [""] * len(joints)
        try:
            d = bus.sync_read("Present_Velocity", normalize=False)
            return [d.get(j, "") for j in joints]
        except Exception:  # noqa: BLE001
            st["vel_ok"] = False
            print("[rec] Present_Velocity unreadable; velraw_ left blank.", file=sys.stderr)
            return [""] * len(joints)

    # 1) achieved position + measured velocity -> start a new per-tick record
    def _patched_get_obs(self, *a, **k):
        obs = _orig_get_obs(self, *a, **k)
        try:
            joints = st["joints"]
            if joints is None:
                joints = [key[:-4] for key in obs if key.endswith(".pos")]
                st["joints"] = joints
                cols = (["step", "t_obs", "t_cmd", "dt"]
                        + [f"pos_{j}" for j in joints]
                        + [f"cmd_{j}" for j in joints]
                        + [f"sent_{j}" for j in joints]
                        + [f"velraw_{j}" for j in joints]
                        + ["chunk_start"])
                st["fh"] = open(rec_path, "w", newline="", buffering=1)
                st["writer"] = csv.writer(st["fh"])
                st["writer"].writerow(cols)
                print(f"[rec] recording commanded+achieved -> {rec_path}", file=sys.stderr)

            now = time.perf_counter()
            dt = 0.0 if st["prev_t"] is None else now - st["prev_t"]
            st["prev_t"] = now
            st["cur"] = {
                "t_obs": round(time.time(), 4), "dt": round(dt, 5),
                "pos": [float(obs[f"{j}.pos"]) for j in joints],
                "vel": _read_vel(self.bus, joints),
                "chunk_start": 1 if st["boundary"] else 0,
            }
            st["boundary"] = False
        except Exception as exc:  # noqa: BLE001 -- never break inference
            print(f"[rec] get_obs record skipped: {exc}", file=sys.stderr)
        return obs

    # 2) a new chunk starts executing on this tick -> mark it (policy-agnostic).
    def _mark_boundary(orig):
        def wrapper(self, *a, **k):
            st["boundary"] = True
            if st["cur"] is not None:
                st["cur"]["chunk_start"] = 1
            return orig(self, *a, **k)
        return wrapper

    # 3) commanded (in) + sent (out) -> flush the paired row
    def _patched_send(self, action, *a, **k):
        sent = _orig_send(self, action, *a, **k)
        try:
            cur = st["cur"]
            joints = st["joints"]
            if cur is not None and joints is not None:
                cmd = [float(action.get(f"{j}.pos", "nan")) for j in joints]
                snt = [float(sent.get(f"{j}.pos", "nan")) for j in joints]
                row = ([st["step"], cur["t_obs"], round(time.time(), 4), cur["dt"]]
                       + [f"{v:.4f}" for v in cur["pos"]]
                       + [f"{v:.4f}" for v in cmd]
                       + [f"{v:.4f}" for v in snt]
                       + list(cur["vel"])
                       + [cur["chunk_start"]])
                st["writer"].writerow(row)
                st["step"] += 1
                st["cur"] = None
        except Exception as exc:  # noqa: BLE001
            print(f"[rec] send record skipped: {exc}", file=sys.stderr)
        return sent

    SOFollower.get_observation = _patched_get_obs
    SOFollower.send_action = _patched_send
    # boundary hook for whichever policy is actually used (SmolVLA or pi0.5)
    for mod_name, cls_name, meth in (
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy", "_get_action_chunk"),
        ("lerobot.policies.pi05.modeling_pi05", "PI05Policy", "predict_action_chunk"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            setattr(cls, meth, _mark_boundary(getattr(cls, meth)))
        except Exception:  # noqa: BLE001 -- policy not installed / not used
            pass
    print("[rec] installed: recording cmd/sent/achieved + chunk boundaries per tick.",
          file=sys.stderr)

    sys.argv = [sys.argv[0]] + rest
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    try:
        rollout_main()
    finally:
        if st["fh"] is not None:
            st["fh"].flush()
            st["fh"].close()
            print(f"[rec] wrote {st['step']} rows -> {rec_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
