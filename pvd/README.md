# PVD — Physics-Verified Selection for `lerobot-rollout`

Runs your normal SmolVLA rollout, but between chunk generation and execution it
samples **K** candidate action chunks from one observation, scores each for physical
feasibility (Pinocchio RNEA + finite-difference kinematics), selects one by
**filter-then-prefer**, and logs every candidate's score. Controlled entirely by
`--pvd.*` CLI flags. Selection is **read-only** (never edits a chunk); projection is
a stub for now. When `--pvd.enabled=false` the output is byte-identical to stock.

---

## 1. Get the repo in the right place

The analytic scorer resolves the URDF and `common.py` via `~/Desktop/PVD/...`, so the
simplest setup is to clone there:

```bash
git clone <your-repo-url> ~/Desktop/PVD
```

> Cloning elsewhere works too, but then edit `URDF_PATH` and `COMMON_PATH` at the top
> of `~/Desktop/PVD/score_trajectories.py` to point at your clone. The `pvd/` code
> itself is location-independent (it finds the repo from its own path).

Files the rollout relies on (all tracked in git): `SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf`,
`so_arm_ws/src/so_arm_viz/so_arm_viz/common.py`, `score_trajectories.py`,
`project_trajectories.py`, `pvd/`. **No colcon build / no ROS is needed for PVD** —
`common.py` is imported directly by file path.

## 2. Environment

Use the same conda env you run rollouts in, plus Pinocchio:

```bash
conda activate lerobot_v6          # lerobot 0.6.0 + a CUDA-matched torch
python -c "import pinocchio"        # if this fails:  conda install -c conda-forge pinocchio
```

If the policy is gated on the Hub: `huggingface-cli login`.

## 3. Smoke-test offline (no robot needed)

Confirms Pinocchio, the paths, and the policy un-normalizer all resolve before you
touch hardware:

```bash
python ~/Desktop/PVD/score_trajectories.py
```

Expect a per-candidate feasibility table. If it prints scores, PVD will run.

```bash
chmod +x ~/Desktop/PVD/pvd/*.sh    # after a fresh clone
```

## 4. Run PVD selection (one command)

Replace `<PORT>` / camera indices with your machine's. `--policy.device=cuda` for
generation; the scorer always runs on CPU automatically.

```bash
cd ~/Desktop/PVD/pvd
./run_pvd_rollout.sh \
  --strategy.type=base \
  --policy.path=qualia-robotics/smolvla-so101-candy-33c62cfe --policy.device=cuda \
  --robot.type=so101_follower --robot.port=<PORT> --robot.id=my_follower \
  --robot.cameras="{ top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --task="Put the black candy in the box" --duration=30 --robot.max_relative_target=10 \
  --pvd.enabled=true --pvd.num_samples=32 --pvd.mode=selection --pvd.threshold=5.0
```

## 5. Run the baseline (byte-identical stock)

```bash
# same command, but:
  --pvd.enabled=false
# (or drop all --pvd.* flags entirely)
```

## 6. CLI flags

| flag | default | meaning |
|---|---|---|
| `--pvd.enabled` | `false` | PVD on/off. `false` ⇒ identical to stock SmolVLA. |
| `--pvd.num_samples` | `1` | K candidate chunks per inference (batched into one forward pass). |
| `--pvd.mode` | `selection` | `selection` = execute the winner UNMODIFIED; `projection` = execute the winner REPAIRED to feasibility. |
| `--pvd.threshold` | `5.0` | feasibility hard-reject on Φ. |
| `--pvd.kp` | `300` | projection tracker stiffness (higher = tighter tracking of the policy chunk). |
| `--pvd.kd` | `2√kp` | projection tracker damping (default = critical). |
| `--pvd.log_path` | auto | JSONL log; default `~/Desktop/PVD/pvd_logs/pvd_run_<timestamp>.jsonl`. |

### selection vs projection

- **selection** samples K, picks by filter-then-prefer (Φ ≤ threshold → policy-preferred;
  else least-infeasible), and executes that candidate **unmodified**.
- **projection** does the same pick, then **repairs** the chosen chunk with the
  bounded-acceleration tracker (reused from `project_trajectories.py`) — starting at the
  real current pose `q0` and hard-clamping velocity/acceleration/joint limits — and
  executes the **modified**, feasible chunk. At `--pvd.num_samples=1` this is pure
  projection of the policy's own chunk. The log adds `phi_before`/`phi_after` and
  `projected_terms` so you can see the repair.

Run projection (K=1 is fine and cheapest):

```bash
./run_pvd_rollout.sh <your usual robot/policy/cameras/task flags> \
  --pvd.enabled=true --pvd.mode=projection --pvd.num_samples=1 --pvd.kp=300
```

`--robot.max_relative_target` still applies underneath PVD as the last-resort cap.

## 7. Log schema & diversity check

First line is `{"_meta":{…config, limits…}}`; then one JSON record per
chunk-generation step:

```json
{"step":0,"num_candidates":32,"chunk_shape":[50,6],"chosen_index":7,
 "reason":"filter+policy_preferred","fallback":false,"threshold":5.0,
 "q0_source":"robot_obs","q0_rad":[...6...],
 "candidates":[{"index":0,"phi":..,"feasible":..,"S_pos":..,"S_vel":..,
                "S_acc":..,"S_torque":..,"S_cont":..}, ...]}
```

`reason` = `filter+policy_preferred` (a survivor won) or `fallback_least_infeasible`
(none passed → lowest-Φ; `fallback:true`).

Confirm the K candidates actually differ:

```bash
python - <<'PY'
import json, glob, os
f=sorted(glob.glob(os.path.expanduser("~/Desktop/PVD/pvd_logs/pvd_run_*.jsonl")))[-1]
for l in open(f):
    r=json.loads(l)
    if "candidates" not in r: continue
    p=[c["phi"] for c in r["candidates"]]
    print(f"step {r['step']}: {len(set(round(x,3) for x in p))}/{len(p)} distinct Φ, "
          f"min={min(p):.2f} max={max(p):.2f} chosen={r['chosen_index']} ({r['reason']})")
PY
```

## 8. Note on the threshold

Raw policy chunks are acceleration-infeasible (Φ typically ~190–350 with the current
**placeholder** limits in `score_trajectories.py`). So `--pvd.threshold=5.0` will make
every step fall back to the least-infeasible candidate. To get real filtering either
raise the threshold (e.g. `--pvd.threshold=250`) or calibrate the measured
`q̇_max / q̈_max / τ_max` in `score_trajectories.py`. The log records the full
per-candidate breakdown either way.
