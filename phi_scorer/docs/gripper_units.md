# Gripper units — do we handle a rad/s (or mm, or deg) gripper?

**Short answer: no, not right now.** The gripper unit is hardcoded to LeRobot's SO-101
convention (0–100 %), in two places:

```python
# config.py
SCALE = np.array([DEG2RAD]*5 + [GRIPPER_PCT_TO_RAD])   # gripper assumed = percent
# features.to_urdf_rad
out[:, 5] = JAW_LO + chunk[:, 5] * GRIPPER_PCT_TO_RAD    # %→rad with an offset
```

If a user's gripper reports **radians** (or degrees, or mm for a linear gripper), that
branch mis-converts it — it treats a radian value as a percentage and applies the
`[JAW_LO, JAW_HI]` map on top.

## Where it actually bites (and where it doesn't)

The gripper is 1 of 6 channels; mis-scaling it only corrupts the features/gates that use
its *absolute* or *velocity* value:

| affected | why |
|---|---|
| **`S_pos`** (gripper) | checks \|q_rad\| vs limit — wrong q_rad |
| **`S_vel`** (gripper) | \|v\| vs q̇_max — wrong units on both sides |
| **`seam_v`** | it's a **max over joints**, so a wrongly-large gripper term can dominate |
| `p99_v` / `p99_a` | mild — 99th pct is usually set by the fast body joints, but a bad gripper can skew it |

**Immune:** `min_invk` (drops the gripper column entirely), `R` (a per-joint *ratio* →
scale-invariant), `fracE_bw` (a per-joint *energy fraction* → scale-invariant). So the
corruption is contained, but `S_pos`/`S_vel`/`seam_v` on the gripper would be wrong.

## The right fix (part of the user-friendly refactor)

Make the unit conversion **per-joint and configurable** instead of hardcoding
"gripper = percent." In the `RobotProfile`, each joint carries a small `(scale, offset)`
(or a named unit):

```python
joint_units = {
    "shoulder_pan":  ("deg",  ...),          # rad = deg * pi/180
    ...
    "gripper":       ("percent", JAW_LO, JAW_HI),   # rad = lo + pct*(hi-lo)/100
    # a rad gripper would just be:  ("rad",)         -> identity
    # a linear mm gripper:          ("linear", mm_per_unit)
}
```

`to_urdf_rad` then applies each joint's own map — a rad gripper becomes identity, a `%`
gripper keeps today's behavior, a linear gripper gets its own scale. This is a clean
generalization: `SCALE` is already a per-joint array; we'd just extend it to
`(scale, offset, kind)` per joint and drop the hardcoded index-5 special case.

**So:** today it's SO-101-only on the gripper; to be portable it needs per-joint units in
the profile. Folding **per-joint unit specs** into the `RobotProfile` refactor (alongside
finishing the collision-safe `measure.py`) is the change that makes a rad/mm/deg gripper
"just work."
