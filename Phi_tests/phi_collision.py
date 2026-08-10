"""Section 3b -- self-collision and clearance via Pinocchio + coal (hpp-fcl).

Builds the geometry model from the URDF meshes, drops adjacent-link pairs (there
is no SRDF in this repo, so adjacency is derived from the kinematic tree), and
scores the minimum signed clearance over each chunk. Sub-samples between the
30 Hz waypoints, since a chunk can pass through a collision between two
collision-free samples.
"""

import os
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_terms import JOINTS, RUNS, SCALE, chunk_bounds  # noqa: E402

URDF = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF/urdf/so101_new_calib.urdf")
MESH_DIR = os.path.expanduser("~/Desktop/PVD/SO-ARM_ROS2_URDF")
SUBSAMPLE = 4          # interpolated configs between consecutive waypoints


def build():
    model = pin.buildModelFromUrdf(URDF)
    # package://so_arm_description/meshes/... resolves under MESH_DIR/meshes
    geom = pin.buildGeomFromUrdf(model, URDF, pin.GeometryType.COLLISION,
                                 package_dirs=[MESH_DIR,
                                               os.path.join(MESH_DIR, ".."),
                                               os.path.expanduser(
                                                   "~/Desktop/PVD/so_arm_ws/install/"
                                                   "so_arm_description/share")])
    # Raw STL mesh-mesh distance costs ~204 ms per full query on this model. Every
    # geometry is therefore replaced by its CONVEX HULL. The survey's claimed
    # "sub-millisecond per configuration" holds only for primitive/convex shapes,
    # not for the detailed meshes this URDF actually ships.
    import coal
    for go in geom.geometryObjects:
        g = go.geometry
        if isinstance(g, coal.BVHModelBase):
            g.buildConvexHull(True, "Qt")
            go.geometry = g.convex
    geom.addAllCollisionPairs()

    # drop pairs on the same link and on parent/child links: they always touch
    drop = []
    for k, cp in enumerate(geom.collisionPairs):
        j1 = geom.geometryObjects[cp.first].parentJoint
        j2 = geom.geometryObjects[cp.second].parentJoint
        if j1 == j2:
            drop.append(k)
        elif model.parents[max(j1, j2)] == min(j1, j2):
            drop.append(k)
    for k in reversed(drop):
        geom.removeCollisionPair(geom.collisionPairs[k])
    return model, geom


def main():
    model, geom = build()
    data = model.createData()
    gdata = geom.createData()
    print(f"geometry objects: {len(geom.geometryObjects)}   "
          f"collision pairs after dropping adjacent: {len(geom.collisionPairs)}")
    if len(geom.collisionPairs) == 0:
        print("no non-adjacent pairs -> self-collision is structurally impossible "
              "for this arm's collision model; term is vacuous here.")
        return

    def clearance(q):
        pin.computeDistances(model, data, geom, gdata, q)
        return min(r.min_distance for r in gdata.distanceResults)

    # workspace reference
    rng = np.random.default_rng(0)
    Q = rng.uniform(model.lowerPositionLimit, model.upperPositionLimit, size=(400, model.nq))
    ref = np.array([clearance(q) for q in Q])
    print(f"workspace reference (400 random poses): min {ref.min():.4f} m, "
          f"p1 {np.percentile(ref,1):.4f}, median {np.median(ref):.4f}, "
          f"fraction in collision {100*np.mean(ref<0):.1f}%")

    print(f"\n{'run':<12}{'chunks':>7}{'min clearance m':>17}{'mean of per-chunk min':>23}"
          f"{'chunks colliding':>18}")
    store = {}
    for lab, path in RUNS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        a = df[[f"cmd_{j}" for j in JOINTS]].to_numpy(float) * SCALE
        starts = chunk_bounds(df["chunk_start"].to_numpy())
        mins = []
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else len(df)
            blk = a[s:e]
            if len(blk) < 4:
                continue
            dense = []
            for i in range(len(blk) - 1):
                for f in np.linspace(0, 1, SUBSAMPLE, endpoint=False):
                    dense.append(blk[i] * (1 - f) + blk[i + 1] * f)
            dense.append(blk[-1])
            mins.append(min(clearance(q) for q in dense))
        store[lab] = np.array(mins)
        print(f"{lab:<12}{len(mins):>7}{store[lab].min():>17.4f}"
              f"{store[lab].mean():>23.4f}{int((store[lab]<0).sum()):>18}")

    from scipy.stats import mannwhitneyu
    sm = np.concatenate([store[k] for k in store if k.startswith("smolvla")])
    pi = np.concatenate([store[k] for k in store if k.startswith("pi0.5")])
    u = mannwhitneyu(sm, pi, alternative="two-sided")
    print(f"\nSEPARATION  clearance: smolvla {sm.mean():.4f} m   pi0.5 {pi.mean():.4f} m   "
          f"AUC {u.statistic/(len(sm)*len(pi)):.3f}  p {u.pvalue:.3g}")


if __name__ == "__main__":
    main()
