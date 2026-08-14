"""phi_scorer.collision -- URDF self-collision checking for safe measurement sweeps.

measure.py must never command a configuration in which the arm hits itself. This
builds the URDF collision model and exposes is_collision(q) so a sweep can be halted
BEFORE it drives a joint into the body.

Adjacent links touch by design (they share a joint), so addAllCollisionPairs() reports
the arm as "colliding" even at rest. We exclude every pair that collides at the neutral
pose -- those are the always-touching adjacent links -- leaving only pairs that would
be a REAL self-collision.
"""

import numpy as np
import pinocchio as pin

from . import config

# Where 'package://so_arm_description/meshes/...' resolves to.
MESH_PKG_DIR = config.REPO + "/so_arm_ws/install/so_arm_description/share"


class SelfCollision:
    def __init__(self, urdf_path=config.URDF_PATH, mesh_dir=MESH_PKG_DIR):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.geom = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=mesh_dir)
        self.geom.addAllCollisionPairs()
        gd = self.geom.createData()
        # drop pairs already in contact at neutral (adjacent links) -> only real ones left
        q0 = pin.neutral(self.model)
        pin.computeCollisions(self.model, self.data, self.geom, gd, q0, False)
        touching = [i for i in range(len(self.geom.collisionPairs))
                    if gd.collisionResults[i].isCollision()]
        for i in sorted(touching, reverse=True):
            self.geom.removeCollisionPair(self.geom.collisionPairs[i])
        self.gd = self.geom.createData()
        self.n_pairs = len(self.geom.collisionPairs)

    def is_collision(self, q_rad):
        """True if configuration q (URDF radians, 6-vector) is a self-collision."""
        return bool(pin.computeCollisions(
            self.model, self.data, self.geom, self.gd, np.asarray(q_rad, float), True))
