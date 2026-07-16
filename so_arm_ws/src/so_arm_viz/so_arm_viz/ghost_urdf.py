"""Generate a name-prefixed 'ghost' URDF so a second RobotModel can live in the
same RViz TF tree as the live arm.

RViz2's RobotModel display derives each link's TF frame from the URDF link name
and has no TF-prefix field, so two models sharing link names ('base', 'shoulder',
...) would collide. We therefore prefix every link/joint name (e.g. 'ghost_base',
'ghost_Rotation') and, optionally, recolor the whole model to a single translucent
color so it reads as a ghost.

Pure stdlib (xml) -- safe to import from a launch file under system python.
"""

import xml.etree.ElementTree as ET


def prefix_urdf(urdf_str: str, prefix: str = "ghost_", ghost_rgba: str | None = None) -> str:
    """Return a copy of `urdf_str` with all link/joint names prefixed.

    Args:
        urdf_str: the source URDF XML string.
        prefix: string prepended to every link and joint name.
        ghost_rgba: if given (e.g. "0.1 0.8 1.0 1.0"), overwrite every material
            color so the whole model is one uniform color.
    """
    root = ET.fromstring(urdf_str)

    for link in root.findall("link"):
        link.set("name", prefix + link.get("name"))

    for joint in root.findall("joint"):
        joint.set("name", prefix + joint.get("name"))
        for tag in ("parent", "child"):
            e = joint.find(tag)
            if e is not None and e.get("link") is not None:
                e.set("link", prefix + e.get("link"))

    # transmissions reference joints by name (rsp ignores them, but keep valid)
    for trans in root.findall("transmission"):
        for j in trans.findall("joint"):
            if j.get("name") is not None:
                j.set("name", prefix + j.get("name"))

    if ghost_rgba is not None:
        for mat in root.findall("material"):
            color = mat.find("color")
            if color is not None:
                color.set("rgba", ghost_rgba)

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    import sys

    src = open(sys.argv[1]).read()
    sys.stdout.write(prefix_urdf(src, ghost_rgba="0.1 0.8 1.0 1.0"))
