"""Robot visualization: two Rizon 4s arms on the Vention pedestal, in Rerun.

A 3D scene logged into the dashboard's **metrics** recording (see :func:`attach`),
so the Metrics viewer shows the arms in the 3D panel that replaced the old
end-effector trace — one viewer for the arms *and* the time-series metrics. The
same geometry + FK helpers also serve the Storage-tab **replay** viewer, which logs
them into its own per-episode recording.

Geometry
--------
* **Pedestal** — the Vention assembly mesh (``assets/robot/pedestal.glb``, converted
  from the STEP). The GLB is Y-up; we rotate it to the world's Z-up and translate it
  so the midpoint of the two **45°-outward mount plates** (outer faces, near the top
  of the tower) sits at the world origin — each arm bolts onto its plate, leaning
  45° outward.
* **Arms** — two Rizon 4s from the vendor URDF (``assets/robot/urdf/Rizon4s.urdf``,
  factory-calibrated joint origins) with its real visual meshes
  (``assets/robot/meshes/Rizon4s/visual/*.obj``). The OBJ+MTL pairs are parsed
  here into per-material ``Mesh3D`` (Rerun's ``Asset3D`` OBJ path ignores MTL
  colours) — which also lets the solid arm be *tinted* red when it has no live
  joint data. If the mesh files are absent, each link falls back to the old
  kinematic skeleton (tube-between-joints + joint balls).

Everything is Z-up, right-handed (the Rizon base ``+Z`` is up). Mount poses and the
pedestal transform are the constants below — tweak them to match the real rig.
"""

from __future__ import annotations

import math
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr

_ASSETS = Path(__file__).resolve().parent / "assets" / "robot"
URDF_PATH = _ASSETS / "urdf" / "Rizon4s.urdf"
PEDESTAL_GLB = _ASSETS / "pedestal.glb"

# -- placement (from the Vention CAD) ----------------------------------------
#
# Pedestal: the GLB is Y-up (verified from the mount plates' normals); +90° about X
# brings it to the world's Z-up. The arms bolt onto the two **45°-outward mount
# plates** near the top of the tower (223×223 mm, GLB nodes `_154`/`_153` —
# component names were stripped by the STEP→GLB conversion; identified
# geometrically as the only plates with 45° normals, mirrored across ±X). The
# translation puts the midpoint of their *outer faces* at the world origin, so
# each arm base sits flush on its plate. Outer-face centres (GLB frame):
# (∓1.1104/−0.6657, 0.6457, −0.3835); world normals (±0.7071, 0, 0.7071).
_PED_QUAT_XYZW = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # +90° about X
_PED_TRANSLATION = (0.8881, -0.3835, -0.6457)  # plate-pair outer-face midpoint -> origin

#: Mount-plate outer-face centre separation along world X (from the CAD).
MOUNT_SEP_M = 0.4447
#: Each arm leans outward by 45° (rotation about world Y), matching its plate normal.
MOUNT_TILT_RAD = math.pi / 4
#: Each arm is bolted to its plate rotated 90° about the plate normal (matched
#: against the real rig by eye). Composed after the outward tilt, i.e. a yaw in
#: the arm's own base frame.
MOUNT_YAW_RAD = math.pi / 2
# _MOUNTS itself is defined below the quaternion helpers it needs.

#: Skeleton styling (per arm), matching the metrics 3D-track colours.
_ARM_COLOR = {"left": [80, 160, 255], "right": [255, 140, 80]}
#: Solid-arm colour when there is NO live measured ``q`` for that arm (control boxes
#: off / no system publishing ``<side>/q``): a muted red so it reads as "stale, not
#: tracking". The arm also stops moving (see :func:`~.runner._emit_collection_view`).
_STALE_COLOR = [200, 90, 85]
#: Translucent "ghost" of each arm at the *commanded* teleop config (RGBA): the same
#: URDF visual meshes as the solid arm, flat-tinted in the side's colour at 20%
#: alpha (Rerun 0.33 honours alpha for Mesh3D/Capsules3D but not LineStrips3D).
_GHOST_COLOR = {"left": [130, 190, 255, 51], "right": [255, 190, 140, 51]}
#: Translucent purple ghost at the policy's horizon-END joint target (eval), plus the
#: solid purple used for the current-EEF → horizon-EEF trace (LineStrips3D ignore
#: alpha in 0.33, so the trace is solid on purpose). Same purple for both sides —
#: the target reads as "policy intent", not as belonging to an arm's colour.
_TARGET_COLOR = [168, 110, 255, 85]
#: Calibration uses the same target language, but its entity subtree is separate
#: from the eval horizon so entering/leaving Calibration cannot disturb policy state.
_CALIBRATION_TARGET_COLOR = [168, 110, 255, 105]
_TRACE_COLOR = [190, 130, 255]
_BONE_RADIUS = 0.028
_JOINT_RADIUS = 0.045
_GHOST_RADIUS = 0.020
_TRACE_RADIUS = 0.006
_TRACE_TIP_RADIUS = 0.016
#: Timeline the live pose updates are logged on (same name the metrics emitter uses).
_POSE_TIMELINE = "elapsed"


# ---------------------------------------------------------------------------
# small quaternion helpers (xyzw), so we need no scipy/transforms3d
# ---------------------------------------------------------------------------


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF fixed-axis rpy (R = Rz·Ry·Rx) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy -> 3x3 rotation matrix (R = Rz·Ry·Rx), matching
    :func:`_quat_from_rpy`."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quat_from_axis_angle(axis, angle: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9 or abs(angle) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    a = a / n
    s = math.sin(angle / 2)
    return (a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2))


def _quat_mul(q1, q2) -> tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rot_from_axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues: rotation matrix about ``axis`` by ``angle`` (matches
    :func:`_quat_from_axis_angle`)."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    a = a / n
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _quat_z_to(direction) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) rotating +Z onto ``direction`` — orients a capsule along a bone.

    Capsules3D lie along their local +Z from the origin, so a bone whose local vector
    is ``child_offset`` needs +Z rotated onto that offset's direction.
    """
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, d))
    if s < 1e-9:  # parallel (c≈1) or anti-parallel (c≈-1)
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    return _quat_from_axis_angle(axis / s, math.atan2(s, c))


#: Arm mounts on the two 45°-outward plates (see the placement block above): each
#: base sits at its plate's outer-face centre, tilted ``MOUNT_TILT_RAD`` outward
#: about world Y so base +Z matches the plate normal, then spun ``MOUNT_YAW_RAD``
#: about that normal (the bolt orientation). ``rot`` is the same rotation as
#: ``quat_xyzw`` in matrix form (for numeric FK in :func:`fk_world_eef`).
def _mount(sep_sign: float, tilt: float) -> dict:
    tilt_q = _quat_from_axis_angle((0.0, 1.0, 0.0), tilt)
    yaw_q = _quat_from_axis_angle((0.0, 0.0, 1.0), MOUNT_YAW_RAD)
    return {
        "translation": (sep_sign * MOUNT_SEP_M / 2, 0.0, 0.0),
        "quat_xyzw": _quat_mul(tilt_q, yaw_q),
        "rot": _rot_from_axis_angle((0.0, 1.0, 0.0), tilt)
        @ _rot_from_axis_angle((0.0, 0.0, 1.0), MOUNT_YAW_RAD),
    }


_MOUNTS = {
    "left": _mount(-1.0, -MOUNT_TILT_RAD),
    "right": _mount(1.0, MOUNT_TILT_RAD),
}


# ---------------------------------------------------------------------------
# URDF -> kinematic chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Visual:
    """One ``<visual>`` mesh of a link: file + its origin offset in the link frame."""

    name: str  # URDF visual name ("shell", "ring", ...)
    mesh: Path  # resolved mesh file (OBJ)
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class _Link:
    """One link along the serial chain: how it attaches to its parent + its child bone."""

    name: str
    xyz: tuple[float, float, float]  # joint origin translation (parent frame)
    rpy: tuple[float, float, float]  # joint origin rotation
    axis: tuple[float, float, float]  # joint rotation axis (revolute)
    child_offset: tuple[float, float, float] | None  # bone end = next joint's xyz
    visuals: tuple[_Visual, ...] = ()  # real link meshes (empty -> skeleton fallback)


def _parse_visuals(root: ET.Element, urdf_path: Path) -> dict[str, tuple[_Visual, ...]]:
    """Per-link ``<visual>`` meshes; ``filename`` resolved relative to the URDF file."""
    out: dict[str, tuple[_Visual, ...]] = {}
    for link_el in root.findall("link"):
        visuals: list[_Visual] = []
        for i, v in enumerate(link_el.findall("visual")):
            mesh_el = v.find("geometry/mesh")
            if mesh_el is None or not mesh_el.get("filename"):
                continue
            o = v.find("origin")
            xyz = tuple(float(x) for x in o.get("xyz", "0 0 0").split()) if o is not None else (0.0, 0.0, 0.0)
            rpy = tuple(float(x) for x in o.get("rpy", "0 0 0").split()) if o is not None else (0.0, 0.0, 0.0)
            scale = tuple(float(x) for x in mesh_el.get("scale", "1 1 1").split())
            visuals.append(
                _Visual(
                    name=v.get("name") or f"visual_{i}",
                    mesh=(urdf_path.parent / mesh_el.get("filename")).resolve(),
                    xyz=xyz,
                    rpy=rpy,
                    scale=scale,
                )
            )
        out[link_el.get("name")] = tuple(visuals)
    return out


def parse_chain(urdf_path: Path = URDF_PATH, root_link: str = "base_link") -> list[_Link]:
    """Walk the URDF serial chain from ``root_link`` to the leaf, returning ordered links.

    Each returned :class:`_Link` carries the transform that places it in its
    parent's frame, the offset to its own child (the skeleton bone), and its
    ``<visual>`` mesh references (drawn as the real arm when the files exist).
    """
    root = ET.parse(urdf_path).getroot()
    by_parent: dict[str, tuple[str, dict]] = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        o = j.find("origin")
        xyz = tuple(float(v) for v in (o.get("xyz", "0 0 0").split())) if o is not None else (0, 0, 0)
        rpy = tuple(float(v) for v in (o.get("rpy", "0 0 0").split())) if o is not None else (0, 0, 0)
        ax = j.find("axis")
        axis = tuple(float(v) for v in ax.get("xyz").split()) if ax is not None else (0.0, 0.0, 1.0)
        by_parent[parent] = (child, {"xyz": xyz, "rpy": rpy, "axis": axis})

    # ordered joints parent->child from the root leaf-ward
    seq: list[tuple[str, dict]] = []
    cur = root_link
    while cur in by_parent:
        child, meta = by_parent[cur]
        seq.append((child, meta))
        cur = child

    visuals = _parse_visuals(root, urdf_path)
    links: list[_Link] = [_Link(root_link, (0, 0, 0), (0, 0, 0), (0, 0, 1), None)]
    for child, meta in seq:
        links.append(_Link(child, meta["xyz"], meta["rpy"], meta["axis"], None))
    # child_offset[k] = translation of the joint leading to k+1 (the bone from link k)
    out: list[_Link] = []
    for k, lk in enumerate(links):
        nxt = links[k + 1].xyz if k + 1 < len(links) else None
        out.append(
            _Link(lk.name, lk.xyz, lk.rpy, lk.axis, nxt, visuals.get(lk.name, ()))
        )
    return out


#: Parse the URDF chain once; the live pose updater reuses it every tick.
_CHAIN: list[_Link] | None = None


def _chain() -> list[_Link]:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = parse_chain()
    return _CHAIN


# ---------------------------------------------------------------------------
# OBJ + MTL loading (the vendor visual meshes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MeshPart:
    """One ``usemtl`` group of an OBJ: unindexed triangles + its material colour."""

    material: str
    albedo: tuple[float, float, float]  # MTL Kd, 0..1
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray | None  # (N, 3) float32
    indices: np.ndarray  # (N/3, 3) uint32


def _parse_mtl(path: Path) -> dict[str, tuple[float, float, float]]:
    """Material name -> diffuse ``Kd`` from an MTL file (missing file -> empty)."""
    mats: dict[str, tuple[float, float, float]] = {}
    if not path.is_file():
        return mats
    current = None
    for raw in path.read_text().splitlines():
        t = raw.split()
        if not t:
            continue
        if t[0] == "newmtl":
            current = t[1]
        elif t[0] == "Kd" and current is not None:
            mats[current] = tuple(float(x) for x in t[1:4])
    return mats


_MESH_CACHE: dict[Path, tuple[_MeshPart, ...]] = {}
_FALLBACK_ALBEDO = (0.8, 0.8, 0.8)


def _load_obj(path: Path) -> tuple[_MeshPart, ...]:
    """Minimal OBJ loader: one :class:`_MeshPart` per ``usemtl`` group.

    Rerun's ``Asset3D`` OBJ path drops the MTL colours, so we parse both files
    ourselves into ``Mesh3D``-ready arrays (per-part ``Kd`` as the albedo).
    Corners are expanded unindexed (v/vn have separate OBJ index spaces); the
    link meshes are small enough that deduplication isn't worth the code.
    Vertex values are metres (the exporter's "MilliMeters" comment is wrong).
    Cached per path — the two arms and every recolour reuse the same parts.
    """
    cached = _MESH_CACHE.get(path)
    if cached is not None:
        return cached
    v: list[list[float]] = []
    vn: list[list[float]] = []
    mats: dict[str, tuple[float, float, float]] = {}
    groups: dict[str, list[str]] = {}
    current = ""
    for raw in path.read_text().splitlines():
        t = raw.split()
        if not t:
            continue
        k = t[0]
        if k == "v":
            v.append([float(x) for x in t[1:4]])
        elif k == "vn":
            vn.append([float(x) for x in t[1:4]])
        elif k == "mtllib":
            mats = _parse_mtl(path.parent / t[1])
        elif k == "usemtl":
            current = t[1]
        elif k == "f":  # fan-triangulate (the vendor OBJs are already triangles)
            c = t[1:]
            corners = groups.setdefault(current, [])
            for i in range(1, len(c) - 1):
                corners.extend((c[0], c[i], c[i + 1]))
    pos = np.asarray(v, dtype=np.float32)
    nrm = np.asarray(vn, dtype=np.float32) if vn else None
    parts: list[_MeshPart] = []
    for material, corners in groups.items():
        split = [c.split("/") for c in corners]
        vi = np.asarray([int(s[0]) for s in split], dtype=np.int64) - 1
        normals = None
        if nrm is not None and len(split[0]) > 1 and split[0][-1]:
            normals = nrm[np.asarray([int(s[-1]) for s in split], dtype=np.int64) - 1]
        parts.append(
            _MeshPart(
                material=material or "default",
                albedo=mats.get(material, _FALLBACK_ALBEDO),
                positions=pos[vi],
                normals=normals,
                indices=np.arange(len(vi), dtype=np.uint32).reshape(-1, 3),
            )
        )
    out = tuple(parts)
    _MESH_CACHE[path] = out
    return out


# ---------------------------------------------------------------------------
# scene logging (static)
# ---------------------------------------------------------------------------


def _arm_root(side: str, ghost: bool) -> str:
    """Entity subtree for one arm: ``robot/<side>`` (real) or ``robot/<side>_ghost``."""
    return f"robot/{side}_ghost" if ghost else f"robot/{side}"


def _log_real_skeleton(rec, side: str, chain: list[_Link], color, *, static: bool) -> None:
    """(Re)log the solid arm's joint balls + bones in ``color``.

    Used both for the initial static geometry and to **recolor** the arm live when its
    measured-``q`` availability flips (normal ↔ :data:`_STALE_COLOR`). Geometry is in
    each link's local frame; the FK transforms that place it come from :func:`_log_arm_pose`.
    """
    path = _arm_root(side, ghost=False)
    for lk in chain:
        path = f"{path}/{lk.name}"
        rec.log(path, rr.Points3D([[0, 0, 0]], radii=_JOINT_RADIUS, colors=[color]), static=static)
        if lk.child_offset is not None and np.linalg.norm(lk.child_offset) > 1e-6:
            rec.log(
                path,
                rr.LineStrips3D([[[0, 0, 0], list(lk.child_offset)]], radii=_BONE_RADIUS, colors=[color]),
                static=static,
            )


def _has_mesh_visuals(chain: list[_Link]) -> bool:
    """True when the URDF's visual mesh files are present on disk."""
    return any(vis.mesh.is_file() for lk in chain for vis in lk.visuals)


def _mesh_entities(side: str, chain: list[_Link]):
    """Yield ``(entity, visual, part)`` for every mesh part of one solid arm."""
    path = _arm_root(side, ghost=False)
    for lk in chain:
        path = f"{path}/{lk.name}"
        for vis in lk.visuals:
            if not vis.mesh.is_file():
                continue
            for part in _load_obj(vis.mesh):
                yield f"{path}/visual/{vis.name}/{part.material}", vis, part


def _log_arm_meshes(rec, side: str, chain: list[_Link]) -> None:
    """Log the solid arm's real link meshes (static), nested under the FK entities.

    Each ``<visual>`` gets its origin offset (+ scale — the ring mesh is a unit
    disc scaled per link) as a child ``Transform3D``, then one ``Mesh3D`` per
    material group so the MTL colours survive (see :func:`_load_obj`).
    """
    logged_transform = set()
    for entity, vis, part in _mesh_entities(side, chain):
        parent = entity.rsplit("/", 1)[0]
        if parent not in logged_transform:
            logged_transform.add(parent)
            rec.log(
                parent,
                rr.Transform3D(
                    translation=vis.xyz,
                    quaternion=rr.Quaternion(xyzw=_quat_from_rpy(*vis.rpy)),
                    scale=vis.scale,
                ),
                static=True,
            )
        rec.log(
            entity,
            rr.Mesh3D(
                vertex_positions=part.positions,
                triangle_indices=part.indices,
                vertex_normals=part.normals,
                albedo_factor=part.albedo,
            ),
            static=True,
        )


def _tint_arm_meshes(rec, side: str, chain: list[_Link], *, stale: bool) -> None:
    """Recolour the solid arm's meshes: red when stale, back to MTL colours when live.

    Partial component update (``from_fields``) — only the albedo is re-sent, not
    the geometry. Static, so it replaces the albedo logged by :func:`_log_arm_meshes`
    (a temporal write would be shadowed by that static one).
    """
    stale_albedo = tuple(c / 255.0 for c in _STALE_COLOR)
    for entity, _vis, part in _mesh_entities(side, chain):
        rec.log(
            entity,
            rr.Mesh3D.from_fields(albedo_factor=stale_albedo if stale else part.albedo),
            static=True,
        )


def _log_arm_geometry(rec, side: str, chain: list[_Link], *, ghost: bool) -> None:
    """Log one arm's constant geometry (static) in each link's frame.

    Real arm = the URDF's visual meshes (or the ``LineStrips3D`` + ``Points3D``
    skeleton if the mesh files are missing). Ghost arm = those same meshes
    flat-tinted translucent (capsule bones when the mesh files are missing).
    The FK ``Transform3D``s that place these are logged separately by
    :func:`_log_arm_pose`, so animating the arm re-logs only the transforms,
    not this geometry.
    """
    root = _arm_root(side, ghost)
    mount = _MOUNTS[side]
    rec.log(
        root,
        rr.Transform3D(translation=mount["translation"], quaternion=rr.Quaternion(xyzw=mount["quat_xyzw"])),
        static=True,
    )
    if not ghost:
        if _has_mesh_visuals(chain):
            _log_arm_meshes(rec, side, chain)
        else:
            _log_real_skeleton(rec, side, chain, _ARM_COLOR[side], static=True)
        return
    _log_ghost_geometry(rec, root, chain, _GHOST_COLOR[side], static=True)


def _log_ghost_geometry(rec, root: str, chain: list[_Link], color, *, static: bool) -> None:
    """A ghost arm under ``root``: translucent URDF meshes, or capsule bones without meshes."""
    if _has_mesh_visuals(chain):
        _log_ghost_meshes(rec, root, chain, color, static=static)
    else:
        _log_ghost_capsules(rec, root, chain, color, static=static)


def _log_ghost_meshes(rec, root: str, chain: list[_Link], color, *, static: bool) -> None:
    """The full URDF visual meshes as a translucent ghost arm under ``root``.

    Mirrors :func:`_log_arm_meshes`'s entity layout (cumulative link nesting +
    per-visual origin transform) so the same FK transforms from
    :func:`_log_arm_pose` pose it — but every mesh part gets the flat RGBA
    ``color`` as its albedo (alpha included; Mesh3D honours it in Rerun 0.33)
    instead of its MTL colour, so the ghost reads as one translucent tint.
    """
    albedo = tuple(c / 255.0 for c in color)
    path = root
    logged_transform = set()
    for lk in chain:
        path = f"{path}/{lk.name}"
        for vis in lk.visuals:
            if not vis.mesh.is_file():
                continue
            parent = f"{path}/visual/{vis.name}"
            if parent not in logged_transform:
                logged_transform.add(parent)
                rec.log(
                    parent,
                    rr.Transform3D(
                        translation=vis.xyz,
                        quaternion=rr.Quaternion(xyzw=_quat_from_rpy(*vis.rpy)),
                        scale=vis.scale,
                    ),
                    static=static,
                )
            for part in _load_obj(vis.mesh):
                rec.log(
                    f"{parent}/{part.material}",
                    rr.Mesh3D(
                        vertex_positions=part.positions,
                        triangle_indices=part.indices,
                        vertex_normals=part.normals,
                        albedo_factor=albedo,
                    ),
                    static=static,
                )


def _log_ghost_capsules(rec, root: str, chain: list[_Link], color, *, static: bool) -> None:
    """Translucent capsule bones for a ghost arm under ``root`` (RGBA ``color``)."""
    path = root
    for lk in chain:
        path = f"{path}/{lk.name}"
        if lk.child_offset is not None and np.linalg.norm(lk.child_offset) > 1e-6:
            offset = np.asarray(lk.child_offset, dtype=float)  # translucent capsule to next joint
            rec.log(
                path,
                rr.Capsules3D(
                    lengths=[float(np.linalg.norm(offset))],
                    radii=[_GHOST_RADIUS],
                    translations=[[0.0, 0.0, 0.0]],
                    quaternions=[list(_quat_z_to(offset))],
                    colors=[color],
                ),
                static=static,
            )


def _log_arm_pose(
    rec,
    side: str,
    chain: list[_Link],
    q: np.ndarray | None,
    *,
    ghost: bool,
    static: bool,
    root: str | None = None,
) -> None:
    """Log the per-link FK ``Transform3D`` chain placing one arm at joint config ``q``.

    ``q`` is the 7 joint angles (rad); ``None`` → zeros (home). Each link entity is
    nested under its parent so Rerun composes the FK. ``static=True`` sets the idle
    home pose; ``static=False`` writes to the current timeline for live animation.
    ``root`` overrides the entity subtree (default: the real/ghost arm root) — used
    for extra arm instances like the eval horizon-target ghost.
    """
    q = np.zeros(7) if q is None else np.asarray(q, dtype=float)
    path = root if root is not None else _arm_root(side, ghost)
    ji = 0  # index into q for revolute joints (links 1..7)
    for k, lk in enumerate(chain):
        path = f"{path}/{lk.name}"
        quat = _quat_from_rpy(*lk.rpy)
        if k > 0 and ji < len(q):
            quat = _quat_mul(quat, _quat_from_axis_angle(lk.axis, float(q[ji])))
            ji += 1
        rec.log(
            path,
            rr.Transform3D(translation=lk.xyz, quaternion=rr.Quaternion(xyzw=quat)),
            static=static,
        )


def log_scene(rec, joint_angles: dict[str, np.ndarray] | None = None) -> None:
    """Log the whole static scene: pedestal + both arms (solid real + ghost) at home."""
    rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if PEDESTAL_GLB.exists():
        rec.log(
            "robot/pedestal",
            rr.Transform3D(translation=_PED_TRANSLATION, quaternion=rr.Quaternion(xyzw=_PED_QUAT_XYZW)),
            static=True,
        )
        rec.log("robot/pedestal", rr.Asset3D(path=str(PEDESTAL_GLB)), static=True)

    chain = _chain()
    # Geometry is timeless, but poses must not be static: a static Transform3D
    # shadows later temporal Transform3D updates on the same entity and freezes
    # the solid/ghost arms at startup even while live samples are being logged.
    rec.set_time(_POSE_TIMELINE, duration=0.0)
    for side in ("left", "right"):
        q = None if joint_angles is None else joint_angles.get(side)
        for ghost in (False, True):
            _log_arm_geometry(rec, side, chain, ghost=ghost)
            _log_arm_pose(rec, side, chain, q, ghost=ghost, static=False)  # initial home pose


#: Last-known measured-``q`` presence, so the solid arm is recoloured only when its
#: state flips (normal ↔ stale-red) rather than every tick. Keyed by ``(id(rec), side)``
#: so independent recordings (the live robot viewer vs. a replay) never share tint
#: state — one bleeding into the other would suppress the flip-triggered recolour.
_arm_has_live: dict[tuple[int, str], bool] = {}


def update_poses(
    rec,
    real_q: dict[str, np.ndarray],
    ghost_q: dict[str, np.ndarray],
    t: float,
) -> None:
    """Live-update the arms: solid arm(s) at measured ``real_q``, ghost(s) at commanded ``ghost_q``.

    Both dicts are ``{side: (7,) rad}``; a side absent from a dict is left at its last
    pose. A solid arm with **no** measured ``q`` this tick (side missing from
    ``real_q``) is tinted :data:`_STALE_COLOR` (muted red) and not moved, so the viewer
    never shows motion the real arm isn't making; it returns to its normal colour when
    live data resumes. Only the FK transforms (and, on a state flip, the colour) are
    re-logged — the geometry from :func:`log_scene` rides along.
    """
    rec.set_time(_POSE_TIMELINE, duration=t)
    chain = _chain()
    for side in ("left", "right"):
        has_live = real_q.get(side) is not None
        key = (id(rec), side)
        if _arm_has_live.get(key) != has_live:  # presence flipped -> recolour the solid arm
            _arm_has_live[key] = has_live
            if _has_mesh_visuals(chain):
                _tint_arm_meshes(rec, side, chain, stale=not has_live)
            else:
                _log_real_skeleton(
                    rec, side, chain, _ARM_COLOR[side] if has_live else _STALE_COLOR, static=False
                )
        if has_live:
            _log_arm_pose(rec, side, chain, real_q[side], ghost=False, static=False)
        if ghost_q.get(side) is not None:
            _log_arm_pose(rec, side, chain, ghost_q[side], ghost=True, static=False)


# ---------------------------------------------------------------------------
# eval horizon target: purple ghost + current-EEF → horizon-EEF trace
# ---------------------------------------------------------------------------


def fk_world_eef(side: str, q: np.ndarray) -> np.ndarray:
    """World-frame position of the chain's leaf link (flange) at joint config ``q``.

    Numeric FK composing the same mount + URDF-joint transforms Rerun composes
    from :func:`_log_arm_pose`'s entity hierarchy, so points computed here land
    exactly on the rendered skeleton (used for the horizon-target EEF trace).
    """
    q = np.asarray(q, dtype=float)
    mount = _MOUNTS[side]
    t = np.asarray(mount["translation"], dtype=float)
    rot = np.asarray(mount["rot"], dtype=float)  # 45° outward tilt (see _MOUNTS)
    ji = 0
    for k, lk in enumerate(_chain()):
        t = t + rot @ np.asarray(lk.xyz, dtype=float)
        rot = rot @ _rot_from_rpy(*lk.rpy)
        if k > 0 and ji < len(q):
            rot = rot @ _rot_from_axis_angle(lk.axis, float(q[ji]))
            ji += 1
    return t


def _target_root(side: str) -> str:
    """Entity subtree of one arm's horizon-target ghost (mount-anchored FK chain)."""
    return f"robot/{side}_target"


def _trace_entities(side: str) -> tuple[str, str]:
    """World-frame trace entities (line + tip). Direct children of ``robot/`` so no
    mount/FK transform applies — the points are logged in world coordinates."""
    return f"robot/{side}_target_trace", f"robot/{side}_target_tip"


#: Sides whose horizon-target ghost geometry has been logged (lazy: the purple ghost
#: only exists in the scene while an eval run publishes joint targets).
_shown_targets: set[str] = set()
#: Sides whose trace/tip entities have been logged (superset of ghost sides: an
#: eef-only prediction draws a trace with no ghost), for clean removal.
_shown_traces: set[str] = set()


def mount_world_point(side: str, p_base) -> np.ndarray:
    """One arm's base-frame point -> the world frame, via its mount transform.

    The measured ``<side>/eef`` pose and the eval node's ``eef_horizon`` estimate
    are in the arm's own base frame; this places them in the rendered scene.
    """
    mount = _MOUNTS[side]
    return np.asarray(mount["rot"], dtype=float) @ np.asarray(p_base, dtype=float)[:3] + np.asarray(
        mount["translation"], dtype=float
    )


def _log_trace(rec, side: str, p_now: np.ndarray | None, p_end: np.ndarray) -> None:
    """The purple current→predicted EEF trace (world-frame): tip always, line when
    the current endpoint is known."""
    _shown_traces.add(side)
    trace_path, tip_path = _trace_entities(side)
    rec.log(
        tip_path,
        rr.Points3D([p_end.tolist()], radii=_TRACE_TIP_RADIUS, colors=[_TRACE_COLOR]),
    )
    if p_now is not None:
        rec.log(
            trace_path,
            rr.LineStrips3D([[p_now.tolist(), p_end.tolist()]],
                            radii=_TRACE_RADIUS, colors=[_TRACE_COLOR]),
        )


def update_horizon_targets(
    rec,
    target_q: dict,
    real_q: dict,
    t: float,
    target_eef: dict | None = None,
    real_eef: dict | None = None,
) -> None:
    """Pose the purple horizon prediction(s): ghost + trace, or trace alone.

    ``target_q`` maps side -> the policy's horizon-END joint target (from the
    ``eval/<side>/q_horizon`` stream) — posed as the purple ghost, with the
    current→target EEF trace between the FK of the measured ``real_q`` and of the
    target (with no measured ``q`` the ghost is still posed, the trace skipped).
    ``target_eef`` maps side -> the estimated horizon-END base-frame TCP position
    (``eval/<side>/eef_horizon``, cartesian control kinds): no joint target exists
    to pose a ghost, so only the trace + tip are drawn — from the measured
    ``real_eef`` TCP position (preferred: TCP→TCP, no flange/tool offset) or,
    lacking that, the FK of ``real_q`` (the flange — a tool shows as a small
    gap); with neither, the predicted tip alone. Sides absent from both target
    dicts are left untouched. Geometry is logged lazily on a side's first target
    (on the timeline, not static, so :func:`clear_horizon_targets` removes it
    cleanly when the run ends).
    """
    rec.set_time(_POSE_TIMELINE, duration=t)
    chain = _chain()
    for side, q in target_q.items():
        if q is None or side not in _MOUNTS:
            continue
        root = _target_root(side)
        if side not in _shown_targets:
            _shown_targets.add(side)
            mount = _MOUNTS[side]
            rec.log(
                root,
                rr.Transform3D(
                    translation=mount["translation"],
                    quaternion=rr.Quaternion(xyzw=mount["quat_xyzw"]),
                ),
                static=False,
            )
            _log_ghost_geometry(rec, root, chain, _TARGET_COLOR, static=False)
        _log_arm_pose(rec, side, chain, q, ghost=False, static=False, root=root)

        rq = real_q.get(side)
        if rq is not None:
            _log_trace(rec, side, fk_world_eef(side, rq), fk_world_eef(side, q))

    for side, p in (target_eef or {}).items():
        if p is None or side not in _MOUNTS or side in target_q:
            continue
        p_end = mount_world_point(side, p)
        p_meas = (real_eef or {}).get(side)
        rq = real_q.get(side)
        p_now = (
            mount_world_point(side, p_meas) if p_meas is not None
            else fk_world_eef(side, rq) if rq is not None
            else None
        )
        _log_trace(rec, side, p_now, p_end)


def clear_horizon_targets(rec) -> None:
    """Remove the horizon-target ghosts + traces (eval run ended / stream gone)."""
    if rec is None or (not _shown_targets and not _shown_traces):
        return
    for side in list(_shown_targets):
        rec.log(_target_root(side), rr.Clear(recursive=True))
    for side in list(_shown_traces):
        for entity in _trace_entities(side):
            rec.log(entity, rr.Clear(recursive=False))
    _shown_targets.clear()
    _shown_traces.clear()


# ---------------------------------------------------------------------------
# calibration target: selected reference pose as a purple ghost
# ---------------------------------------------------------------------------


def _calibration_target_root(side: str) -> str:
    """Entity subtree for the selected calibration reference-pose ghost."""
    return f"robot/{side}_calibration_target"


#: Calibration targets are timeless because the operator may pause/scrub the live
#: timeline while matching a pose. The app clears them explicitly on leaving the
#: Calibration controls, so they never leak into Experiment mode.
_shown_calibration_targets: set[str] = set()
_calibration_target_q: dict[str, tuple[float, ...]] = {}


def show_calibration_target(side: str, q) -> None:
    """Show ``side`` at calibration reference configuration ``q`` in purple.

    Uses the existing metrics recording and robot scene—there is no calibration
    viewer or recording. Only the currently selected leader side remains visible.
    Geometry is installed lazily and the FK transforms are updated only when the
    selected reference pose changes.
    """
    rec = _REC
    if rec is None:
        return
    if side not in _MOUNTS:
        raise ValueError(f"unknown calibration target side {side!r}")
    q_tuple = tuple(float(v) for v in np.asarray(q, dtype=float).ravel())

    for stale_side in _shown_calibration_targets - {side}:
        rec.log(
            _calibration_target_root(stale_side),
            rr.Clear(recursive=True),
            static=True,
        )
        _shown_calibration_targets.discard(stale_side)
        _calibration_target_q.pop(stale_side, None)

    root = _calibration_target_root(side)
    chain = _chain()
    if side not in _shown_calibration_targets:
        mount = _MOUNTS[side]
        rec.log(
            root,
            rr.Transform3D(
                translation=mount["translation"],
                quaternion=rr.Quaternion(xyzw=mount["quat_xyzw"]),
            ),
            static=True,
        )
        _log_ghost_geometry(
            rec, root, chain, _CALIBRATION_TARGET_COLOR, static=True
        )
        _shown_calibration_targets.add(side)
    if _calibration_target_q.get(side) != q_tuple:
        _log_arm_pose(
            rec,
            side,
            chain,
            np.asarray(q_tuple),
            ghost=False,
            static=True,
            root=root,
        )
        _calibration_target_q[side] = q_tuple


def clear_calibration_targets() -> None:
    """Hide every calibration ghost from the shared experiment viewer."""
    rec = _REC
    if rec is not None:
        for side in list(_shown_calibration_targets):
            rec.log(
                _calibration_target_root(side),
                rr.Clear(recursive=True),
                static=True,
            )
    _shown_calibration_targets.clear()
    _calibration_target_q.clear()


# ---------------------------------------------------------------------------
# RGB-D overlay (Robot-tab depth checkbox)
# ---------------------------------------------------------------------------

#: Entity holding the projected RGB-D cloud, sibling of the arms under ``robot/``.
_DEPTH_ENTITY = "robot/depth_cloud"
_DEPTH_POINT_RADIUS = 0.004
#: Keep every Nth point of the RGB-D cloud before logging (viewer perf).
_DEPTH_DOWNSAMPLE = 50


def camera_world_pose(cam) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` posing a camera's optical frame (X right, Y down, Z forward) in
    the robot world frame.

    ``cam`` is a :class:`~dual_flexiv_control.configs.CameraCfg`: its
    ``pose_xyz``/``pose_rpy`` are applied relative to ``pose_frame`` — ``world``
    (the mount-plate-midpoint origin) or an arm's mount point, ``mount_left`` /
    ``mount_right``. Only the mount's *translation* is used as the anchor — the
    45° plate tilt is deliberately not applied (the camera is fixed to the rig,
    not bolted to a tilted plate). World-frame points are then
    ``p_world = p_cam @ R.T + t``.
    """
    anchors = {
        "world": (0.0, 0.0, 0.0),
        "mount_left": _MOUNTS["left"]["translation"],
        "mount_right": _MOUNTS["right"]["translation"],
    }
    if cam.pose_frame not in anchors:
        raise ValueError(
            f"camera pose_frame {cam.pose_frame!r}; expected one of {sorted(anchors)}"
        )
    rot = _rot_from_rpy(*cam.pose_rpy)
    t = np.asarray(anchors[cam.pose_frame], dtype=float) + np.asarray(
        list(cam.pose_xyz), dtype=float
    )
    return rot, t


#: Parent entity for camera frustums, sibling of the arms under ``robot/``.
_CAMERA_ROOT = "robot/camera"
#: How far out (m) the viewer draws the frustum's image plane.
_FRUSTUM_PLANE_M = 0.4


def log_camera_frustum(name: str, cam) -> None:
    """(Re)log a camera's pose + pinhole frustum into the robot scene.

    Entity ``robot/camera/<name>`` gets the world transform from
    :func:`camera_world_pose` plus a ``Pinhole`` built from the camera's
    ``hfov_deg``/resolution — Rerun renders that as a wireframe frustum, so the
    scene shows where the RGB-D overlay is projected from. Logged static
    (latest-write-wins): re-logging after a ``conf/camera`` edit moves it.
    No-op until the viewer is started.
    """
    if _REC is None:
        return
    rot, t = camera_world_pose(cam)
    fx = (cam.width / 2.0) / math.tan(math.radians(float(cam.hfov_deg)) / 2.0)
    entity = f"{_CAMERA_ROOT}/{name}"
    _REC.log(entity, rr.Transform3D(translation=t, mat3x3=rot), static=True)
    _REC.log(
        entity,
        rr.Pinhole(
            resolution=[cam.width, cam.height],
            focal_length=[fx, fx],
            camera_xyz=rr.ViewCoordinates.RDF,  # ZED optical: X right, Y down, Z forward
            image_plane_distance=_FRUSTUM_PLANE_M,
        ),
        static=True,
    )


def log_depth_points(points: np.ndarray, colors: np.ndarray) -> None:
    """(Re)log the RGB-D overlay cloud (world-frame points, per-point RGB).

    Logged **static** so each refresh replaces the previous cloud (latest-write-
    wins) independent of the pose timeline. No-op until the viewer is started.

    The cloud is downsampled ``_DEPTH_DOWNSAMPLE``x before logging to keep the
    viewer responsive — the full RGB-D cloud is far denser than the overlay needs.
    """
    if _REC is not None:
        points = points[::_DEPTH_DOWNSAMPLE]
        colors = colors[::_DEPTH_DOWNSAMPLE]
        _REC.log(
            _DEPTH_ENTITY,
            rr.Points3D(points, colors=colors, radii=_DEPTH_POINT_RADIUS),
            static=True,
        )


def clear_depth_points() -> None:
    """Remove the RGB-D overlay cloud (depth checkbox switched off)."""
    if _REC is not None:
        _REC.log(
            _DEPTH_ENTITY,
            rr.Points3D(np.zeros((0, 3), dtype=np.float32)),
            static=True,
        )


def robot_recording():
    """The metrics recording the robot scene is logged into (``None`` until
    :func:`attach` binds it).

    The live pose updater logs the arms to this — the *same* recording the metrics
    time series stream into — so the arms animate in lock-step with the plots on the
    shared ``"elapsed"`` timeline, inside one embedded viewer.
    """
    return _REC


# ---------------------------------------------------------------------------
# attach to the dashboard's metrics recording (singleton per process)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
#: The recording the robot scene lives in — the dashboard's process-global metrics
#: recording once :func:`attach` binds it (``None`` before that). Module-scoped so
#: every helper here (pose updates, depth overlay, camera frustums) logs into it.
_REC = None


def attach(rec=None):
    """Bind the robot scene to the dashboard's **metrics** recording and log it there.

    The robot scene shares the metrics recording rather than running its own viewer,
    so the metrics 3D panel shows the arms in place of the old EEF trace. ``rec``
    defaults to the process-global metrics recording (installed by
    :func:`~.viewer.start_servers`'s ``rr.init``). Idempotent per process: the first
    call logs the static scene (pedestal + both arms at home); later calls return the
    already-bound recording without re-logging.
    """
    global _REC
    with _LOCK:
        if _REC is not None:
            return _REC
        rec = rec if rec is not None else rr.get_global_data_recording()
        if rec is not None:
            log_scene(rec)
        _REC = rec
        return _REC


def reset() -> None:
    """Forget the bound metrics recording so the next :func:`attach` rebinds.

    Used by the dashboard's *Reset services* action after :func:`~.viewer.teardown`
    drops the recording the scene was logged into; without this, :func:`attach`
    would keep returning the stale (now dead) handle instead of binding the fresh
    recording that the restarted servers install.
    """
    global _REC
    with _LOCK:
        _REC = None
        _shown_calibration_targets.clear()
        _calibration_target_q.clear()
