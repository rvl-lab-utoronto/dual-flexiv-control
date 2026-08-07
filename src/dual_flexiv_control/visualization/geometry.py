"""Backend-neutral Rizon/FACTR kinematics and rig placement.

The constants and FK match the established Rerun scene, but this module imports
no viewer SDK.  A backend receives link-local meshes and relative transforms and
owns only their presentation.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "dashboard" / "assets" / "robot"
URDF_PATH = ASSETS / "urdf" / "Rizon4s.urdf"
PEDESTAL_GLB = ASSETS / "pedestal.glb"
GRAV_GRIPPER_STL = ASSETS / "meshes" / "GRAV" / "Grav-PVT-50mm.stl"

PED_QUAT_XYZW = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
PED_TRANSLATION = (0.8881, -0.3835, -0.6457)
MOUNT_SEP_M = 0.4447
MOUNT_TILT_RAD = math.pi / 4
MOUNT_YAW_RAD = math.pi / 2
FACTR_BASE_YAW_RAD = math.pi
FACTR_LIVE_DISPLAY_SCALE = 2.0


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rpy_from_rot(rot: np.ndarray):
    pitch = math.atan2(-float(rot[2, 0]), math.hypot(rot[0, 0], rot[1, 0]))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll, yaw = math.atan2(-float(rot[1, 2]), float(rot[1, 1])), 0.0
    return roll, pitch, yaw


def quat_from_axis_angle(axis, angle: float):
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-9 or abs(angle) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    a /= norm
    s = math.sin(angle / 2)
    return (a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2))


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def rot_from_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    a /= norm
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def xyzw_to_wxyz(q):
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def wxyz_from_rot(rot: np.ndarray):
    """Stable rotation-matrix to Viser ``(w,x,y,z)`` quaternion."""
    m = np.asarray(rot, dtype=float)
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
                (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                0.25 * s, (m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s, 0.25 * s)


def _mount(sep_sign: float, tilt: float) -> dict:
    tilt_q = quat_from_axis_angle((0.0, 1.0, 0.0), tilt)
    yaw_q = quat_from_axis_angle((0.0, 0.0, 1.0), MOUNT_YAW_RAD)
    return {
        "translation": (sep_sign * MOUNT_SEP_M / 2, 0.0, 0.0),
        "quat_xyzw": quat_mul(tilt_q, yaw_q),
        "rot": rot_from_axis_angle((0.0, 1.0, 0.0), tilt)
        @ rot_from_axis_angle((0.0, 0.0, 1.0), MOUNT_YAW_RAD),
    }


MOUNTS = {
    "left": _mount(-1.0, -MOUNT_TILT_RAD),
    "right": _mount(1.0, MOUNT_TILT_RAD),
}


def factr_mount(side: str) -> dict:
    mount = MOUNTS[side]
    correction_q = quat_from_axis_angle((0.0, 0.0, 1.0), FACTR_BASE_YAW_RAD)
    correction_r = rot_from_axis_angle((0.0, 0.0, 1.0), FACTR_BASE_YAW_RAD)
    return {
        "translation": mount["translation"],
        "quat_xyzw": quat_mul(mount["quat_xyzw"], correction_q),
        "rot": np.asarray(mount["rot"]) @ correction_r,
    }


@dataclass(frozen=True)
class Visual:
    name: str
    mesh: Path
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class Link:
    name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    child_offset: tuple[float, float, float] | None
    visuals: tuple[Visual, ...] = ()


def _parse_visuals(root: ET.Element, urdf_path: Path):
    out = {}
    for link_el in root.findall("link"):
        visuals = []
        for i, visual in enumerate(link_el.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            origin = visual.find("origin")
            xyz = (
                tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
                if origin is not None else (0.0, 0.0, 0.0)
            )
            rpy = (
                tuple(float(x) for x in origin.get("rpy", "0 0 0").split())
                if origin is not None else (0.0, 0.0, 0.0)
            )
            scale = tuple(float(x) for x in mesh.get("scale", "1 1 1").split())
            visuals.append(Visual(
                visual.get("name") or f"visual_{i}",
                (urdf_path.parent / mesh.get("filename")).resolve(),
                xyz, rpy, scale,
            ))
        out[link_el.get("name")] = tuple(visuals)
    return out


@lru_cache(maxsize=8)
def parse_chain(urdf_path: Path = URDF_PATH, root_link: str = "base_link") -> tuple[Link, ...]:
    root = ET.parse(urdf_path).getroot()
    by_parent = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = (
            tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            if origin is not None else (0.0, 0.0, 0.0)
        )
        rpy = (
            tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            if origin is not None else (0.0, 0.0, 0.0)
        )
        axis_el = joint.find("axis")
        axis = (
            tuple(float(v) for v in axis_el.get("xyz").split())
            if axis_el is not None else (0.0, 0.0, 1.0)
        )
        by_parent[parent] = (child, xyz, rpy, axis)
    sequence = []
    current = root_link
    while current in by_parent:
        child, xyz, rpy, axis = by_parent[current]
        sequence.append(Link(child, xyz, rpy, axis, None))
        current = child
    visuals = _parse_visuals(root, urdf_path)
    links = [Link(root_link, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), None), *sequence]
    return tuple(
        replace(
            link,
            child_offset=links[i + 1].xyz if i + 1 < len(links) else None,
            visuals=visuals.get(link.name, ()),
        )
        for i, link in enumerate(links)
    )


def localize_factr_base_meshes(chain: tuple[Link, ...]) -> tuple[Link, ...]:
    out = []
    translation, rotation = np.zeros(3), np.eye(3)
    for index, link in enumerate(chain):
        if index:
            translation = translation + rotation @ np.asarray(link.xyz)
            rotation = rotation @ rot_from_rpy(*link.rpy)
        localized = []
        for visual in link.visuals:
            explicitly_placed = (
                not np.allclose(visual.xyz, (0.0, 0.0, 0.0))
                or not np.allclose(visual.rpy, (0.0, 0.0, 0.0))
                or not np.allclose(visual.scale, (1.0, 1.0, 1.0))
            )
            if "ros_zup_local" in str(visual.mesh) or explicitly_placed:
                localized.append(visual)
            else:
                inv = rotation.T
                localized.append(replace(
                    visual,
                    xyz=tuple(float(v) for v in (-inv @ translation)),
                    rpy=rpy_from_rot(inv),
                ))
        out.append(replace(link, visuals=tuple(localized)))
    return tuple(out)


def fk_world_eef(side: str, q: np.ndarray, chain: tuple[Link, ...] | None = None) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    mount = MOUNTS[side]
    translation = np.asarray(mount["translation"], dtype=float)
    rotation = np.asarray(mount["rot"], dtype=float)
    joint_index = 0
    for index, link in enumerate(chain or parse_chain()):
        translation = translation + rotation @ np.asarray(link.xyz)
        rotation = rotation @ rot_from_rpy(*link.rpy)
        if index > 0 and joint_index < len(q):
            rotation = rotation @ rot_from_axis_angle(link.axis, float(q[joint_index]))
            joint_index += 1
    return translation


def mount_world_point(side: str, point) -> np.ndarray:
    mount = MOUNTS[side]
    return (
        np.asarray(mount["rot"]) @ np.asarray(point, dtype=float)[:3]
        + np.asarray(mount["translation"])
    )


def camera_world_pose(cam) -> tuple[np.ndarray, np.ndarray]:
    anchors = {
        "world": (0.0, 0.0, 0.0),
        "mount_left": MOUNTS["left"]["translation"],
        "mount_right": MOUNTS["right"]["translation"],
    }
    if cam.pose_frame not in anchors:
        raise ValueError(f"camera pose_frame {cam.pose_frame!r}; expected one of {sorted(anchors)}")
    rotation = rot_from_rpy(*cam.pose_rpy)
    translation = np.asarray(anchors[cam.pose_frame]) + np.asarray(cam.pose_xyz)
    return rotation, translation
