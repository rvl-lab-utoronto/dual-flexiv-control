"""Viser adapter for the backend-neutral robot geometry."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..visualization import geometry

log = logging.getLogger(__name__)

ARM_COLOR = {"left": (80, 160, 255), "right": (255, 140, 80)}
STALE_COLOR = (200, 90, 85)
GHOST_COLOR = {"left": (130, 190, 255), "right": (255, 190, 140)}
FACTR_COLOR = (70, 235, 155)
TARGET_COLOR = (168, 110, 255)
TRACE_COLOR = (190, 130, 255)

# The external FACTR CAD is roughly 1.18 million triangles per arm. Most of
# those triangles are threaded hardware and internal CAD shells. A raw edge
# collapse destroys these STL triangle soups, so we first weld/split them,
# replace each meaningful shell with its outer hull. This preserves the CAD
# silhouette and leaves about 20k faces per arm (roughly 50x smaller).
FACTR_MESH_DECIMATION_FACTOR = 50
_MESH_CACHE_VERSION = 3
_MIN_COMPONENT_VOLUME_FRACTION = 0.001


def _mesh_cache_path(path: Path, decimation_factor: int) -> Path:
    """Content-versioned cache path for one display-only simplified mesh."""
    override = os.environ.get("DFC_VISER_MESH_CACHE", "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        root = (
            Path(xdg).expanduser()
            if xdg else Path.home() / ".cache"
        ) / "dual-flexiv-control" / "viser-meshes"
    stat = path.stat()
    identity = "\0".join((
        str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns),
        str(decimation_factor), str(_MESH_CACHE_VERSION),
    ))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return root / f"{path.stem}-{digest}.npz"


def _read_simplified_cache(path: Path, decimation_factor: int):
    import trimesh

    cache_path = _mesh_cache_path(path, decimation_factor)
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float32)
            faces = np.asarray(payload["faces"], dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"invalid cached vertex shape {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"invalid cached face shape {faces.shape}")
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:  # noqa: BLE001 - regenerate a damaged local cache
        log.warning("ignoring invalid simplified mesh cache %s: %s", cache_path, exc)
        return None


def _write_simplified_cache(path: Path, decimation_factor: int, mesh) -> None:
    cache_path = _mesh_cache_path(path, decimation_factor)
    temporary: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, prefix=".mesh-", suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(
            temporary,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
        )
        os.replace(temporary, cache_path)
        temporary = None
    except Exception as exc:  # noqa: BLE001 - cache failure must not kill viewer
        log.warning("could not cache simplified viewer mesh %s: %s", path, exc)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _repair_for_viewer(mesh):
    """Turn multi-shell CAD tessellation into a bounded low-poly display mesh."""
    import trimesh

    parts = list(mesh.split(only_watertight=False))
    total_volume = sum(abs(float(part.volume)) for part in parts)
    volume_floor = max(total_volume * _MIN_COMPONENT_VOLUME_FRACTION, 1e-15)
    meaningful = [part for part in parts if abs(float(part.volume)) >= volume_floor]
    if not meaningful:
        meaningful = [max(parts, key=lambda part: len(part.faces))]
    hulls = [part.convex_hull for part in meaningful]

    # Hull construction is the topology-safe reduction step. Further support-
    # point resampling was visually too aggressive and could erase recognizable
    # leader-arm detail even though its bounds remained valid.
    simplified = trimesh.util.concatenate(hulls)

    if not len(simplified.faces) or not np.isfinite(simplified.vertices).all():
        raise ValueError("topology repair produced empty or non-finite geometry")
    if not np.allclose(simplified.extents, mesh.extents, rtol=0.03, atol=1e-6):
        raise ValueError(
            f"topology repair changed bounds {mesh.extents} -> {simplified.extents}"
        )
    return simplified


@lru_cache(maxsize=64)
def _load_mesh(path: Path, decimation_factor: int = 1):
    import trimesh

    path = Path(path)
    decimation_factor = max(1, int(decimation_factor))
    if decimation_factor > 1:
        cached = _read_simplified_cache(path, decimation_factor)
        if cached is not None:
            return cached
    # Processing welds each STL's three-vertices-per-triangle export into the
    # closed shells needed by the topology repair. Normal follower meshes keep
    # their original material-friendly, unprocessed representation.
    loaded = trimesh.load(
        path, force="mesh", process=decimation_factor > 1
    )
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangle mesh")
    if decimation_factor > 1 and len(loaded.faces) > 4:
        original_faces = len(loaded.faces)
        try:
            loaded = _repair_for_viewer(loaded)
        except Exception as exc:  # noqa: BLE001 - never make the arm disappear
            log.warning(
                "could not repair/simplify viewer mesh %s; using full CAD: %s",
                path, exc,
            )
            return loaded
        log.info(
            "repaired viewer mesh %s: %d -> %d triangles (%.1fx)",
            path.name, original_faces, len(loaded.faces),
            original_faces / max(1, len(loaded.faces)),
        )
        _write_simplified_cache(path, decimation_factor, loaded)
    return loaded


class RobotModel:
    """One independently poseable URDF chain under a Viser parent frame."""

    def __init__(
        self,
        server,
        root: str,
        chain: tuple[geometry.Link, ...],
        mount: dict,
        *,
        color: tuple[int, int, int] | None = None,
        opacity: float | None = None,
        scale: float = 1.0,
        mesh_decimation: int = 1,
        follower_gripper: bool = False,
        visible: bool = True,
    ) -> None:
        self.server = server
        self.root = root
        self.chain = chain
        # Viser's Frame ``scale`` changes only the displayed coordinate axes;
        # unlike a normal scene-graph transform it is deliberately not inherited
        # by children.  Keep the model scale here and apply it explicitly to link
        # offsets, visual origins, and mesh nodes.
        self.model_scale = float(scale)
        self.mesh_decimation = max(1, int(mesh_decimation))
        self.root_handle = server.scene.add_frame(
            root,
            show_axes=False,
            position=mount["translation"],
            wxyz=geometry.xyzw_to_wxyz(mount["quat_xyzw"]),
            visible=visible,
        )
        self.frames = []
        path = root
        for link in chain:
            path = f"{path}/{link.name}"
            frame = server.scene.add_frame(path, show_axes=False)
            self.frames.append(frame)
            for visual in link.visuals:
                self._add_visual(path, visual, color=color, opacity=opacity)
        if follower_gripper:
            self._add_gripper(color=color, opacity=opacity)
        self.update(np.zeros(7))

    @property
    def visible(self) -> bool:
        return bool(self.root_handle.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self.root_handle.visible = bool(value)

    def _add_visual(self, parent: str, visual: geometry.Visual, *, color, opacity) -> None:
        if not visual.mesh.is_file():
            return
        visual_path = f"{parent}/visual/{visual.name}"
        mesh_scale = tuple(
            self.model_scale * float(component) for component in visual.scale
        )
        self.server.scene.add_frame(
            visual_path,
            show_axes=False,
            position=self.model_scale * np.asarray(visual.xyz, dtype=float),
            wxyz=geometry.xyzw_to_wxyz(geometry.quat_from_rpy(*visual.rpy)),
        )
        try:
            mesh = _load_mesh(visual.mesh, self.mesh_decimation).copy()
            if color is None:
                self.server.scene.add_mesh_trimesh(
                    f"{visual_path}/mesh", mesh, scale=mesh_scale
                )
            else:
                self.server.scene.add_mesh_simple(
                    f"{visual_path}/mesh",
                    np.asarray(mesh.vertices, dtype=np.float32),
                    np.asarray(mesh.faces, dtype=np.uint32),
                    color=color,
                    opacity=opacity,
                    side="double",
                    scale=mesh_scale,
                )
        except Exception as exc:  # noqa: BLE001 - one visual cannot kill the viewer
            log.warning("could not load viewer mesh %s: %s", visual.mesh, exc)

    def _add_gripper(self, *, color, opacity) -> None:
        if not geometry.GRAV_GRIPPER_STL.is_file():
            return
        try:
            flange_index = next(i for i, link in enumerate(self.chain) if link.name == "flange")
        except StopIteration:
            return
        parent = self.frames[flange_index].name
        path = f"{parent}/visual/grav_gripper"
        self.server.scene.add_frame(
            path,
            show_axes=False,
            position=self.model_scale * np.asarray((0.0, 0.0, 0.075038)),
            wxyz=geometry.xyzw_to_wxyz(
                geometry.quat_from_rpy(np.pi / 2, 0.0, np.pi / 2)
            ),
        )
        mesh = _load_mesh(
            geometry.GRAV_GRIPPER_STL, self.mesh_decimation
        ).copy()
        mesh_scale = (0.001 * self.model_scale,) * 3
        if color is None:
            self.server.scene.add_mesh_trimesh(
                f"{path}/mesh", mesh, scale=mesh_scale
            )
        else:
            self.server.scene.add_mesh_simple(
                f"{path}/mesh",
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=opacity,
                side="double",
                scale=mesh_scale,
            )

    def update(self, q) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        joint_index = 0
        for index, (link, frame) in enumerate(zip(self.chain, self.frames)):
            quat = geometry.quat_from_rpy(*link.rpy)
            if index > 0 and joint_index < q.size:
                quat = geometry.quat_mul(
                    quat,
                    geometry.quat_from_axis_angle(link.axis, float(q[joint_index])),
                )
                joint_index += 1
            frame.position = self.model_scale * np.asarray(link.xyz, dtype=float)
            frame.wxyz = geometry.xyzw_to_wxyz(quat)


class RobotScene:
    """The complete dual-arm scene shared by live view and episode replay."""

    def __init__(self, server, factr_urdfs: dict[str, Path] | None = None) -> None:
        self.server = server
        self.chain = geometry.parse_chain()
        self.real: dict[str, RobotModel] = {}
        self.stale_labels = {}
        self.ghost: dict[str, RobotModel] = {}
        self.factr: dict[str, RobotModel] = {}
        self.target: dict[str, RobotModel] = {}
        self._traces = {}
        self._tips = {}
        self._depth = None
        self._frustums = {}

        server.scene.set_up_direction("+z")
        server.scene.add_grid(
            "/world/grid", width=4.0, height=4.0, cell_size=0.1,
            section_size=0.5, cell_color=(55, 62, 75),
            section_color=(88, 98, 116), plane="xy",
        )
        if geometry.PEDESTAL_GLB.is_file():
            server.scene.add_glb(
                "/robot/pedestal",
                geometry.PEDESTAL_GLB.read_bytes(),
                position=geometry.PED_TRANSLATION,
                wxyz=geometry.xyzw_to_wxyz(geometry.PED_QUAT_XYZW),
            )
        for side in geometry.MOUNTS:
            mount = geometry.MOUNTS[side]
            self.real[side] = RobotModel(
                server, f"/robot/{side}/measured", self.chain, mount,
                color=ARM_COLOR[side], follower_gripper=True, visible=True,
            )
            self.stale_labels[side] = server.scene.add_label(
                f"/robot/{side}/stale_warning",
                f"{side.upper()} · NO LIVE q",
                position=np.asarray(mount["translation"]) + np.asarray((0.0, 0.0, 0.18)),
                visible=True,
            )
            self.ghost[side] = RobotModel(
                server, f"/robot/{side}/command", self.chain, mount,
                color=GHOST_COLOR[side], opacity=0.2,
                follower_gripper=True, visible=False,
            )
            self.target[side] = RobotModel(
                server, f"/robot/{side}/target", self.chain, mount,
                color=TARGET_COLOR, opacity=0.34, visible=False,
            )
            factr_chain = self.chain
            factr_mesh_decimation = 1
            factr_path = (factr_urdfs or {}).get(side)
            if factr_path is not None:
                try:
                    factr_chain = geometry.localize_factr_base_meshes(
                        geometry.parse_chain(factr_path, "base_link")
                    )
                    factr_mesh_decimation = FACTR_MESH_DECIMATION_FACTOR
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not load FACTR %s URDF: %s", side, exc)
            self.factr[side] = RobotModel(
                server, f"/robot/{side}/factr", factr_chain,
                geometry.factr_mount(side), color=FACTR_COLOR, opacity=0.45,
                scale=geometry.FACTR_LIVE_DISPLAY_SCALE, visible=False,
                mesh_decimation=factr_mesh_decimation,
            )

    def update_followers(self, configs: dict[str, np.ndarray]) -> None:
        for side in geometry.MOUNTS:
            q = configs.get(side)
            live = q is not None and np.asarray(q).size >= 7
            self.real[side].visible = True
            self.stale_labels[side].visible = not live
            if live:
                self.real[side].update(np.asarray(q)[:7])

    def update_commands(self, configs: dict[str, np.ndarray]) -> None:
        for side, model in self.ghost.items():
            q = configs.get(side)
            model.visible = q is not None and np.asarray(q).size >= 7
            if model.visible:
                model.update(np.asarray(q)[:7])

    def update_factr(self, configs: dict[str, np.ndarray]) -> None:
        for side, model in self.factr.items():
            q = configs.get(side)
            model.visible = q is not None and np.asarray(q).size >= 7
            if model.visible:
                model.update(np.asarray(q)[:7])

    def update_targets(
        self,
        target_q: dict[str, np.ndarray],
        real_q: dict[str, np.ndarray],
        *,
        q_paths: dict[str, np.ndarray] | None = None,
        target_eef: dict[str, np.ndarray] | None = None,
        real_eef: dict[str, np.ndarray] | None = None,
    ) -> None:
        q_paths, target_eef, real_eef = q_paths or {}, target_eef or {}, real_eef or {}
        active = set(target_q) | set(target_eef)
        for side, model in self.target.items():
            q = target_q.get(side)
            model.visible = q is not None and np.asarray(q).size >= 7
            if model.visible:
                model.update(np.asarray(q)[:7])
            if side not in active:
                self._hide_trace(side)
                continue
            p_now = None
            if side in real_eef:
                p_now = geometry.mount_world_point(side, real_eef[side])
            elif side in real_q:
                p_now = geometry.fk_world_eef(side, real_q[side])
            points = [] if p_now is None else [p_now]
            if side in q_paths:
                path = np.asarray(q_paths[side])
                points.extend(geometry.fk_world_eef(side, row[:7]) for row in path)
                p_end = points[-1]
            elif q is not None:
                p_end = geometry.fk_world_eef(side, np.asarray(q)[:7])
                points.append(p_end)
            else:
                p_end = geometry.mount_world_point(side, target_eef[side])
                points.append(p_end)
            self._show_trace(side, np.asarray(points), np.asarray(p_end))

    def clear_targets(self) -> None:
        for side, model in self.target.items():
            model.visible = False
            self._hide_trace(side)

    def _show_trace(self, side: str, points: np.ndarray, tip: np.ndarray) -> None:
        tip_points = np.asarray(tip, dtype=np.float32).reshape(1, 3)
        handle = self._tips.get(side)
        if handle is None:
            handle = self.server.scene.add_point_cloud(
                f"/robot/{side}/target_tip", tip_points, TRACE_COLOR,
                point_size=0.035, point_shape="circle", precision="float32",
            )
            self._tips[side] = handle
        else:
            handle.points, handle.visible = tip_points, True
        if len(points) < 2:
            return
        segments = np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)
        line = self._traces.get(side)
        if line is None:
            line = self.server.scene.add_line_segments(
                f"/robot/{side}/target_trace", segments, TRACE_COLOR, line_width=4.0
            )
            self._traces[side] = line
        else:
            line.points, line.visible = segments, True

    def _hide_trace(self, side: str) -> None:
        for handles in (self._tips, self._traces):
            if side in handles:
                handles[side].visible = False

    def show_calibration(self, side: str, q) -> None:
        for candidate, model in self.target.items():
            model.visible = candidate == side
        if side in self.target:
            self.target[side].update(np.asarray(q, dtype=float)[:7])

    def clear_calibration(self) -> None:
        for model in self.target.values():
            model.visible = False

    def add_camera_frustum(self, name: str, cam) -> None:
        rotation, translation = geometry.camera_world_pose(cam)
        # Viser/OpenGL cameras look down -Z with +Y up; rig optical frames are
        # +Z forward, +Y down.
        rotation = rotation @ np.diag((1.0, -1.0, -1.0))
        handle = self.server.scene.add_camera_frustum(
            f"/robot/camera/{name}",
            fov=np.deg2rad(float(cam.hfov_deg)),
            aspect=float(cam.width) / float(cam.height),
            scale=0.4,
            color=(160, 190, 220),
            position=translation,
            wxyz=geometry.wxyz_from_rot(rotation),
        )
        self._frustums[name] = handle

    def update_depth(self, points: np.ndarray | None, colors: np.ndarray | None) -> None:
        if points is None or colors is None or not len(points):
            if self._depth is not None:
                self._depth.visible = False
            return
        points = np.asarray(points, dtype=np.float32)[::2]
        colors = np.asarray(colors, dtype=np.uint8)[::2]
        if self._depth is None:
            self._depth = self.server.scene.add_point_cloud(
                "/robot/depth_cloud", points, colors,
                point_size=0.004, precision="float32",
            )
        else:
            self._depth.points = points
            self._depth.colors = colors
            self._depth.visible = True
