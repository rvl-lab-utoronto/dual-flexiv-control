"""Viser adapter for the backend-neutral robot geometry."""

from __future__ import annotations

import logging
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


@lru_cache(maxsize=64)
def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangle mesh")
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
            mesh = _load_mesh(visual.mesh).copy()
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
        mesh = _load_mesh(geometry.GRAV_GRIPPER_STL).copy()
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
            factr_path = (factr_urdfs or {}).get(side)
            if factr_path is not None:
                try:
                    factr_chain = geometry.localize_factr_base_meshes(
                        geometry.parse_chain(factr_path, "base_link")
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not load FACTR %s URDF: %s", side, exc)
            self.factr[side] = RobotModel(
                server, f"/robot/{side}/factr", factr_chain,
                geometry.factr_mount(side), color=FACTR_COLOR, opacity=0.45,
                scale=geometry.FACTR_LIVE_DISPLAY_SCALE, visible=False,
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
