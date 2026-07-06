"""Camera discovery + live frames for the dashboard's camera tab.

Camera keys are the canonical ``cam/<camera>/<view>`` stream names, taken from
the composed Hydra config (so they track ``conf/camera`` and any overrides).

A view's frame is read **live** from the running system's shared-memory stream
when something is producing it — the dashboard attaches a short-lived
:class:`~dual_flexiv_control.streams.StreamReader` to the newest run in
``runtime_dir`` that publishes the key, pulls the latest sample, and reshapes it
with the camera's known shape. When nothing is producing (the system isn't up),
a synthetic animated placeholder is returned instead, so the tab is always live —
the same philosophy as the dashboard's placeholder metrics.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraView:
    """One selectable camera view (one shared-memory stream)."""

    key: str  # stream name, e.g. "cam/static/left"
    camera: str  # "static"
    view: str  # "left" | "right" | "depth"
    width: int
    height: int
    channels: int
    dtype: str

    @property
    def shape(self) -> tuple[int, ...]:
        if self.channels > 1:
            return (self.height, self.width, self.channels)
        return (self.height, self.width)


_LOCK = threading.Lock()
_VIEWS: tuple[CameraView, ...] | None = None
_CAMS: dict | None = None  # camera name -> plain CameraCfg (typed via the schema)


def discover_camera_views() -> list[CameraView]:
    """Every camera view in the composed config (computed once, then cached)."""
    global _VIEWS
    with _LOCK:
        if _VIEWS is None:
            _VIEWS = tuple(_compose_views())
        return list(_VIEWS)


def camera_cfg(name: str):
    """The composed :class:`~dual_flexiv_control.configs.CameraCfg` for one camera."""
    discover_camera_views()  # populates _CAMS on first compose
    return _CAMS[name]


def depth_cameras() -> list[str]:
    """Cameras publishing both a ``left`` RGB view and a ``depth`` view."""
    have: dict[str, set[str]] = {}
    for v in discover_camera_views():
        have.setdefault(v.camera, set()).add(v.view)
    return [c for c, vs in have.items() if "left" in vs and "depth" in vs]


def reset() -> None:
    """Drop the cached camera views so the next call re-composes from ``conf``."""
    global _VIEWS, _CAMS
    with _LOCK:
        _VIEWS = None
        _CAMS = None


def _compose_views() -> list[CameraView]:
    from hydra import compose
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from dual_flexiv_control import cameras as cam_mod
    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config")

    global _CAMS
    _CAMS = {name: OmegaConf.to_object(cam) for name, cam in cfg.cameras.items()}

    views: list[CameraView] = []
    for name, cam in cfg.cameras.items():
        for view in cam.views:
            views.append(
                CameraView(
                    key=cam_mod.camera_stream_name(name, view),
                    camera=name,
                    view=view,
                    width=int(cam.width),
                    height=int(cam.height),
                    channels=cam_mod.view_channels(view),
                    dtype=cam_mod.view_dtype(view),
                )
            )
    return views


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def get_frame(view: CameraView, runtime_dir: str | None = None) -> tuple[np.ndarray, str]:
    """Latest display-ready frame for ``view`` and its source.

    Returns ``(uint8 array, "live" | "placeholder")``. The array is RGB
    ``(H,W,3)`` for colour views and grayscale ``(H,W)`` for depth.
    """
    live = _read_live_frame(view, runtime_dir)
    if live is not None:
        return _to_display(live, view), "live"
    return _synthetic_frame(view), "placeholder"


def _runtime_root(runtime_dir: str | None) -> Path:
    rd = runtime_dir or os.environ.get("DFC_RUNTIME_DIR", "runtime")
    return Path(rd if os.path.isabs(rd) else os.path.join(os.getcwd(), rd))


def _read_live_frame(view: CameraView, runtime_dir: str | None) -> np.ndarray | None:
    """The newest shm frame for ``view`` across live runs, or None if none.

    Attaches and detaches a reader per call (cheap mmap) so there are no stale
    handles to invalidate when a run ends.
    """
    root = _runtime_root(runtime_dir)
    if not root.is_dir():
        return None
    try:
        from dual_flexiv_control.streams import StreamReader
        from dual_flexiv_control.streams import StreamRegistry
    except Exception:  # noqa: BLE001 - streams stack unavailable -> placeholder
        return None

    run_dirs = sorted(
        (p for p in root.iterdir() if (p / "streams").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        try:
            entry = StreamRegistry(str(root), run_dir.name).get(view.key)
            if entry is None:
                continue
            reader = StreamReader.attach(entry)
            try:
                samples = reader.latest()
                if samples.n > 0:
                    return np.asarray(samples.newest).reshape(view.shape).copy()
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - dead run / lapped buffer -> try next
            continue
    return None


def depth_point_cloud(
    camera: str,
    stride: int = 4,
    max_depth_m: float = 5.0,
    runtime_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The latest RGB-D frame pair of ``camera`` as a coloured point cloud.

    Back-projects the ``depth`` view through a pinhole model (``hfov_deg`` from
    the camera's config; principal point at the image centre) and colours each
    point from the ``left`` RGB view (the ZED depth map is registered to the
    left sensor). Returns ``(points (N,3) float32 [m], colors (N,3) uint8)`` in
    the camera's **optical frame** (X right, Y down, Z forward) — pose it into
    the robot scene with :func:`~.robot_view.camera_world_pose`. ``None`` when
    either stream has no live frame.

    ``stride`` subsamples pixels (720p @ 4 -> ≤57.6k points); invalid depth
    (NaN/Inf/0) and anything beyond ``max_depth_m`` is dropped.
    """
    views = {v.view: v for v in discover_camera_views() if v.camera == camera}
    depth_view, rgb_view = views.get("depth"), views.get("left")
    if depth_view is None or rgb_view is None:
        return None
    depth = _read_live_frame(depth_view, runtime_dir)
    rgb = _read_live_frame(rgb_view, runtime_dir)
    if depth is None or rgb is None:
        return None

    h, w = depth.shape
    fx = (w / 2.0) / math.tan(math.radians(float(camera_cfg(camera).hfov_deg)) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    d = depth[::stride, ::stride]
    c = rgb[::stride, ::stride]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    valid = np.isfinite(d) & (d > 0.2) & (d <= max_depth_m)
    if not valid.any():
        return None
    z = d[valid]
    pts = np.stack(
        [(us[valid] - cx) * z / fx, (vs[valid] - cy) * z / fx, z], axis=1
    ).astype(np.float32)
    return pts, np.ascontiguousarray(c[valid])


def _to_display(frame: np.ndarray, view: CameraView) -> np.ndarray:
    """Coerce a raw frame to a uint8 array Streamlit can show directly."""
    if view.channels == 1:  # depth (float32 metres) -> normalized grayscale
        finite = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        hi = float(finite.max()) or 1.0
        return np.clip(finite / hi * 255.0, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8, copy=False)


def _synthetic_frame(view: CameraView, max_width: int = 480) -> np.ndarray:
    """An animated placeholder, distinct per camera key, downscaled for the wire."""
    scale = min(1.0, max_width / max(1, view.width))
    w = max(160, int(view.width * scale))
    h = max(120, int(view.height * scale))
    t = time.time()
    base = (hash(view.key) % 360) / 360.0  # stable hue per camera
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (xx / w + 0.20 * t + base))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (yy / h + 0.15 * t + base + 1 / 3))
    b = 0.5 + 0.5 * np.sin(2 * np.pi * ((xx + yy) / (w + h) + 0.10 * t + base + 2 / 3))
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
