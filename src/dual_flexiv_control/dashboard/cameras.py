"""Camera discovery + live frames for the dashboard's camera tab.

Camera keys are the canonical ``cam/<camera>/<view>`` stream names, taken from
the composed Hydra config (so they track ``conf/camera`` and any overrides).

A view's frame is read **live** from the running system's shared-memory stream
when something is producing it — the dashboard attaches a short-lived
:class:`~dual_flexiv_control.streams.StreamReader` to the newest run in
``runtime_dir`` that publishes the key, pulls the latest sample, and reshapes it
with the camera's known shape. When nothing is producing (the system isn't up, or
a camera failed to open), the frame reads as **missing** — nothing is fabricated,
so the dashboard surfaces the gap as an error rather than disguising a dead camera
with a synthetic feed.
"""

from __future__ import annotations

import math
import os
import threading
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

    from .arms import compose_overrides

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config", overrides=compose_overrides())

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


def get_frame(view: CameraView, runtime_dir: str | None = None) -> tuple[np.ndarray | None, str]:
    """Latest display-ready frame for ``view`` and its source.

    Returns ``(uint8 array, "live")`` when a running system publishes the view's
    stream — RGB ``(H,W,3)`` for colour views, grayscale ``(H,W)`` for depth — else
    ``(None, "missing")``. Nothing is fabricated: a camera that is not producing
    reads as **missing** so the caller can surface it as an error instead of
    disguising it with a synthetic frame.
    """
    live = _read_live_frame(view, runtime_dir)
    if live is not None:
        return _to_display(live, view), "live"
    return None, "missing"


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
    except Exception:  # noqa: BLE001 - streams stack unavailable -> missing
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


@dataclass(frozen=True)
class CameraStatus:
    """Per-camera live-detection snapshot for the dashboard's Status panel."""

    camera: str
    views: tuple[str, ...]  # every configured view, e.g. ("left", "right", "depth")
    live_views: tuple[str, ...]  # the subset currently producing frames

    @property
    def detected(self) -> bool:
        """True when a running system is publishing at least one of this camera's views."""
        return bool(self.live_views)


def read_camera_statuses(runtime_dir: str | None = None) -> list[CameraStatus]:
    """Per-camera live-detection status, one entry per configured camera.

    A camera reads as **detected** when a running system is publishing at least one
    of its views' shared-memory streams. Nothing is fabricated: a camera that is not
    producing (system down, or the camera failed to open) reads as undetected so the
    Status panel surfaces the gap rather than disguising it.
    """
    by_cam: dict[str, list[CameraView]] = {}
    for v in discover_camera_views():
        by_cam.setdefault(v.camera, []).append(v)
    statuses: list[CameraStatus] = []
    for camera, views in by_cam.items():
        live = tuple(v.view for v in views if _stream_is_live(v.key, runtime_dir))
        statuses.append(CameraStatus(camera, tuple(v.view for v in views), live))
    return statuses


def _stream_is_live(key: str, runtime_dir: str | None) -> bool:
    """Whether any live run currently holds ≥1 sample for ``key`` (no frame decode)."""
    root = _runtime_root(runtime_dir)
    if not root.is_dir():
        return False
    try:
        from dual_flexiv_control.streams import StreamReader
        from dual_flexiv_control.streams import StreamRegistry
    except Exception:  # noqa: BLE001 - streams stack unavailable -> not detected
        return False

    run_dirs = sorted(
        (p for p in root.iterdir() if (p / "streams").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        try:
            entry = StreamRegistry(str(root), run_dir.name).get(key)
            if entry is None:
                continue
            reader = StreamReader.attach(entry)
            try:
                if reader.latest().n > 0:
                    return True
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - dead run / lapped buffer -> try next
            continue
    return False


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
