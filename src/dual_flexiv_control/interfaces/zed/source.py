"""Data sources for a single ZED camera.

:class:`ZedSource` wraps a real ``pyzed.sl`` camera. The same SDK and code path
serve a ZED 2 (USB3 stereo) and a ZED X / ZED X Nano (GMSL2 capture) — the SDK
auto-detects the device; only :class:`~dual_flexiv_control.configs.CameraCfg`
fields (resolution, serial, depth) differ. :class:`FakeZedSource` synthesises
animated frames so the whole multiprocess + shared-memory pipeline runs with no
hardware (``runtime.sim=true``), mirroring ``FakeFlexivSource``.

Targets the **ZED SDK 4.x** Python API (``pyzed``) — e.g. the
``get_camera_information().camera_configuration.resolution`` path and
``InitParameters.set_from_serial_number`` used below are 4.x. ``pyzed`` is not on
PyPI (so it is not a project dependency); install it with Stereolabs'
``get_python_api.py`` from a matching SDK. Only the real path needs it — sim does
not import ``pyzed`` at all.

Both expose the same tiny interface: ``open()`` then repeated ``read()``
returning ``{view: ndarray}`` — RGB views are ``(H,W,3)`` uint8, depth is
``(H,W)`` float32 metres — or ``None`` when no new frame is ready this tick.

Like ``flexivrdk``, the ZED SDK starts native capture/CUDA threads, so
``pyzed.sl`` is imported lazily inside :meth:`ZedSource.open` (i.e. in the child
process), never at module import — consistent with the spawn-only mandate (see
:mod:`dual_flexiv_control.process`).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ...cameras import VIEW_DEPTH
from ...cameras import VIEW_RIGHT
from ...configs import CameraCfg

log = logging.getLogger(__name__)


def _parse_serial(serial: str) -> int | None:
    """ZED serials are numeric; return the int, or ``None`` to open the first camera."""
    serial = (serial or "").strip()
    if not serial:
        return None
    try:
        return int(serial)
    except ValueError:
        log.warning("ZED serial %r is not numeric; opening first available camera", serial)
        return None


class ZedSource:
    """Real ``pyzed.sl`` connection to one ZED camera; read-only frame capture."""

    def __init__(self, cam: CameraCfg) -> None:
        self.cam = cam
        self._cam = None
        self._runtime = None
        self._mats: dict = {}
        self._sl = None

    def open(self) -> None:
        import pyzed.sl as sl  # imported in the child process only (native threads)

        cam = self.cam
        if VIEW_DEPTH in cam.views and cam.depth_mode.upper() == "NONE":
            raise ValueError(
                f"camera {cam.placement!r} publishes a 'depth' view but depth_mode is NONE; "
                "set depth_mode (e.g. PERFORMANCE/NEURAL) to enable depth"
            )

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, cam.resolution)
        init.camera_fps = int(cam.fps)
        init.depth_mode = getattr(sl.DEPTH_MODE, cam.depth_mode)
        init.coordinate_units = sl.UNIT.METER
        serial = _parse_serial(cam.serial)
        if serial is not None:
            init.set_from_serial_number(serial)

        handle = sl.Camera()
        status = handle.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED open failed ({cam.placement}, serial={cam.serial!r}): {status}"
            )

        # Validate the SDK's actual resolution matches the configured dims, so a
        # wrong `resolution` enum fails fast instead of silently producing
        # wrong-sized frames that mismatch the declared stream `dim`.
        res = handle.get_camera_information().camera_configuration.resolution
        if (int(res.width), int(res.height)) != (cam.width, cam.height):
            handle.close()
            raise RuntimeError(
                f"ZED {cam.placement} resolution {cam.resolution!r} yields "
                f"{res.width}x{res.height}, but config says {cam.width}x{cam.height}"
            )

        self._sl = sl
        self._cam = handle
        self._runtime = sl.RuntimeParameters()
        self._mats = {view: sl.Mat() for view in cam.views}
        log.info(
            "ZED %s open (serial=%s, %dx%d @ %.0f Hz, views=%s)",
            cam.placement, cam.serial, cam.width, cam.height, cam.fps, list(cam.views),
        )

    def read(self):
        sl = self._sl
        if self._cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None  # no new frame ready this tick -> skip (stream stays live)
        out: dict[str, np.ndarray] = {}
        try:
            for view, mat in self._mats.items():
                if view == VIEW_DEPTH:
                    self._cam.retrieve_measure(mat, sl.MEASURE.DEPTH)
                    # F32_C1 depth in metres (may contain NaN/Inf for invalid pixels).
                    out[view] = np.ascontiguousarray(mat.get_data(), dtype=np.float32)
                else:
                    self._cam.retrieve_image(
                        mat, sl.VIEW.RIGHT if view == VIEW_RIGHT else sl.VIEW.LEFT
                    )
                    # ZED images are BGRA uint8; drop alpha and reorder BGR->RGB. The
                    # advanced index returns a fresh C-contiguous copy, so it is safe
                    # to keep after the SDK reuses `mat` on the next grab.
                    out[view] = np.ascontiguousarray(mat.get_data()[:, :, [2, 1, 0]])
        except Exception:  # noqa: BLE001 - a transient retrieve error should skip
            # the tick (same as a failed grab), not crash the whole camera process
            # and lose the stream. Don't return a partially-filled frame.
            log.exception("[%s] frame retrieve failed; skipping tick", self.cam.placement)
            return None
        return out

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            finally:
                self._cam = None
                self._mats = {}
                self._sl = None


class FakeZedSource:
    """Synthetic animated frames for hardware-free runs. Same shapes as ZedSource."""

    def __init__(self, cam: CameraCfg, seed: int = 0) -> None:
        self.cam = cam
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(seed)
        # Precompute the per-pixel coordinate ramps once (HD frames are large).
        self._xs = np.linspace(0.0, 255.0, int(cam.width), dtype=np.float32)[None, :]
        self._ys = np.linspace(0.0, 255.0, int(cam.height), dtype=np.float32)[:, None]

    def open(self) -> None:
        self._t0 = time.monotonic()

    def read(self):
        t = time.monotonic() - self._t0
        return {view: self._frame(view, t) for view in self.cam.views}

    def _frame(self, view: str, t: float) -> np.ndarray:
        if view == VIEW_DEPTH:
            return self._depth(t)
        return self._rgb(view, t)

    def _rgb(self, view: str, t: float) -> np.ndarray:
        # Moving diagonal gradient; a per-view tint makes left != right so a
        # stereo consumer can tell the views apart.
        base = (self._xs + self._ys + (t * 40.0)) % 256.0       # (H,W) float32
        tint = 40.0 if view == VIEW_RIGHT else 0.0
        h, w = int(self.cam.height), int(self.cam.width)
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[..., 0] = base.astype(np.uint8)
        img[..., 1] = ((base + 85.0 + tint) % 256.0).astype(np.uint8)
        img[..., 2] = ((base + 170.0) % 256.0).astype(np.uint8)
        return img

    def _depth(self, t: float) -> np.ndarray:
        # Smooth radial field oscillating ~1.0-2.0 m.
        ys = np.linspace(-1.0, 1.0, int(self.cam.height), dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, int(self.cam.width), dtype=np.float32)[None, :]
        r = np.sqrt(xs * xs + ys * ys)
        return np.ascontiguousarray(1.5 + 0.5 * np.sin(2.0 * r - t), dtype=np.float32)

    def close(self) -> None:
        pass
