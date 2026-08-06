"""Live Intel RealSense acquisition without SDK bag-file recording."""

from __future__ import annotations

import logging
import time

import numpy as np

from ...cameras import VIEW_COLOR
from ...cameras import VIEW_DEPTH
from ...configs import CameraCfg

log = logging.getLogger(__name__)


class RealSenseSource:
    """Capture synchronized color/depth frames from one RealSense pipeline.

    Frames in a pipeline frameset are device-synchronized. Their streams receive
    one host monotonic timestamp when the producer publishes them, which makes
    them participate in the application's existing software synchronization.
    No ``enable_record_to_file`` call is made: persistence remains LeRobot's job.
    """

    def __init__(self, cam: CameraCfg) -> None:
        self.cam = cam
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001

    def open(self) -> None:
        import pyrealsense2 as rs  # lazy: the SDK starts native threads

        unsupported = set(self.cam.views) - {VIEW_COLOR, VIEW_DEPTH}
        if unsupported:
            raise ValueError(f"RealSense views must be color/depth, got {sorted(unsupported)}")
        if not self.cam.views:
            raise ValueError("RealSense camera must declare at least one view")

        pipeline = rs.pipeline()
        config = rs.config()
        serial = self.cam.serial.strip()
        if self.cam.auto_serial:
            devices = list(rs.context().query_devices())
            if not devices:
                raise RuntimeError(
                    f"camera {self.cam.placement!r} has auto_serial=true but no "
                    "RealSense is connected"
                )
            if len(devices) > 1:
                found = [d.get_info(rs.camera_info.serial_number) for d in devices]
                log.warning(
                    "auto_serial (%s): multiple RealSense devices %s; using first",
                    self.cam.placement,
                    found,
                )
            serial = devices[0].get_info(rs.camera_info.serial_number)
        if serial:
            config.enable_device(serial)
        fps = int(self.cam.fps)
        if VIEW_COLOR in self.cam.views:
            config.enable_stream(
                rs.stream.color, self.cam.width, self.cam.height, rs.format.rgb8, fps
            )
        if VIEW_DEPTH in self.cam.views:
            config.enable_stream(
                rs.stream.depth, self.cam.width, self.cam.height, rs.format.z16, fps
            )

        profile = pipeline.start(config)
        if VIEW_DEPTH in self.cam.views:
            self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self._align = (
            rs.align(rs.stream.color)
            if self.cam.align_depth and VIEW_COLOR in self.cam.views and VIEW_DEPTH in self.cam.views
            else None
        )
        self._rs, self._pipeline = rs, pipeline
        log.info(
            "RealSense %s open (serial=%s, %dx%d @ %d Hz, views=%s, align_depth=%s)",
            self.cam.placement, serial or "first-available", self.cam.width,
            self.cam.height, fps, list(self.cam.views), bool(self._align),
        )

    def read(self):
        # A short timeout keeps shutdown and supervision responsive.
        try:
            timeout_ms = max(100, int(2000 / self.cam.fps))
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            if self._align is not None:
                frames = self._align.process(frames)
            out = {}
            if VIEW_COLOR in self.cam.views:
                frame = frames.get_color_frame()
                if not frame:
                    return None
                out[VIEW_COLOR] = np.ascontiguousarray(
                    np.asanyarray(frame.get_data()), dtype=np.uint8
                )
            if VIEW_DEPTH in self.cam.views:
                frame = frames.get_depth_frame()
                if not frame:
                    return None
                out[VIEW_DEPTH] = np.ascontiguousarray(
                    np.asanyarray(frame.get_data()).astype(np.float32) * self._depth_scale
                )
            return out
        except RuntimeError as exc:
            log.warning(
                "[%s] RealSense frame unavailable; skipping tick: %s",
                self.cam.placement,
                exc,
            )
            return None

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = self._align = self._rs = None


class FakeRealSenseSource:
    """Small synthetic source matching RealSense color/depth output."""

    def __init__(self, cam: CameraCfg) -> None:
        self.cam = cam
        self._t0 = time.monotonic()

    def open(self) -> None:
        self._t0 = time.monotonic()

    def read(self):
        h, w = self.cam.height, self.cam.width
        t = time.monotonic() - self._t0
        out = {}
        if VIEW_COLOR in self.cam.views:
            x = np.arange(w, dtype=np.uint16)[None, :]
            y = np.arange(h, dtype=np.uint16)[:, None]
            out[VIEW_COLOR] = np.stack(np.broadcast_arrays(
                (x + int(t * 30)) % 256, (y + 80) % 256, (x + y + 160) % 256
            ), axis=-1).astype(np.uint8)
        if VIEW_DEPTH in self.cam.views:
            out[VIEW_DEPTH] = np.full((h, w), 1.0 + 0.1 * np.sin(t), dtype=np.float32)
        return out

    def close(self) -> None:
        pass
