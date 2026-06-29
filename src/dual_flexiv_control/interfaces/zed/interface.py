"""The ZED interface node: one process per camera, publishing image streams."""

from __future__ import annotations

import numpy as np

from ...cameras import camera_stream_name
from ...cameras import camera_streams_to_specs
from ...configs import CameraCfg
from ...configs import RuntimeCfg
from ...process import StreamProducerNode
from ...streams.spec import StreamSpec
from .source import FakeZedSource
from .source import ZedSource


class ZedInterface(StreamProducerNode):
    """Reads one ZED camera and publishes its views as image streams.

    Stream names are ``"cam/<name>/<view>"`` (e.g. ``cam/wrist_left/left``), with
    dims derived from the camera resolution. One :class:`ZedInterface` runs per
    camera, so the bimanual rig (two ZED X Nano wrist cams + one static ZED 2)
    spawns three. The capture rate (``cam.fps``) is this producer's loop rate.
    """

    def __init__(self, name: str, cam: CameraCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(
            name=f"zed:{name}",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=cam.fps,
        )
        self.cam_name = name
        self.cam = cam
        self.sim = runtime.sim
        self._source: ZedSource | FakeZedSource | None = None

    def declare_streams(self) -> list[StreamSpec]:
        return camera_streams_to_specs(self.cam_name, self.cam)

    def open_source(self) -> None:
        self._source = FakeZedSource(self.cam) if self.sim else ZedSource(self.cam)
        self._source.open()

    def poll(self) -> dict[str, np.ndarray] | None:
        frames = self._source.read()
        if not frames:                       # no new frame this tick -> skip writes
            return None
        # Flatten each frame to the ring's fixed-dim vector: (H,W,3) for RGB or
        # (H,W) for depth. The writer casts to the stream dtype (uint8 RGB /
        # float32 depth); consumers reshape with cameras.reshape_frame.
        return {
            camera_stream_name(self.cam_name, view): np.ascontiguousarray(img).reshape(-1)
            for view, img in frames.items()
        }

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None
