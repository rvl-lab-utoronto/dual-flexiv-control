"""RealSense producer using the same shared-memory stream path as ZED."""

from __future__ import annotations

import numpy as np

from ...cameras import camera_stream_name
from ...cameras import camera_streams_to_specs
from ...configs import CameraCfg
from ...configs import RuntimeCfg
from ...process import StreamProducerNode
from ...streams.spec import StreamSpec
from .source import FakeRealSenseSource
from .source import RealSenseSource


class RealSenseInterface(StreamProducerNode):
    def __init__(self, name: str, cam: CameraCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(f"realsense:{name}", runtime.runtime_dir, run_id, cam.fps)
        self.cam_name = name
        self.cam = cam
        self.sim = runtime.sim
        self._source: RealSenseSource | FakeRealSenseSource | None = None

    def declare_streams(self) -> list[StreamSpec]:
        return camera_streams_to_specs(self.cam_name, self.cam)

    def open_source(self) -> None:
        self._source = FakeRealSenseSource(self.cam) if self.sim else RealSenseSource(self.cam)
        self._source.open()

    def poll(self) -> dict[str, np.ndarray] | None:
        frames = self._source.read()
        if not frames:
            return None
        return {
            camera_stream_name(self.cam_name, view): np.ascontiguousarray(frame).reshape(-1)
            for view, frame in frames.items()
        }

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None
