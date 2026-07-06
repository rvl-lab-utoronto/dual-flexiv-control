"""Generic camera interface node: one process per camera."""

from __future__ import annotations

import numpy as np

from ...cameras import camera_stream_name
from ...cameras import camera_streams_to_specs
from ...configs import CameraCfg
from ...configs import RuntimeCfg
from ...process import StreamProducerNode
from ...streams.spec import StreamSpec
from .source import CameraSource
from .source import make_camera_source


class CameraInterface(StreamProducerNode):
    """Reads one camera source and publishes configured image views as streams.

    Sources return shaped frames keyed by view (for example ``{"left": HxWx3}``).
    This interface owns the common stream behavior: flattening each frame and
    writing it to ``cam/<camera>/<view>`` shared-memory streams.
    """

    def __init__(
        self,
        name: str,
        cam: CameraCfg,
        runtime: RuntimeCfg,
        run_id: str,
        *,
        node_prefix: str = "camera",
    ) -> None:
        super().__init__(
            name=f"{node_prefix}:{name}",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=cam.fps,
        )
        self.cam_name = name
        self.cam = cam
        self.sim = runtime.sim
        self._source: CameraSource | None = None

    def declare_streams(self) -> list[StreamSpec]:
        return camera_streams_to_specs(self.cam_name, self.cam)

    def open_source(self) -> None:
        self._source = make_camera_source(self.cam, runtime_sim=self.sim)
        self._source.open()

    def poll(self) -> dict[str, np.ndarray] | None:
        frames = self._source.read()
        if not frames:
            return None
        self._sample_t_ns = getattr(self._source, "latest_t_ns", None)
        return {
            camera_stream_name(self.cam_name, view): np.ascontiguousarray(img).reshape(-1)
            for view, img in frames.items()
        }

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None
