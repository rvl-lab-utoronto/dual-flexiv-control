"""ZED camera sources for the generic camera interface.

Serves both the ZED X Nano wrist cameras and the static ZED 2 external stereo
camera. New code should use :class:`dual_flexiv_control.interfaces.camera.CameraInterface`
with ``CameraCfg.source == "zed"``; ``ZedInterface`` remains as a compatibility
wrapper for older imports.
"""

from .interface import ZedInterface
from .source import FakeZedSource
from .source import ZedSource

__all__ = ["ZedInterface", "ZedSource", "FakeZedSource"]
