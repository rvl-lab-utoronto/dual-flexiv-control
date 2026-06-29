"""ZED camera interface: real ``pyzed.sl`` capture, one process per camera.

Serves both the ZED X Nano wrist cameras and the static ZED 2 external stereo
camera. Publishes image streams named ``"cam/<name>/<view>"`` (uint8 RGB / float32
depth); see :mod:`dual_flexiv_control.cameras` for the view/spec conventions.
"""

from .interface import ZedInterface
from .source import FakeZedSource
from .source import ZedSource

__all__ = ["ZedInterface", "ZedSource", "FakeZedSource"]
