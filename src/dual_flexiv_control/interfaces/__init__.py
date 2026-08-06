"""Hardware/system interfaces.

The Flexiv (arm proprioception), camera, and FACTR (leader-arm
WebSockets) interfaces are stream producers. Each publishes shared-memory data
that the brain consumes without opening hardware/network connections itself.
"""

from .factr import FactrClient
from .flexiv import FlexivInterface
from .realsense import RealSenseInterface
from .zed import ZedInterface

__all__ = ["FlexivInterface", "ZedInterface", "RealSenseInterface", "FactrClient"]
