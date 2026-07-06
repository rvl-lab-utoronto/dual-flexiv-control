"""Hardware/system interfaces.

The Flexiv (arm proprioception) and camera-frame interfaces are stream producers
— one process each (per arm / per camera). FACTR is an on-request HTTP client
(no stream) — the brain holds one and queries it on demand.
"""

from .camera import CameraInterface
from .factr import FactrClient
from .flexiv import FlexivInterface

__all__ = ["FlexivInterface", "CameraInterface", "FactrClient"]
