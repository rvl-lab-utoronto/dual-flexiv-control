"""Hardware/system interfaces.

The Flexiv (arm proprioception) and ZED (camera frames) interfaces are stream
producers — one process each (per arm / per camera). FACTR is an on-request HTTP
client (no stream) — the brain holds one and queries it on demand.
"""

from .factr import FactrClient
from .flexiv import FlexivInterface
from .zed import ZedInterface

__all__ = ["FlexivInterface", "ZedInterface", "FactrClient"]
