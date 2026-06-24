"""Hardware/system interfaces.

The Flexiv interfaces are stream producers (one process per arm). FACTR is an
on-request HTTP client (no stream) — the brain holds one and queries it on demand.
"""

from .factr import FactrClient
from .flexiv import FlexivInterface

__all__ = ["FlexivInterface", "FactrClient"]
