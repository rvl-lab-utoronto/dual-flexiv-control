"""Deprecated Rerun stream visualizer.

Viser is the dashboard default.  This module keeps the previous backend usable
as an explicitly started, read-only stream consumer during the transition.
"""

from .service import RerunService
from .service import start_service

__all__ = ["RerunService", "start_service"]
