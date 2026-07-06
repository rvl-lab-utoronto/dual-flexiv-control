"""Teleoperated data collection: read FACTR + proprio, command the arms, record.

The collection loop runs at :attr:`CollectionCfg.frequency_hz` (default 15 Hz),
posts FACTR-derived joint setpoints to every control-enabled arm, samples all
cameras software-synchronised, and exports demonstrations in the LeRobot format.

* :class:`FrameBuilder` — config -> LeRobot ``features`` + per-tick frame packing.
* :class:`CollectionLoop` — the reusable teleop+record core (testable in-process).
* :class:`CollectionNode` — the spawned-process node wired into the live system.
* :class:`LeRobotRecorder` — the LeRobot dataset sink (lazy ``lerobot`` import).
"""

from .features import FrameBuilder
from .loop import CollectionLoop
from .loop import CollectionNode
from .recorder import LeRobotRecorder
from .recorder import Recorder
from .recorder import RecorderUnavailable

__all__ = [
    "FrameBuilder",
    "CollectionLoop",
    "CollectionNode",
    "LeRobotRecorder",
    "Recorder",
    "RecorderUnavailable",
]
