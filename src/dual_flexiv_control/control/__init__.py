"""Control channel: the brain→arm command IPC category (distinct from streams).

Telemetry *streams* flow arm→brain (SPMC, latest-wins-with-history). The
*control channel* is the inverse: the brain posts commands to each arm. It reuses
the same :class:`~dual_flexiv_control.streams.ring.SharedRingBuffer` transport but
with two delivery disciplines:

* **setpoint** — a latest-wins mailbox for high-rate target updates (the brain
  posts the freshest control target; the arm reads ``latest()`` and drops stale).
* **command** — a reliable, in-order stream of discrete events (home, stop,
  switch-mode), consumed gap-free via :class:`CommandCursor`.

Channels live under ``<run_id>/control/`` (a sibling of ``<run_id>/streams/``) so
the two categories never alias during discovery.
"""

from .channel import COMMAND
from .channel import SETPOINT
from .channel import CommandCursor
from .channel import control_channel_name
from .channel import control_specs
from .channel import pack_streamed
from .channel import setpoint_dim
from .channel import slice_streamed
from .channel import streamed_layout
from .convention import convert_factr_to_rizon
from .message import CommandKind
from .message import ControlCommand

__all__ = [
    "CommandKind",
    "ControlCommand",
    "CommandCursor",
    "control_channel_name",
    "control_specs",
    "setpoint_dim",
    "streamed_layout",
    "pack_streamed",
    "slice_streamed",
    "convert_factr_to_rizon",
    "SETPOINT",
    "COMMAND",
]
