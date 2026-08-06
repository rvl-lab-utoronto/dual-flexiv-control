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
from .channel import GRIPPER
from .channel import SETPOINT
from .channel import CommandCursor
from .channel import action_dim
from .channel import action_field
from .channel import action_hold_fields
from .channel import control_channel_name
from .channel import control_specs
from .channel import estimate_chunk_end
from .channel import estimate_chunk_trajectory
from .channel import gripper_channel_name
from .channel import gripper_spec
from .channel import horizon_kind
from .channel import horizon_signals
from .channel import pack_action
from .channel import pack_streamed
from .channel import setpoint_dim
from .channel import slice_streamed
from .channel import streamed_layout
from .convention import convert_factr_to_rizon
from .convention import normalize_gripper
from .convention import offsets_from_straight_pose
from .message import CommandKind
from .message import ControlCommand

__all__ = [
    "CommandKind",
    "ControlCommand",
    "CommandCursor",
    "action_field",
    "action_dim",
    "action_hold_fields",
    "control_channel_name",
    "control_specs",
    "estimate_chunk_end",
    "estimate_chunk_trajectory",
    "gripper_channel_name",
    "gripper_spec",
    "horizon_kind",
    "horizon_signals",
    "setpoint_dim",
    "streamed_layout",
    "pack_action",
    "pack_streamed",
    "slice_streamed",
    "convert_factr_to_rizon",
    "normalize_gripper",
    "offsets_from_straight_pose",
    "SETPOINT",
    "COMMAND",
    "GRIPPER",
]
