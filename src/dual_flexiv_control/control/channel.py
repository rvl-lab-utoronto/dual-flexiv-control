"""Naming, spec builders, and the reliable-command cursor for control channels.

The transport is the existing :class:`~dual_flexiv_control.streams.stream.StreamWriter`
/ :class:`StreamReader` pair — they already have exactly the ownership semantics a
control channel needs (the writer owns and unlinks the segment, the reader attaches
and never unlinks). This module adds only what is *new* relative to telemetry:

* the ``cmd/<side>/<leaf>`` naming,
* turning a :class:`ControlCfg` into the per-channel :class:`StreamSpec`s, and
* :class:`CommandCursor`, the gap-free in-order consumer for discrete commands.

The high-rate *setpoint* channel needs no new consumer logic: the arm just calls
``reader.latest()`` (already latest-wins / drop-stale).
"""

from __future__ import annotations

import logging

import numpy as np

from ..streams.spec import StreamSpec
from ..streams.stream import StreamReader

log = logging.getLogger(__name__)

#: Logical leaf names for an arm's two control channels.
SETPOINT = "setpoint"
COMMAND = "command"


def control_channel_name(side: str, leaf: str) -> str:
    """The logical name of a control channel, e.g. ``cmd/left/setpoint``."""
    return f"cmd/{side}/{leaf}"


def setpoint_dim(ctrl_cfg) -> int:
    """Width of the setpoint vector: the sum of the *streamed* command fields.

    ``ControlCfg.command`` maps every command-struct field to its dim; only the
    fields listed in ``ControlCfg.streamed`` are sent per tick (the rest, e.g.
    ``dq_max``/``ddq_max``, are static limits taken from the coefficients).
    """
    return int(sum(ctrl_cfg.command[f] for f in ctrl_cfg.streamed))


def streamed_layout(ctrl_cfg) -> list[tuple[str, int, int]]:
    """``[(field, start, end)]`` slices of the setpoint vector, in ``streamed`` order."""
    out: list[tuple[str, int, int]] = []
    off = 0
    for f in ctrl_cfg.streamed:
        d = int(ctrl_cfg.command[f])
        out.append((f, off, off + d))
        off += d
    return out


def pack_streamed(ctrl_cfg, fields: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the ``streamed`` fields into one setpoint vector (brain side)."""
    parts = [np.asarray(fields[f], dtype=np.float64).ravel() for f in ctrl_cfg.streamed]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def slice_streamed(ctrl_cfg, vec: np.ndarray) -> dict[str, np.ndarray]:
    """Split a received setpoint vector back into its named fields (arm side)."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    return {f: v[s:e].copy() for f, s, e in streamed_layout(ctrl_cfg)}


#: The one streamed command field a learned policy's action fills, per control
#: kind — its *action space*. The policy emits this field (per side); the arm's
#: RDK mode (set from ``ControlCfg.mode``) defines how it's realised. Everything
#: else the kind streams is filled by :func:`pack_action` (feedforward → zeros;
#: absolute non-primary targets → held at the measured value).
_ACTION_PRIMARY = {
    "qpos": "q_d",          # joint positions
    "qvel": "dq_d",         # joint velocities (arm integrates to q_d)
    "end_effector": "pose_d",   # TCP pose
    "eef_vel": "twist_d",   # TCP twist (arm integrates to pose_d)
    "force": "wrench_d",    # TCP wrench on force-controlled axes
}

#: Streamed fields that are *absolute* targets (not feedforward): when such a
#: field is streamed but is NOT the primary, it must be held at the measured value
#: rather than zeroed (e.g. ``pose_d`` on the motion axes under ``force`` control).
_ABSOLUTE_FIELDS = frozenset({"q_d", "pose_d"})


def action_field(ctrl_cfg) -> str:
    """The streamed field a policy action fills for this control kind (its action space)."""
    try:
        return _ACTION_PRIMARY[ctrl_cfg.kind]
    except KeyError:
        raise ValueError(
            f"control kind {ctrl_cfg.kind!r} has no policy action mapping "
            f"(known: {sorted(_ACTION_PRIMARY)})"
        ) from None


def action_dim(ctrl_cfg) -> int:
    """Width of one side's policy action for this kind (the primary field's dim)."""
    return int(ctrl_cfg.command[action_field(ctrl_cfg)])


def action_hold_fields(ctrl_cfg) -> list[str]:
    """Streamed fields that must be *held at measured* (absolute, non-primary).

    These need a measured value supplied to :func:`pack_action` (via ``held``);
    for the wired kinds only ``force`` has one (``pose_d`` on the motion axes).
    """
    primary = action_field(ctrl_cfg)
    return [f for f in ctrl_cfg.streamed if f != primary and f in _ABSOLUTE_FIELDS]


def pack_action(ctrl_cfg, primary_value, held: dict | None = None) -> np.ndarray:
    """One setpoint vector from a policy action for *any* control kind.

    ``primary_value`` is the policy's per-side output (the :func:`action_field`);
    it fills the primary streamed field. Remaining streamed fields are filled as:
    feedforward fields (``dq_d``/``twist_d``) → zeros; absolute non-primary fields
    (see :func:`action_hold_fields`) → the corresponding entry in ``held`` (the
    measured value), which must be supplied. The result matches the width the arm's
    :func:`slice_streamed` expects, so the same setpoint channel serves every kind.
    """
    primary = action_field(ctrl_cfg)
    held = held or {}
    fields: dict[str, np.ndarray] = {}
    for f in ctrl_cfg.streamed:
        if f == primary:
            fields[f] = np.asarray(primary_value, dtype=np.float64).ravel()
        elif f in _ABSOLUTE_FIELDS:
            if f not in held:
                raise ValueError(
                    f"control kind {ctrl_cfg.kind!r} streams absolute field {f!r} "
                    f"that the policy does not emit; a measured value to hold is required"
                )
            fields[f] = np.asarray(held[f], dtype=np.float64).ravel()
        else:
            fields[f] = np.zeros(int(ctrl_cfg.command[f]), dtype=np.float64)
    return pack_streamed(ctrl_cfg, fields)


def control_specs(side: str, ctrl_cfg) -> dict[str, StreamSpec]:
    """Build the ``{SETPOINT, COMMAND}`` channel specs for one arm's control."""
    ch = ctrl_cfg.channel
    return {
        SETPOINT: StreamSpec(
            name=control_channel_name(side, SETPOINT),
            dim=setpoint_dim(ctrl_cfg),
            capacity=ch.setpoint_capacity,
            dtype=ch.dtype,
            rate_hz=ch.rate_hz,
        ),
        COMMAND: StreamSpec(
            name=control_channel_name(side, COMMAND),
            dim=ch.command_dim,
            capacity=ch.command_capacity,
            dtype=ch.dtype,
            rate_hz=ch.rate_hz,
        ),
    }


class CommandCursor:
    """Reliable, in-order consumer of one arm's discrete command channel.

    Wraps a :class:`StreamReader` and yields only command rows published *after*
    this cursor was created. Baselining at the current head on attach is the fix
    for the startup race: discrete commands the brain may have posted before the
    arm began consuming (e.g. a STOP from a prior aborted attempt) are **not**
    replayed. ``drain_new`` also detects (and loudly logs) lost commands when the
    arm has fallen so far behind that the ring lapped it — for a low-rate command
    channel that is a real fault signal, not something to absorb silently.
    """

    def __init__(self, reader: StreamReader) -> None:
        self._reader = reader
        head = reader.latest()
        self._last_seq = int(head.seq[-1]) if head.n else -1

    @property
    def name(self) -> str:
        return self._reader.name

    def drain_new(self) -> list[np.ndarray]:
        """Return every command row with ``seq > cursor``, oldest → newest."""
        batch = self._reader.last(self._reader.capacity)
        if batch.n == 0:
            return []
        first = int(batch.seq[0])
        last = int(batch.seq[-1])
        if self._last_seq >= 0 and first > self._last_seq + 1:
            log.error(
                "control command channel %s: %d command(s) lost (cursor at %d, "
                "oldest still available %d) — consumer fell behind ring capacity",
                self._reader.name,
                first - self._last_seq - 1,
                self._last_seq,
                first,
            )
        rows = [
            np.array(batch.data[i], copy=True)
            for i in range(batch.n)
            if int(batch.seq[i]) > self._last_seq
        ]
        self._last_seq = last
        return rows

    def close(self) -> None:
        self._reader.close()
