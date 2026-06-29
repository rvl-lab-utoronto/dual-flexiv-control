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
