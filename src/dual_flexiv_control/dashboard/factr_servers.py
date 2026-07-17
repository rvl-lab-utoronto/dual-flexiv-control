"""Read the FACTR-Server teleops' health logs for the dashboard's Grav comp panel.

The grav-comp teleops narrate their whole lifecycle on stderr (which the
:class:`~dual_flexiv_control.interfaces.factr.FactrServerSupervisor` tees to
``logs/factr_health_<side>.log`` — the same files the FACTR-Server repo's Tail
tasks follow):

* boot: ``FACTR TELEOP <name>: best offsets: […]`` — calibration done;
* then ``… Please match starting joint pos. Current error: <e> | …`` twice a
  second until the operator moves the leader to the follower's start pose;
* then ``… Initial joint position matched.`` and the grav-comp loop begins,
  emitting one ``[health] id1:T=41C I=-241/910  id2:…`` line per second
  (per-servo temperature and present/limit current), plus loud
  ``[health] servo idN HARDWARE ERROR …`` / ``… disabled its own torque …``
  lines when a servo latches a fault (the classic silent grav-comp killer).

This module turns the tail of such a log into a :class:`TeleopHealth` snapshot
— the boot phase, the newest servo readings, and any recent fault lines — so
the panel can show *why* a leader isn't streaming yet (still calibrating? wants
the start pose matched?) and which servo is cooking, without anyone tailing
logs by hand. Reading is filesystem-only (the daemon and dashboard share the
machine); nothing here talks to the processes.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from dataclasses import field

#: Boot phases, in lifecycle order (values are what the UI prints).
CALIBRATING = "calibrating"
MATCHING = "match the start pose"
RUNNING = "grav comp running"

#: How much of the log tail to inspect. Health lines are ~200 B at 1 Hz, so
#: this covers minutes of quiet operation while staying a single small read.
_TAIL_BYTES = 32_768
#: Fault lines older than this many lines from the end are history, not alerts.
_ALERT_WINDOW_LINES = 40

#: ``id4:T=43C I=-89/1193`` -> (4, 43, -89, 1193)
_SERVO_RE = re.compile(r"id(\d+):T=(-?\d+)C I=(-?\d+)/(\d+)")
#: The rclpy log prefix carries a wall-clock epoch: ``[1784237385.093860358]``.
_ROS_TS_RE = re.compile(r"\[(\d{9,10}\.\d+)\]")
#: ``Current error: 8.312`` from the match-phase nag line.
_MATCH_ERR_RE = re.compile(r"Current error: ([0-9.]+)")


@dataclass(frozen=True)
class ServoHealth:
    """One Dynamixel servo's newest reading from a ``[health]`` summary line."""

    sid: int
    temp_c: int
    current: int          # present current (signed, raw units)
    current_limit: int


@dataclass(frozen=True)
class TeleopHealth:
    """One teleop's lifecycle snapshot, parsed from its health-log tail."""

    phase: str                        # CALIBRATING | MATCHING | RUNNING
    #: Match-phase distance to the start pose (the operator's progress bar).
    match_error: float | None = None
    #: Wall-clock stamp of the newest ``[health]`` summary, and its age now.
    ts: float | None = None
    age_s: float | None = None
    servos: tuple = ()                # ServoHealth, ascending id
    #: Recent fault lines (HARDWARE ERROR / self-disabled torque), newest last.
    alerts: tuple = ()

    @property
    def hottest(self) -> ServoHealth | None:
        return max(self.servos, key=lambda s: s.temp_c) if self.servos else None


def tail_lines(path: str, max_bytes: int = _TAIL_BYTES) -> list[str]:
    """The last lines of ``path`` (single bounded read), [] if unreadable."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read(max_bytes)
    except OSError:
        return []
    text = data.decode(errors="replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is almost surely cut mid-way
    return lines


def parse_servos(line: str) -> tuple:
    """Every ``idN:T=..C I=../..`` reading on a ``[health]`` summary line."""
    return tuple(
        ServoHealth(sid=int(m[0]), temp_c=int(m[1]), current=int(m[2]), current_limit=int(m[3]))
        for m in _SERVO_RE.findall(line)
    )


def _ros_ts(line: str) -> float | None:
    m = _ROS_TS_RE.search(line)
    return float(m.group(1)) if m else None


def read_teleop_health(log_path: str) -> TeleopHealth | None:
    """Parse one teleop's health log tail; None when there is nothing yet.

    The phase is the newest lifecycle marker in the tail (a fresh log is a boot
    in its calibration read); servo readings come from the newest ``[health]``
    summary; fault lines within the last few dozen lines surface as alerts.
    """
    lines = tail_lines(log_path)
    if not lines:
        return None

    alerts = tuple(
        line.strip() for line in lines[-_ALERT_WINDOW_LINES:]
        if "HARDWARE ERROR" in line or "disabled its own torque" in line
    )

    phase: str | None = None
    match_error: float | None = None
    ts: float | None = None
    servos: tuple = ()
    for line in reversed(lines):
        if "[health] servo" in line or "[health] read failed" in line:
            continue  # fault/noise lines say nothing about the phase
        if "[health]" in line:
            phase = RUNNING
            servos = parse_servos(line)
            ts = _ros_ts(line)
            break
        if "Initial joint position matched" in line:
            phase = RUNNING  # grav-comp loop starts right after; no summary yet
            break
        if "Please match starting joint pos" in line:
            phase = MATCHING
            m = _MATCH_ERR_RE.search(line)
            match_error = float(m.group(1)) if m else None
            break
        if "best offsets" in line:
            phase = MATCHING  # calibration done; the match loop is next
            break
    if phase is None:
        phase = CALIBRATING  # booted, no marker yet: the calibration read

    age_s = None if ts is None else max(0.0, time.time() - ts)
    return TeleopHealth(
        phase=phase, match_error=match_error, ts=ts, age_s=age_s,
        servos=servos, alerts=alerts,
    )
