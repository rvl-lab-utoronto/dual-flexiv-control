"""Canonical proprioception signal definitions for a Flexiv arm.

These are the six per-arm signals the brain observes, mapped one-to-one onto
Flexiv RDK 1.8 ``RobotStates`` fields (see ``interfaces.flexiv.states``). Each
becomes its own stream, named ``"<side>/<signal>"`` (e.g. ``"right/tau"``).

Stream *dimensions* are no longer hard-coded here — they come from the Hydra
config (``arm.streams.<signal>.dim``). :func:`streams_to_specs` turns a config
``streams`` mapping into :class:`StreamSpec` objects.
"""

from __future__ import annotations

from enum import Enum

from .streams.spec import StreamSpec

#: Serial-arm joint-space DoF for a single Flexiv arm (``kSerialJointDoF``).
DEFAULT_DOF = 7

#: Cartesian DoF (force/moment, linear/angular velocity).
CART_DOF = 6

#: Pose size: 3 position + 4 quaternion (``kPoseSize``).
POSE_SIZE = 7


class Side(str, Enum):
    """Which arm of the bimanual setup a stream belongs to."""

    LEFT = "left"
    RIGHT = "right"


#: Stable ordering / canonical set of the proprio signals.
PROPRIO_SIGNALS: tuple[str, ...] = ("q", "dq", "tau", "wrench", "eef", "eef_vel")


def stream_name(side: "Side | str", signal: str) -> str:
    s = side.value if isinstance(side, Side) else side
    return f"{s}/{signal}"


def proprio_stream_names(side: "Side | str") -> list[str]:
    return [stream_name(side, sig) for sig in PROPRIO_SIGNALS]


def streams_to_specs(prefix: str, streams_cfg) -> list[StreamSpec]:
    """Build :class:`StreamSpec` objects from a config ``streams`` mapping.

    ``streams_cfg`` maps a leaf name (a proprio signal like ``q``, or a FACTR side
    like ``left``) to an object with ``dim``/``dtype``/``capacity``/``rate_hz``
    (i.e. a ``StreamCfg``). Each becomes a stream named ``"<prefix>/<leaf>"``.
    Insertion order is preserved.
    """
    return [
        StreamSpec(
            name=f"{prefix}/{leaf}",
            dim=cfg.dim,
            capacity=cfg.capacity,
            dtype=cfg.dtype,
            rate_hz=cfg.rate_hz,
        )
        for leaf, cfg in streams_cfg.items()
    ]
