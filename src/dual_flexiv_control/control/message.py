"""Discrete control commands (brain→arm) carried on the reliable command channel.

The control channel's rings hold fixed-width ``float64`` vectors, so a discrete
command is encoded as ``[kind, arg0, arg1, ...]``. Ordering and de-duplication are
driven by the ring's per-slot sequence stamp (see :class:`CommandCursor`), not by
anything in the payload — so the opcode vector carries only the kind and its args.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

#: Index of the command kind within an encoded opcode vector; args follow.
CMD_KIND_IDX = 0
CMD_ARG0_IDX = 1


class CommandKind(IntEnum):
    """Discrete control events the brain can post to an arm.

    ``NONE`` is the zero value, so a freshly zeroed (never-written) command slot
    decodes to a no-op rather than an accidental STOP/HOME.
    """

    NONE = 0
    HOME = 1          # smooth move to a home posture (args optional: target q)
    STOP = 2          # stop the robot and return control to IDLE
    SWITCH_MODE = 3   # reserved: change control kind mid-run (arg0 = kind code)
    SERVO_ON = 4      # (re-)enable servos


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """A discrete control event, encodable to / decodable from a ring vector."""

    kind: CommandKind
    args: tuple[float, ...] = ()

    def encode(self, dim: int) -> np.ndarray:
        """Pack into a ``(dim,)`` float64 vector ``[kind, *args]`` (zero-padded)."""
        if CMD_ARG0_IDX + len(self.args) > dim:
            raise ValueError(
                f"command {self.kind.name} with {len(self.args)} args does not fit "
                f"in a width-{dim} command channel"
            )
        v = np.zeros(dim, dtype=np.float64)
        v[CMD_KIND_IDX] = float(int(self.kind))
        if self.args:
            v[CMD_ARG0_IDX : CMD_ARG0_IDX + len(self.args)] = self.args
        return v

    @classmethod
    def decode(cls, vec: np.ndarray) -> "ControlCommand":
        """Recover a command from a ring row (trailing zeros are kept as args)."""
        kind = CommandKind(int(round(float(vec[CMD_KIND_IDX]))))
        args = tuple(float(x) for x in np.asarray(vec[CMD_ARG0_IDX:]).ravel())
        return cls(kind=kind, args=args)
