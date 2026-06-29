"""FACTR leader → Rizon follower joint-space convention (pure math, no SDK).

Captured verbatim from the hardware-validated teleop test (``rizon_tests``): the
FACTR FastAPI server returns ``DoF+1`` joint values in radians (the arm joints
plus a trailing gripper value). Mapping to Rizon joint targets is: drop the
gripper, convert to degrees, add per-joint offsets, flip the sign of selected
joints, wrap to ``[-180, 180]``, convert back to radians.

This lives on the **brain** side — the brain reads FACTR and posts the converted
radian setpoints onto the control channel; the arm process never talks to FACTR.
Only the *left* arm's offsets/sign-flips are known from the test; the right arm's
convention must be measured (do not assume symmetry) — see ``conf/arm/flexiv.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle with configs
    from ..configs import JointConventionCfg


def convert_factr_to_rizon(q_leader_rad, conv: "JointConventionCfg") -> np.ndarray:
    """Convert a FACTR leader sample (rad, ``DoF+1``) to Rizon joint targets (rad).

    Returns a ``(DoF,)`` float64 array of joint positions in **radians** (the
    setpoint-channel convention). The MoveJ bootstrap on the arm converts these
    to degrees for ``JPos``; ``SendJointPosition`` consumes the radians directly.
    """
    q = np.asarray(q_leader_rad, dtype=np.float64).ravel()
    if conv.drop_trailing:
        q = q[: len(q) - conv.drop_trailing]          # drop trailing gripper value(s)
    deg = np.degrees(q) + np.asarray(conv.offsets_deg[: len(q)], dtype=np.float64)
    for j in conv.sign_flip_joints:                   # flip AFTER offsets, BEFORE wrap
        deg[j] = -deg[j]
    if conv.wrap_deg:
        deg = (deg + 180.0) % 360.0 - 180.0
    return np.radians(deg)
