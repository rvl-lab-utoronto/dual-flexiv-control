"""FACTR leader → Rizon follower joint-space convention (pure math, no SDK).

Captured verbatim from the hardware-validated teleop test (``rizon_tests``): the
The FACTR WebSocket returns ``DoF+1`` joint values in radians (the arm joints
plus a trailing gripper value). Mapping to Rizon joint targets is: drop the
gripper, convert to degrees, add per-joint offsets, flip the sign of selected
joints, and convert back to radians. Targets deliberately remain on their
continuous branch; independently wrapping samples would create artificial
``2π`` command jumps at the branch cut.

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
    for j in conv.sign_flip_joints:                   # flip after offsets
        deg[j] = -deg[j]
    return np.radians(deg)


def offsets_from_straight_pose(q_leader_rad, conv: "JointConventionCfg") -> list[float]:
    """Solve for ``offsets_deg`` that map a captured *straight* leader pose to zero.

    Calibration inverse of :func:`convert_factr_to_rizon`: pose the leader so the
    follower would stand at its URDF home (every joint at 0 — all links straight),
    read one leader sample ``q_leader_rad`` (rad, ``DoF+1``), and this returns the
    per-joint ``offsets_deg`` that make ``convert_factr_to_rizon(q_leader_rad, …)``
    come out all-zero. Since the convention adds the offset *before* the sign flip
    and the target is zero, the flip is irrelevant here, so::

        offsets_deg[j] = wrap(-degrees(q_leader_arm[j]))   # wrap to [-180, 180]

    The returned list has one entry per follower joint (gripper dropped via
    ``conv.drop_trailing``). Offset normalization is safe here because offsets are
    static calibration parameters, not live command samples.
    """
    q = np.asarray(q_leader_rad, dtype=np.float64).ravel()
    if conv.drop_trailing:
        q = q[: len(q) - conv.drop_trailing]          # drop trailing gripper value(s)
    offsets = -np.degrees(q)
    offsets = (offsets + 180.0) % 360.0 - 180.0       # canonical static offsets
    return [float(o) for o in offsets]


def normalize_gripper(raw_value, conv: "JointConventionCfg") -> float:
    """Map FACTR's trailing gripper value (raw servo radians) to a 0..1 fraction.

    FACTR serves the gripper as an un-normalized servo angle in radians (no 0..1
    mapping exists anywhere in FACTR). When both ``gripper_open`` and
    ``gripper_closed`` are set on the convention, this linearly maps
    ``gripper_open → 0.0`` and ``gripper_closed → 1.0`` and clips to ``[0, 1]``;
    when either is unset (or they are equal), the raw value is returned unchanged so
    an uncalibrated setup records exactly what it did before. Endpoints may be given
    in either order — a closed reading below the open reading still maps correctly.
    """
    lo = conv.gripper_open
    hi = conv.gripper_closed
    raw = float(raw_value)
    if lo is None or hi is None or lo == hi:
        return raw
    frac = (raw - lo) / (hi - lo)
    return float(min(1.0, max(0.0, frac)))
