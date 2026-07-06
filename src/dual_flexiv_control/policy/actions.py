"""Action-vector layout for eval: mirrors collection's ``action`` feature.

A policy emits, per control-enabled arm (side order), the action for that arm's
control kind followed by one gripper scalar. The *kind* determines the primary
field the policy outputs — ``q_d`` for ``qpos``, ``dq_d`` for ``qvel``, ``pose_d``
for ``end_effector``, ``twist_d`` for ``eef_vel``, ``wrench_d`` for ``force`` (see
:func:`~dual_flexiv_control.control.action_field`). For ``qpos`` this reproduces
collection's ``[q_d, gripper]`` layout exactly (``FrameBuilder.action_names``);
for other kinds it's the analogous ``[<primary>, gripper]``.

:class:`ActionLayout` computes that layout from each arm's ``control`` config and
splits a flat policy action back into per-side ``{<field>: vector, "gripper": s}``.
"""

from __future__ import annotations

import numpy as np

from ..collection.features import sorted_sides
from ..control import action_dim
from ..control import action_field


class ActionLayout:
    """Per-side slicing of a flat policy action vector, driven by control kind.

    ``arms`` maps side -> ``ArmCfg``; ``sides`` is the subset the policy drives
    (the control-enabled arms), ordered deterministically like collection. Each
    side contributes its control kind's primary field (:func:`action_field`,
    width :func:`action_dim`) plus a trailing gripper scalar.
    """

    def __init__(self, arms: dict, sides) -> None:
        unknown = [s for s in sides if s not in arms]
        if unknown:
            raise ValueError(f"unknown action side(s) {unknown}; arms: {sorted(arms)}")
        self.sides = [s for s in sorted_sides(arms) if s in sides]
        self.names: list[str] = []
        self._field: dict[str, str] = {}     # side -> primary field name (e.g. "q_d")
        self._prim: dict[str, slice] = {}     # side -> slice of the primary field
        self._grip: dict[str, int] = {}       # side -> index of the gripper scalar
        offset = 0
        for side in self.sides:
            ctrl = arms[side].control
            field = action_field(ctrl)
            dim = action_dim(ctrl)
            self._field[side] = field
            self._prim[side] = slice(offset, offset + dim)
            self._grip[side] = offset + dim
            self.names += [f"{side}.{field}.{j}" for j in range(dim)]
            self.names.append(f"{side}.gripper")
            offset += dim + 1
        self.dim = offset

    def field(self, side: str) -> str:
        """The primary action field name for ``side`` (e.g. ``"q_d"``, ``"dq_d"``)."""
        return self._field[side]

    def primary_slice(self, side: str) -> slice:
        """Where ``side``'s primary control field lands inside the action vector."""
        return self._prim[side]

    def split(self, action) -> dict[str, dict]:
        """``{side: {<primary field>: (dim,) array, "gripper": float}}`` from one action.

        The primary field is keyed by its real name, so ``qpos`` yields ``"q_d"``,
        ``qvel`` yields ``"dq_d"``, etc.
        """
        arr = np.asarray(action, dtype=np.float64).ravel()
        if arr.shape != (self.dim,):
            raise ValueError(
                f"action has shape {arr.shape}, expected ({self.dim},) for layout {self.names}"
            )
        return {
            side: {
                self._field[side]: arr[self._prim[side]].copy(),
                "gripper": float(arr[self._grip[side]]),
            }
            for side in self.sides
        }
