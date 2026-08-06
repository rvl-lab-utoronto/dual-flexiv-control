"""Deprecated action-only layout wrapper.

A policy emits, per control-enabled arm (side order), the action for that arm's
control kind followed by one gripper scalar. The *kind* determines the primary
field the policy outputs — ``q_d`` for ``qpos``, ``dq_d`` for ``qvel``, ``pose_d``
for ``end_effector``, ``twist_d`` for ``eef_vel``, ``wrench_d`` for ``force`` (see
:func:`~dual_flexiv_control.control.action_field`). For ``qpos`` this reproduces
collection's ``[q_d, gripper]`` layout exactly (``FrameBuilder.action_names``);
for other kinds it's the analogous ``[<primary>, gripper]``.

Use :class:`dual_flexiv_control.layout.DFCStateActionLayout`, which owns both
state and action layouts and is shared with collection.
"""

from __future__ import annotations

from ..layout import DFCStateActionLayout


class ActionLayout(DFCStateActionLayout):
    """Per-side slicing of a flat policy action vector, driven by control kind.

    ``arms`` maps side -> ``ArmCfg``; ``sides`` is the subset the policy drives
    (the control-enabled arms), ordered deterministically like collection. Each
    side contributes its control kind's primary field (:func:`action_field`,
    width :func:`action_dim`) plus a trailing gripper scalar.
    """

    def __init__(self, arms: dict, sides) -> None:
        super().__init__(arms, ["q"], sides)
        self.names = self.action_names
        self.dim = self.action_dim
