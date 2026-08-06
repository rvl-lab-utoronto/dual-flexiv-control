"""Canonical DFC state/action vector layout.

This is the global DFC contract shared by collection, online evaluation, and
policy-serving overlays. Endpoint adapters may project these vectors into a
checkpoint-specific representation, but they do not redefine their meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import action_dim
from .control import action_field


def sorted_sides(mapping: dict) -> list[str]:
    """Deterministic side ordering (left, right, then any additional names)."""
    order = {"left": 0, "right": 1}
    return sorted(mapping, key=lambda side: (order.get(side, 99), side))


@dataclass(frozen=True)
class _StateBlock:
    side: str
    signal: str
    stream: str
    vector_slice: slice


class DFCStateActionLayout:
    """Names, dimensions, slices, and packing for canonical DFC vectors."""

    def __init__(self, arms: dict, state_signals, action_sides) -> None:
        unknown = [side for side in action_sides if side not in arms]
        if unknown:
            raise ValueError(f"unknown action side(s) {unknown}; arms: {sorted(arms)}")

        self.state_signals = list(state_signals)
        self.state_sides = sorted_sides(arms)
        self.sides = [side for side in self.state_sides if side in action_sides]
        self.state_names: list[str] = []
        self.action_names: list[str] = []
        self._state_blocks: list[_StateBlock] = []
        self._state: dict[tuple[str, str], slice] = {}
        self._field: dict[str, str] = {}
        self._primary: dict[str, slice] = {}
        self._gripper: dict[str, int] = {}

        offset = 0
        for side in self.state_sides:
            arm = arms[side]
            for signal in self.state_signals:
                if signal not in arm.streams:
                    raise ValueError(
                        f"state signal {signal!r} not among {side} arm streams "
                        f"{sorted(arm.streams)}"
                    )
                dim = int(arm.streams[signal].dim)
                vector_slice = slice(offset, offset + dim)
                self._state[(side, signal)] = vector_slice
                self._state_blocks.append(
                    _StateBlock(side, signal, f"{side}/{signal}", vector_slice)
                )
                self.state_names.extend(f"{side}.{signal}.{i}" for i in range(dim))
                offset += dim
        self.state_dim = offset

        offset = 0
        for side in self.sides:
            control = arms[side].control
            field = action_field(control)
            dim = action_dim(control)
            self._field[side] = field
            self._primary[side] = slice(offset, offset + dim)
            self._gripper[side] = offset + dim
            self.action_names.extend(f"{side}.{field}.{i}" for i in range(dim))
            self.action_names.append(f"{side}.gripper")
            offset += dim + 1
        self.action_dim = offset
        # Compatibility with the former action-only layout API.
        self.names = self.action_names
        self.dim = self.action_dim

    @classmethod
    def from_arms(cls, arms: dict, state_signals, action_sides):
        return cls(arms, state_signals, action_sides)

    @classmethod
    def bimanual_qpos(cls) -> "DFCStateActionLayout":
        """The deployed two-Rizon qpos contract without requiring config objects."""
        obj = cls.__new__(cls)
        obj.state_signals = ["q"]
        obj.state_sides = ["left", "right"]
        obj.sides = ["left", "right"]
        obj.state_names = [
            *(f"left.q.{i}" for i in range(7)),
            *(f"right.q.{i}" for i in range(7)),
        ]
        obj.action_names = [
            *(f"left.q_d.{i}" for i in range(7)),
            "left.gripper",
            *(f"right.q_d.{i}" for i in range(7)),
            "right.gripper",
        ]
        obj._state = {
            ("left", "q"): slice(0, 7),
            ("right", "q"): slice(7, 14),
        }
        obj._state_blocks = [
            _StateBlock("left", "q", "left/q", slice(0, 7)),
            _StateBlock("right", "q", "right/q", slice(7, 14)),
        ]
        obj._field = {"left": "q_d", "right": "q_d"}
        obj._primary = {"left": slice(0, 7), "right": slice(8, 15)}
        obj._gripper = {"left": 7, "right": 15}
        obj.state_dim = 14
        obj.action_dim = 16
        obj.names = obj.action_names
        obj.dim = obj.action_dim
        return obj

    @property
    def state_stream_names(self) -> list[str]:
        return [block.stream for block in self._state_blocks]

    def state_slice(self, side: str, signal: str) -> slice:
        try:
            return self._state[(side, signal)]
        except KeyError:
            raise KeyError(
                f"({side!r}, {signal!r}) is not in the DFC state layout; "
                f"available: {sorted(self._state)}"
            ) from None

    def field(self, side: str) -> str:
        return self._field[side]

    def primary_slice(self, side: str) -> slice:
        return self._primary[side]

    def gripper_index(self, side: str) -> int:
        return self._gripper[side]

    def action_dof(self, side: str) -> int:
        vector_slice = self._primary[side]
        return vector_slice.stop - vector_slice.start

    def state_to_action_joint_slices(self) -> list[tuple[slice, slice]]:
        """Canonical measured-q/commanded-q_d pairs used by delta transforms."""
        pairs = []
        for side in self.sides:
            if (side, "q") in self._state and self._field[side] == "q_d":
                pairs.append((self._state[(side, "q")], self._primary[side]))
        return pairs

    def pack_state(self, values: dict[tuple[str, str], object]) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for key, vector_slice in self._state.items():
            value = np.asarray(values[key], dtype=np.float32).ravel()
            expected = vector_slice.stop - vector_slice.start
            if value.size != expected:
                raise ValueError(f"state block {key} has {value.size} values; expected {expected}")
            state[vector_slice] = value
        return state

    def pack_action(self, primary: dict, gripper: dict) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        for side in self.sides:
            value = np.asarray(primary[side], dtype=np.float32).ravel()
            vector_slice = self._primary[side]
            expected = vector_slice.stop - vector_slice.start
            if value.size != expected:
                raise ValueError(
                    f"action block {side!r} has {value.size} values; expected {expected}"
                )
            action[vector_slice] = value
            action[self._gripper[side]] = float(gripper.get(side, 0.0))
        return action

    def split(self, action) -> dict[str, dict]:
        arr = np.asarray(action, dtype=np.float64).ravel()
        if arr.shape != (self.action_dim,):
            raise ValueError(
                f"action has shape {arr.shape}, expected ({self.action_dim},) "
                f"for layout {self.action_names}"
            )
        return {
            side: {
                self._field[side]: arr[self._primary[side]].copy(),
                "gripper": float(arr[self._gripper[side]]),
            }
            for side in self.sides
        }
