"""Canonical DFC state/action layout and OpenPI-side model transforms.

DFC owns the 14-state / 16-action format. Endpoint projection—including which
policy indices correspond to each arm—belongs to ``conf/policy/*.yaml`` in the
robot client. The transforms here begin *after* that wire layout: they adapt a
DFC-native OpenPI request to the model's internal image/action representation.

This module deliberately has no OpenPI imports. Keeping the transformation
NumPy-only makes the canonical layout independently testable.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from dual_flexiv_control.layout import DFCStateActionLayout


DFC_STATE_ACTION_LAYOUT = DFCStateActionLayout.bimanual_qpos()
STATE_DIM = DFC_STATE_ACTION_LAYOUT.state_dim
ACTION_DIM = DFC_STATE_ACTION_LAYOUT.action_dim


def _parse_image(value: object, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{key} must be a 3-D image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have 3 color channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class OpenPIInputs:
    """Map a DFC-native request into OpenPI's internal model representation.

    The current checkpoint consumes one base-camera slot and two optional wrist
    slots. ``static_left`` fills the base view; absent wrist views are black and
    masked. This is checkpoint/model adaptation, not the DFC endpoint schema.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"DFC state must have shape ({STATE_DIM},), got {state.shape}"
            )
        base_image = _parse_image(
            data["observation/images/static_left"],
            "observation/images/static_left",
        )
        padding = np.zeros_like(base_image)
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": padding,
                "right_wrist_0_rgb": padding.copy(),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim < 1 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"DFC actions must end in {ACTION_DIM} values, got {actions.shape}"
                )
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return result


@dataclasses.dataclass(frozen=True)
class OpenPIActionDeltas:
    """Convert canonical DFC joint targets between absolute and delta space.

    OpenPI's generic ``DeltaActions`` assumes state and action columns align.
    DFC's canonical action interleaves grippers, so the global layout supplies
    the state/action joint-block pairs. Grippers remain absolute.
    """

    inverse: bool = False
    layout: DFCStateActionLayout = DFC_STATE_ACTION_LAYOUT

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if state.shape[-1] < self.layout.state_dim:
            raise ValueError(
                f"DFC state must contain {self.layout.state_dim} joints, got {state.shape}"
            )
        if actions.shape[-1] < self.layout.action_dim:
            raise ValueError(
                f"DFC actions must contain {self.layout.action_dim} values, got {actions.shape}"
            )
        sign = 1.0 if self.inverse else -1.0
        for state_slice, action_slice in self.layout.state_to_action_joint_slices():
            actions[..., action_slice] += sign * state[..., None, state_slice]
        result = dict(data)
        result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class OpenPIOutputs:
    """Crop OpenPI's padded action head to the canonical DFC action width."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                "π0.5 output must be (horizon, >=16), "
                f"got shape {actions.shape}"
            )
        return {"actions": actions[..., :ACTION_DIM]}
