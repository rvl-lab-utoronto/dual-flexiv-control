"""Dual-Flexiv transforms for an OpenPI π0.5 policy.

This module deliberately has no OpenPI imports: OpenPI accepts ordinary callable
transforms, and keeping the robot-shape logic NumPy-only makes it possible to test
the exact 14-state / 16-action contract on the robot workstation.
"""

from __future__ import annotations

import dataclasses

import numpy as np

STATE_DIM = 14
ACTION_DIM = 16


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
class DFCInputs:
    """Map the dashboard's flat slash-keyed request into the π0.5 model shape.

    The physical rig currently has one stereo ZED and no wrist cameras. π0.5 has
    one base-camera slot and two wrist-camera slots, so ``static_left`` is the
    base view and the wrist slots are explicitly masked. ``static_right`` remains
    on the wire for recording/visualization but is not mislabeled as a wrist view.
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
class DFCJointDeltas:
    """Convert the two 7-DoF joint blocks between absolute and delta space.

    DFC observations contain 14 joints but actions interleave two gripper values:

    ``left q[7], left grip, right q[7], right grip``.

    OpenPI's generic ``DeltaActions`` assumes state and action columns align, so
    it cannot represent this layout. This transform handles the two joint slices
    explicitly and leaves both grippers absolute.
    """

    inverse: bool = False

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if state.shape[-1] < STATE_DIM:
            raise ValueError(
                f"DFC state must contain {STATE_DIM} joints, got {state.shape}"
            )
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                f"DFC actions must contain {ACTION_DIM} values, got {actions.shape}"
            )
        sign = 1.0 if self.inverse else -1.0
        actions[..., 0:7] += sign * state[..., None, 0:7]
        actions[..., 8:15] += sign * state[..., None, 7:14]
        result = dict(data)
        result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class DFCOutputs:
    """Crop the padded π0.5 action head back to DFC's 16-value action layout."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                "π0.5 output must be (horizon, >=16), "
                f"got shape {actions.shape}"
            )
        return {"actions": actions[..., :ACTION_DIM]}
