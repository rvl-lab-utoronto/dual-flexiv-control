"""Endpoint adapters between canonical DFC frames and policy-server protocols."""

from __future__ import annotations

import abc

import numpy as np

from .client import PolicyError

_IMAGE_PREFIX = "observation.images."
_ADAPTERS: dict[str, type] = {}


def register_adapter(name: str):
    """Make an endpoint adapter selectable as ``policy.adapter=<name>``."""

    def decorator(cls):
        _ADAPTERS[name] = cls
        return cls

    return decorator


def build_adapter(cfg) -> "PolicyEndpointAdapter":
    try:
        cls = _ADAPTERS[cfg.adapter]
    except KeyError:
        raise ValueError(
            f"unknown policy adapter {cfg.adapter!r}; registered: {sorted(_ADAPTERS)}"
        ) from None
    return cls(cfg)


class PolicyEndpointAdapter(abc.ABC):
    """Pure payload mapping; transports own I/O and serialization."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def encode_request(self, observation: dict) -> dict:
        """Encode one canonical DFC observation for the endpoint."""

    @abc.abstractmethod
    def decode_actions(self, response: dict) -> np.ndarray:
        """Decode an endpoint response into canonical DFC actions."""


@register_adapter("openpi")
class OpenPIEndpointAdapter(PolicyEndpointAdapter):
    """OpenPI endpoint conventions with YAML-owned embodiment projections."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._last_dfc_state: np.ndarray | None = None

    @staticmethod
    def _indices(values, size: int, label: str) -> list[int]:
        indices = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise PolicyError(f"{label} must contain integer indices, got {value!r}")
            index = int(value)
            if index < 0 or index >= size:
                raise PolicyError(f"{label} index {index} is outside [0, {size})")
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise PolicyError(f"{label} contains duplicate indices: {indices}")
        return indices

    @classmethod
    def _pairs(cls, item, source_size: int, target_size: int, label: str):
        source = cls._indices(item.source, source_size, f"{label} source")
        target = cls._indices(item.target, target_size, f"{label} target")
        if len(source) != len(target):
            raise PolicyError(
                f"{label} source/target lengths differ: {len(source)} != {len(target)}"
            )
        return source, target

    @staticmethod
    def _claim(mapped: set[int], target: list[int], label: str) -> None:
        overlap = mapped.intersection(target)
        if overlap:
            raise PolicyError(f"{label} overwrites mapped target indices {sorted(overlap)}")
        mapped.update(target)

    @staticmethod
    def _image(image, layout: str, label: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise PolicyError(f"OpenPI image {label!r} must be rank 3, got {array.shape}")
        if array.shape[-1] == 3:
            hwc = array
        elif array.shape[0] == 3:
            hwc = np.moveaxis(array, 0, -1)
        else:
            raise PolicyError(
                f"OpenPI image {label!r} must be HWC or CHW RGB, got {array.shape}"
            )
        if hwc.dtype != np.uint8:
            if np.issubdtype(hwc.dtype, np.floating) and hwc.size and np.nanmax(hwc) <= 1:
                hwc = hwc * 255.0
            hwc = np.clip(hwc, 0, 255).astype(np.uint8)
        if layout == "hwc":
            return np.ascontiguousarray(hwc)
        if layout == "chw":
            return np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
        raise PolicyError(f"unknown OpenPI image layout {layout!r}; use 'hwc' or 'chw'")

    def encode_request(self, observation: dict) -> dict:
        try:
            dfc_state = np.asarray(
                observation["observation.state"], dtype=np.float32
            ).ravel()
        except KeyError:
            raise PolicyError("canonical observation has no 'observation.state'") from None

        request = self.cfg.openpi.request
        state_cfg = request.state
        state_dim = int(state_cfg.dim)
        if state_dim < 0:
            raise PolicyError(f"OpenPI request state dim cannot be negative: {state_dim}")
        if state_dim == 0:
            state = dfc_state.copy()
        else:
            state = np.zeros(state_dim, dtype=np.float32)
            mapped: set[int] = set()
            for item in state_cfg.slices:
                label = item.name or "unnamed state copy"
                source, target = self._pairs(item, dfc_state.size, state_dim, label)
                self._claim(mapped, target, label)
                state[target] = dfc_state[source]
            for item in state_cfg.constants:
                label = item.name or "unnamed state constant"
                target = self._indices(item.target, state_dim, f"{label} target")
                self._claim(mapped, target, label)
                state[target] = float(item.value)
            missing = sorted(set(range(state_dim)) - mapped)
            if missing:
                raise PolicyError(f"OpenPI request leaves state indices unmapped: {missing}")

        payload: dict = {state_cfg.key: state}
        if request.images:
            images = {}
            for image_cfg in request.images:
                canonical = _IMAGE_PREFIX + image_cfg.source
                if canonical not in observation:
                    available = sorted(
                        key[len(_IMAGE_PREFIX):]
                        for key in observation
                        if key.startswith(_IMAGE_PREFIX)
                    )
                    raise PolicyError(
                        f"OpenPI image {image_cfg.key!r} maps to DFC view "
                        f"{image_cfg.source!r}, absent from observation (available: {available})"
                    )
                images[image_cfg.key] = self._image(
                    observation[canonical], str(image_cfg.layout), str(image_cfg.source)
                )
            if request.images_key:
                payload[request.images_key] = images
            else:
                payload.update(images)
        else:
            for key, value in observation.items():
                if not key.startswith(_IMAGE_PREFIX):
                    continue
                camera = key[len(_IMAGE_PREFIX):]
                wire_key = request.image_keys.get(
                    camera, request.image_key_template.format(camera=camera)
                )
                if wire_key:
                    payload[wire_key] = self._image(value, "hwc", camera)

        payload[request.prompt_key] = observation["task"]
        self._last_dfc_state = dfc_state.copy()
        return payload

    def decode_actions(self, response: dict) -> np.ndarray:
        cfg = self.cfg.openpi.response.actions
        if not isinstance(response, dict) or cfg.key not in response:
            keys = sorted(response) if isinstance(response, dict) else type(response).__name__
            raise PolicyError(f"policy response has no {cfg.key!r} key (got: {keys})")
        raw = np.asarray(response[cfg.key], dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise PolicyError(f"expected OpenPI actions of shape (horizon, dim), got {raw.shape}")

        output_dim = int(cfg.dim)
        if output_dim < 0:
            raise PolicyError(f"OpenPI response action dim cannot be negative: {output_dim}")
        if output_dim == 0:
            return raw
        output = np.zeros((raw.shape[0], output_dim), dtype=np.float64)
        mapped: set[int] = set()
        for item in cfg.slices:
            label = item.name or "unnamed action copy"
            source, target = self._pairs(item, raw.shape[1], output_dim, label)
            self._claim(mapped, target, label)
            output[:, target] = raw[:, source]
        if cfg.state_holds:
            if self._last_dfc_state is None:
                raise PolicyError(
                    "cannot map OpenPI actions before mapping a DFC observation"
                )
            for item in cfg.state_holds:
                label = item.name or "unnamed measured-state hold"
                source, target = self._pairs(
                    item, self._last_dfc_state.size, output_dim, label
                )
                self._claim(mapped, target, label)
                output[:, target] = self._last_dfc_state[source]
        for item in cfg.constants:
            label = item.name or "unnamed action constant"
            target = self._indices(item.target, output_dim, f"{label} target")
            self._claim(mapped, target, label)
            output[:, target] = float(item.value)
        missing = sorted(set(range(output_dim)) - mapped)
        if missing:
            raise PolicyError(f"OpenPI response leaves DFC action indices unmapped: {missing}")
        return output


@register_adapter("acme")
class AcmeEndpointAdapter(PolicyEndpointAdapter):
    """ACME multipart endpoint mapped from the canonical DFC observation."""

    def encode_request(self, observation: dict) -> dict:
        state = np.asarray(observation["observation.state"], dtype=np.float32).ravel()
        lo, hi = self.cfg.qpos_slice
        qpos = state[lo:hi]
        if qpos.shape[0] != hi - lo or hi > state.shape[0]:
            raise PolicyError(
                f"qpos_slice {[lo, hi]} does not fit observation.state length {state.shape[0]}"
            )
        images = {}
        for slot, camera in self.cfg.acme_image_keys.items():
            if not camera:
                continue
            canonical = _IMAGE_PREFIX + camera
            try:
                images[slot] = observation[canonical]
            except KeyError:
                available = sorted(
                    key[len(_IMAGE_PREFIX):]
                    for key in observation
                    if key.startswith(_IMAGE_PREFIX)
                )
                raise PolicyError(
                    f"ACME image slot {slot!r} maps to absent view {camera!r} "
                    f"(available: {available})"
                ) from None
        return {
            "images": images,
            "lowdim": {
                "qpos": np.ascontiguousarray(qpos, dtype=np.float32),
                "gripper_force": np.array([float(self.cfg.gripper_force)], dtype=np.float32),
            },
            "form": {
                "prompt": observation["task"],
                "obs_steps": int(self.cfg.obs_steps),
            },
        }

    def decode_actions(self, response: dict) -> np.ndarray:
        if not isinstance(response, dict) or "action" not in response:
            keys = sorted(response) if isinstance(response, dict) else type(response).__name__
            raise PolicyError(f"ACME response has no 'action' key (got: {keys})")
        actions = np.asarray(response["action"], dtype=np.float64)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise PolicyError(f"ACME returned a batch of {actions.shape[0]}; expected 1")
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise PolicyError(
                f"expected ACME action of shape (horizon, 8), got {actions.shape}"
            )
        return actions

