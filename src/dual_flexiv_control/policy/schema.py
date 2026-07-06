"""Wire schemas: canonical observation -> one policy server's request format.

The eval loop always builds the *canonical* observation (the LeRobot-keyed
training frame — see :mod:`.observation`). A :class:`PolicySchema` translates
it into the payload one family of policy servers expects, and parses that
server's response back into an action chunk ``(horizon, action_dim)``.

Supporting a new policy server means registering a new schema::

    @register_schema("my_server")
    class MySchema(PolicySchema):
        def request(self, obs): ...
        def actions(self, response): ...

then selecting it with ``policy.schema=my_server``. Schemas are pure mappings
(no I/O): the transport (:mod:`.client`) is chosen independently, so an
HTTP-served policy with openpi-shaped payloads needs no new schema, and vice
versa.
"""

from __future__ import annotations

import abc

import numpy as np

from .client import PolicyError

#: Canonical-observation key prefix for camera views.
_IMAGE_PREFIX = "observation.images."

_SCHEMAS: dict[str, type] = {}


def register_schema(name: str):
    """Class decorator: make a :class:`PolicySchema` selectable as ``policy.schema=<name>``."""

    def deco(cls):
        _SCHEMAS[name] = cls
        return cls

    return deco


def build_schema(cfg) -> "PolicySchema":
    """Instantiate the schema registered under ``cfg.schema``."""
    try:
        cls = _SCHEMAS[cfg.schema]
    except KeyError:
        raise ValueError(
            f"unknown policy schema {cfg.schema!r}; registered: {sorted(_SCHEMAS)}"
        ) from None
    return cls(cfg)


class PolicySchema(abc.ABC):
    """Maps the canonical observation to one server's wire format and back."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def request(self, obs: dict) -> dict:
        """Server request payload from a canonical observation dict."""

    @abc.abstractmethod
    def actions(self, response: dict) -> np.ndarray:
        """Action chunk ``(horizon, action_dim)`` from a server response."""


@register_schema("openpi")
class OpenPISchema(PolicySchema):
    """openpi policy servers: flat slash-keyed observation + ``prompt``.

    Request (key names all come from :class:`~dual_flexiv_control.configs.PolicyCfg`)::

        {state_key: (state_dim,) float32,
         <image key per camera>: (H, W, 3) uint8,
         prompt_key: str}

    Image keys default to ``image_key_template.format(camera=<canonical key>)``;
    ``image_keys`` overrides per camera, and mapping a camera to ``""`` drops
    its view from requests (bandwidth for views the checkpoint ignores).
    Response: ``{actions_key: (horizon, action_dim)}``.
    """

    def request(self, obs: dict) -> dict:
        req: dict = {self.cfg.state_key: obs["observation.state"]}
        for key, value in obs.items():
            if not key.startswith(_IMAGE_PREFIX):
                continue
            wire_key = self._image_key(key[len(_IMAGE_PREFIX):])
            if wire_key:
                req[wire_key] = value
        req[self.cfg.prompt_key] = obs["task"]
        return req

    def _image_key(self, camera: str) -> str:
        if camera in self.cfg.image_keys:
            return self.cfg.image_keys[camera]  # "" drops the view
        return self.cfg.image_key_template.format(camera=camera)

    def actions(self, response: dict) -> np.ndarray:
        if not isinstance(response, dict) or self.cfg.actions_key not in response:
            keys = sorted(response) if isinstance(response, dict) else type(response).__name__
            raise PolicyError(
                f"policy response has no {self.cfg.actions_key!r} key (got: {keys})"
            )
        arr = np.asarray(response[self.cfg.actions_key], dtype=np.float64)
        if arr.ndim == 1:  # a single action -> a chunk of one
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise PolicyError(f"expected actions of shape (horizon, dim), got {arr.shape}")
        return arr


@register_schema("acme")
class AcmeSchema(PolicySchema):
    """ACME policy servers (HTTP ``POST /predict``, multipart/form-data).

    Single-arm franka recipe: three camera views + a 7-dim ``qpos``, and a
    language ``prompt``. Rather than serialize here (a schema is a pure
    mapping), :meth:`request` returns a *structured* payload the ACME transport
    (:class:`~.client.AcmeHttpTransport`) encodes onto the wire — image tensors
    ``torch.save``'d as ``(B, T, C, H, W)`` uint8, lowdim bundled into
    ``lowdim_data.npz`` as ``(B, T, D)``::

        {"images": {<acme slot>: (H, W, 3) uint8, ...},
         "lowdim": {"qpos": (7,) float32},
         "form":   {"prompt": str, "obs_steps": int}}

    ``acme_image_keys`` maps each ACME slot to a canonical camera view; the
    server validates strictly, so every mapped slot must be present. Response:
    ``action`` of shape ``(B, H, 8)`` — absolute joint targets ``[q1..q7, grip]``
    already integrated server-side; ``B == 1`` for this single-frame client.
    """

    def request(self, obs: dict) -> dict:
        state = np.asarray(obs["observation.state"], dtype=np.float32).ravel()
        lo, hi = self.cfg.qpos_slice
        qpos = state[lo:hi]
        if qpos.shape[0] != hi - lo or hi > state.shape[0]:
            raise PolicyError(
                f"qpos_slice {[lo, hi]} does not fit observation.state of "
                f"length {state.shape[0]}"
            )

        images: dict = {}
        for slot, camera in self.cfg.acme_image_keys.items():
            if not camera:  # "" drops the slot (the server must drop its key too)
                continue
            canonical = _IMAGE_PREFIX + camera
            try:
                images[slot] = obs[canonical]
            except KeyError:
                available = sorted(k[len(_IMAGE_PREFIX):] for k in obs if k.startswith(_IMAGE_PREFIX))
                raise PolicyError(
                    f"ACME image slot {slot!r} maps to camera view {camera!r}, "
                    f"absent from the observation (available: {available})"
                ) from None

        return {
            "images": images,
            "lowdim": {
                "qpos": np.ascontiguousarray(qpos, dtype=np.float32),
                "gripper_force": np.array([float(self.cfg.gripper_force)], dtype=np.float32),
            },
            "form": {"prompt": obs["task"], "obs_steps": int(self.cfg.obs_steps)},
        }

    def actions(self, response: dict) -> np.ndarray:
        if not isinstance(response, dict) or "action" not in response:
            keys = sorted(response) if isinstance(response, dict) else type(response).__name__
            raise PolicyError(f"ACME response has no 'action' key (got: {keys})")
        arr = np.asarray(response["action"], dtype=np.float64)
        if arr.ndim == 3:  # (B, H, 8) — single-item batch
            if arr.shape[0] != 1:
                raise PolicyError(f"ACME returned a batch of {arr.shape[0]}; expected 1")
            arr = arr[0]
        if arr.ndim == 1:  # a single action -> a chunk of one
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise PolicyError(f"expected ACME action of shape (horizon, 8), got {arr.shape}")
        return arr
