"""LeRobot feature schema + per-tick frame assembly (pure, hardware-free).

The collection loop records one LeRobot *frame* per tick. This module turns the
run's config (arms, cameras, which proprio signals form the state) into:

* a LeRobot ``features`` dict (``LeRobotDataset.create(features=...)``), and
* a :class:`FrameBuilder` that maps a stream observation snapshot + the tick's
  commanded joint targets into the ``frame`` dict handed to ``add_frame``.

Conventions (LeRobot standard):

* ``observation.state`` — measured follower proprio, concatenated per arm in
  side order over :attr:`CollectionCfg.state_signals` (default just ``q``).
* ``action`` — the teleop command: per arm, the converted Rizon joint targets
  (``q_d``, ``dof`` values) plus the leader's trailing gripper scalar.
* ``observation.images.<cam>[_<view>]`` — one RGB view per camera stream,
  reshaped ``(H, W, 3)`` uint8. Depth views are not exported (non-standard as
  video); a warning is logged once if any are configured.

Keeping this pure (numpy + config, no SDK, no lerobot import) makes the schema
and frame packing unit-testable without hardware or the lerobot dependency.
"""

from __future__ import annotations

import logging

import numpy as np

from ..cameras import RGB_VIEWS
from ..cameras import camera_stream_name
from ..cameras import view_shape
from ..interfaces.factr.interface import factr_stream_name
from ..interfaces.factr.interface import raw_factr_stream_name
from ..layout import DFCStateActionLayout
from ..layout import sorted_sides

log = logging.getLogger(__name__)

#: LeRobot dtype for an image feature depending on the storage mode.
_IMAGE_DTYPE = {True: "video", False: "image"}


class FrameBuilder:
    """Builds the LeRobot ``features`` spec and packs one frame per tick.

    ``arms`` maps side -> ``ArmCfg`` (all arms, for state); ``teleop_sides`` is
    the subset the loop actuates + records as ``action`` (control-enabled arms).
    ``cameras`` maps name -> ``CameraCfg``.
    """

    def __init__(
        self,
        arms: dict,
        teleop_sides: list[str],
        cameras: dict,
        instruction: str,
        state_signals: list[str],
        video: bool = True,
    ) -> None:
        self.arms = arms
        self.instruction = instruction
        self.state_signals = list(state_signals)
        self.video = video
        self.layout = DFCStateActionLayout.from_arms(arms, self.state_signals, teleop_sides)
        self._state_sides = self.layout.state_sides
        self._action_sides = self.layout.sides
        self._state_reads = self.layout.state_stream_names
        self.state_names = self.layout.state_names
        self.state_dim = self.layout.state_dim
        self.action_names = self.layout.action_names
        self.action_dim = self.layout.action_dim
        self._factr_streams = [
            name for side in self._action_sides
            for name in (factr_stream_name(side), raw_factr_stream_name(side))
        ]

        # -- image layout: one RGB view per camera stream ----------------------
        # (name, view, feature_key, stream_name, shape)
        self._images: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for name, cam in cameras.items():
            rgb = [v for v in cam.views if v in RGB_VIEWS]
            depth = [v for v in cam.views if v not in RGB_VIEWS]
            if depth:
                log.warning(
                    "camera %s: depth view(s) %s are not exported to LeRobot "
                    "(only RGB views become observation.images.*)",
                    name, depth,
                )
            for view in rgb:
                suffix = "" if len(rgb) == 1 else f"_{view}"
                key = f"observation.images.{name}{suffix}"
                self._images.append(
                    (name, view, key, camera_stream_name(name, view), view_shape(cam, view))
                )

    # -- layout accessors -----------------------------------------------------

    @property
    def action_sides(self) -> list[str]:
        """Sides whose teleop command is recorded as ``action`` (in vector order)."""
        return list(self._action_sides)

    def action_dof(self, side: str) -> int:
        """DoF of a side's recorded ``q_d`` block."""
        return self.layout.action_dof(side)

    def missing_streams(self, observation: dict) -> list[str]:
        """Subscribed streams that have no sample yet (block frame recording).

        A frame can't be assembled until every proprio-state and camera stream has
        produced at least one sample; this reports which are still empty so a stuck
        recording (e.g. a camera that opened but never grabbed a frame) is visible.
        """
        missing: list[str] = []
        for name in self.stream_names:
            samples = observation.get(name)
            if samples is None or samples.newest is None:
                missing.append(name)
        return missing

    # -- subscription ---------------------------------------------------------

    @property
    def stream_names(self) -> list[str]:
        """Every stream the loop must subscribe to (proprio state + camera views)."""
        return list(self._state_reads) + self._factr_streams + [img[3] for img in self._images]

    @property
    def image_keys(self) -> list[str]:
        """The frame's image keys, e.g. ``observation.images.wrist_left``."""
        return [img[2] for img in self._images]

    # -- LeRobot features spec ------------------------------------------------

    def features(self) -> dict:
        """The ``features`` dict for ``LeRobotDataset.create``."""
        feats: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": self.state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": self.action_names,
            },
        }
        for side in self._action_sides:
            dim = int(self.arms[side].dof) + 1
            names = [f"{side}.joint.{j}" for j in range(dim - 1)] + [f"{side}.gripper"]
            feats[f"observation.factr.{side}"] = {
                "dtype": "float32", "shape": (dim,), "names": names,
            }
            feats[f"observation.factr_raw.{side}"] = {
                "dtype": "float32", "shape": (dim,), "names": names,
            }
        for _name, _view, key, _stream, shape in self._images:
            feats[key] = {
                "dtype": _IMAGE_DTYPE[self.video],
                "shape": tuple(int(x) for x in shape),
                "names": ["height", "width", "channels"],
            }
        return feats

    # -- per-tick action from ingestion-converted leader samples --------------

    def actions_from_leaders(self, leaders: dict, conventions=None) -> dict:
        """``{side: q_d}`` targets + ``{side: gripper}`` from FACTR streams.

        ``leaders`` maps side -> DFC-coordinate sample (``dof+1``, rad), converted
        once by :class:`FactrInterface`. Sides absent from ``leaders`` are skipped.
        The trailing value is already normalized to a 0..1 gripper fraction using
        calibration owned by the FACTR leader YAML. ``conventions`` is accepted only
        for compatibility with older callers and is ignored.
        """
        q_d: dict = {}
        grip: dict = {}
        for side in self._action_sides:
            raw = leaders.get(side)
            if raw is None:
                continue
            raw = np.asarray(raw, dtype=np.float64).ravel()
            q_d[side] = raw[:self.layout.action_dof(side)]
            grip[side] = float(raw[-1]) if raw.size else 0.0
        return {"q_d": q_d, "gripper": grip}

    # -- frame assembly -------------------------------------------------------

    def build(self, observation: dict, q_d: dict, gripper: dict):
        """Assemble one ``add_frame`` dict, or ``None`` if the tick is incomplete.

        ``observation`` maps stream name -> ``Samples`` (from ``brain.observe``).
        ``q_d``/``gripper`` map side -> commanded targets / gripper (from
        :meth:`actions_from_leaders`). Returns ``None`` when any required proprio,
        action, or camera frame is not yet available (the caller skips the tick).
        """
        # observation.state
        state_values: dict[tuple[str, str], np.ndarray] = {}
        for side in self._state_sides:
            for signal in self.state_signals:
                stream = f"{side}/{signal}"
                newest = observation.get(stream)
                newest = newest.newest if newest is not None else None
                if newest is None:
                    return None
                state_values[(side, signal)] = newest
        state = self.layout.pack_state(state_values)

        # action (per teleop arm: q_d + gripper)
        for side in self._action_sides:
            if side not in q_d:
                return None
        action = self.layout.pack_action(q_d, gripper)

        frame: dict = {
            "observation.state": state,
            "action": action,
            "task": self.instruction,
        }
        for side in self._action_sides:
            converted = observation.get(factr_stream_name(side))
            raw = observation.get(raw_factr_stream_name(side))
            if converted is None or converted.newest is None or raw is None or raw.newest is None:
                return None
            frame[f"observation.factr.{side}"] = np.asarray(
                converted.newest, dtype=np.float32
            ).ravel()
            frame[f"observation.factr_raw.{side}"] = np.asarray(
                raw.newest, dtype=np.float32
            ).ravel()

        # camera images (software-synchronised: newest of each at this tick)
        for _name, _view, key, stream, shape in self._images:
            samples = observation.get(stream)
            newest = samples.newest if samples is not None else None
            if newest is None:
                return None
            frame[key] = np.asarray(newest, dtype=np.uint8).reshape(shape)

        return frame
