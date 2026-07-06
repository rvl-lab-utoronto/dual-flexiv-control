"""Canonical eval observation: the LeRobot training frame, minus the action.

Eval-time model inputs must match the data the policy was trained on, so rather
than define a second observation schema, :class:`ObservationBuilder` reuses the
collection :class:`~dual_flexiv_control.collection.features.FrameBuilder` with
no teleop sides. The canonical observation IS the LeRobot frame::

    observation.state                     float32 (state_dim,)  per-arm proprio concat
    observation.images.<cam>[_<view>]     uint8   (H, W, 3)     one per RGB camera view
    task                                  str                   language instruction

built from the same config inputs (arms, cameras, ``state_signals``,
instruction) as collection — identical layout by construction, not by
convention. Policy schemas (:mod:`.schema`) then map this canonical dict onto
each policy server's wire format.
"""

from __future__ import annotations

from ..collection.features import FrameBuilder
from ..collection.features import sorted_sides


class ObservationBuilder:
    """Builds the canonical observation dict from a stream snapshot.

    ``arms`` maps side -> ``ArmCfg``; ``cameras`` maps name -> ``CameraCfg``;
    ``state_signals`` must equal what the policy was trained with — it is
    task-level (``task.state_signals``), shared by collection and eval by
    construction.
    """

    def __init__(self, arms: dict, cameras: dict, instruction: str, state_signals) -> None:
        # No teleop sides: the frame has an empty action, which build() drops.
        self._frames = FrameBuilder(arms, [], cameras, instruction, list(state_signals))

        # Where each (side, signal) lands inside observation.state — same
        # iteration order as FrameBuilder's state layout (sides, then signals).
        self._state_slices: dict[tuple[str, str], slice] = {}
        offset = 0
        for side in sorted_sides(arms):
            for sig in state_signals:
                dim = int(arms[side].streams[sig].dim)
                self._state_slices[(side, sig)] = slice(offset, offset + dim)
                offset += dim

    # -- schema ----------------------------------------------------------------

    @property
    def stream_names(self) -> list[str]:
        """Every stream to subscribe to (proprio state + camera views)."""
        return self._frames.stream_names

    @property
    def state_dim(self) -> int:
        return self._frames.state_dim

    @property
    def image_keys(self) -> list[str]:
        """Canonical image keys, e.g. ``observation.images.wrist_left``."""
        return self._frames.image_keys

    def state_slice(self, side: str, signal: str) -> slice:
        """Where ``(side, signal)`` lands inside ``observation.state``."""
        try:
            return self._state_slices[(side, signal)]
        except KeyError:
            raise KeyError(
                f"({side!r}, {signal!r}) is not part of observation.state; "
                f"available: {sorted(self._state_slices)}"
            ) from None

    def missing(self, observation: dict) -> list[str]:
        """Subscribed streams with no sample yet in a snapshot (why build() is None)."""
        empty = []
        for name in self.stream_names:
            samples = observation.get(name)
            if samples is None or samples.n == 0:
                empty.append(name)
        return empty

    # -- per-tick assembly -------------------------------------------------------

    def build(self, observation: dict) -> dict | None:
        """Canonical observation dict, or ``None`` while any stream is unproduced.

        ``observation`` maps stream name -> ``Samples`` (from ``brain.observe``).
        ``None`` means the tick is incomplete (a stream not yet warm) and the
        caller should hold.
        """
        frame = self._frames.build(observation, {}, {})
        if frame is None:
            return None
        frame.pop("action")  # zero-length (no teleop sides) — not an observation
        return frame
