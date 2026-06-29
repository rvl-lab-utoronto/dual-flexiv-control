"""The brain: the main processing pipeline.

The brain subscribes to streams (by attaching read-only views of the producers'
shared memory) and observes them — pulling the latest sample or the last ``k``
samples of any signal. It owns no streams of its own; it is a pure consumer.

:class:`Brain` is the reusable API (also usable in-process, e.g. tests).
:class:`BrainNode` wraps it in the standard process loop for the live system.
"""

from __future__ import annotations

import logging

import numpy as np

from ..configs import ArmCfg
from ..configs import BrainCfg
from ..configs import FactrCfg
from ..configs import RuntimeCfg
from ..control import ControlCommand
from ..control import SETPOINT
from ..control import COMMAND
from ..control import control_specs
from ..control import convert_factr_to_rizon
from ..control import pack_streamed
from ..interfaces.factr import FactrClient
from ..interfaces.factr.client import FactrError
from ..process import ProcessNode
from ..process import RateLimiter
from ..streams.registry import AttachAborted
from ..streams.registry import StreamRegistry
from ..streams.ring import Samples
from ..streams.stream import StreamReader
from ..streams.stream import StreamWriter

log = logging.getLogger(__name__)


def default_stream_names(arms) -> list[str]:
    """Every stream the brain observes by default: each arm's proprio signals.

    ``arms`` maps side -> ``ArmCfg`` (with ``.streams``); names follow the
    ``"<side>/<signal>"`` paths. FACTR is queried on request, not streamed.
    """
    names: list[str] = []
    for side, arm in arms.items():
        names.extend(f"{side}/{sig}" for sig in arm.streams)
    return names


class Brain:
    """Subscribes to streams and observes them; gets the last ``k`` elements."""

    def __init__(
        self,
        registry: StreamRegistry,
        stream_names: list[str],
        attach_timeout_s: float = 10.0,
        factr: FactrClient | None = None,
    ) -> None:
        self._registry = registry
        self._stream_names = list(stream_names)
        self._attach_timeout_s = attach_timeout_s
        self._readers: dict[str, StreamReader] = {}
        #: On-request FACTR client; query with :meth:`factr_joint_positions`.
        self.factr = factr
        #: brain→arm control-channel writers (created in :meth:`open_control`). The
        #: brain OWNS these segments (inverse of telemetry) and unlinks them on close.
        self._setpoint_writers: dict[str, StreamWriter] = {}
        self._command_writers: dict[str, StreamWriter] = {}

    def attach(self, stop_event=None) -> None:
        """Block until all subscribed streams exist, then attach read-only views.

        If ``stop_event`` is provided and set during the wait, raises
        :class:`AttachAborted` so a shutdown requested mid-attach unwinds promptly.
        """
        entries = self._registry.wait_for(
            self._stream_names, self._attach_timeout_s, stop_event=stop_event
        )
        for name in self._stream_names:
            self._readers[name] = StreamReader.attach(entries[name])
        log.info("brain attached to %d streams", len(self._readers))

    @property
    def stream_names(self) -> list[str]:
        return list(self._readers) or list(self._stream_names)

    def last(self, name: str, k: int) -> Samples:
        """Last ``k`` samples of one stream, oldest -> newest."""
        return self._readers[name].last(k)

    def latest(self, name: str) -> Samples:
        """The single newest sample of one stream."""
        return self._readers[name].latest()

    def observe(self) -> dict[str, Samples]:
        """Snapshot the newest sample of every subscribed stream."""
        return {name: reader.latest() for name, reader in self._readers.items()}

    def observe_last(self, k: int) -> dict[str, Samples]:
        """Snapshot the last ``k`` samples of every subscribed stream."""
        return {name: reader.last(k) for name, reader in self._readers.items()}

    def factr_joint_positions(self):
        """Fetch the FACTR leaders' joint positions on request: ``{side: array}``.

        Raises if no FACTR client was configured, or :class:`FactrError` if the
        server is unreachable / returns a bad response.
        """
        if self.factr is None:
            raise RuntimeError("brain has no FACTR client configured")
        return self.factr.get_joint_positions()

    # -- control channel (brain -> arm) ---------------------------------------

    def open_control(self, control_registry: StreamRegistry, specs_by_side: dict) -> None:
        """Create the setpoint + command writers for each controlled arm.

        ``specs_by_side`` maps ``side -> {SETPOINT: StreamSpec, COMMAND: StreamSpec}``
        (from :func:`dual_flexiv_control.control.control_specs`). The arms publish
        telemetry first and only then wait for these channels, so opening them here —
        after the brain has attached telemetry — cannot deadlock.
        """
        for side, specs in specs_by_side.items():
            self._setpoint_writers[side] = StreamWriter.create(
                specs[SETPOINT], control_registry.run_id, control_registry
            )
            self._command_writers[side] = StreamWriter.create(
                specs[COMMAND], control_registry.run_id, control_registry
            )
        if specs_by_side:
            log.info("brain opened control channels for %s", list(specs_by_side))

    def command(self, side: str, setpoint: np.ndarray) -> None:
        """Post a high-rate follower setpoint (latest-wins). No flexivrdk involved."""
        self._setpoint_writers[side].write(np.ascontiguousarray(setpoint, dtype=np.float64))

    def send_command(self, side: str, command: ControlCommand) -> None:
        """Post a discrete, reliable control event (home/stop/switch-mode)."""
        writer = self._command_writers[side]
        writer.write(command.encode(writer.spec.dim))

    @property
    def controlled_sides(self) -> list[str]:
        return list(self._setpoint_writers)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        for writers in (self._setpoint_writers, self._command_writers):
            for writer in writers.values():
                try:
                    writer.close()
                    writer.unlink()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("error releasing control writer %s", writer.name)
            writers.clear()
        if self.factr is not None:
            self.factr.close()


class BrainNode(ProcessNode):
    """Runs the brain in its own process: attach, then observe at a fixed rate."""

    def __init__(
        self,
        brain: BrainCfg,
        runtime: RuntimeCfg,
        factr: FactrCfg,
        run_id: str,
        stream_names: list[str],
        arms: dict[str, ArmCfg],
    ) -> None:
        self.name = "brain"
        self.cfg = brain
        self.runtime = runtime
        self.factr_cfg = factr
        self.run_id = run_id
        self.stream_names = stream_names
        self.arms = arms
        self._brain: Brain | None = None
        #: control-enabled arms whose setpoints this node posts each tick.
        self._teleop: dict[str, ArmCfg] = {}
        self._warned_kinds: set[str] = set()

    def run(self, stop_event) -> None:
        registry = StreamRegistry(self.runtime.runtime_dir, self.run_id)
        factr = FactrClient.from_config(self.factr_cfg, sim=self.runtime.sim)
        brain = Brain(registry, self.stream_names, self.cfg.attach_timeout_s, factr=factr)
        try:
            brain.attach(stop_event=stop_event)
        except AttachAborted:
            log.info("brain attach aborted by shutdown")
            brain.close()
            return
        self._brain = brain

        # Open brain→arm control channels for every control-enabled arm. The arms
        # publish telemetry first and only then wait for these, so this is safe to
        # do after our telemetry attach above (no deadlock).
        control_registry = StreamRegistry(self.runtime.runtime_dir, self.run_id, sub="control")
        specs_by_side = {
            side: control_specs(side, arm.control)
            for side, arm in self.arms.items()
            if arm.control_enabled
        }
        brain.open_control(control_registry, specs_by_side)
        self._teleop = {side: self.arms[side] for side in specs_by_side}

        rate = RateLimiter(self.cfg.rate_hz)
        rate.reset()
        ticks = 0
        heartbeat_every = max(1, int(self.cfg.rate_hz))  # ~1 s
        try:
            while not stop_event.is_set():
                observation = brain.observe()
                self.process(observation)
                ticks += 1
                if ticks % heartbeat_every == 0:
                    self._heartbeat(observation)
                rate.sleep()
        finally:
            brain.close()
            self._brain = None

    def process(self, observation: dict[str, Samples]) -> None:
        """FACTR→follower teleoperation: post each control-enabled arm's setpoint.

        Reads the FACTR leaders on request, converts to Rizon joint targets via the
        per-arm convention, and posts them on the setpoint channel (latest-wins). This
        is pure numpy + the control channel — **no flexivrdk** here. Only joint-position
        (``qpos``) kinds map directly from FACTR joint readings; other kinds expect a
        policy setpoint source (not wired here) and are skipped with a one-time note.

        Override this for a policy: build the setpoint vector with
        :func:`~dual_flexiv_control.control.pack_streamed` and call ``self._brain.command``.
        """
        if not self._teleop or self._brain is None or self._brain.factr is None:
            return
        try:
            leaders = self._brain.factr_joint_positions()
        except FactrError as exc:
            log.warning("FACTR read failed; holding (no setpoint posted this tick): %s", exc)
            return
        for side, arm in self._teleop.items():
            q_leader = leaders.get(side)
            if q_leader is None:
                continue
            ctrl = arm.control
            if ctrl.kind != "qpos":
                if ctrl.kind not in self._warned_kinds:
                    self._warned_kinds.add(ctrl.kind)
                    log.warning(
                        "[%s] control kind %r has no FACTR teleop source; setpoints "
                        "must come from a policy (override process()). Skipping.",
                        side, ctrl.kind,
                    )
                continue
            q_d = convert_factr_to_rizon(q_leader, arm.convention)
            setpoint = pack_streamed(ctrl, {"q_d": q_d, "dq_d": np.zeros_like(q_d)})
            self._brain.command(side, setpoint)

    def _heartbeat(self, observation: dict[str, Samples]) -> None:
        fresh = sum(1 for s in observation.values() if s.n > 0)
        log.info("brain heartbeat: %d/%d streams fresh", fresh, len(observation))
