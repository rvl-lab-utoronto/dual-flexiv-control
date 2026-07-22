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
from ..control import CommandKind
from ..control import ControlCommand
from ..control import SETPOINT
from ..control import COMMAND
from ..control import GRIPPER
from ..control import control_specs
from ..control import gripper_spec
from ..control import pack_streamed
from ..interfaces.factr import fresh_leader_positions
from ..interfaces.factr import leader_stream_names
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
    ``"<side>/<signal>"`` paths. Consumers that read the FACTR leaders append
    the ``factr/<side>`` names themselves (see
    :func:`~dual_flexiv_control.interfaces.factr.leader_stream_names`) — the
    brain itself is a generic stream consumer with no FACTR knowledge.
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
    ) -> None:
        self._registry = registry
        self._stream_names = list(stream_names)
        self._attach_timeout_s = attach_timeout_s
        self._readers: dict[str, StreamReader] = {}
        #: brain→arm control-channel writers (created in :meth:`open_control`). The
        #: brain OWNS these segments (inverse of telemetry) and unlinks them on close.
        self._setpoint_writers: dict[str, StreamWriter] = {}
        self._command_writers: dict[str, StreamWriter] = {}
        #: optional per-side gripper mailbox writers (only for arms with a gripper).
        self._gripper_writers: dict[str, StreamWriter] = {}

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

    # -- control channel (brain -> arm) ---------------------------------------

    def open_control(self, control_registry: StreamRegistry, specs_by_side: dict) -> None:
        """Create the setpoint + command (+ optional gripper) writers per controlled arm.

        ``specs_by_side`` maps ``side -> {SETPOINT: StreamSpec, COMMAND: StreamSpec,
        [GRIPPER: StreamSpec]}`` (from :func:`dual_flexiv_control.control.control_specs`
        plus, for gripper-enabled arms, :func:`~dual_flexiv_control.control.gripper_spec`).
        The arms publish telemetry first and only then wait for these channels, so
        opening them here — after the brain has attached telemetry — cannot deadlock.

        The gripper mailbox (when present) is created *before* the setpoint/command
        channels so that once the arm sees those two appear (its attach condition),
        the gripper channel it optionally discovers is already registered — no race.
        """
        for side, specs in specs_by_side.items():
            if GRIPPER in specs:
                self._gripper_writers[side] = StreamWriter.create(
                    specs[GRIPPER], control_registry.run_id, control_registry
                )
            self._setpoint_writers[side] = StreamWriter.create(
                specs[SETPOINT], control_registry.run_id, control_registry
            )
            self._command_writers[side] = StreamWriter.create(
                specs[COMMAND], control_registry.run_id, control_registry
            )
        if specs_by_side:
            log.info(
                "brain opened control channels for %s (gripper: %s)",
                list(specs_by_side), list(self._gripper_writers) or "none",
            )

    def command(self, side: str, setpoint: np.ndarray) -> None:
        """Post a high-rate follower setpoint (latest-wins). No flexivrdk involved."""
        self._setpoint_writers[side].write(np.ascontiguousarray(setpoint, dtype=np.float64))

    def send_command(self, side: str, command: ControlCommand) -> None:
        """Post a discrete, reliable control event (home/stop/switch-mode)."""
        writer = self._command_writers[side]
        writer.write(command.encode(writer.spec.dim))

    def command_gripper(self, side: str, value: float) -> None:
        """Post a normalized (0..1) gripper target for one arm (latest-wins).

        No-op for an arm without a gripper channel (gripper disabled) — the caller
        may post unconditionally. The arm maps the fraction to a physical width and
        drives the gripper; see :class:`~dual_flexiv_control.configs.GripperCfg`.
        """
        writer = self._gripper_writers.get(side)
        if writer is not None:
            writer.write(np.array([float(value)], dtype=np.float64))

    @property
    def gripper_sides(self) -> list[str]:
        """Controlled arms with an open gripper channel (posted by the teleop loop)."""
        return list(self._gripper_writers)

    @property
    def controlled_sides(self) -> list[str]:
        return list(self._setpoint_writers)

    def stop_arms(self) -> None:
        """Post STOP to every controlled arm — the clean end-of-run handoff.

        Called by the consumers on their way out (before :meth:`close` unlinks the
        channels) so session-hosted arms leave their control session immediately
        instead of riding out the deadman. Best-effort and exception-safe: teardown
        must never raise, and an arm that already dropped its reader just won't see
        the command (its deadman/exit already handled it).
        """
        for side, writer in self._command_writers.items():
            try:
                writer.write(ControlCommand(CommandKind.STOP).encode(writer.spec.dim))
            except Exception:  # noqa: BLE001 - teardown must not raise
                log.exception("error sending STOP to %s", side)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        for writers in (self._setpoint_writers, self._command_writers, self._gripper_writers):
            for writer in writers.values():
                try:
                    writer.close()
                    writer.unlink()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("error releasing control writer %s", writer.name)
            writers.clear()


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
        # Subscribe the leader streams alongside the given ones: the brain itself
        # is a generic stream consumer; FACTR semantics stay in this node.
        subscribe = list(self.stream_names) + [
            n for n in leader_stream_names(self.factr_cfg)
            if n not in self.stream_names
        ]
        brain = Brain(registry, subscribe, self.cfg.attach_timeout_s)
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
        specs_by_side = {}
        for side, arm in self.arms.items():
            if not arm.control_enabled:
                continue
            specs = control_specs(side, arm.control)
            if arm.gripper.enabled:
                specs[GRIPPER] = gripper_spec(side, arm.control.channel)
            specs_by_side[side] = specs
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

        Reads the latest converted FACTR leader streams and posts them on the
        setpoint channel (latest-wins). This
        is pure numpy + the control channel — **no flexivrdk** here. Only joint-position
        (``qpos``) kinds map directly from FACTR joint readings; other kinds expect a
        policy setpoint source (not wired here) and are skipped with a one-time note.

        Override this for a policy: build the setpoint vector with
        :func:`~dual_flexiv_control.control.pack_streamed` and call ``self._brain.command``.
        """
        if not self._teleop or self._brain is None or not self.factr_cfg.servers:
            return
        # Fresh sides only — a dropped-out leader is simply absent, and its arm
        # keeps riding the last posted setpoint (the arm-side deadman covers a
        # long dropout). Never fabricate a target for a missing leader.
        leaders = fresh_leader_positions(self._brain, self.factr_cfg)
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
            # FACTR interface already publishes arm joints in the canonical DFC/Rizon
            # convention; ingestion is the only raw-leader conversion boundary.
            q_d = np.asarray(q_leader, dtype=np.float64).ravel()[:arm.dof]
            setpoint = pack_streamed(ctrl, {"q_d": q_d, "dq_d": np.zeros_like(q_d)})
            self._brain.command(side, setpoint)
            # Gripper rides its own latest-wins mailbox. FACTR ingestion already
            # normalized the trailing value using the leader-owned convention.
            q_leader = np.asarray(q_leader, dtype=np.float64).ravel()
            if q_leader.size:
                self._brain.command_gripper(side, float(q_leader[-1]))

    def _heartbeat(self, observation: dict[str, Samples]) -> None:
        fresh = sum(1 for s in observation.values() if s.n > 0)
        log.info("brain heartbeat: %d/%d streams fresh", fresh, len(observation))
