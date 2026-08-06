"""The FACTR interface node: ONE process consuming leader WebSockets, publishing streams.

FACTR leader data used to be fetched over HTTP independently by every consumer
(the collection loop, the brain node, the dashboard mirror, the status probe) —
four keep-alive connections that could disagree about reachability and payload
(the viewer showing live motion while control read the server's zeros
placeholder). This node makes FACTR a first-class stream producer like the arms
and cameras: it is the **only** network reader, and everyone else attaches
read-only to the shared-memory streams it publishes — so the
viewer and the control path see the *same samples* by construction. Each per-arm
client now drains a push WebSocket in the background, so this node samples the
freshest received frame without creating request traffic or a receive backlog.

Stream contract, each dim ``server.dof`` at :attr:`FactrCfg.rate_hz`:

* ``factr/raw/<side>`` — untouched server/Dynamixel payload.
* ``factr/<side>`` — arm joints converted once to DFC/Rizon coordinates, with
  the untouched trailing gripper retained.
* ``factr/telemetry/<side>/<field>`` — one explicit live FACTR model/control
  signal per stream (7-D joint vectors or 1-D scalar gains/timestamp).

Outage behavior: per-side tolerant, never crashes. An unreachable leader is
logged once (and once on recovery) and simply publishes nothing — its stream
goes stale, which readers detect by sample age (:attr:`FactrCfg.max_age_s`).
The stream segments survive the outage, so attached readers resume seamlessly
when the server comes back; consumers hold their last real target meanwhile.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ...configs import FactrCfg
from ...configs import JointConventionCfg
from ...configs import RuntimeCfg
from ...control.convention import convert_factr_to_rizon
from ...control.convention import normalize_gripper
from ...process import StreamProducerNode
from ...streams.registry import StreamRegistry
from ...streams.spec import StreamSpec
from ...streams.stream import StreamReader
from .client import FactrClient
from .client import FactrError

log = logging.getLogger(__name__)


def factr_stream_name(side: str) -> str:
    """DFC-coordinate leader pose plus trailing raw gripper value."""
    return f"factr/{side}"


def raw_factr_stream_name(side: str) -> str:
    """Unmodified FACTR-server/Dynamixel payload."""
    return f"factr/raw/{side}"


TELEMETRY_VECTOR_FIELDS = (
    "raw_q_rad",
    "model_q_rad",
    "model_dq_rad_s",
    "home_error_rad",
    "joint_offsets_rad",
    "model_signs",
    "limit_torque_nm",
    "null_torque_nm",
    "gravity_torque_nm",
    "friction_torque_nm",
    "force_feedback_torque_nm",
    "applied_torque_nm",
)
TELEMETRY_SCALAR_FIELDS = (
    "stamp_monotonic_ns",
    "grav_comp_gain",
    "grav_comp_gain_target",
    "force_feedback_gain",
    "force_feedback_gain_target",
)


def factr_telemetry_stream_name(side: str, field: str) -> str:
    """One explicit live telemetry signal from a FACTR leader."""
    return f"factr/telemetry/{side}/{field}"


def leader_stream_names(cfg: FactrCfg) -> list[str]:
    """The ``factr/<side>`` stream names for every configured leader."""
    return [factr_stream_name(side) for side in cfg.servers]


def fresh_leader_positions(source, cfg: FactrCfg) -> dict[str, np.ndarray]:
    """The leaders' newest *fresh* joint positions from their streams: ``{side: array}``.

    ``source`` is any attached stream consumer with ``latest(name) -> Samples``
    (a :class:`~dual_flexiv_control.brain.Brain`); the FACTR semantics — which
    streams are leaders, and how old a sample may be (``cfg.max_age_s``) — live
    here, not in the consumer. A side whose stream is unsubscribed, empty, or
    stale (leader dropout) is simply omitted: callers hold their last real
    target for missing sides, never fabricate one.
    """
    max_age_ns = int(float(cfg.max_age_s) * 1e9)
    now_ns = time.monotonic_ns()
    out: dict[str, np.ndarray] = {}
    for side in cfg.servers:
        try:
            s = source.latest(factr_stream_name(side))
        except KeyError:
            continue
        if s.n == 0 or (now_ns - s.newest_t_ns) > max_age_ns:
            continue
        out[side] = np.asarray(s.newest, dtype=np.float64)
    return out


def wait_leaders_fresh(source, cfg: FactrCfg, timeout_s: float, stop_event=None) -> None:
    """Block until EVERY configured leader stream carries a fresh sample.

    The launch-time teleop preflight, judged on the streams the run actually
    reads: the producer publishes ``factr/<side>`` the moment it starts (so a
    successful attach alone proves nothing about data), and this waits for real
    samples. Raises :class:`FactrError` naming the missing side(s) on timeout —
    the caller surfaces it as a failed run. Returns quietly if ``stop_event``
    fires (shutdown unwinds normally). No-op with no configured leaders.
    """
    if not cfg.servers:
        return
    deadline = time.monotonic() + timeout_s
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        missing = set(cfg.servers) - set(fresh_leader_positions(source, cfg))
        if not missing:
            return
        if time.monotonic() > deadline:
            raise FactrError(
                f"FACTR teleop preflight failed — no fresh leader data on "
                f"stream(s): {sorted(missing)} (is the FACTR server up and its "
                "leader publisher running?)"
            )
        time.sleep(0.05)


class FactrInterface(StreamProducerNode):
    """Samples every configured FACTR WebSocket cache; publishes ``factr/<side>``.

    One node covers all leaders (their streams aren't exclusive devices),
    so a rig spawns exactly one ``factr`` process. ``runtime.sim`` propagates to
    the client, which then fabricates positions without a network.
    """

    def __init__(self, cfg: FactrCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(
            name="factr",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=cfg.rate_hz,
        )
        self.cfg = cfg
        self.sim = runtime.sim
        self._client: FactrClient | None = None
        self._conventions: dict[str, JointConventionCfg] = {}
        self._telemetry_versions: dict[str, int] = {}
        #: Read-only views of each follower's estimated external joint torque.
        #: They are discovered lazily because the arm and FACTR processes start
        #: concurrently. This process owns the duplex leader sockets, so it is the
        #: one place that can carry follower torque back over those same sockets.
        self._feedback_registry: StreamRegistry | None = None
        self._feedback_readers: dict[str, StreamReader] = {}
        self._feedback_last_seq: dict[str, int] = {}
        self._feedback_errored: set[str] = set()
        #: sides currently failing, so outages log once (and once on recovery).
        self._errored: set[str] = set()

    @property
    def sides(self) -> list[str]:
        return list(self.cfg.servers)

    def declare_streams(self) -> list[StreamSpec]:
        specs = []
        for side, server in self.cfg.servers.items():
            for name in (factr_stream_name(side), raw_factr_stream_name(side)):
                specs.append(StreamSpec(
                    name=name, dim=server.dof, capacity=4096,
                    dtype="float64", rate_hz=self.cfg.rate_hz,
                ))
            for field in TELEMETRY_VECTOR_FIELDS:
                specs.append(StreamSpec(
                    name=factr_telemetry_stream_name(side, field),
                    dim=server.dof - 1, capacity=4096,
                    dtype="float64", rate_hz=self.cfg.rate_hz,
                ))
            for field in TELEMETRY_SCALAR_FIELDS:
                specs.append(StreamSpec(
                    name=factr_telemetry_stream_name(side, field),
                    dim=1, capacity=4096,
                    dtype="float64", rate_hz=self.cfg.rate_hz,
                ))
        return specs

    def open_source(self) -> None:
        self._client = FactrClient.from_config(self.cfg, sim=self.sim)
        self._feedback_registry = StreamRegistry(self.runtime_dir, self.run_id)
        for side, server in self.cfg.servers.items():
            leader = self.cfg.leaders.get(side)
            if leader is None:
                raise RuntimeError(f"FACTR {side} has a server but no DFC leader config")
            self._conventions[side] = self._validate_convention(
                side, server.dof, leader.raw_to_dfc
            )

    @staticmethod
    def _validate_convention(
        side: str, dof: int, conv: JointConventionCfg
    ) -> JointConventionCfg:
        """Validate and return DFC's persisted raw-leader convention."""
        drop = int(conv.drop_trailing)
        arm_dof = dof - drop
        offsets = [float(x) for x in conv.offsets_deg]
        flips = [int(x) for x in conv.sign_flip_joints]
        if drop < 1 or arm_dof <= 0 or len(offsets) != arm_dof:
            raise RuntimeError(
                f"FACTR {side} invalid leader convention: dof={dof}, drop={drop}, "
                f"offset count={len(offsets)}"
            )
        if len(set(flips)) != len(flips) or any(i < 0 or i >= arm_dof for i in flips):
            raise RuntimeError(f"FACTR {side} invalid sign-flip indices: {flips}")
        if conv.gripper_open is None or conv.gripper_closed is None:
            raise RuntimeError(f"FACTR {side} leader gripper calibration is missing")
        opened = float(conv.gripper_open)
        closed = float(conv.gripper_closed)
        if not np.all(np.isfinite(offsets + [opened, closed])) or opened == closed:
            raise RuntimeError(f"FACTR {side} leader calibration contains invalid values")
        validated = JointConventionCfg(
            offsets_deg=offsets,
            sign_flip_joints=flips,
            wrap_deg=bool(conv.wrap_deg),
            drop_trailing=drop,
            gripper_open=opened,
            gripper_closed=closed,
        )
        log.info("[factr] %s convention loaded from DFC leader config: %s", side, validated)
        return validated

    def poll(self) -> dict[str, np.ndarray] | None:
        sample: dict[str, np.ndarray] = {}
        for side in self._client.sides:
            try:
                jp = self._client.get_joint_positions_for(side)
            except FactrError as exc:
                if side not in self._errored:
                    self._errored.add(side)
                    log.warning(
                        "[factr] %s leader not reachable (stream goes stale until "
                        "it recovers): %s", side, exc,
                    )
                continue
            if side in self._errored:
                self._errored.discard(side)
                log.info("[factr] %s leader recovered", side)
            conv = self._conventions[side]
            q_dfc = convert_factr_to_rizon(jp, conv)
            converted = np.append(q_dfc, normalize_gripper(jp[-1], conv))
            sample[raw_factr_stream_name(side)] = jp
            sample[factr_stream_name(side)] = converted
            self._append_telemetry(side, sample)
            self._forward_force_feedback(side)
        return sample or None

    def _append_telemetry(self, side: str, sample: dict[str, np.ndarray]) -> None:
        """Publish each newly received telemetry field as its own typed stream."""
        cached = self._client.get_telemetry_for(side)
        if cached is None:
            return
        version, telemetry = cached
        if self._telemetry_versions.get(side) == version:
            return
        arm_dof = self.cfg.servers[side].dof - 1
        parsed: dict[str, np.ndarray] = {}
        try:
            for field in TELEMETRY_VECTOR_FIELDS:
                value = np.asarray(telemetry[field], dtype=np.float64)
                if value.shape != (arm_dof,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{field} has shape {value.shape}, expected ({arm_dof},)")
                parsed[factr_telemetry_stream_name(side, field)] = value
            for field in TELEMETRY_SCALAR_FIELDS:
                value = float(telemetry[field])
                if not np.isfinite(value):
                    raise ValueError(f"{field} is not finite")
                parsed[factr_telemetry_stream_name(side, field)] = np.asarray([value])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("[factr] dropping malformed %s telemetry frame: %s", side, exc)
            self._telemetry_versions[side] = version
            return
        sample.update(parsed)
        self._telemetry_versions[side] = version

    def _forward_force_feedback(self, side: str) -> None:
        """Send one fresh follower ``tau_ext`` sample to its matching leader.

        ``RobotStates.tau_ext`` is the RDK estimate of torque exerted *on the
        follower* by external contact. FACTR's original ``torque_feedback()`` law
        expects the opposite convention (follower-on-environment) and negates its
        input, so this transport negates RDK ``tau_ext`` once at the convention
        boundary. The two negatives cancel: a positive external follower torque
        produces a positive leader feedback torque. Measured actuator torque
        (``tau``) includes the robot's own dynamics and is deliberately not used.

        Samples are latest-wins and sent once each; if the arm stream disappears
        or goes stale, sending stops and the leader teleop's short timeout removes
        the feedback.
        """
        if self.sim or self._client is None or self._feedback_registry is None:
            return
        stream_name = f"{side}/tau_ext"
        reader = self._feedback_readers.get(side)
        if reader is None:
            entry = self._feedback_registry.get(stream_name)
            if entry is None:
                return
            try:
                reader = StreamReader.attach(entry)
            except FileNotFoundError:
                return  # producer restarted between discovery and attach
            self._feedback_readers[side] = reader
        try:
            samples = reader.latest()
        except (FileNotFoundError, ValueError):
            reader.close()
            self._feedback_readers.pop(side, None)
            self._feedback_last_seq.pop(side, None)
            return
        if samples.n == 0:
            return
        # Never keep replaying an old contact after the follower producer stalls.
        max_age_ns = int(float(self.cfg.max_age_s) * 1e9)
        if time.monotonic_ns() - samples.newest_t_ns > max_age_ns:
            return
        seq = int(samples.seq[-1])
        if self._feedback_last_seq.get(side) == seq:
            return
        try:
            # RDK: environment-on-follower. FACTR input: follower-on-environment.
            self._client.send_force_feedback_for(side, -samples.newest)
        except FactrError as exc:
            if side not in self._feedback_errored:
                self._feedback_errored.add(side)
                log.warning(
                    "[factr] %s follower external torque could not reach leader: %s",
                    side, exc,
                )
            return
        self._feedback_last_seq[side] = seq
        if side in self._feedback_errored:
            self._feedback_errored.discard(side)
            log.info("[factr] %s follower external-torque feedback recovered", side)

    def close_source(self) -> None:
        for reader in self._feedback_readers.values():
            reader.close()
        self._feedback_readers.clear()
        self._feedback_last_seq.clear()
        self._feedback_registry = None
        if self._client is not None:
            self._client.close()
            self._client = None
