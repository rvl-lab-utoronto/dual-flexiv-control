"""The Flexiv interface node: one process per arm.

Read-only by default (``control_enabled=false``): it declares its proprio streams
and runs the base producer loop, publishing states and never touching the robot's
control. With ``control_enabled=true`` it becomes **producer + consumer** in the
same process — the single ``flexivrdk.Robot`` connection per arm forces the control
loop to live here, sharing that one handle. It then also attaches the brain's
control channels, drives the robot to the first commanded target, and tracks the
latest-wins setpoint mailbox (with a deadman) until asked to stop.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ...configs import ArmCfg
from ...configs import ControlCoeffsCfg
from ...configs import RuntimeCfg
from ...control import COMMAND
from ...control import SETPOINT
from ...control import CommandCursor
from ...control import CommandKind
from ...control import ControlCommand
from ...control import control_channel_name
from ...control import slice_streamed
from ...process import RateLimiter
from ...process import StreamProducerNode
from ...streams.registry import AttachAborted
from ...streams.registry import StreamRegistry
from ...streams.stream import StreamReader
from ...streams.stream import StreamWriter
from ...proprio import streams_to_specs
from ...streams.spec import StreamSpec
from .source import FakeFlexivSource
from .source import FlexivSource
from .source import SafetyHalt
from .states import map_states

log = logging.getLogger(__name__)

#: Leaf name of the per-arm status stream the dashboard reads to tell "connected"
#: from "disconnected" (and to show operation mode + E-stop). Published as
#: ``"<side>/status"`` — the consumer counterpart is ``dashboard.arms.STATUS_STREAM``.
STATUS_SIGNAL = "status"
#: 2-vector ``[operational_status_code, estop_pressed]``.
STATUS_DIM = 2
#: Refresh the (slow-changing) operation-mode/E-stop read at ~10 Hz regardless of the
#: telemetry/control loop rate: the RDK status calls are pointless to run at 1 kHz and
#: this keeps them off the hot control path's per-tick budget.
STATUS_RATE_HZ = 10.0


class FlexivInterface(StreamProducerNode):
    """Reads one Flexiv arm and publishes its proprio signals as streams.

    Stream names are ``"<side>/<signal>"`` (e.g. ``left/q``, ``right/tau``), with
    dims/dtype/capacity taken from ``arm.streams`` in the config. One
    :class:`FlexivInterface` runs per arm, so a bimanual setup spawns two. When the
    arm's ``control_enabled`` is set, :meth:`run` is overridden to additionally
    consume the brain's control channels and actuate (see :meth:`_run_control`).
    """

    def __init__(
        self,
        side: str,
        arm: ArmCfg,
        runtime: RuntimeCfg,
        run_id: str,
        coeffs: ControlCoeffsCfg | None = None,
    ) -> None:
        super().__init__(
            name=f"flexiv:{side}",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=arm.control_rate_hz if arm.control_enabled else arm.rate_hz,
        )
        self.side = side
        self.arm = arm
        self.sim = runtime.sim
        #: Active-phase controller coefficients (applied only when control_enabled).
        self.coeffs = coeffs if coeffs is not None else ControlCoeffsCfg()
        self._source: FlexivSource | FakeFlexivSource | None = None
        #: Cached ``[op_status_code, estop_pressed]`` + next-refresh deadline; the
        #: status read is throttled to ``STATUS_RATE_HZ`` (see :meth:`_status_signal`).
        self._status_cache: np.ndarray | None = None
        self._status_next_ns: int = 0

    def declare_streams(self) -> list[StreamSpec]:
        # Proprio streams (from config) + the dashboard-facing status stream. The
        # latter is published like the camera streams: not part of the brain's
        # observation set, only the dashboard's connection/mode indicator reads it.
        specs = streams_to_specs(self.side, self.arm.streams)
        specs.append(
            StreamSpec(
                name=f"{self.side}/{STATUS_SIGNAL}",
                dim=STATUS_DIM,
                capacity=64,
                dtype="float64",
                rate_hz=STATUS_RATE_HZ,
            )
        )
        return specs

    def _status_signal(self, now_ns: int) -> np.ndarray:
        """Cached ``[op_status_code, estop_pressed]``, refreshed at ~``STATUS_RATE_HZ``.

        A transient status read failure is swallowed (reusing the last value, or
        ``UNKNOWN``/clear on the very first tick): the status stream is a cosmetic
        dashboard indicator and must never take down the arm's data/control loop.
        """
        if self._status_cache is None or now_ns >= self._status_next_ns:
            try:
                self._status_cache = self._source.read_status()
            except Exception:  # noqa: BLE001 - status is cosmetic; never kill the loop
                log.debug("[%s] status read failed; reusing last", self.name, exc_info=True)
                if self._status_cache is None:
                    self._status_cache = np.zeros(STATUS_DIM)  # UNKNOWN, estop clear
            self._status_next_ns = now_ns + int(1e9 / STATUS_RATE_HZ)
        return self._status_cache

    def open_source(self) -> None:
        # A stream marked `dummy` means "fabricate, don't read hardware". One robot
        # connection feeds all of an arm's proprio streams, so it's all-or-nothing:
        # fabricate the whole arm (no flexivrdk connect) when every stream is dummy.
        streams = self.arm.streams.values()
        dummy = bool(streams) and all(getattr(s, "dummy", False) for s in streams)
        any_dummy = any(getattr(s, "dummy", False) for s in streams)
        if any_dummy and not dummy:
            log.warning(
                "[%s] some but not all proprio streams are dummy — a real robot can't "
                "fabricate a subset; treating the arm as REAL (connecting hardware).",
                self.name,
            )
        if self.sim or dummy:
            if dummy and not self.sim:
                log.info("[%s] publishing DUMMY (fabricated) proprio — no robot connection", self.name)
            self._source = FakeFlexivSource(self.arm.serial, dof=self.arm.dof)
        else:
            self._source = FlexivSource(
                self.arm.serial,
                dof=self.arm.dof,
                require_operational=self.arm.require_operational,
                verbose=self.arm.verbose_rdk,
            )
        self._source.open()

    def poll(self) -> dict[str, np.ndarray] | None:
        rs = self._source.read()
        # map_states emits float64; each writer casts to its stream's dtype.
        signals = map_states(rs, self.arm.wrench_frame)
        out = {f"{self.side}/{sig}": vec for sig, vec in signals.items()}
        out[f"{self.side}/{STATUS_SIGNAL}"] = self._status_signal(time.monotonic_ns())
        return out

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    # -- control loop (control_enabled only) ----------------------------------

    def run(self, stop_event) -> None:
        """Read-only base loop, or the merged producer+consumer control loop."""
        if not self.arm.control_enabled:
            super().run(stop_event)
            return
        self._run_control(stop_event)

    def _run_control(self, stop_event) -> None:
        registry = StreamRegistry(self.runtime_dir, self.run_id)               # telemetry (own)
        control_reg = StreamRegistry(self.runtime_dir, self.run_id, sub="control")  # commands (attach)
        ctrl = self.arm.control
        ch = ctrl.channel
        sp_reader: StreamReader | None = None
        cmd_cursor: CommandCursor | None = None
        try:
            # 1. Publish telemetry streams (producer), then connect the robot.
            for spec in self.declare_streams():
                self._writers[spec.name] = StreamWriter.create(spec, self.run_id, registry)
            self.open_source()
            log.info("[%s] published %d streams; source open", self.name, len(self._writers))

            # 2. Wait for the brain's control channels and attach (consumer). The
            #    brain publishes these only after it has attached our telemetry, so
            #    publishing telemetry first (above) is what prevents a deadlock.
            sp_name = control_channel_name(self.side, SETPOINT)
            cmd_name = control_channel_name(self.side, COMMAND)
            entries = control_reg.wait_for(
                [sp_name, cmd_name], self.arm.control_attach_timeout_s, stop_event=stop_event
            )
            sp_reader = StreamReader.attach(entries[sp_name])
            cmd_cursor = CommandCursor(StreamReader.attach(entries[cmd_name]))

            # 3. Enable servos, wait for the first setpoint, then bootstrap + switch.
            #    A STOP on the command channel (distinct from stop_event) must abort
            #    even during the blocking bootstrap, so both the first-setpoint wait
            #    and the MoveJ poll drain commands via this predicate.
            self._source.enter_control()
            abort = lambda: stop_event.is_set() or self._drain_commands(cmd_cursor)  # noqa: E731
            first = self._await_first_setpoint(sp_reader, stop_event, cmd_cursor)
            if first is None:
                return
            first_fields = slice_streamed(ctrl, first)
            if not self._source.start_control(
                ctrl, self.coeffs, first_fields, self._source.read(), abort=abort
            ):
                log.info("[%s] control bootstrap aborted before start", self.name)
                return
            log.info(
                "[%s] control loop started (kind=%s @ %.0f Hz)",
                self.name, ctrl.kind, self.arm.control_rate_hz,
            )

            # 4. The merged loop: publish telemetry, watch faults/commands, track the
            #    latest setpoint with a deadman, and actuate.
            rate = RateLimiter(self.arm.control_rate_hz)
            rate.reset()
            period = 1.0 / self.arm.control_rate_hz
            deadman_ns = int(ch.deadman_ms * 1e6)
            deadman_hard_ns = int(ch.deadman_hard_ms * 1e6)
            while not stop_event.is_set():
                t_ns = time.monotonic_ns()
                rs = self._source.read()

                # (a) telemetry out (+ the dashboard status stream, self-throttled)
                for sig, vec in map_states(rs, self.arm.wrench_frame).items():
                    self._writers[f"{self.side}/{sig}"].write(vec, t_ns)
                self._writers[f"{self.side}/{STATUS_SIGNAL}"].write(
                    self._status_signal(t_ns), t_ns
                )

                # (b) fault watchdog
                if self._source.fault():
                    log.error("[%s] robot fault during control; stopping", self.name)
                    break

                # (c) discrete commands (reliable, in order)
                if self._drain_commands(cmd_cursor):
                    break  # STOP requested

                # (d) latest-wins setpoint + deadman
                s = sp_reader.latest()
                if s.n == 0:
                    # Transient torn read of the single newest slot. After bootstrap
                    # the mailbox always holds >=1 sample, so n==0 is never "no setpoint
                    # ever" — soft-hold this tick rather than hard-stopping on a race.
                    rate.sleep()
                    continue
                age_ns = t_ns - s.newest_t_ns
                if age_ns > deadman_hard_ns:
                    log.error(
                        "[%s] setpoint stale > %.0f ms (deadman); hard stop",
                        self.name, ch.deadman_hard_ms,
                    )
                    break
                if age_ns > deadman_ns:
                    rate.sleep()   # soft hold: NRT motion generator parks at last target
                    continue

                # (e) actuate; the joint L-inf safety gate is enforced inside
                #     send_control against the effective commanded target (covers qvel).
                fields = slice_streamed(ctrl, s.newest)
                try:
                    self._source.send_control(
                        ctrl, self.coeffs, fields, rs, period,
                        safety_check=self.arm.control_safety_check,
                        tolerance=self.arm.control_tolerance,
                    )
                except SafetyHalt as exc:
                    log.error("[%s] %s; halting", self.name, exc)
                    break
                rate.sleep()
        except AttachAborted:
            log.info("[%s] control attach aborted by shutdown", self.name)
        finally:
            self._control_teardown(registry, sp_reader, cmd_cursor)

    def _await_first_setpoint(self, sp_reader: StreamReader, stop_event, cmd_cursor):
        """Block until the brain posts the first setpoint (or stop/STOP-command/timeout).

        The arm is already servoed and operational here, so a command-channel STOP must
        abort this wait too — it is drained each iteration (stop_event covers only
        process shutdown, not an operator/brain STOP).

        Telemetry keeps streaming during the wait: a policy-driven brain (eval)
        computes its first setpoint FROM the measured state, so withholding
        telemetry until the first setpoint would deadlock the bootstrap
        (observation incomplete <-> no setpoint). Teleop brains don't need this
        but the dashboard still benefits (live state while waiting).
        """
        deadline = time.monotonic() + self.arm.control_attach_timeout_s
        while not stop_event.is_set():
            if self._drain_commands(cmd_cursor):
                log.info("[%s] STOP during first-setpoint wait; aborting", self.name)
                return None
            s = sp_reader.latest()
            if s.n > 0:
                return s.newest
            if time.monotonic() > deadline:
                log.error(
                    "[%s] no setpoint within %.1fs; aborting control",
                    self.name, self.arm.control_attach_timeout_s,
                )
                return None
            t_ns = time.monotonic_ns()
            for sig, vec in map_states(self._source.read(), self.arm.wrench_frame).items():
                self._writers[f"{self.side}/{sig}"].write(vec, t_ns)
            self._writers[f"{self.side}/{STATUS_SIGNAL}"].write(
                self._status_signal(t_ns), t_ns
            )
            time.sleep(0.01)
        return None

    def _drain_commands(self, cmd_cursor: CommandCursor) -> bool:
        """Apply discrete commands; return True if a STOP was requested."""
        for row in cmd_cursor.drain_new():
            cmd = ControlCommand.decode(row)
            if cmd.kind == CommandKind.STOP:
                log.info("[%s] STOP command received", self.name)
                return True
            if cmd.kind == CommandKind.NONE:
                continue
            # HOME / SWITCH_MODE / SERVO_ON: surfaced for now; richer handling later.
            log.info("[%s] control command %s (args=%s)", self.name, cmd.kind.name, cmd.args)
        return False

    def _control_teardown(self, registry, sp_reader, cmd_cursor) -> None:
        try:
            if self._source is not None:
                self._source.stop()
        except Exception:  # noqa: BLE001 - teardown must not raise
            log.exception("[%s] error stopping robot", self.name)
        try:
            self.close_source()
        except Exception:  # noqa: BLE001
            log.exception("[%s] error closing source", self.name)
        for name, writer in self._writers.items():
            try:
                writer.close()
                writer.unlink()
            except Exception:  # noqa: BLE001
                log.exception("[%s] error releasing stream %s", self.name, name)
            registry.remove(name)
        for reader in (sp_reader, cmd_cursor):
            if reader is not None:
                try:
                    reader.close()   # readers never unlink (the brain owns the segments)
                except Exception:  # noqa: BLE001
                    log.exception("[%s] error closing control reader", self.name)
        log.info("[%s] control stopped", self.name)
