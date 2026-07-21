"""The Flexiv interface node: one process per arm.

Read-only by default (``control_enabled=false``): it declares its proprio streams
and runs the base producer loop, publishing states and never touching the robot's
control. With ``control_enabled=true`` it becomes **producer + consumer** in the
same process — the single ``flexivrdk.Robot`` connection per arm forces the control
loop to live here, sharing that one handle. It then also attaches the brain's
control channels, drives the robot to the first commanded target, and tracks the
latest-wins setpoint mailbox (with a deadman) until asked to stop.

A control-enabled arm is an **idle ↔ control state machine**:

* **IDLE** — connected read-only, publishing telemetry + status at ``arm.rate_hz``;
  no control action of any kind (the session's VIEWING mode).
* **CONTROL** — entered on an :class:`EnterControl` message (which carries the
  active phase's coefficients): enable servos (eagerly, overlapping the
  consumer's own startup — ``arm.control_eager_enable`` reverts to enabling
  after the channels attach), attach the brain's channels, bootstrap to the
  first setpoint, then run the merged telemetry+control loop at
  ``arm.control_rate_hz``. *Every* exit (STOP command, deadman, fault, safety
  halt, aborted bootstrap) routes through ``robot.Stop()`` back to IDLE — the
  connection and the published streams survive across control sessions.

Two hosting modes select how sessions arrive: a **session queue** (``session_q``,
from the long-lived session daemon) delivers any number of :class:`EnterControl`
messages over the process's life; without one (the one-shot CLI), the node runs a
single control session with its spawn-time coefficients and then exits — exactly
the legacy ``runtime.phase=<p>`` behavior.
"""

from __future__ import annotations

import logging
import queue as queue_mod
import time
from dataclasses import dataclass
from dataclasses import field

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
from ...control import gripper_channel_name
from ...control import slice_streamed
from ...process import RateLimiter
from ...process import StreamProducerNode
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
#: 5-vector ``[operational_status_code, estop_pressed, control_active,
#: servo_enabled, mode_code]``. The third
#: element is 1.0 while this arm is inside a control session (COLLECTION/EVAL) and
#: 0.0 while IDLE — the session supervisor watches it to detect an arm that dropped
#: out of control mid-run (fault/deadman/safety halt), and the dashboard shows it.
STATUS_DIM = 5
#: Refresh the (slow-changing) operation-mode/E-stop read at ~10 Hz regardless of the
#: telemetry/control loop rate: the RDK status calls are pointless to run at 1 kHz and
#: this keeps them off the hot control path's per-tick budget.
STATUS_RATE_HZ = 10.0


@dataclass(frozen=True)
class EnterControl:
    """Session-layer message: begin one control session with these coefficients.

    Sent by the session supervisor on an arm's session queue when a collection or
    eval run starts (``coeffs`` are the active task's per-phase controller
    coefficients — the values :meth:`FlexivSource._apply_coeffs` applies after
    ``SwitchMode``). Plain dataclass of plain dataclasses, so it pickles across the
    spawn boundary. The session *ends* via the control channel (a STOP command or
    the deadman), not via a queue message.
    """

    coeffs: ControlCoeffsCfg = field(default_factory=ControlCoeffsCfg)
    phase: str = "collection"  # collection | eval (logging/diagnostics only)


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
        session_q=None,
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
        #: One-shot-mode controller coefficients (used when no session queue is
        #: given); a session-hosted arm gets per-run coeffs via :class:`EnterControl`.
        self.coeffs = coeffs if coeffs is not None else ControlCoeffsCfg()
        #: Session-daemon command queue (``mp.Queue`` of :class:`EnterControl`), or
        #: None for the legacy one-shot mode (single control session, then exit).
        self._session_q = session_q
        self._source: FlexivSource | FakeFlexivSource | None = None
        #: Cached ``[op_status_code, estop_pressed]`` + next-refresh deadline; the
        #: status read is throttled to ``STATUS_RATE_HZ`` (see :meth:`_status_signal`).
        self._status_cache: np.ndarray | None = None
        self._status_next_ns: int = 0
        #: True while inside a control session; published as the status stream's
        #: third element (see :data:`STATUS_DIM`).
        self._control_active = False

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
        """Operational/E-stop/control/servo/mode vector for the status stream.

        The mode/E-stop half is read from the source, cached, and refreshed at
        ~``STATUS_RATE_HZ``; ``control_active`` is this node's own live state and is
        appended fresh every call (so a session transition shows immediately). A
        transient status read failure is swallowed (reusing the last value, or
        ``UNKNOWN``/clear on the very first tick): the status stream is a monitoring
        indicator and must never take down the arm's data/control loop.
        """
        if self._status_cache is None or now_ns >= self._status_next_ns:
            try:
                self._status_cache = self._source.read_status()
            except Exception:  # noqa: BLE001 - status is cosmetic; never kill the loop
                log.debug("[%s] status read failed; reusing last", self.name, exc_info=True)
                if self._status_cache is None:
                    self._status_cache = np.zeros(4)  # UNKNOWN, clear, servo off, UNKNOWN mode
            self._status_next_ns = now_ns + int(1e9 / STATUS_RATE_HZ)
        return np.concatenate((
            self._status_cache[:2],
            [1.0 if self._control_active else 0.0],
            self._status_cache[2:4],
        ))

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

    # -- state machine (control_enabled only) ----------------------------------

    def run(self, stop_event) -> None:
        """Read-only base loop, or the idle ↔ control state machine."""
        if not self.arm.control_enabled:
            super().run(stop_event)
            return
        registry = StreamRegistry(self.runtime_dir, self.run_id)
        try:
            # Publish telemetry streams + connect the robot ONCE; both survive
            # across control sessions (the dashboard's "connected" indicator is
            # exactly these streams staying alive).
            for spec in self.declare_streams():
                self._writers[spec.name] = StreamWriter.create(spec, self.run_id, registry)
            self.open_source()
            log.info("[%s] published %d streams; source open", self.name, len(self._writers))
            if self._session_q is None:
                # One-shot hosting (the CLI): a single control session with the
                # spawn-time coeffs, then exit — the legacy runtime.phase behavior.
                # An attach timeout propagates (run_node exits non-zero: a crash).
                self._control_session(stop_event, self.coeffs)
            else:
                self._run_session_loop(stop_event)
        finally:
            self._teardown(registry)

    def _run_session_loop(self, stop_event) -> None:
        """IDLE ↔ CONTROL for the session daemon's lifespan.

        IDLE publishes read-only telemetry at ``arm.rate_hz`` and polls the session
        queue; an :class:`EnterControl` message runs one control session (at
        ``arm.control_rate_hz``, with that run's coeffs), after which the arm drops
        back to IDLE. Only ``stop_event`` ends the process. A control session that
        never starts (the consumer died before opening its channels) is logged and
        absorbed — the session supervisor records the run's outcome from the
        consumer's exit code; the arm itself just returns to IDLE.
        """
        rate = RateLimiter(self.arm.rate_hz)
        rate.reset()
        while not stop_event.is_set():
            msg = self._poll_session_q()
            if msg is not None:
                log.info("[%s] entering control session (phase=%s)", self.name, msg.phase)
                try:
                    self._control_session(stop_event, msg.coeffs)
                except TimeoutError as exc:
                    log.error("[%s] control session never started: %s", self.name, exc)
                log.info("[%s] control session ended; IDLE", self.name)
                rate = RateLimiter(self.arm.rate_hz)  # re-pace after the control-rate loop
                rate.reset()
                continue
            self._write_telemetry(time.monotonic_ns())
            rate.sleep()

    def _poll_session_q(self) -> EnterControl | None:
        """Next session message, or None. Non-blocking; never raises."""
        try:
            msg = self._session_q.get_nowait()
        except queue_mod.Empty:
            return None
        except Exception:  # noqa: BLE001 - a broken queue must not kill telemetry
            log.exception("[%s] session queue read failed", self.name)
            return None
        if not isinstance(msg, EnterControl):
            log.error("[%s] unknown session message %r; ignoring", self.name, msg)
            return None
        return msg

    def _write_telemetry(self, t_ns: int):
        """One telemetry tick: read the source, write every stream. Returns the states."""
        rs = self._source.read()
        for sig, vec in map_states(rs, self.arm.wrench_frame).items():
            self._writers[f"{self.side}/{sig}"].write(vec, t_ns)
        self._writers[f"{self.side}/{STATUS_SIGNAL}"].write(self._status_signal(t_ns), t_ns)
        return rs

    def _control_session(self, stop_event, coeffs: ControlCoeffsCfg) -> None:
        """One control session: attach channels → servo on → bootstrap → merged loop.

        EVERY exit — STOP command, deadman, robot fault, safety halt, aborted
        bootstrap, first-setpoint timeout, process shutdown — routes through the
        ``finally``: ``robot.Stop()`` then close the channel readers, so the caller
        always gets the arm back IDLE (connected, telemetry flowing, no control
        state). Raises ``TimeoutError`` only if the brain's channels never appear
        (before any servo action) — the one failure the caller may want to treat
        as a crash (one-shot mode does; the session loop absorbs it).
        """
        control_reg = StreamRegistry(self.runtime_dir, self.run_id, sub="control")
        ctrl = self.arm.control
        ch = ctrl.channel
        sp_reader: StreamReader | None = None
        cmd_cursor: CommandCursor | None = None
        grip_reader: StreamReader | None = None
        self._control_active = True
        try:
            # 0. Eager servo-on (config-gated): Enable/brake release takes seconds
            #    and needs nothing from the brain, so start it BEFORE waiting on
            #    the channels — it overlaps the consumer's spawn + imports + attach
            #    instead of serializing after them. Every failure past this point
            #    still routes through the finally's robot.Stop() → IDLE. Tradeoff:
            #    a consumer that dies before publishing its channels leaves the
            #    arm enabled-but-idle for the attach timeout (motion-safe).
            if self.arm.control_eager_enable:
                self._source.enter_control()

            # 1. Wait for the brain's control channels and attach (consumer). The
            #    brain publishes these only after it has attached our telemetry, so
            #    telemetry being published first is what prevents a deadlock.
            entries = self._await_channels(control_reg, stop_event)
            if entries is None:
                return  # shutdown while waiting
            sp_name = control_channel_name(self.side, SETPOINT)
            cmd_name = control_channel_name(self.side, COMMAND)
            sp_reader = StreamReader.attach(entries[sp_name])
            cmd_cursor = CommandCursor(StreamReader.attach(entries[cmd_name]))
            # Optional gripper mailbox: attach it if this arm has a gripper AND the
            # brain published the channel for this session (teleop does; a policy
            # eval run does not). The brain creates it *before* setpoint/command, so
            # by now it is registered if it exists at all — a single lookup suffices.
            if self.arm.gripper.enabled:
                grip_entry = control_reg.get(gripper_channel_name(self.side))
                if grip_entry is not None:
                    grip_reader = StreamReader.attach(grip_entry)
                else:
                    log.info(
                        "[%s] gripper enabled but no gripper channel this session "
                        "(non-teleop run?); gripper not actuated", self.name,
                    )

            # 2. Enable servos (unless already eagerly enabled above), wait for the
            #    first setpoint, then bootstrap + switch. A STOP on the command
            #    channel (distinct from stop_event) must abort even during the
            #    blocking bootstrap, so both the first-setpoint wait and the MoveJ
            #    poll drain commands via this predicate — which also ticks
            #    telemetry, keeping status/proprio fresh (dashboard + the session
            #    supervisor's freshness watchdog) through the multi-second MoveJ.
            if not self.arm.control_eager_enable:
                self._source.enter_control()

            def abort() -> bool:
                self._write_telemetry(time.monotonic_ns())
                return stop_event.is_set() or self._drain_commands(cmd_cursor)
            first = self._await_first_setpoint(sp_reader, stop_event, cmd_cursor)
            if first is None:
                return
            first_fields = slice_streamed(ctrl, first)
            if not self._source.start_control(
                ctrl, coeffs, first_fields, self._source.read(), abort=abort
            ):
                log.info("[%s] control bootstrap aborted before start", self.name)
                return
            # Enable + home the gripper once, now (blocking is fine during bootstrap,
            # not inside the loop). If setup fails/declines, drop the reader so the
            # loop skips gripper work entirely.
            if grip_reader is not None and not self._source.setup_gripper(self.arm.gripper):
                grip_reader.close()
                grip_reader = None
            log.info(
                "[%s] control loop started (kind=%s @ %.0f Hz, gripper=%s)",
                self.name, ctrl.kind, self.arm.control_rate_hz,
                "on" if grip_reader is not None else "off",
            )

            # 3. The merged loop: publish telemetry, watch faults/commands, track the
            #    latest setpoint with a deadman, and actuate.
            rate = RateLimiter(self.arm.control_rate_hz)
            rate.reset()
            period = 1.0 / self.arm.control_rate_hz
            deadman_ns = int(ch.deadman_ms * 1e6)
            deadman_hard_ns = int(ch.deadman_hard_ms * 1e6)
            while not stop_event.is_set():
                t_ns = time.monotonic_ns()

                # (a) telemetry out (+ the dashboard status stream, self-throttled)
                rs = self._write_telemetry(t_ns)

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
                        ctrl, coeffs, fields, rs, period,
                        safety_check=self.arm.control_safety_check,
                        tolerance=self.arm.control_tolerance,
                    )
                except SafetyHalt as exc:
                    log.error("[%s] %s; halting", self.name, exc)
                    break

                # (f) gripper: track the leader's latest normalized target (its own
                #     latest-wins mailbox). Self-throttled/deadbanded in send_gripper,
                #     so calling it every tick is cheap. Only alongside live actuation
                #     — a stale/held joint setpoint already `continue`d above. A gripper
                #     fault disables the gripper for the session but never stops the arm.
                if grip_reader is not None:
                    g = grip_reader.latest()
                    if g.n:
                        try:
                            self._source.send_gripper(float(g.newest[0]), t_ns)
                        except Exception:  # noqa: BLE001 - gripper is non-critical to arm motion
                            log.exception("[%s] gripper command failed; disabling for session", self.name)
                            self._source.stop_gripper()
                            grip_reader.close()
                            grip_reader = None
                rate.sleep()
        finally:
            self._control_active = False
            try:
                if self._source is not None:
                    self._source.stop_gripper()  # best-effort; never raises
                    self._source.stop()  # blocking stop -> IDLE; connection retained
            except Exception:  # noqa: BLE001 - teardown must not raise
                log.exception("[%s] error stopping robot", self.name)
            for reader in (sp_reader, cmd_cursor, grip_reader):
                if reader is not None:
                    try:
                        reader.close()  # readers never unlink (the brain owns the segments)
                    except Exception:  # noqa: BLE001
                        log.exception("[%s] error closing control reader", self.name)
            log.info("[%s] control stopped", self.name)

    def _await_channels(self, control_reg: StreamRegistry, stop_event):
        """Wait for this arm's control channels, ticking telemetry meanwhile.

        Returns the registry entries, or None if ``stop_event`` fired. Raises
        ``TimeoutError`` past ``control_attach_timeout_s``. Telemetry keeps flowing
        during the wait (unlike a blocking ``wait_for``) so the dashboard's live
        view never stalls across a session start.
        """
        wanted = (
            control_channel_name(self.side, SETPOINT),
            control_channel_name(self.side, COMMAND),
        )
        deadline = time.monotonic() + self.arm.control_attach_timeout_s
        while not stop_event.is_set():
            found = control_reg.discover()
            if all(name in found for name in wanted):
                return {name: found[name] for name in wanted}
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"control channels not published within "
                    f"{self.arm.control_attach_timeout_s}s: {list(wanted)}"
                )
            self._write_telemetry(time.monotonic_ns())
            time.sleep(0.02)
        return None

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
            self._write_telemetry(time.monotonic_ns())
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
