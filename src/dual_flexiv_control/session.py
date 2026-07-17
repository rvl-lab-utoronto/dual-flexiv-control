r"""The session daemon: persistent hardware + a VIEWING ↔ COLLECTION ↔ EVAL state machine.

One long-lived process owns the rig for the dashboard's lifespan::

    dfc-session rig=bimanual runtime.sim=false

At startup it spawns the persistent hardware nodes (one process per arm, one per
camera) exactly once. Arms idle **read-only** (VIEWING: telemetry + status streams
flow, no control of any kind). A ``start`` command composes the selected task's
config, hands each control-enabled arm that phase's coefficients (an
:class:`~.interfaces.flexiv.EnterControl` on its session queue), and spawns the
phase's consumer (:class:`~.collection.CollectionNode` or
:class:`~.policy.EvalNode`) — the same node the one-shot CLI runs. When the
consumer exits (user stop, self-finish, or crash) its outcome is recorded and the
session drops back to VIEWING with the hardware untouched.

**Protocol** (dashboard ↔ daemon):

* commands — JSON lines on **stdin**: ``{"cmd": "start", "phase": "collection",
  "task": "<name>"}`` (eval may add ``"policy": <conf/policy name>``,
  ``"host": <str>``, and/or ``"port": <int>`` to point at a different policy
  server; a ``"skill"`` phase takes ``"skill": <saved skill name>`` instead of a
  task), ``{"cmd": "switch", …same fields…}`` (atomic mode switch: stops any
  active run, then starts this one the moment the session is idle — see
  ``state.pending``), ``{"cmd": "stop"}`` (also cancels a pending switch),
  ``{"cmd": "reconnect_arm", "side": "<side>"}`` / ``{"cmd": "respawn_camera",
  "name": "zed:<name>"}`` (replace that hardware node now, when no run needs
  it), ``{"cmd": "shutdown"}``. stdin EOF == shutdown, so a dead dashboard can
  never leave an orphaned daemon holding robots.
* state — ``<runtime_dir>/session.json``, atomically replaced (same pattern as the
  stream-manifest registry): current mode, active run, a heartbeat timestamp, and
  the last run's outcome. The dashboard polls it.

**Supervision invariants**: a consumer exit returns the session to VIEWING (its
crashed control channels are swept so the next run can't attach a dead segment).
An **arm** node exit is NOT fatal: the arm is reported down (``arms_down`` in the
state file), a run it was commanding is stopped (with its save grace), its
leaked streams are swept and its session queue drained (a stale ``EnterControl``
must never fire on respawn), and the node is respawned periodically while no run
is active — a fresh process makes a fresh RDK connection (flexivrdk cannot
re-init in-process) and comes back **IDLE, read-only**; entering control is
always user-initiated. An alive-but-stale arm is only *reported* (never
auto-killed: a wedged RDK handle is indistinguishable from a long Enable/MoveJ);
the ``reconnect_arm`` command replaces it on demand.
A **camera** node exit or stall is likewise NOT fatal: the camera is reported
down (``cameras_down``), runs are refused while any rig camera is down
(recording needs every view), and the node is respawned periodically — a
stalled-but-alive node is killed first, because a ZED handle never recovers
once its device dropped — so a recovered camera (e.g. after a USB replug) is
picked back up automatically. Arm telemetry and VIEWING are never held hostage
by a flaky camera.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import signal
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from .configs import Config
from .configs import register_configs
from .orphans import sweep as orphan_sweep
from .orphans import tag_supervisor
from .process import run_node
from .streams.registry import StreamRegistry
from .streams.registry import cleanup_run
from .streams.registry import cleanup_sub
from .streams.stream import StreamReader
from .system import _escalate
from .system import _shutdown
from .system import active_coeffs
from .system import build_consumer
from .system import build_hardware_nodes
from .system import make_run_id
from .interfaces.flexiv.interface import EnterControl

log = logging.getLogger(__name__)

#: Session states surfaced in ``session.json``.
VIEWING = "viewing"
COLLECTION = "collection"
EVAL = "eval"
SKILL = "skill"
SAVING = "saving"

PHASES = (COLLECTION, EVAL, SKILL)

#: Main supervise-loop cadence.
_TICK_S = 0.05
#: Heartbeat / state-file refresh cadence (also written immediately on any change).
_HEARTBEAT_S = 1.0
#: Arm-status poll cadence during a run (watches ``control_active`` for dropouts).
_STATUS_POLL_S = 0.5
#: How often a dead camera node is respawned (a recovered camera — e.g. after a
#: USB replug — comes back automatically without restarting the session).
_CAMERA_RESPAWN_S = 30.0
#: A live camera process gets this long after spawn to start producing frames
#: before it reads as down (a real ZED open takes several seconds).
_CAMERA_BOOT_GRACE_S = 20.0
#: A camera whose newest published frame is older than this reads as down.
_CAMERA_FRESH_S = 5.0
#: Cooperative grace when killing a stalled-but-alive camera node before its
#: respawn (a wedged ZED grab may not unwind; SIGTERM/SIGKILL follow promptly).
_CAMERA_KILL_GRACE_S = 2.0
#: How often a dead arm node is respawned while no run is active (a fresh
#: process makes a fresh RDK connection; the arm comes back IDLE, read-only).
_ARM_RESPAWN_S = 20.0
#: A live arm process gets this long after spawn to connect + start publishing
#: before it reads as down (a real RDK open retries for ~30 s).
_ARM_BOOT_GRACE_S = 45.0
#: An arm whose newest published status sample is older than this reads as down.
#: Generous: ``enter_control()`` (Enable/brake release) blocks without ticking
#: telemetry for several seconds.
_ARM_FRESH_S = 20.0
#: Cooperative grace when replacing a live arm node on ``reconnect_arm``.
_ARM_KILL_GRACE_S = 2.0
#: How long a queued (pending) start waits for the arms to finish winding down
#: out of their previous control session before it is abandoned. Counts from
#: the moment the session is otherwise idle (a stopping run's save does not
#: eat into it — an episode save may legitimately take minutes).
_PENDING_WINDDOWN_S = 20.0
#: Periodic orphan-sweep cadence (tag-based detector only; the boot sweep also
#: runs the legacy heuristic — see :mod:`dual_flexiv_control.orphans`).
_ORPHAN_SWEEP_S = 300.0

#: session.json filename under the runtime dir (one session per runtime dir).
STATE_BASENAME = "session.json"


def state_path(runtime_dir: str | os.PathLike[str]) -> str:
    return os.path.join(str(runtime_dir), STATE_BASENAME)


# ---------------------------------------------------------------------------
# Protocol helpers (pure, testable)
# ---------------------------------------------------------------------------


def parse_command(line: str) -> dict | None:
    """One stdin line -> a command dict, or None if malformed (logged by caller)."""
    line = line.strip()
    if not line:
        return None
    try:
        cmd = json.loads(line)
    except json.JSONDecodeError:
        return None
    return cmd if isinstance(cmd, dict) and isinstance(cmd.get("cmd"), str) else None


def classify_outcome(phase: str, exitcode: int | None, stopping: bool) -> tuple[str, str]:
    """(outcome, human detail) for a consumer that exited with ``exitcode``.

    Mirrors the dashboard's historical wording: a clean exit after a user stop
    means the episode saved (collection) / the rollout stopped (eval); a clean
    self-exit means the run hit its target; anything non-zero is a crash.
    """
    if exitcode == 0:
        if stopping:
            if phase == COLLECTION:
                return "saved", "collection stopped — episode saved to the dataset."
            if phase == SKILL:
                return "stopped", "skill replay stopped."
            return "stopped", "eval stopped."
        if phase == COLLECTION:
            return "finished", "collection finished (target episodes reached)."
        if phase == SKILL:
            return "finished", "skill replay finished."
        return "finished", "eval finished (rollout horizon reached)."
    if stopping:
        return (
            "stopped-error",
            f"run stopped but exited abnormally (code {exitcode}); "
            "a collection episode may not have saved.",
        )
    if phase == SKILL:
        return (
            "crashed",
            f"skill replay failed (exit code {exitcode}) — e.g. the arm never "
            "reached the skill's start pose (E-stop/fault/safety halt), or a "
            "source failed.",
        )
    return (
        "crashed",
        f"run stopped on its own (exit code {exitcode}) — a source likely failed "
        "(e.g. a camera, the FACTR server, or the policy server).",
    )


@dataclass
class SessionState:
    """What ``session.json`` carries; the dashboard's whole view of the daemon."""

    pid: int
    run_id: str
    rig: str
    sim: bool
    state: str = VIEWING
    task: str | None = None
    phase: str | None = None
    run_seq: int = 0
    run_started_ts: float | None = None
    heartbeat_ts: float = 0.0
    #: Transient operator feedback (e.g. a refused start); cleared on the next start.
    message: str | None = None
    #: The most recently finished run: {phase, task, outcome, exitcode, detail}.
    last_outcome: dict | None = None
    #: Camera nodes currently down (crashed / not producing). Non-fatal: VIEWING
    #: continues on the arms, but runs are refused until every rig camera is back.
    cameras_down: list = field(default_factory=list)
    #: Arm nodes currently down (crashed / not publishing). Non-fatal: a dead arm
    #: is respawned to IDLE telemetry automatically; runs it would feed are
    #: refused until it is back. Never auto-enters control.
    arms_down: list = field(default_factory=list)
    #: A queued start waiting for the session to go idle (the ``switch``
    #: command's tail): ``{phase, task, skill}``, or None. The UI keys its
    #: "switching…" affordance on this — there is no separate state value.
    pending: dict | None = None


class StateFile:
    """Atomically-replaced JSON state file (same pattern as the stream registry)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, state: SessionState) -> None:
        state.heartbeat_ts = time.time()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(state), f)
        os.replace(tmp, self.path)

    def remove(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def read_state(runtime_dir: str | os.PathLike[str]) -> dict | None:
    """Parse ``session.json`` under ``runtime_dir``, or None (absent/torn)."""
    try:
        with open(state_path(runtime_dir)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Config composition (per session at startup; per run for the task's coeffs)
# ---------------------------------------------------------------------------


def compose_config(overrides: list[str]) -> Config:
    """Compose + validate the full config for ``overrides`` (plain dataclasses).

    Same result as the ``@hydra.main`` path in :mod:`.system` (schema-validated,
    ``runtime_dir`` resolved absolute against the cwd), via the compose API so the
    daemon can re-compose per run (``task=<name> runtime.phase=<phase>``).
    """
    from hydra import compose
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config", overrides=overrides)
    config: Config = OmegaConf.to_object(cfg)
    if not os.path.isabs(config.runtime.runtime_dir):
        config.runtime.runtime_dir = os.path.abspath(config.runtime.runtime_dir)
    return config


def run_overrides(
    session_overrides: list[str], task: str, phase: str, extra: list[str] = (),
) -> list[str]:
    """The compose overrides for one run: the session's, re-pinned to task + phase.

    ``extra`` overrides (e.g. a per-run ``policy.port``) are appended last and
    likewise re-pinned: a session override of the same key is dropped so the
    per-run value wins.
    """
    pinned = {"task", "runtime.phase", *(e.split("=", 1)[0].lstrip("+~") for e in extra)}
    keep = [
        o for o in session_overrides
        if o.split("=", 1)[0].lstrip("+~") not in pinned
    ]
    return [*keep, f"task={task}", f"runtime.phase={phase}", *extra]


def _override_value(overrides: list[str], key: str) -> str | None:
    for o in overrides:
        k, _, v = o.partition("=")
        if k.lstrip("+~") == key:
            return v
    return None


# ---------------------------------------------------------------------------
# Run preflight (fail fast BEFORE the arms are told to enter control)
# ---------------------------------------------------------------------------

#: TCP probe budget for a remote eval policy server.
_PREFLIGHT_TCP_TIMEOUT_S = 3.0


def preflight_run(run_config: Config, phase: str) -> str | None:
    """Why this run would die on its external dependencies, or None to proceed.

    Historically an unreachable FACTR leader / policy server was discovered
    *inside* the consumer, after the arms had already been handed a control
    session — the consumer crashed, the arms waited out their channel timeout,
    and the next start was refused mid-wind-down. Checking here (cost ≤ ~1 s,
    touches no robot hardware) turns that into an instant, actionable refusal
    with the arms never leaving IDLE.

    * collection — nothing here: leader liveness is judged on the
      ``factr/<side>`` streams the run will actually read (the daemon's
      :meth:`SessionDaemon._factr_stale_sides` gate in ``_handle_start``), not
      on a separate HTTP probe that could disagree with them.
    * eval — for a ``remote`` policy, a bare TCP connect to ``host:port``:
      protocol-agnostic (openpi websocket + ACME HTTP alike) and catches the
      dominant failure, connection refused. The consumer's own connect stays
      the backstop for servers that accept then fail. Non-remote kinds (e.g.
      ``hold``) need no server.
    * skill — nothing external (arm proprio only).
    """
    if phase == EVAL:
        policy = run_config.policy
        if getattr(policy, "kind", None) != "remote":
            return None
        import socket

        host, port = policy.host, int(policy.port)
        try:
            socket.create_connection((host, port), timeout=_PREFLIGHT_TCP_TIMEOUT_S).close()
        except OSError as exc:
            return (
                f"start refused — policy server {host}:{port} is not reachable "
                f"({exc}). Start the policy server (or fix the host/port "
                "overrides), then retry."
            )
        return None
    return None  # SKILL: no external dependency


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


@dataclass
class _ActiveRun:
    """The consumer process of the run in flight, plus its stop bookkeeping."""

    phase: str
    task: str
    proc: mp.process.BaseProcess
    stop_event: object                      # the consumer's own mp Event
    command_sides: list[str]
    stopping: bool = False
    #: Dropout watchdog arms once every commanded side has reported control_active=1.
    watchdog_armed: bool = False
    active_sides: set = field(default_factory=set)


@dataclass
class _PendingStart:
    """A validated start waiting for the session to go idle (``switch``'s tail)."""

    cmd: dict
    #: Monotonic deadline for the arm wind-down wait; None while a run is still
    #: stopping/saving (the countdown arms once the consumer is reaped).
    deadline: float | None = None


@dataclass
class _CameraUnit:
    """One camera node + its live process; respawned (paced) when it dies or stalls.

    Each camera gets its OWN stop event: ``run_node`` sets its stop event on a
    crash ("bring the system down with us" — the right call for a one-shot run),
    so sharing the session-wide event would let one dying camera cooperatively
    stop the arms and every other camera. Isolated events contain the blast
    radius to the camera itself.
    """

    node: object                            # ZedInterface (picklable, re-spawnable)
    proc: mp.process.BaseProcess
    stop_event: object                      # this camera's own mp Event
    spawned_at: float
    announced_down: bool = False
    stalled_since: float | None = None      # alive but not producing since (monotonic)


@dataclass
class _ArmUnit:
    """One arm node + its live process; respawned (paced) when it dies.

    Per-arm stop events for the same blast-radius reason as cameras. Unlike a
    camera, an alive-but-stale arm is never auto-killed (a wedged RDK handle is
    indistinguishable from a long Enable/MoveJ) — only a *dead* process is
    respawned, and the respawn comes back IDLE (read-only telemetry): entering
    control stays user-initiated, always.
    """

    node: object                            # FlexivInterface (picklable, re-spawnable)
    side: str
    proc: mp.process.BaseProcess
    stop_event: object                      # this arm's own mp Event
    spawned_at: float
    announced_down: bool = False
    #: Death already handled (joined + streams swept + session queue drained)?
    #: Reset on respawn. Keeps the one-shot cleanup from re-running every tend.
    reaped: bool = False


@dataclass
class _FactrUnit:
    """The FACTR producer node + its live process; respawned (paced) if it dies.

    Its own stop event for the same blast-radius reason as cameras. Like an arm
    (and unlike a camera), an alive-but-stale producer is never killed: stale
    ``factr/<side>`` streams mean the FACTR *server* is unreachable — the node
    reconnects per request and recovers on its own, while killing it would
    unlink the stream segments under every attached reader for no gain.
    """

    node: object                            # FactrInterface (picklable, re-spawnable)
    proc: mp.process.BaseProcess
    stop_event: object                      # the producer's own mp Event
    spawned_at: float
    announced_down: bool = False


class SessionDaemon:
    """Owns the hardware processes and the mode state machine; see the module doc."""

    def __init__(self, config: Config, overrides: list[str]) -> None:
        self.config = config
        self.overrides = overrides
        self.run_id = make_run_id()
        self.ctx = mp.get_context("spawn")  # never fork: flexivrdk has live threads
        #: Session-wide stop for _shutdown's escalation; every hardware unit ALSO
        #: has its own event (set first in _teardown) so its crash stays contained.
        self.hw_stop = self.ctx.Event()
        #: side -> session queue, for every control-enabled arm.
        self.session_qs = {
            side: self.ctx.Queue()
            for side, arm in config.arms.items()
            if arm.control_enabled
        }
        self.arms: list[_ArmUnit] = []
        self.cameras: list[_CameraUnit] = []
        self.factr: _FactrUnit | None = None
        self.run: _ActiveRun | None = None
        self.pending: _PendingStart | None = None
        self.cmd_q: queue_mod.Queue = queue_mod.Queue()
        self.state = SessionState(
            pid=os.getpid(),
            run_id=self.run_id,
            rig=_override_value(overrides, "rig") or "default",
            sim=bool(config.runtime.sim),
        )
        self.state_file = StateFile(state_path(config.runtime.runtime_dir))
        self._registry = StreamRegistry(config.runtime.runtime_dir, self.run_id)
        self._fatal: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start_hardware(self) -> None:
        nodes = build_hardware_nodes(self.config, self.run_id, self.session_qs)
        # Arms, cameras and the FACTR producer are all supervised individually
        # (down + respawn), never session-fatal.
        arm_nodes = [
            n for n in nodes if not n.name.startswith("zed:") and n.name != "factr"
        ]
        cam_nodes = [n for n in nodes if n.name.startswith("zed:")]
        factr_node = next((n for n in nodes if n.name == "factr"), None)
        self.arms = [self._make_arm_unit(node) for node in arm_nodes]
        self.cameras = [self._make_camera_unit(node) for node in cam_nodes]
        self.factr = None if factr_node is None else self._make_factr_unit(factr_node)
        log.info(
            "session %s: %d arm node(s) + %d camera node(s) + %s up "
            "(rig=%s sim=%s); VIEWING",
            self.run_id, len(self.arms), len(self.cameras),
            "the factr node" if self.factr is not None else "no factr node",
            self.state.rig, self.state.sim,
        )

    def _make_arm_unit(self, node) -> _ArmUnit:
        stop_event = self.ctx.Event()  # per-arm: its crash must not stop the rest
        proc = self.ctx.Process(target=run_node, args=(node, stop_event), name=node.name)
        proc.start()
        return _ArmUnit(
            node=node, side=node.side, proc=proc, stop_event=stop_event,
            spawned_at=time.monotonic(),
        )

    def _make_camera_unit(self, node) -> _CameraUnit:
        stop_event = self.ctx.Event()  # per-camera: its crash must not stop the rest
        proc = self.ctx.Process(target=run_node, args=(node, stop_event), name=node.name)
        proc.start()
        return _CameraUnit(
            node=node, proc=proc, stop_event=stop_event, spawned_at=time.monotonic()
        )

    def _make_factr_unit(self, node) -> _FactrUnit:
        stop_event = self.ctx.Event()  # its own: a factr crash must not stop the rest
        proc = self.ctx.Process(target=run_node, args=(node, stop_event), name=node.name)
        proc.start()
        return _FactrUnit(
            node=node, proc=proc, stop_event=stop_event, spawned_at=time.monotonic()
        )

    def _sweep_node_streams(self, node) -> None:
        """Reclaim a killed node's leaked segments so its respawn can re-create them.

        A node that died cleanly unlinked its own streams (no-op here); one that
        was SIGKILLed mid-life leaks them, and ``StreamWriter.create`` on the same
        names would then fail every respawn. For a dead arm this also drops its
        frozen ``<side>/status`` manifest — a stale ``control_active=1`` in it
        would otherwise wedge :meth:`_arms_still_in_control` into refusing every
        future start.
        """
        from multiprocessing import shared_memory

        from .streams.registry import shm_name_for

        for spec in node.declare_streams():
            try:
                shm = shared_memory.SharedMemory(
                    name=shm_name_for(self.run_id, spec.name), create=False
                )
                shm.close()
                shm.unlink()
                log.info("swept leaked segment of %s", spec.name)
            except FileNotFoundError:
                pass
            self._registry.remove(spec.name)

    def serve(self) -> int:
        """The supervise loop; returns the process exit code."""
        self._start_stdin_listener()
        self.start_hardware()
        self.state_file.write(self.state)
        last_beat = time.monotonic()
        last_status_poll = 0.0
        last_hw_tend = 0.0
        last_orphan_sweep = time.monotonic()
        try:
            while True:
                changed = self._drain_commands()
                if self._fatal == "shutdown":
                    break
                changed |= self._reap_consumer()
                changed |= self._tend_pending()
                now = time.monotonic()
                if self.run is not None and now - last_status_poll >= _STATUS_POLL_S:
                    last_status_poll = now
                    self._watch_arm_dropout()
                if now - last_hw_tend >= 1.0:
                    last_hw_tend = now
                    changed |= self._tend_cameras()
                    changed |= self._tend_arms()
                    self._tend_factr()
                if now - last_orphan_sweep >= _ORPHAN_SWEEP_S:
                    last_orphan_sweep = now
                    # Tag-based detector only while running (the boot sweep in
                    # main() also ran the legacy heuristic); keeps other dead
                    # supervisors' debris from wedging OUR devices mid-session.
                    orphan_sweep(
                        self.config.runtime.runtime_dir,
                        keep_run_ids=(self.run_id,),
                        include_legacy=False,
                    )
                if changed or now - last_beat >= _HEARTBEAT_S:
                    last_beat = now
                    self.state_file.write(self.state)
                time.sleep(_TICK_S)
        finally:
            self._teardown()
        if self._fatal and self._fatal != "shutdown":
            log.error("session fatal: %s", self._fatal)
            return 1
        return 0

    def _teardown(self) -> None:
        # Stop the run first (the episode gets its save grace), then the hardware
        # (short grace; a wedged RDK handle is force-killed promptly), then sweep shm.
        if self.run is not None:
            log.info("shutdown: stopping active %s run (save grace %.0fs)…",
                     self.run.phase, self.config.runtime.save_grace_s)
            self.run.stop_event.set()
            _escalate([self.run.proc], cooperative_s=self.config.runtime.save_grace_s)
            self.run = None
        factr_units = [self.factr] if self.factr is not None else []
        for unit in (*self.arms, *self.cameras, *factr_units):  # each via its own event
            unit.stop_event.set()
        hw_procs = [unit.proc for unit in (*self.arms, *self.cameras, *factr_units)]
        _shutdown(hw_procs, self.hw_stop, save_grace_s=0.0)
        n = cleanup_run(self.config.runtime.runtime_dir, self.run_id)
        self.state_file.remove()
        log.info("session shutdown complete; unlinked %d shm segment(s)", n)

    # -- command intake ----------------------------------------------------------

    def _start_stdin_listener(self) -> None:
        def _listen() -> None:
            for line in sys.stdin:
                cmd = parse_command(line)
                if cmd is None:
                    log.warning("ignoring malformed command line: %r", line.strip())
                    continue
                self.cmd_q.put(cmd)
            self.cmd_q.put({"cmd": "shutdown"})  # EOF: the dashboard is gone

        threading.Thread(target=_listen, name="dfc-session-stdin", daemon=True).start()

    def install_signal_handlers(self) -> None:
        def _handle(signum, _frame):  # noqa: ANN001
            log.info("session received signal %s -> shutdown", signum)
            self.cmd_q.put({"cmd": "shutdown"})

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _drain_commands(self) -> bool:
        changed = False
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue_mod.Empty:
                return changed
            changed = True
            kind = cmd.get("cmd")
            if kind == "shutdown":
                self._fatal = "shutdown"
                return changed
            if kind == "start":
                self._handle_start(cmd)
            elif kind == "switch":
                self._handle_switch(cmd)
            elif kind == "stop":
                self._handle_stop()
            elif kind == "reconnect_arm":
                self._handle_reconnect_arm(cmd)
            elif kind == "respawn_camera":
                self._handle_respawn_camera(cmd)
            else:
                self.state.message = f"unknown command {kind!r}"
                log.warning("unknown command: %r", cmd)

    # -- run lifecycle -----------------------------------------------------------

    @staticmethod
    def _validate_start(cmd: dict) -> str | None:
        """Why this start/switch command is malformed, or None if well-formed.

        Purely structural: live-state gates (active run, wind-down, cameras/arms
        down, preflight) stay in :meth:`_handle_start`, where they apply at
        execution time — a queued switch re-passes them when it actually fires.
        """
        phase, task = cmd.get("phase"), cmd.get("task")
        # A skill replay is task-independent; any composable task satisfies the
        # schema, so a missing task just takes the default.
        if phase == SKILL and not task:
            task = "default"
        if phase not in PHASES or not task:
            return f"bad start command (phase={phase!r}, task={task!r})"
        skill_name = cmd.get("skill")
        if skill_name is not None and not (
            isinstance(skill_name, str) and skill_name
            and all(c.isalnum() or c in "._-" for c in skill_name)
        ):
            return f"bad start command (skill={skill_name!r})"
        if phase == SKILL and not skill_name:
            return "bad start command (a skill run needs a skill name)"
        port = cmd.get("port")
        if port is not None and not (isinstance(port, int) and 0 < port < 65536):
            return f"bad start command (port={port!r})"
        host = cmd.get("host")
        if host is not None and not (
            isinstance(host, str) and host
            and all(c.isalnum() or c in "._-" for c in host)
        ):
            return f"bad start command (host={host!r})"
        policy = cmd.get("policy")
        if policy is not None and not (
            isinstance(policy, str) and policy
            and all(c.isalnum() or c in "._-" for c in policy)
        ):
            return f"bad start command (policy={policy!r})"
        return None

    def _handle_start(self, cmd: dict) -> None:
        error = self._validate_start(cmd)
        if error is not None:
            self.state.message = error
            return
        if self.run is not None:
            self.state.message = (
                "a run is already active — stop it first (or switch, which does both)"
            )
            return
        phase, task = cmd.get("phase"), cmd.get("task")
        if phase == SKILL and not task:
            task = "default"
        skill_name = cmd.get("skill")
        port = cmd.get("port")
        host = cmd.get("host")
        policy = cmd.get("policy")
        busy = self._arms_still_in_control()
        if busy:
            # The arms exit control on their own within seconds of a run ending;
            # queue the start instead of bouncing "retry shortly" at the operator.
            self._set_pending(cmd, deadline=time.monotonic() + _PENDING_WINDDOWN_S)
            self.state.message = (
                f"arm(s) still winding down a control session: {sorted(busy)} — "
                f"start queued (fires when idle, waits ≤{_PENDING_WINDDOWN_S:.0f}s)"
            )
            return
        if self.state.cameras_down and phase != SKILL:
            # Recording/eval sample every rig camera; a missing view would stall
            # the run at frame 0 (or record holes). Refuse with the reason —
            # "still starting" (boot window) reads differently from "down".
            # A skill replay observes only arm proprio, so it is not camera-gated.
            names = ", ".join(self.state.cameras_down)
            booting_only = all(
                not unit.announced_down
                for unit in self.cameras
                if unit.node.name in self.state.cameras_down
            )
            if booting_only:
                self.state.message = (
                    f"camera(s) still starting: {names} — retry in a few seconds."
                )
            else:
                self.state.message = (
                    f"camera(s) down: {names} — runs need every rig camera. "
                    "Fix/replug the camera (it is retried automatically) or pick "
                    "a rig without it."
                )
            return
        if self.state.arms_down and phase != SKILL:
            # Collection/eval observe EVERY rig arm's proprio, so a missing arm
            # would stall the consumer at attach. A skill run touches only the
            # sides it drives — gated below, after drive_sides narrowing.
            names = ", ".join(self.state.arms_down)
            booting_only = all(
                not unit.announced_down
                for unit in self.arms
                if unit.node.name in self.state.arms_down
            )
            if booting_only:
                self.state.message = (
                    f"arm(s) still starting: {names} — retry in a few seconds."
                )
            else:
                self.state.message = (
                    f"arm(s) down: {names} — reconnecting automatically; retry "
                    "when the arm is back."
                )
            return
        extra = []
        if policy is not None:
            # The group selection first: policy.host/port refine the chosen type.
            extra.append(f"policy={policy}")
        if host is not None:
            extra.append(f"policy.host={host}")
        if port is not None:
            extra.append(f"policy.port={port}")
        if phase == SKILL:
            extra.append(f"skill.name={skill_name}")
        try:
            run_config = compose_config(run_overrides(self.overrides, task, phase, extra))
            coeffs = active_coeffs(run_config)
            # For a skill run this also loads the skill file, so a missing/corrupt
            # skill surfaces here as a refused start, not a crashed run.
            consumer = build_consumer(run_config, self.run_id)
        except Exception as exc:  # noqa: BLE001 - bad task/config must not kill the session
            log.exception("start %s task=%s failed to compose", phase, task)
            self.state.message = f"start failed: {exc}"
            return
        # Fail fast on unreachable external dependencies (FACTR leaders, remote
        # policy server) BEFORE any arm is told to enter control: a refusal here
        # leaves the arms IDLE and the session instantly ready for the next try.
        refusal = preflight_run(run_config, phase)
        if refusal is None and phase == COLLECTION:
            # Teleop needs live leader data, judged on the SAME factr/<side>
            # streams the run will read (not a separate HTTP probe that can
            # disagree with them). Stale = server down OR serving without its
            # ROS producer — either way the leaders aren't feeding teleop.
            stale = self._factr_stale_sides()
            if stale:
                refusal = (
                    f"start refused — FACTR leader stream(s) not fresh: "
                    f"{', '.join(stale)}. Start/check the FACTR server(s) "
                    "(and their leader publishers), then retry."
                )
        if refusal is not None:
            log.error("start %s task=%s refused by preflight: %s", phase, task, refusal)
            self.state.message = refusal
            return
        command_sides = [
            s for s, arm in run_config.arms.items()
            if arm.control_enabled and s in self.session_qs
        ]
        # A consumer that declares which sides it will feed (a skill drives only
        # the taught sides) narrows the EnterControl fan-out: an arm handed a
        # control session nobody feeds would just time out waiting for setpoints.
        drive_sides = getattr(consumer, "drive_sides", None)
        if drive_sides is not None:
            command_sides = [s for s in command_sides if s in drive_sides]
            if not command_sides:
                self.state.message = (
                    f"skill {skill_name!r} has no driveable side on this rig — it "
                    "needs a control-enabled arm (qpos control) matching the "
                    "skill's taught side(s)"
                )
                return
        down_commanded = sorted(self._down_sides() & set(command_sides))
        if down_commanded:
            self.state.message = (
                f"arm(s) down: {', '.join(down_commanded)} — reconnecting "
                "automatically; retry when the arm is back."
            )
            return
        for side in command_sides:
            self.session_qs[side].put(EnterControl(coeffs=coeffs, phase=phase))
        stop_event = self.ctx.Event()
        proc = self.ctx.Process(target=run_node, args=(consumer, stop_event), name=consumer.name)
        proc.start()
        # A skill run is labelled by its skill (the composed task is incidental).
        label = skill_name if phase == SKILL else task
        self.run = _ActiveRun(
            phase=phase, task=label, proc=proc,
            stop_event=stop_event, command_sides=command_sides,
        )
        self.state.state = phase
        self.state.task = label
        self.state.phase = phase
        self.state.run_seq += 1
        self.state.run_started_ts = time.time()
        self.state.message = None
        self._clear_pending()  # a queued start firing later would just be refused
        log.info("run %d: %s task=%s%s (consumer pid %s; commanding %s)",
                 self.state.run_seq, phase, task,
                 f" skill={skill_name}" if skill_name else "",
                 proc.pid, command_sides)

    def _handle_switch(self, cmd: dict) -> None:
        """Atomic mode switch: stop the active run (if any), then start ``cmd``.

        One command replaces the stop → poll → retry-start dance: the start is
        queued (:class:`_PendingStart`) and fires from :meth:`_tend_pending` the
        moment the session is idle, re-passing every normal start gate. A second
        ``switch`` replaces the queued one; ``stop`` cancels it. No new state
        value: the observable sequence stays e.g. ``collection → saving →
        viewing → eval``, with ``state.pending`` carrying the "switching…" hint.
        """
        error = self._validate_start(cmd)
        if error is not None:
            self.state.message = error
            return
        if self.run is None and not self._arms_still_in_control():
            self._handle_start(cmd)  # already idle: a switch is just a start
            return
        phase = cmd.get("phase")
        if self.run is not None:
            log.info("switch to %s requested: stopping the active %s run",
                     phase, self.run.phase)
            self._handle_stop()  # before _set_pending: stop cancels any older pending
            self._set_pending(cmd, deadline=None)  # countdown arms once reaped
            self.state.message = (
                f"switching to {phase} — stopping the current run "
                "(the episode saves first)…"
            )
        else:  # no run, but arms still winding down out of the previous session
            self._set_pending(cmd, deadline=time.monotonic() + _PENDING_WINDDOWN_S)
            self.state.message = (
                f"switching to {phase} — waiting for the arm(s) to go idle…"
            )

    def _set_pending(self, cmd: dict, deadline: float | None) -> None:
        self.pending = _PendingStart(cmd=dict(cmd), deadline=deadline)
        self.state.pending = {
            "phase": cmd.get("phase"),
            "task": cmd.get("task"),
            "skill": cmd.get("skill"),
        }

    def _clear_pending(self) -> None:
        self.pending = None
        self.state.pending = None

    def _tend_pending(self) -> bool:
        """Fire the queued start once the session is idle; abandon it if stale.

        While a run is still stopping/saving nothing counts down (a save may
        legitimately take minutes); once the consumer is reaped the wind-down
        deadline arms, covering the seconds the arms take to exit control.
        Returns True when the pending start fired or was abandoned.
        """
        p = self.pending
        if p is None or self.run is not None:
            return False
        now = time.monotonic()
        if p.deadline is None:
            p.deadline = now + _PENDING_WINDDOWN_S
        if self._arms_still_in_control():
            if now < p.deadline:
                return False
            self._clear_pending()
            self.state.message = (
                f"queued {p.cmd.get('phase')} start abandoned — arm(s) still in "
                f"a control session after {_PENDING_WINDDOWN_S:.0f}s"
            )
            log.error("%s", self.state.message)
            return True
        self._clear_pending()
        log.info("session idle — firing the queued %s start", p.cmd.get("phase"))
        self._handle_start(p.cmd)
        return True

    def _handle_stop(self) -> None:
        cancelled = self.pending is not None
        if cancelled:
            log.info("queued %s start cancelled", self.pending.cmd.get("phase"))
            self._clear_pending()
        if self.run is None:
            self.state.message = (
                "queued start cancelled" if cancelled else "no active run to stop"
            )
            return
        if not self.run.stopping:
            self.run.stopping = True
            self.run.stop_event.set()
            self.state.state = SAVING
            log.info("stop requested: %s run winding down (episode saving)…", self.run.phase)

    def _handle_reconnect_arm(self, cmd: dict) -> None:
        """Replace an arm node NOW (fresh process → fresh RDK connection, IDLE).

        The operator's escape hatch for a wedged-alive arm (which the watchdog
        deliberately never auto-kills) — and an immediate retry for a dead one,
        bypassing the respawn pacing. Refused mid-run / mid-control: killing a
        controlling arm's node would drop the robot mid-motion.
        """
        side = cmd.get("side")
        unit = next((u for u in self.arms if u.side == side), None)
        if unit is None:
            self.state.message = f"unknown arm side {side!r}"
            return
        if self.run is not None or self.pending is not None:
            self.state.message = "cannot reconnect an arm during a run — stop it first"
            return
        if side in self._arms_still_in_control():
            self.state.message = (
                f"arm {side} is still winding down a control session — retry shortly"
            )
            return
        if unit.proc.is_alive():
            log.warning("reconnect_arm %s: replacing the live node "
                        "(fresh RDK connection)", side)
            unit.stop_event.set()
            _escalate([unit.proc], cooperative_s=_ARM_KILL_GRACE_S)
        else:
            unit.proc.join()
        self._sweep_node_streams(unit.node)
        self._drain_session_q(unit.side)
        fresh_unit = self._make_arm_unit(unit.node)
        unit.proc, unit.stop_event = fresh_unit.proc, fresh_unit.stop_event
        unit.spawned_at = fresh_unit.spawned_at
        unit.reaped = False
        self.state.message = (
            f"arm {side}: reconnecting (fresh RDK connection; back to IDLE telemetry)…"
        )
        log.info("reconnect_arm %s: node respawned (IDLE, read-only)", side)

    def _handle_respawn_camera(self, cmd: dict) -> None:
        """Replace a camera node NOW (the "I replugged it" button).

        Skips the ``_CAMERA_RESPAWN_S`` pacing of the automatic watchdog.
        Refused during a recording/eval run (they sample every camera); a skill
        run observes only arm proprio, so it does not block this.
        """
        name = cmd.get("name")
        unit = next((u for u in self.cameras if u.node.name == name), None)
        if unit is None:
            self.state.message = f"unknown camera {name!r}"
            return
        if self.run is not None and self.run.phase != SKILL:
            self.state.message = (
                "cannot respawn a camera during a recording/eval run — stop it first"
            )
            return
        if unit.proc.is_alive():
            unit.stop_event.set()
            _escalate([unit.proc], cooperative_s=_CAMERA_KILL_GRACE_S)
        else:
            unit.proc.join()
        self._sweep_node_streams(unit.node)
        fresh_unit = self._make_camera_unit(unit.node)
        unit.proc, unit.stop_event = fresh_unit.proc, fresh_unit.stop_event
        unit.spawned_at = fresh_unit.spawned_at
        unit.stalled_since = None
        self.state.message = f"camera {name}: respawning now…"
        log.info("respawn_camera %s: node respawned", name)

    def _reap_consumer(self) -> bool:
        """Harvest a finished consumer: outcome -> state, sweep channels, VIEWING."""
        run = self.run
        if run is None or run.proc.is_alive():
            return False
        run.proc.join()
        outcome, detail = classify_outcome(run.phase, run.proc.exitcode, run.stopping)
        self.state.last_outcome = {
            "run_seq": self.state.run_seq,  # lets the dashboard alert exactly once
            "phase": run.phase, "task": run.task,
            "outcome": outcome, "exitcode": run.proc.exitcode, "detail": detail,
        }
        # If the consumer died without brain.close() (crash/SIGKILL), its control
        # channel segments + manifests leak — sweep them so the next run's arms
        # can never attach a dead channel.
        swept = cleanup_sub(self.config.runtime.runtime_dir, self.run_id, sub="control")
        if swept:
            log.info("swept %d leaked control channel segment(s)", swept)
        self.state.state = VIEWING
        self.state.task = None
        self.state.phase = None
        self.state.run_started_ts = None
        self.run = None
        log.info("run ended: %s (%s)", outcome, detail)
        return True

    # -- watchdogs ---------------------------------------------------------------

    def _arm_status(self, side: str) -> np.ndarray | None:
        """Newest ``<side>/status`` vector, or None (not yet published / torn)."""
        entry = self._registry.get(f"{side}/status")
        if entry is None:
            return None
        try:
            reader = StreamReader.attach(entry)
            try:
                s = reader.latest()
                return np.asarray(s.newest) if s.n > 0 else None
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - racing a producer teardown
            return None

    def _arms_still_in_control(self) -> list[str]:
        """Sides whose status stream still reports an active control session."""
        busy = []
        for side in self.session_qs:
            vec = self._arm_status(side)
            if vec is not None and len(vec) >= 3 and vec[2] >= 0.5:
                busy.append(side)
        return busy

    def _watch_arm_dropout(self) -> None:
        """Auto-stop the run when a commanded arm falls out of control mid-run.

        The watchdog arms only after EVERY commanded side has reported
        ``control_active=1`` (session startup is not a dropout), and stands down
        while the run is already stopping (arms exiting control is then expected).
        A dropout means the arm hit its deadman / a fault / a safety halt — the
        consumer would otherwise keep recording a frozen arm.
        """
        run = self.run
        if run is None or run.stopping or not run.command_sides:
            return
        for side in run.command_sides:
            vec = self._arm_status(side)
            active = vec is not None and len(vec) >= 3 and vec[2] >= 0.5
            if active:
                run.active_sides.add(side)
            elif run.watchdog_armed and side in run.active_sides:
                log.error(
                    "arm %s dropped out of control mid-run (deadman/fault/safety halt); "
                    "stopping the run", side,
                )
                self.state.message = f"arm {side} dropped out of control; run stopped"
                self._handle_stop()
                return
        if not run.watchdog_armed and run.active_sides >= set(run.command_sides):
            run.watchdog_armed = True

    def _arms_alive(self) -> bool:
        return all(unit.proc.is_alive() for unit in self.arms)

    # -- arm supervision (non-fatal) ------------------------------------------------

    def _down_sides(self) -> set:
        """Sides of the arms currently reported down."""
        return {
            unit.side for unit in self.arms
            if unit.node.name in self.state.arms_down
        }

    def _arm_fresh(self, unit: _ArmUnit) -> bool:
        """Is this arm publishing fresh status samples?

        Same rationale as :meth:`_camera_producing`: process liveness lies during
        a doomed RDK connect. The ``<side>/status`` stream ticks with telemetry
        (and throughout control sessions), so its newest sample's age is the
        truthful signal.
        """
        entry = self._registry.get(f"{unit.side}/status")
        if entry is None:
            return False
        try:
            reader = StreamReader.attach(entry)
            try:
                s = reader.latest()
                if s.n == 0:
                    return False
                return (time.monotonic_ns() - s.newest_t_ns) < _ARM_FRESH_S * 1e9
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - racing the producer's teardown
            return False

    def _drain_session_q(self, side: str) -> None:
        """Discard queued session messages for ``side``.

        Called when the side's arm node died: a stale ``EnterControl`` left in
        the queue would otherwise fire the moment the respawned arm reaches its
        session loop — a respawned arm must come back IDLE, never in control.
        """
        q = self.session_qs.get(side)
        if q is None:
            return
        drained = 0
        while True:
            try:
                q.get_nowait()
                drained += 1
            except queue_mod.Empty:
                break
            except Exception:  # noqa: BLE001 - a broken queue: nothing left to drain
                break
        if drained:
            log.warning("drained %d stale session message(s) for arm %s", drained, side)

    def _reap_dead_arm(self, unit: _ArmUnit) -> None:
        """One-shot cleanup for a dead arm node: join, sweep streams, drain queue,
        and stop a run it was feeding (the dropout watchdog reads the now-frozen
        status stream, so it cannot see a process death)."""
        unit.reaped = True
        unit.proc.join()
        log.error(
            "arm node %s exited (code %s) — session continues; respawning to "
            "IDLE telemetry every %.0fs while no run is active",
            unit.node.name, unit.proc.exitcode, _ARM_RESPAWN_S,
        )
        self._sweep_node_streams(unit.node)
        self._drain_session_q(unit.side)
        run = self.run
        if run is None or run.stopping:
            return
        # Commanded sides lose their controller outright; collection/eval also
        # observe every arm's proprio, so a dead read-only arm would freeze the
        # recording. Either way the run cannot continue meaningfully.
        if unit.side in run.command_sides or run.phase != SKILL:
            self.state.message = f"arm {unit.side} node died mid-run; run stopped"
            log.error("arm %s node died mid-run; stopping the %s run",
                      unit.side, run.phase)
            self._handle_stop()

    def _tend_arms(self) -> bool:
        """Per-arm supervision: report down arms, respawn dead ones (paced).

        A *dead* arm process is cleaned up immediately (:meth:`_reap_dead_arm`)
        and respawned — only while no run is active — at most every
        ``_ARM_RESPAWN_S``; the fresh process reconnects RDK from scratch and
        idles read-only. An *alive* arm that stopped publishing past its boot
        grace is only reported: unlike a ZED, a live RDK handle mid-Enable/MoveJ
        looks identical to a wedged one, so replacing it is the operator's call
        (``reconnect_arm``). Returns True when ``state.arms_down`` changed.
        """
        now = time.monotonic()
        down: list[str] = []
        for unit in self.arms:
            alive = unit.proc.is_alive()
            if alive:
                fresh = self._arm_fresh(unit)
                in_grace = (now - unit.spawned_at) < _ARM_BOOT_GRACE_S
                if fresh or in_grace:
                    if fresh and unit.announced_down:
                        unit.announced_down = False
                        log.info("arm node %s recovered — telemetry flowing (IDLE)",
                                 unit.node.name)
                    if not fresh:
                        down.append(unit.node.name)  # still connecting: gate, no alarm
                    continue
                down.append(unit.node.name)
                if not unit.announced_down:
                    unit.announced_down = True
                    log.error(
                        "arm node %s is alive but not publishing — NOT auto-killed "
                        "(may be mid-Enable/MoveJ); use reconnect_arm to replace it",
                        unit.node.name,
                    )
                continue
            down.append(unit.node.name)
            if not unit.reaped:
                unit.announced_down = True
                self._reap_dead_arm(unit)
            if self.run is not None or now - unit.spawned_at < _ARM_RESPAWN_S:
                continue
            fresh_unit = self._make_arm_unit(unit.node)
            unit.proc, unit.stop_event = fresh_unit.proc, fresh_unit.stop_event
            unit.spawned_at = fresh_unit.spawned_at
            unit.reaped = False
            log.info("respawned arm node %s (IDLE, read-only)", unit.node.name)
        down.sort()
        if down != self.state.arms_down:
            self.state.arms_down = down
            return True
        return False

    # -- camera supervision (non-fatal) -------------------------------------------

    def _camera_producing(self, unit: _CameraUnit) -> bool:
        """Is this camera actually publishing fresh frames?

        Liveness of the process is not enough: a wedged ZED takes seconds to fail
        *after* its streams are declared, so "manifest exists" and "process alive"
        both lie during the doomed-open window. The newest sample's age on the
        camera's first view is the truthful signal (samples only flow once the
        camera opened).
        """
        from .cameras import camera_stream_name

        node = unit.node
        views = list(node.cam.views)
        if not views:
            return True  # nothing to produce; don't hold runs hostage
        entry = self._registry.get(camera_stream_name(node.cam_name, views[0]))
        if entry is None:
            return False
        try:
            reader = StreamReader.attach(entry)
            try:
                s = reader.latest()
                if s.n == 0:
                    return False
                return (time.monotonic_ns() - s.newest_t_ns) < _CAMERA_FRESH_S * 1e9
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - racing the producer's teardown
            return False

    def _tend_cameras(self) -> bool:
        """Per-camera supervision: report down cameras, respawn them (paced).

        A camera is *down* when its process died, or when it stopped producing
        fresh frames past its boot grace. Both are respawned at most every
        ``_CAMERA_RESPAWN_S``: a dead process directly; a stalled-but-alive one
        is killed first — the ZED SDK never revives a handle whose device
        dropped, so after an unplug the old process keeps failing ``grab()``
        forever even once the camera is back on the bus, and only a fresh
        ``open()`` picks it up. Returns True when ``state.cameras_down``
        changed (the caller then rewrites the state file).
        """
        now = time.monotonic()
        down: list[str] = []
        for unit in self.cameras:
            alive = unit.proc.is_alive()
            producing = alive and self._camera_producing(unit)
            in_grace = alive and (now - unit.spawned_at) < _CAMERA_BOOT_GRACE_S
            if producing or in_grace:
                if producing:
                    unit.stalled_since = None
                if producing and unit.announced_down:
                    unit.announced_down = False
                    log.info("camera node %s recovered — producing again", unit.node.name)
                if not producing:
                    down.append(unit.node.name)  # still booting: gate runs, no alarm
                continue
            down.append(unit.node.name)
            if not unit.announced_down:
                unit.announced_down = True
                log.error(
                    "camera node %s is down (%s) — session continues on the arms; "
                    "respawning it every %.0fs",
                    unit.node.name,
                    f"exited code {unit.proc.exitcode}" if not alive else "not producing",
                    _CAMERA_RESPAWN_S,
                )
            if alive:
                if unit.stalled_since is None:
                    unit.stalled_since = now
                if now - unit.stalled_since < _CAMERA_RESPAWN_S:
                    continue
                log.warning("killing stalled camera node %s for respawn", unit.node.name)
                unit.stop_event.set()
                _escalate([unit.proc], cooperative_s=_CAMERA_KILL_GRACE_S)
            elif now - unit.spawned_at < _CAMERA_RESPAWN_S:
                continue
            self._sweep_node_streams(unit.node)
            fresh = self._make_camera_unit(unit.node)
            unit.proc, unit.stop_event = fresh.proc, fresh.stop_event
            unit.spawned_at = fresh.spawned_at
            unit.stalled_since = None
            log.info("respawned camera node %s", unit.node.name)
        down.sort()
        if down != self.state.cameras_down:
            self.state.cameras_down = down
            return True
        return False

    def _tend_factr(self) -> None:
        """Supervise the FACTR producer: respawn it (paced) only when it DIES.

        Staleness is deliberately not a kill signal (see :class:`_FactrUnit`):
        the producer logs and rides out server outages on the same stream
        segments, so attached readers resume seamlessly. Data-level freshness is
        enforced where it matters — the collection start gate
        (:meth:`_factr_stale_sides`) and every reader's ``max_age_s`` check.
        """
        unit = self.factr
        if unit is None:
            return
        if unit.proc.is_alive():
            if unit.announced_down:
                unit.announced_down = False
                log.info("factr node recovered — process back up")
            return
        if not unit.announced_down:
            unit.announced_down = True
            log.error(
                "factr node is down (exited code %s) — leader streams frozen; "
                "respawning it every %.0fs", unit.proc.exitcode, _CAMERA_RESPAWN_S,
            )
        if time.monotonic() - unit.spawned_at < _CAMERA_RESPAWN_S:
            return
        self._sweep_node_streams(unit.node)
        fresh = self._make_factr_unit(unit.node)
        unit.proc, unit.stop_event = fresh.proc, fresh.stop_event
        unit.spawned_at = fresh.spawned_at
        log.info("respawned factr node")

    def _factr_stale_sides(self) -> list[str]:
        """Configured leader sides whose ``factr/<side>`` stream is not fresh.

        The collection start gate: a side is *stale* when its stream is absent,
        empty, or its newest sample is older than ``factr.max_age_s`` — i.e. the
        leader is not actually feeding teleop right now, however reachable its
        server looks. Empty when the rig configures no leaders (nothing to gate).
        """
        from .interfaces.factr import factr_stream_name

        if self.factr is None:
            return []
        max_age_ns = float(self.config.factr.max_age_s) * 1e9
        stale: list[str] = []
        for side in self.config.factr.servers:
            fresh = False
            entry = self._registry.get(factr_stream_name(side))
            if entry is not None:
                try:
                    reader = StreamReader.attach(entry)
                    try:
                        s = reader.latest()
                        fresh = s.n > 0 and (
                            time.monotonic_ns() - s.newest_t_ns
                        ) < max_age_ns
                    finally:
                        reader.close()
                except Exception:  # noqa: BLE001 - racing the producer's teardown
                    fresh = False
            if not fresh:
                stale.append(side)
        return stale


def main() -> None:
    """Console entry point: ``dfc-session [hydra overrides…]``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )
    overrides = sys.argv[1:]
    config = compose_config(overrides)

    # One session per runtime dir: a live daemon already owning it is an error
    # (two sessions would fight over the arms); a stale file from a dead one is
    # overwritten.
    existing = read_state(config.runtime.runtime_dir)
    if existing and _pid_alive(existing.get("pid")):
        raise SystemExit(
            f"another session (pid {existing['pid']}) already owns "
            f"{config.runtime.runtime_dir!r}"
        )

    # Make this daemon's future children provably ours, then reclaim any dead
    # supervisor's debris (orphaned nodes holding ZED/RDK handles, leaked shm,
    # stale run dirs) — boot is exactly when the devices must be reclaimable,
    # so the legacy (pre-tag) heuristic runs here too.
    tag_supervisor()
    orphan_sweep(config.runtime.runtime_dir, include_legacy=True)

    daemon = SessionDaemon(config, overrides)
    daemon.install_signal_handlers()
    raise SystemExit(daemon.serve())


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
