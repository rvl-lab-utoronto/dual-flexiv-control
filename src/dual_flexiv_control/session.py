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
  task), ``{"cmd": "stop"}``, ``{"cmd": "shutdown"}``. stdin EOF == shutdown,
  so a dead dashboard can never leave an orphaned daemon holding robots.
* state — ``<runtime_dir>/session.json``, atomically replaced (same pattern as the
  stream-manifest registry): current mode, active run, a heartbeat timestamp, and
  the last run's outcome. The dashboard polls it.

**Supervision invariants**: a consumer exit returns the session to VIEWING (its
crashed control channels are swept so the next run can't attach a dead segment).
An **arm** node exit is session-fatal — everything stops (the in-progress episode
gets its save grace) and the daemon exits non-zero for the dashboard to surface.
A **camera** node exit or stall is NOT fatal: the camera is reported down
(``cameras_down`` in the state file), runs are refused while any rig camera is
down (recording needs every view), and the node is respawned periodically — a
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


class SessionDaemon:
    """Owns the hardware processes and the mode state machine; see the module doc."""

    def __init__(self, config: Config, overrides: list[str]) -> None:
        self.config = config
        self.overrides = overrides
        self.run_id = make_run_id()
        self.ctx = mp.get_context("spawn")  # never fork: flexivrdk has live threads
        self.hw_stop = self.ctx.Event()
        #: side -> session queue, for every control-enabled arm.
        self.session_qs = {
            side: self.ctx.Queue()
            for side, arm in config.arms.items()
            if arm.control_enabled
        }
        self.arm_procs: list = []
        self.cameras: list[_CameraUnit] = []
        self.run: _ActiveRun | None = None
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
        # Arms are the session's reason to exist — an arm node dying is fatal.
        # Cameras are supervised individually (down + respawn), never fatal.
        arm_nodes = [n for n in nodes if not n.name.startswith("zed:")]
        cam_nodes = [n for n in nodes if n.name.startswith("zed:")]
        self.arm_procs = [
            self.ctx.Process(target=run_node, args=(node, self.hw_stop), name=node.name)
            for node in arm_nodes
        ]
        for proc in self.arm_procs:
            proc.start()
        self.cameras = [self._make_camera_unit(node) for node in cam_nodes]
        log.info(
            "session %s: %d arm node(s) + %d camera node(s) up (rig=%s sim=%s); VIEWING",
            self.run_id, len(self.arm_procs), len(self.cameras),
            self.state.rig, self.state.sim,
        )

    def _make_camera_unit(self, node) -> _CameraUnit:
        stop_event = self.ctx.Event()  # per-camera: its crash must not stop the rest
        proc = self.ctx.Process(target=run_node, args=(node, stop_event), name=node.name)
        proc.start()
        return _CameraUnit(
            node=node, proc=proc, stop_event=stop_event, spawned_at=time.monotonic()
        )

    def _sweep_camera_streams(self, node) -> None:
        """Reclaim a killed camera's leaked segments so its respawn can re-create them.

        A camera that died cleanly unlinked its own streams (no-op here); one that
        was SIGKILLed mid-life leaks them, and ``StreamWriter.create`` on the same
        names would then fail every respawn.
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
        last_camera_tend = 0.0
        try:
            while True:
                changed = self._drain_commands()
                if self._fatal == "shutdown":
                    break
                changed |= self._reap_consumer()
                now = time.monotonic()
                if self.run is not None and now - last_status_poll >= _STATUS_POLL_S:
                    last_status_poll = now
                    self._watch_arm_dropout()
                if now - last_camera_tend >= 1.0:
                    last_camera_tend = now
                    changed |= self._tend_cameras()
                if not self._arms_alive():
                    self._fatal = "arm node died"
                    break
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
        for unit in self.cameras:  # cameras stop via their own (isolated) events
            unit.stop_event.set()
        hw_procs = [*self.arm_procs, *(unit.proc for unit in self.cameras)]
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
            elif kind == "stop":
                self._handle_stop()
            else:
                self.state.message = f"unknown command {kind!r}"
                log.warning("unknown command: %r", cmd)

    # -- run lifecycle -----------------------------------------------------------

    def _handle_start(self, cmd: dict) -> None:
        phase, task = cmd.get("phase"), cmd.get("task")
        if self.run is not None:
            self.state.message = "a run is already active — stop it first"
            return
        # A skill replay is task-independent; any composable task satisfies the
        # schema, so a missing task just takes the default.
        if phase == SKILL and not task:
            task = "default"
        if phase not in PHASES or not task:
            self.state.message = f"bad start command (phase={phase!r}, task={task!r})"
            return
        skill_name = cmd.get("skill")
        if skill_name is not None and not (
            isinstance(skill_name, str) and skill_name
            and all(c.isalnum() or c in "._-" for c in skill_name)
        ):
            self.state.message = f"bad start command (skill={skill_name!r})"
            return
        if phase == SKILL and not skill_name:
            self.state.message = "bad start command (a skill run needs a skill name)"
            return
        port = cmd.get("port")
        if port is not None and not (isinstance(port, int) and 0 < port < 65536):
            self.state.message = f"bad start command (port={port!r})"
            return
        host = cmd.get("host")
        if host is not None and not (
            isinstance(host, str) and host
            and all(c.isalnum() or c in "._-" for c in host)
        ):
            self.state.message = f"bad start command (host={host!r})"
            return
        policy = cmd.get("policy")
        if policy is not None and not (
            isinstance(policy, str) and policy
            and all(c.isalnum() or c in "._-" for c in policy)
        ):
            self.state.message = f"bad start command (policy={policy!r})"
            return
        busy = self._arms_still_in_control()
        if busy:
            self.state.message = (
                f"arm(s) still winding down a control session: {sorted(busy)} — retry shortly"
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
        log.info("run %d: %s task=%s%s (consumer pid %s; commanding %s)",
                 self.state.run_seq, phase, task,
                 f" skill={skill_name}" if skill_name else "",
                 proc.pid, command_sides)

    def _handle_stop(self) -> None:
        if self.run is None:
            self.state.message = "no active run to stop"
            return
        if not self.run.stopping:
            self.run.stopping = True
            self.run.stop_event.set()
            self.state.state = SAVING
            log.info("stop requested: %s run winding down (episode saving)…", self.run.phase)

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
        for proc in self.arm_procs:
            if not proc.is_alive():
                log.error("arm node %s exited (code %s) — session-fatal",
                          proc.name, proc.exitcode)
                return False
        return True

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
            self._sweep_camera_streams(unit.node)
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
