"""Own the session daemon (``dfc-session``) for the dashboard's lifespan.

The dashboard never touches robots or spawns per-run systems anymore: one
long-lived daemon holds the rig (see :mod:`dual_flexiv_control.session`) and this
manager is its client — spawn it with the selected rig/sim, restart it when they
change, send JSON-line commands down its stdin, and read its atomically-written
``session.json`` for state. The daemon treats stdin EOF as shutdown, so a dead
dashboard can never orphan a process holding robot connections.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from dual_flexiv_control.session import read_state

log = logging.getLogger(__name__)

#: A heartbeat older than this means the daemon is wedged (or the file is stale).
_HEARTBEAT_STALE_S = 6.0
#: How long a graceful shutdown may take with no run active (hardware teardown).
_SHUTDOWN_GRACE_S = 15.0
#: Respawn pacing after the daemon dies on its own: never sooner than this…
_RESPAWN_BACKOFF_S = 10.0
#: …and a death this soon after spawn counts as an "early death" (a boot crash,
#: e.g. a camera that cannot open) rather than a mid-life failure.
_EARLY_DEATH_S = 30.0
#: Consecutive early deaths before ensure() STOPS auto-respawning (each attempt
#: briefly seizes cameras/arms, so a crash loop is actively harmful). A rig/sim
#: change or *Reset services* re-arms it.
_MAX_EARLY_DEATHS = 3

#: Session states a run occupies (launches are refused while in one).
RUN_STATES = ("collection", "eval", "skill", "saving")


@dataclass(frozen=True)
class SessionView:
    """One snapshot of the daemon, as the UI consumes it.

    ``state`` extends the daemon's own states with two client-side ones:
    ``down`` (no live daemon) and ``starting`` (spawned, hardware still coming up
    — no fresh ``session.json`` yet).
    """

    state: str                    # down | starting | viewing | collection | eval | saving
    rig: str | None = None
    sim: bool | None = None
    run_id: str | None = None
    task: str | None = None
    phase: str | None = None
    run_seq: int = 0
    run_started_ts: float | None = None
    message: str | None = None
    last_outcome: dict | None = None
    #: Camera nodes the daemon reports down/booting (runs are gated on empty).
    cameras_down: tuple = ()
    #: Arm nodes the daemon reports down (auto-respawned to IDLE telemetry;
    #: runs needing them are refused until they are back).
    arms_down: tuple = ()
    #: A queued switch/start waiting for the session to go idle
    #: (``{phase, task, skill}``), or None.
    pending: dict | None = None
    #: Daemon-managed FACTR-Server processes: the supervisor's status dict
    #: (state / countdown_ends_ts / units / down), or None when the daemon does
    #: not manage them (factr.launch disabled, sim, or no daemon state yet).
    factr_servers: dict | None = None
    log_path: str | None = None

    @property
    def run_active(self) -> bool:
        return self.state in RUN_STATES


def _runtime_dir() -> str:
    """The runtime dir the daemon writes ``session.json`` under (shared cwd)."""
    rd = os.environ.get("DFC_RUNTIME_DIR", "runtime")
    return rd if os.path.isabs(rd) else os.path.join(os.getcwd(), rd)


def _new_session_log_path() -> str:
    """A persistent, dashboard-discoverable log file for one daemon lifetime.

    Lives under the outputs root (where CLI runs land) instead of a ``/tmp``
    tempfile, so :func:`~.logs.discover_logs` lists it in the Logs tab even after
    the daemon is gone — a ``/tmp`` file is only reachable while it is the
    *current* daemon (via :func:`~.logs.live_daemon_log`), so every respawn used
    to strand the previous log. One dir per spawn; ``mkdtemp`` keeps the name
    unique when two daemons spawn in the same second.
    """
    from dual_flexiv_control.dashboard import logs as _logs

    root = _logs.outputs_root()
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = tempfile.mkdtemp(prefix=f"session_{ts}_", dir=str(root))
    return os.path.join(run_dir, "session.log")


def _daemon_cmd(rig: str | None, sim: bool) -> list[str]:
    exe = shutil.which("dfc-session")
    if exe:
        cmd = [exe]
    else:  # not on PATH (e.g. odd launch) -> run the module with the same interpreter
        import sys

        cmd = [sys.executable, "-m", "dual_flexiv_control.session"]
    if rig:
        cmd.append(f"rig={rig}")
    cmd.append(f"runtime.sim={'true' if sim else 'false'}")
    return cmd


class SessionManager:
    """Spawn/restart/talk to the one session daemon. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._log_path: str | None = None
        self._rig: str | None = None
        self._sim: bool | None = None
        self._spawned_at: float = 0.0
        #: Crash-loop bookkeeping (a dead daemon must not be respawned hot: every
        #: boot attempt briefly seizes the cameras/arms).
        self._early_deaths = 0
        self._last_exit_code: int | None = None
        self._gave_up = False
        atexit.register(self._atexit)

    # -- lifecycle -------------------------------------------------------------

    def ensure(self, rig: str | None, sim: bool) -> bool:
        """Make the daemon run with (rig, sim); restart it if they changed.

        Returns True when a (re)spawn happened — the caller then refreshes anything
        that caches rig-derived config (e.g. the metrics mirror's FACTR client).
        A change is REFUSED (False, with a log) while a run is active: restarting
        would kill an in-flight episode; the UI disables rig switching during runs,
        so this is a backstop.

        A daemon that died on its own is respawned with **backoff**, and after
        ``_MAX_EARLY_DEATHS`` consecutive boot crashes (e.g. a camera the rig
        expects that cannot open) ensure() stops retrying — the failure is
        surfaced via :meth:`view` and a rig/sim change or *Reset services*
        (:meth:`shutdown`) re-arms it. Without this, every Streamlit rerun would
        hot-loop spawn→crash→spawn against the hardware.
        """
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
            if alive and (rig, sim) == (self._rig, self._sim):
                return False
            if alive:
                if self.view().run_active:
                    log.warning(
                        "refusing session restart to rig=%s sim=%s: a run is active", rig, sim
                    )
                    return False
                self._shutdown_locked()
                self._rearm_locked()  # deliberate change: retry freely on the new target
            else:
                self._harvest_exit_locked()
                if (rig, sim) != (self._rig, self._sim):
                    self._rearm_locked()  # new target: previous crashes don't count
                elif self._gave_up:
                    return False
                elif time.monotonic() - self._spawned_at < _RESPAWN_BACKOFF_S:
                    return False  # too soon after the last attempt; try again later
            self._spawn_locked(rig, sim)
            return True

    def _harvest_exit_locked(self) -> None:
        """Record how a self-died daemon ended; count boot crashes toward give-up."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._last_exit_code = proc.poll()
        uptime = time.monotonic() - self._spawned_at
        if uptime < _EARLY_DEATH_S:
            self._early_deaths += 1
        else:
            self._early_deaths = 1  # a fresh failure streak, not a boot loop yet
        if self._early_deaths >= _MAX_EARLY_DEATHS:
            self._gave_up = True
            log.error(
                "session daemon died %d times within %.0fs of spawning (last exit "
                "code %s); giving up on auto-respawn — see %s",
                self._early_deaths, _EARLY_DEATH_S, self._last_exit_code, self._log_path,
            )

    def _rearm_locked(self) -> None:
        self._early_deaths = 0
        self._last_exit_code = None
        self._gave_up = False

    def _spawn_locked(self, rig: str | None, sim: bool) -> None:
        cmd = _daemon_cmd(rig, sim)
        self._log_path = _new_session_log_path()
        logf = open(self._log_path, "w")
        log.info("spawning session daemon: %s (log %s)", " ".join(cmd), self._log_path)
        try:
            # start_new_session: the daemon must survive Streamlit's own signal
            # handling; its lifetime is governed by our stdin pipe (EOF = shutdown).
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        finally:
            logf.close()  # Popen inherited its own fd
        self._rig, self._sim = rig, sim
        self._spawned_at = time.monotonic()

    def _shutdown_locked(self, grace_s: float = _SHUTDOWN_GRACE_S) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is not None:
            return
        try:  # closing stdin == EOF == shutdown; belt-and-braces with the command
            proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            log.warning("session daemon did not exit in %.0fs; terminating", grace_s)
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            log.error("session daemon unresponsive; killing")
            proc.kill()

    def shutdown(self) -> None:
        """Stop the daemon (graceful; used by Reset services and atexit).

        Also re-arms auto-respawn: an operator reset is the explicit "try again"
        after a crash-loop give-up.
        """
        with self._lock:
            self._shutdown_locked()
            self._rearm_locked()

    def _atexit(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass

    # -- commands ----------------------------------------------------------------

    def send(self, cmd: dict) -> bool:
        """Write one JSON command line to the daemon; False if it is not reachable."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return False
            try:
                proc.stdin.write(json.dumps(cmd) + "\n")
                proc.stdin.flush()
                return True
            except (OSError, ValueError):
                log.exception("session daemon stdin write failed")
                return False

    def start_run(
        self, phase: str, task: str, policy: str | None = None,
        host: str | None = None, port: int | None = None,
        skill: str | None = None,
    ) -> bool:
        return self._send_run_cmd("start", phase, task, policy, host, port, skill)

    def switch_run(
        self, phase: str, task: str, policy: str | None = None,
        host: str | None = None, port: int | None = None,
        skill: str | None = None,
    ) -> bool:
        """Atomic mode switch: the daemon stops any active run (episode saves),
        then starts this one the moment the session is idle — no polling."""
        return self._send_run_cmd("switch", phase, task, policy, host, port, skill)

    def _send_run_cmd(
        self, kind: str, phase: str, task: str, policy: str | None,
        host: str | None, port: int | None, skill: str | None,
    ) -> bool:
        cmd = {"cmd": kind, "phase": phase, "task": task}
        if policy is not None:
            cmd["policy"] = policy
        if host is not None:
            cmd["host"] = host
        if port is not None:
            cmd["port"] = port
        if skill is not None:
            cmd["skill"] = skill
        return self.send(cmd)

    def stop_run(self) -> bool:
        return self.send({"cmd": "stop"})

    def reconnect_arm(self, side: str) -> bool:
        """Replace one arm node now (fresh RDK connection, back to IDLE)."""
        return self.send({"cmd": "reconnect_arm", "side": side})

    def respawn_camera(self, name: str) -> bool:
        """Replace one camera node now (skips the watchdog's respawn pacing)."""
        return self.send({"cmd": "respawn_camera", "name": name})

    def start_factr(self) -> bool:
        """Launch the FACTR-Server processes (pose-then-calibrate countdown)."""
        return self.send({"cmd": "start_factr"})

    def stop_factr(self) -> bool:
        """Stop the FACTR-Server processes (SIGINT — the leaders de-energize)."""
        return self.send({"cmd": "stop_factr"})

    # -- state -------------------------------------------------------------------

    def view(self) -> SessionView:
        """Parse ``session.json`` into a :class:`SessionView` (never raises)."""
        proc = self._proc
        alive = proc is not None and proc.poll() is None
        raw = read_state(_runtime_dir())
        if raw is not None and alive and raw.get("pid") == proc.pid:
            fresh = time.time() - float(raw.get("heartbeat_ts") or 0) < _HEARTBEAT_STALE_S
            return SessionView(
                state=str(raw.get("state")) if fresh else "starting",
                rig=raw.get("rig"),
                sim=raw.get("sim"),
                run_id=raw.get("run_id"),
                task=raw.get("task"),
                phase=raw.get("phase"),
                run_seq=int(raw.get("run_seq") or 0),
                run_started_ts=raw.get("run_started_ts"),
                message=raw.get("message"),
                last_outcome=raw.get("last_outcome"),
                cameras_down=tuple(raw.get("cameras_down") or ()),
                arms_down=tuple(raw.get("arms_down") or ()),
                pending=raw.get("pending"),
                factr_servers=raw.get("factr_servers"),
                log_path=self._log_path,
            )
        if alive:
            # Spawned but no (matching, fresh) state file yet: hardware coming up.
            return SessionView(state="starting", rig=self._rig, sim=self._sim,
                               log_path=self._log_path)
        # Down: say why (and whether we are still retrying) so the UI can show it.
        message = None
        if self._gave_up:
            message = (
                f"session daemon crashed {self._early_deaths}× in a row on startup "
                f"(last exit code {self._last_exit_code}) — auto-restart paused. "
                "Likely a hardware node the rig expects (e.g. a camera) failing to "
                "open; see the log below, then fix the cause or pick a rig matching "
                "the connected hardware, and press Reset services."
            )
        elif self._last_exit_code is not None:
            message = (
                f"session daemon exited (code {self._last_exit_code}); "
                "retrying shortly…"
            )
        return SessionView(state="down", message=message, log_path=self._log_path)

    def log_tail(self, n: int = 25) -> str | None:
        """Last ``n`` daemon log lines (hardware + consumer output), or None."""
        path = self._log_path
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return None
        return "".join(lines[-n:]).strip() or None
