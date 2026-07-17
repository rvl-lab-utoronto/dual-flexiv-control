"""Launch + supervise the external FACTR-Server processes (leader grav-comp + API).

This replaces the FACTR-Server repo's VS Code "Launch EVERYTHING" task: the
session daemon owns the three processes — one grav-comp teleop per leader arm
(``factr_rizon_teleop`` / ``factr_rizon_dual_board``, torque ON) and the FastAPI
relay serving both leaders (:5000 left, :5001 right) — exactly like it owns the
arm/camera nodes, so the dashboard is the one place services are launched,
watched, and stopped.

These are **subprocesses, not nodes**: they need ROS 2 sourced and the system
``/usr/bin/python3`` (conda's interpreter cannot load rclpy), so they cannot be
``multiprocessing`` children of the daemon. Each is spawned via
``bash -c 'source …; exec <python> -m <module>'`` — the ``exec`` makes the
signal target the python process itself, with no shell wrapper in between.

Lifecycle, mirroring the task's semantics:

* **Start is user-initiated, never automatic.** Energizing the leader servos
  runs a calibration read at boot, so the operator must first pose the arms at
  the calibration pose. ``request_start()`` arms a countdown
  (``cfg.calib_delay_s``, surfaced live in the dashboard) and the processes
  spawn when it expires.
* **Stop means SIGINT.** The teleop nodes de-energize (zero + disable torque,
  close the serial port) only from ``KeyboardInterrupt``; SIGTERM kills them
  with the servos still energized. Escalation past ``cfg.stop_grace_s`` is
  therefore a last resort and logged as such.
* **Supervision matches the hardware units' spirit:** a dead API relay is
  respawned (paced) — it never touches the servos. A dead teleop is only
  *reported*: respawning it would re-energize and re-calibrate an unposed arm,
  so relaunching stays the operator's call.

stdout of the teleops is discarded (they clear the screen at 500 Hz); their
stderr — the per-second ``[health]`` telemetry — goes to the same
``logs/factr_health_<side>.log`` files the FACTR-Server tail tasks follow.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field

from ...configs import FactrLaunchCfg

log = logging.getLogger(__name__)

#: Supervisor states, surfaced verbatim in ``session.json`` (``factr_servers``).
OFF = "off"
COUNTDOWN = "countdown"
RUNNING = "running"
STOPPING = "stopping"

#: How often a dead API relay is respawned. Safe to retry hot-ish: it holds no
#: device, just re-binds its ports and re-subscribes to the ROS topics.
_API_RESPAWN_S = 10.0
#: Cooperative window after SIGTERM before SIGKILL, once SIGINT already failed.
_TERM_GRACE_S = 2.0
#: Hard cap on the blocking :meth:`FactrServerSupervisor.shutdown` beyond the
#: cooperative + TERM graces (teardown must never hang the daemon).
_SHUTDOWN_SLACK_S = 5.0


@dataclass
class _ServerUnit:
    """One launchable FACTR-Server process and its live handle."""

    name: str                               # "teleop:<side>" | "api"
    kind: str                               # "teleop" | "api"
    module: str                             # run as `<python> -m <module>`
    log_path: str
    #: Teleop stdout is a 500 Hz screen-clear — discarded; the API's is kept.
    keep_stdout: bool
    proc: subprocess.Popen | None = None
    spawned_at: float = 0.0
    announced_down: bool = False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class FactrServerSupervisor:
    """Own the FACTR-Server process group: countdown → spawn → watch → SIGINT.

    Driven by the session daemon exactly like the hardware units: commands call
    :meth:`request_start` / :meth:`request_stop`, the supervise loop calls
    :meth:`tend` about once a second, and :meth:`status` is what lands in
    ``session.json`` for the dashboard. All methods are cheap and non-blocking
    except :meth:`shutdown` (daemon teardown), which waits out the stop grace.
    """

    def __init__(self, cfg: FactrLaunchCfg, sides: list[str]) -> None:
        self.cfg = cfg
        self.workdir = os.path.expanduser(cfg.workdir)
        log_dir = os.path.join(self.workdir, "logs")
        self.units: list[_ServerUnit] = [
            _ServerUnit(
                name=f"teleop:{side}",
                kind="teleop",
                module=cfg.teleop_modules[side],
                log_path=os.path.join(log_dir, f"factr_health_{side}.log"),
                keep_stdout=False,
            )
            for side in sides
        ]
        self.units.append(
            _ServerUnit(
                name="api",
                kind="api",
                module=cfg.api_module,
                log_path=os.path.join(log_dir, "factr_api.log"),
                keep_stdout=True,
            )
        )
        self.state = OFF
        #: Monotonic deadline the countdown/stop waits against…
        self._deadline: float | None = None
        #: …and its wall-clock twin for the dashboard's live countdown.
        self._countdown_ends_ts: float | None = None
        self._terminated = False  # SIGTERM already sent while stopping

    # -- commands ---------------------------------------------------------------

    def request_start(self) -> tuple[bool, str]:
        """Arm the pose-then-calibrate countdown; processes spawn when it ends."""
        if self.state == COUNTDOWN:
            return False, "FACTR launch already counting down"
        if self.state == STOPPING:
            return False, "FACTR servers are still stopping — retry when they are down"
        if self.state == RUNNING:
            return False, (
                "FACTR servers already launched — stop them first to relaunch "
                "(a relaunch re-runs leader calibration)"
            )
        delay = max(0.0, float(self.cfg.calib_delay_s))
        self.state = COUNTDOWN
        self._deadline = time.monotonic() + delay
        self._countdown_ends_ts = time.time() + delay
        log.info(
            "FACTR launch armed: pose the leader arm(s) at %s — spawning in %.0fs",
            self.cfg.calib_pose, delay,
        )
        return True, (
            f"FACTR launch armed — pose the leader arm(s) at {self.cfg.calib_pose} "
            f"NOW; calibration reads them in {delay:.0f}s"
        )

    def request_stop(self) -> tuple[bool, str]:
        """SIGINT every live process (the only de-energizing stop); cancel a countdown."""
        if self.state == COUNTDOWN:
            self._to_off()
            log.info("FACTR launch countdown cancelled")
            return True, "FACTR launch cancelled (nothing was spawned)"
        if self.state not in (RUNNING, STOPPING):
            return False, "FACTR servers are not running"
        if self.state == STOPPING:
            return False, "FACTR servers already stopping (de-energizing)…"
        for unit in self.units:
            if unit.alive():
                unit.proc.send_signal(signal.SIGINT)
        self.state = STOPPING
        self._terminated = False
        self._deadline = time.monotonic() + float(self.cfg.stop_grace_s)
        log.info("FACTR servers stopping (SIGINT sent; leaders de-energizing)…")
        return True, "FACTR servers stopping — the leader arms are de-energizing"

    # -- supervision ------------------------------------------------------------

    def tend(self) -> None:
        """One supervision beat: fire the countdown, watch/respawn, drive the stop."""
        now = time.monotonic()
        if self.state == COUNTDOWN:
            if now >= self._deadline:
                self._spawn_all()
            return
        if self.state == RUNNING:
            self._tend_running(now)
            return
        if self.state == STOPPING:
            self._tend_stopping(now)

    def _tend_running(self, now: float) -> None:
        for unit in self.units:
            if unit.alive():
                if unit.announced_down:
                    unit.announced_down = False
                    log.info("FACTR %s recovered", unit.name)
                continue
            code = None if unit.proc is None else unit.proc.poll()
            if not unit.announced_down:
                unit.announced_down = True
                if unit.kind == "teleop":
                    log.error(
                        "FACTR %s exited (code %s) — its leader stream goes stale; "
                        "NOT auto-respawned (that would re-energize and re-calibrate "
                        "an unposed arm). Stop + relaunch from the dashboard; see %s",
                        unit.name, code, unit.log_path,
                    )
                else:
                    log.error(
                        "FACTR %s exited (code %s) — every leader stream goes stale; "
                        "respawning it every %.0fs; see %s",
                        unit.name, code, _API_RESPAWN_S, unit.log_path,
                    )
            if unit.kind == "api" and now - unit.spawned_at >= _API_RESPAWN_S:
                self._spawn_unit(unit)
                log.info("respawned FACTR api relay")

    def _tend_stopping(self, now: float) -> None:
        alive = [u for u in self.units if u.alive()]
        if not alive:
            self._to_off()
            log.info("FACTR servers stopped")
            return
        if now < self._deadline:
            return
        if not self._terminated:
            self._terminated = True
            self._deadline = now + _TERM_GRACE_S
            for unit in alive:
                log.error(
                    "FACTR %s ignored SIGINT for %.0fs; sending SIGTERM — it does "
                    "NOT de-energize the servos: if a leader arm stays stiff, cut "
                    "its power", unit.name, float(self.cfg.stop_grace_s),
                )
                unit.proc.terminate()
            return
        for unit in alive:
            log.error("FACTR %s unresponsive; sending SIGKILL", unit.name)
            unit.proc.kill()
        self._to_off()

    def shutdown(self) -> None:
        """Blocking stop for daemon teardown: SIGINT, wait out the grace, escalate."""
        if self.state == COUNTDOWN:
            self._to_off()
            return
        if self.state == OFF:
            return
        if self.state == RUNNING:
            self.request_stop()
        cap = time.monotonic() + float(self.cfg.stop_grace_s) + _TERM_GRACE_S + _SHUTDOWN_SLACK_S
        while self.state == STOPPING and time.monotonic() < cap:
            self.tend()
            time.sleep(0.1)
        if self.state == STOPPING:  # belt-and-braces; tend's escalation should end it
            for unit in self.units:
                if unit.alive():
                    unit.proc.kill()
            self._to_off()

    # -- state ------------------------------------------------------------------

    def status(self) -> dict:
        """The ``session.json`` payload: plain JSON the dashboard renders from."""
        return {
            "state": self.state,
            "countdown_ends_ts": (
                self._countdown_ends_ts if self.state == COUNTDOWN else None
            ),
            "countdown_total_s": float(max(0.0, self.cfg.calib_delay_s)),
            "units": {unit.name: self._unit_state(unit) for unit in self.units},
            "down": sorted(
                unit.name for unit in self.units
                if self.state == RUNNING and not unit.alive()
            ),
            "calib_pose": self.cfg.calib_pose,
            # The daemon and dashboard share the filesystem: the panel tails
            # these for the teleops' boot phase + servo health readouts.
            "logs": {unit.name: unit.log_path for unit in self.units},
        }

    def _unit_state(self, unit: _ServerUnit) -> str:
        if self.state in (OFF, COUNTDOWN):
            return self.state
        if unit.alive():
            return "stopping" if self.state == STOPPING else "running"
        return "down"

    def _to_off(self) -> None:
        for unit in self.units:
            if unit.proc is not None:
                unit.proc = None
            unit.announced_down = False
        self.state = OFF
        self._deadline = None
        self._countdown_ends_ts = None
        self._terminated = False

    # -- spawning ---------------------------------------------------------------

    def _spawn_all(self) -> None:
        self._free_api_ports()
        for unit in self.units:
            self._spawn_unit(unit)
        self.state = RUNNING
        self._deadline = None
        self._countdown_ends_ts = None
        log.info(
            "FACTR servers spawned: %s (workdir %s)",
            ", ".join(u.name for u in self.units), self.workdir,
        )

    def _spawn_unit(self, unit: _ServerUnit) -> None:
        os.makedirs(os.path.dirname(unit.log_path), exist_ok=True)
        script = " && ".join(
            [f"source {shlex.quote(s)}" for s in self.cfg.setup_scripts]
            + [f"exec {shlex.quote(self.cfg.python_exe)} -m {shlex.quote(unit.module)}"]
        )
        # One log per launch (truncate), like the task; Popen inherits its own fd.
        with open(unit.log_path, "w") as logf:
            unit.proc = subprocess.Popen(
                ["bash", "-c", script],
                cwd=self.workdir,
                stdout=logf if unit.keep_stdout else subprocess.DEVNULL,
                stderr=logf,
                # Its own session: the daemon's terminal signals must never reach
                # the servos' processes except through our explicit SIGINT stop.
                start_new_session=True,
            )
        unit.spawned_at = time.monotonic()
        unit.announced_down = False

    def _free_api_ports(self) -> None:
        """Kill whatever still holds the API ports (a stale/orphaned relay).

        Same belt-and-braces as the launch task's ``fuser -k``: the ports must be
        free or the fresh relay dies at bind. Anything legitimately holding them
        is a previous FACTR API this supervisor is about to replace.
        """
        if shutil.which("fuser") is None:
            return
        for port in self.cfg.api_ports:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
