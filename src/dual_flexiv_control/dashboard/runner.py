"""Drive the session daemon from the dashboard and feed the embedded Rerun viewer.

The dashboard owns one long-lived **session daemon** (see
:mod:`dual_flexiv_control.session`) via :class:`~.session.SessionManager`: the
daemon holds the arms + cameras for the dashboard's whole lifespan and runs a
VIEWING ↔ COLLECTION ↔ EVAL state machine. This module is the glue:

* :class:`RunRegistry` — the UI-facing facade: launch/stop are JSON commands to
  the daemon; run state, outcomes, and alerts come from its ``session.json``.
* :class:`SessionMirror` — ONE persistent background thread that mirrors the live
  shared-memory streams (arm proprio, ``factr/<side>`` leaders, the 3D robot
  scene, eval horizon ghosts) into the metrics Rerun recording, in every mode. In
  VIEWING the streams are published by the idle (read-only) arms and the FACTR
  producer, so the viewer is live *before* any run starts. It never opens a robot
  or HTTP connection — shared memory only.

A run is **one episode** (collection = one teleop demo, eval = one policy
rollout); launch one at a time, matching the single bimanual rig. Stopping is
asynchronous: the daemon gives the recording consumer a generous save window
(``RuntimeCfg.save_grace_s``) while the UI shows *saving*.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import rerun as rr

from . import blueprints
from . import robot_view
from .session import RUN_STATES
from .session import SessionManager
from .session import SessionView
from .tasks import TaskInfo

log = logging.getLogger(__name__)

PHASES = ("collection", "eval", "skill")

#: Mirror loop rate (matches the recording rate; leader streams read per tick).
_MIRROR_HZ = 15.0
#: Trailing daemon-log lines surfaced in the UI when a run errors out.
_ERROR_TAIL_LINES = 25

#: Outcomes that read as success (toast) vs failure (modal popup with a log tail).
_OK_OUTCOMES = frozenset({"saved", "finished", "stopped"})


@dataclass
class CollectionStatus:
    """Snapshot of the active run for the status panel."""

    state: str      # "running" | "saving"
    detail: str     # human-readable one-liner
    error_tail: str | None = None


@dataclass
class RunRecord:
    """A run, as surfaced in the dashboard's status panel + history."""

    run_id: str
    task: str
    phase: str
    rig: str
    started_wall: str
    status: str = "running"  # running | stopping | <outcome> (history)


class RunRegistry:
    """UI facade over the session daemon: launches, run state, one-shot alerts."""

    def __init__(self, manager: SessionManager | None = None) -> None:
        self._lock = threading.Lock()
        self.manager = manager or SessionManager()
        self._mirror = SessionMirror(self.manager)
        self._history: list[RunRecord] = []
        #: run_seq of the newest outcome already surfaced as an alert.
        self._seen_seq = 0

    # -- session lifecycle -------------------------------------------------------

    def ensure_session(self, rig: str | None, sim: bool) -> None:
        """Make the daemon match (rig, sim); (re)start the mirror after a respawn.

        Called every app rerun — a no-op when nothing changed. A rig/sim change
        respawns the daemon (refused while a run is active) and restarts the mirror
        so its FACTR client + conventions follow the new rig.
        """
        if self.manager.ensure(rig, sim):
            self._mirror.restart()
        else:
            self._mirror.start()  # idempotent: first call spins the thread up

    def session_view(self) -> SessionView:
        return self.manager.view()

    def restart_mirror(self) -> None:
        """Restart the mirror thread (after *Reset services* rebinds the viewers)."""
        self._mirror.restart()

    # -- run state -----------------------------------------------------------------

    def active(self) -> RunRecord | None:
        view = self.manager.view()
        if not view.run_active:
            return None
        started = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(view.run_started_ts))
            if view.run_started_ts
            else "—"
        )
        return RunRecord(
            run_id=f"{view.run_id or '?'}#{view.run_seq}",
            task=view.task or "?",
            phase=view.phase or "?",
            rig=view.rig or "?",
            started_wall=started,
            status="stopping" if view.state == "saving" else "running",
        )

    def history(self) -> list[RunRecord]:
        with self._lock:
            return list(self._history)

    def collection_status(self) -> CollectionStatus | None:
        """Live state of the active run, or ``None`` when idle."""
        view = self.manager.view()
        if view.state == "saving":
            return CollectionStatus(
                "saving", "saving episode — finalizing the video encode (can take a while)…"
            )
        if view.state == "collection":
            beat = _last_heartbeat(view.log_path)
            return CollectionStatus(
                "running", beat or "recording — the episode saves when you press Stop."
            )
        if view.state == "eval":
            return CollectionStatus("running", "policy rollout in progress.")
        if view.state == "skill":
            return CollectionStatus("running", "skill replay in progress.")
        return None

    def take_alert(self) -> dict | None:
        """Return the newest not-yet-surfaced run outcome as a one-shot alert.

        Outcomes are sequenced by the daemon (``last_outcome.run_seq``); each is
        surfaced exactly once and appended to the history.
        """
        view = self.manager.view()
        outcome = view.last_outcome
        if not outcome:
            return None
        seq = int(outcome.get("run_seq") or 0)
        with self._lock:
            if seq <= self._seen_seq:
                return None
            self._seen_seq = seq
            ok = outcome.get("outcome") in _OK_OUTCOMES
            detail = str(outcome.get("detail") or outcome.get("outcome") or "run ended")
            self._history.append(
                RunRecord(
                    run_id=f"{view.run_id or '?'}#{seq}",
                    task=str(outcome.get("task") or "?"),
                    phase=str(outcome.get("phase") or "?"),
                    rig=view.rig or "?",
                    started_wall=time.strftime("%Y-%m-%d %H:%M:%S"),
                    status=str(outcome.get("outcome") or "?"),
                )
            )
            _log_event(f"run ended: {outcome.get('outcome')} · {detail}")
            return {
                "kind": "info" if ok else "error",
                "run_id": f"{seq}",
                "detail": detail,
                "tail": None if ok else self.manager.log_tail(_ERROR_TAIL_LINES),
            }

    # -- launch / stop ---------------------------------------------------------------

    def _check_launchable(self, rig: str, switching: bool = False) -> None:
        """Raise a ``RuntimeError`` describing why nothing can launch right now.

        ``switching`` skips only the run-active refusal: an atomic switch stops
        the current run itself (daemon-side), so an active run is expected.
        """
        view = self.manager.view()
        if view.run_active and not switching:
            raise RuntimeError(
                "a run is still active (recording or saving its episode) — stop it "
                "and wait for it to finish first (or switch, which does both)"
            )
        if view.state == "down":
            raise RuntimeError(
                "the session daemon is not running — check the Logs tab (or Reset "
                "services) and try again"
            )
        if view.state == "starting":
            raise RuntimeError("the session is still starting (arms coming up) — try again shortly")
        if view.rig and rig and view.rig != rig:
            raise RuntimeError(
                f"the session holds rig {view.rig!r} but {rig!r} is selected — wait "
                "for the session to restart onto the new rig, then launch"
            )

    def launch(
        self, task: TaskInfo, phase: str, rig: str = "bimanual",
        policy: str | None = None, host: str | None = None, port: int | None = None,
        switch: bool = False,
    ) -> None:
        """Ask the daemon to start ``phase`` for ``task``; raises on a refusal.

        The daemon composes ``task=<name>`` itself (validating it), hands the
        phase's coefficients to the arms, and spawns the real consumer — the same
        node the CLI runs. ``rig`` is the *session's* rig; a mismatch means the
        session needs a restart (surfaced, not silently absorbed). ``policy`` /
        ``host`` / ``port`` (eval only) override the ``policy`` group selection
        and ``policy.host`` / ``policy.port`` for this run; None keeps the
        task's policy config. ``switch`` sends an atomic switch instead: the
        daemon stops any active run (its episode saves) and starts this one the
        moment the session is idle.
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (expected one of {PHASES})")
        self._check_launchable(rig, switching=switch)
        send = self.manager.switch_run if switch else self.manager.start_run
        if not send(phase, task.name, policy=policy, host=host, port=port):
            raise RuntimeError("could not reach the session daemon — see the Logs tab")
        detail = ""
        if policy is not None or host is not None or port is not None:
            detail = (
                f" · policy {policy or 'config-type'}"
                f" @ {host or 'config-host'}:{port or 'config-port'}"
            )
        verb = "switch to" if switch else "launch"
        _log_event(f"{verb} {phase} · task={task.name} · rig={rig}{detail}")

    def launch_skill(self, skill: str, rig: str) -> None:
        """Ask the daemon to repeat a taught skill; raises on a refusal.

        A skill run is task-independent (the daemon composes the default task to
        satisfy the schema) and camera-independent (only arm proprio is observed),
        so the only gates are the shared launchability ones.
        """
        self._check_launchable(rig)
        if not self.manager.start_run("skill", "default", skill=skill):
            raise RuntimeError("could not reach the session daemon — see the Logs tab")
        _log_event(f"launch skill · {skill} · rig={rig}")

    def stop_active(self) -> None:
        """Ask the daemon to stop the active run (non-blocking; episode saves)."""
        if self.manager.stop_run():
            _log_event("stop requested — saving episode…")

    def reconnect_arm(self, side: str) -> bool:
        """Ask the daemon to replace one arm node now (fresh RDK connection)."""
        ok = self.manager.reconnect_arm(side)
        if ok:
            _log_event(f"reconnect requested for arm {side}")
        return ok

    def respawn_camera(self, name: str) -> bool:
        """Ask the daemon to replace one camera node now (skip the retry pacing)."""
        ok = self.manager.respawn_camera(name)
        if ok:
            _log_event(f"respawn requested for camera {name}")
        return ok

    def reset(self) -> None:
        """Stop any active run and clear the history — a clean slate."""
        view = self.manager.view()
        if view.run_active:
            self.manager.stop_run()
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------


def log_welcome() -> None:
    """Idle README shown before the session comes up (paired with the welcome blueprint)."""
    md = (
        "# dual-flexiv experiments\n\n"
        "The session daemon is starting: arms connect read-only and stream live "
        "telemetry here (**viewing**). Pick a **rig** and a **task** on the left, "
        "then launch **Collection** (one teleop demo) or **Eval** (one policy "
        "rollout). Each launch runs a **single episode**."
    )
    rr.log(blueprints.README, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)


def reset_viewer() -> None:
    """Return the metrics viewer to its idle welcome state (blueprint + README).

    The Rerun servers themselves keep their bound ports (they are process
    singletons); only the recording's blueprint + README are reset. The session
    mirror re-sends the mode's layout on its next state tick.
    """
    rr.send_blueprint(blueprints.welcome_blueprint())
    log_welcome()
    _log_event("services reset — viewers returned to idle")


_MODE_READMES = {
    "viewing": (
        "_**Viewing** — arms connected read-only; live proprio, FACTR leaders, and "
        "the 3D scene stream with no control of any kind._"
    ),
    "collection": "_**Collection** — teleop demo recording; live mirror of the run._",
    "eval": "_**Eval** — policy rollout; purple ghost = the policy's horizon target._",
    "skill": "_**Skill** — teach-and-repeat replay; purple ghost = the skill's end pose._",
    "saving": "_**Saving** — finalizing the episode's video encode…_",
}


def _log_mode_readme(state: str, task: str | None) -> None:
    title = f"# {task}" if task else "# dual-flexiv experiments"
    body = _MODE_READMES.get(state, "")
    rr.log(
        blueprints.README,
        rr.TextDocument(f"{title}\n\n{body}", media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def _log_event(message: str) -> None:
    rr.log(blueprints.EVENTS, rr.TextLog(message, level="INFO"))


def _style_eef_pos_series(sides) -> None:
    """Name + colour the TCP-position x/y/z series once (static styling).

    ``eef_pos`` logs three scalars per arm; logging a static ``SeriesLines`` with
    ``names`` makes the plot legend read x/y/z (in the classic axis colours) instead
    of anonymous indices. Static so it applies for the whole recording, set once.
    """
    for side in sides:
        rr.log(
            blueprints.proprio_path("eef_pos", side),
            rr.SeriesLines(
                names=list(blueprints.EEF_POS_COMPONENTS), colors=blueprints.EEF_POS_COLORS
            ),
            static=True,
        )


def _style_policy_comm_series() -> None:
    """Name + colour the policy-server comm series once (static styling)."""
    for series, color in zip(blueprints.POLICY_COMM_SERIES, blueprints.POLICY_COMM_COLORS):
        rr.log(
            blueprints.policy_comm_path(series),
            rr.SeriesLines(names=[series.replace("_", "-")], colors=[color]),
            static=True,
        )
    rr.log(
        blueprints.POLICY_LATENCY_PATH,
        rr.SeriesPoints(names=["round trip (ms)"], colors=[[180, 120, 240]]),
        static=True,
    )


def _log_policy_comm(state: dict, t0_ns: int, now_t: float) -> None:
    """Mirror new policy-server comm events into the metrics recording.

    Reads the eval run's ``eval/policy_comm`` stream (None outside eval runs) and
    logs each not-yet-surfaced event at its TRUE position on the viewer timeline
    (the ring's per-sample monotonic timestamps share the mirror's clock):
    cumulative sent/received/error counters, a 0/1 in-flight square wave between
    a send and its response, and a round-trip latency point per completed
    request. ``state`` persists across ticks (next ring seq to surface +
    counters); the stream disappearing (run over) resets it so the next eval
    starts its counters at zero. Restores the tick's ambient time before
    returning.
    """
    from .arms import read_live_policy_comm
    from ..policy.client import COMM_RECV
    from ..policy.client import COMM_SENT

    comm = read_live_policy_comm()
    if comm is None or comm.n == 0:
        if comm is None and state["next_seq"] > 0:
            state.update(next_seq=0, sent=0, received=0, errors=0)
        return
    # Ring restarted under us (a new run's stream): the newest seq fell BELOW the
    # last surfaced one (next_seq - 1; equal just means nothing new yet).
    if int(comm.seq[-1]) + 1 < state["next_seq"]:
        state.update(next_seq=0, sent=0, received=0, errors=0)
    fresh = comm.seq >= state["next_seq"]
    if not fresh.any():
        return
    for row, ev_t_ns in zip(comm.data[fresh], comm.t_ns[fresh]):
        kind, elapsed_s = float(row[0]), float(row[2])
        rr.set_time("elapsed", duration=max(0.0, (int(ev_t_ns) - t0_ns) / 1e9))
        if kind == COMM_SENT:
            state["sent"] += 1
            rr.log(blueprints.policy_comm_path("sent"), rr.Scalars([state["sent"]]))
            rr.log(blueprints.policy_comm_path("in_flight"), rr.Scalars([1.0]))
            continue
        ok = kind == COMM_RECV
        counter = "received" if ok else "errors"
        state[counter] += 1
        rr.log(blueprints.policy_comm_path(counter), rr.Scalars([state[counter]]))
        rr.log(blueprints.policy_comm_path("in_flight"), rr.Scalars([0.0]))
        if ok:
            rr.log(blueprints.POLICY_LATENCY_PATH, rr.Scalars([elapsed_s * 1000.0]))
    state["next_seq"] = int(comm.seq[-1]) + 1
    rr.set_time("elapsed", duration=now_t)  # back to the tick's ambient time


def _log_tail(path: str | None, n: int = _ERROR_TAIL_LINES) -> str | None:
    """Last ``n`` lines of a log file (for surfacing an error), or ``None``."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    return "".join(lines[-n:]).strip() or None


def _last_heartbeat(path: str | None) -> str | None:
    """The collection loop's most recent status line (for the live status panel).

    The loop logs a beat every ~2s — "recording (N frames …)" or a loud
    "NO frames recorded …" warning naming the stuck stream. The consumer's output
    lands in the session daemon's log, so surfacing it verbatim tells the operator
    whether frames are actually flowing.
    """
    tail = _log_tail(path, n=12)
    if not tail:
        return None
    for line in reversed(tail.splitlines()):
        marker = "collection: "
        pos = line.find(marker)
        if pos != -1:
            return line[pos + len(marker):].strip() or None
    return None


# ---------------------------------------------------------------------------
# FACTR leaders (REAL, read-only -> Rerun): live leader joint positions
# ---------------------------------------------------------------------------


def _factr_leader_sides() -> list[str]:
    """The configured leader sides, or empty if no FACTR config composes."""
    from .arms import configured_leader_sides

    try:
        return configured_leader_sides()
    except Exception as exc:  # noqa: BLE001 - no/invalid FACTR config -> skip leaders
        _log_event(f"FACTR leaders unavailable: {exc}")
        return []


def _log_factr_leaders(sides: list[str], errored: set[str]) -> dict:
    """Read each leader's ``factr/<side>`` stream; log joints (``q``) + gripper (``grip``).

    The SAME shared-memory samples the control loop converts into setpoints (one
    source of truth — the viewer can never show leader motion control isn't
    seeing). Per-side tolerant: a stale/absent stream (leader server down, or no
    running producer) is logged once and skipped, leaving its row empty; it is
    picked back up automatically once fresh samples flow again. Each sample is a
    flat ``DoF+1`` vector — arm joints then a trailing gripper scalar.

    Returns ``{side: joint_pos}`` for every fresh leader this tick, so the caller
    can reuse the samples (e.g. to render the commanded-teleop ghost).
    """
    from .arms import read_live_leader

    fetched: dict = {}
    for side in sides:
        jp = read_live_leader(side)
        if jp is None or jp.size == 0:
            if side not in errored:
                errored.add(side)
                _log_event(f"FACTR {side} leader not streaming (server down or stale)")
            continue
        if side in errored:
            errored.discard(side)
            _log_event(f"FACTR {side} leader streaming again")
        fetched[side] = jp
        arm = jp[:-1] if jp.shape[0] > 1 else jp
        rr.log(blueprints.factr_path("q", side), rr.Scalars(arm.tolist()))
        rr.log(blueprints.factr_path("grip", side), rr.Scalars([float(jp[-1])]))
    return fetched


def _ghost_configs(leader_samples: dict, conventions: dict) -> dict:
    """Map raw leader samples to commanded Rizon joint configs — the teleop ghost.

    ``{side: joint_pos(DoF+1 rad)}`` → ``{side: q_cmd(DoF rad)}`` via
    :func:`convert_factr_to_rizon` (the exact brain-side leader→follower mapping), so
    the ghost stands where teleop is *commanding* the follower to go. Sides without a
    known convention are skipped.
    """
    from ..control.convention import convert_factr_to_rizon

    ghost: dict = {}
    for side, jp in leader_samples.items():
        conv = conventions.get(side)
        if conv is not None:
            ghost[side] = convert_factr_to_rizon(jp, conv)
    return ghost


# ---------------------------------------------------------------------------
# The persistent session mirror (shared memory + FACTR -> Rerun; never touches robots)
# ---------------------------------------------------------------------------


#: Proprio streams the mirror follows into the metrics plots. Maps the published
#: stream suffix (``<side>/<suffix>``) to the blueprint row; ``eef`` is a 7-vector
#: pose whose position part feeds the ``eef_pos`` row.
_VIEW_STREAMS = (
    ("q", "q", None),
    ("dq", "dq", None),
    ("tau", "tau", None),
    ("wrench", "wrench", None),
    ("eef_vel", "eef_vel", None),
    ("eef", "eef_pos", 3),  # pose [x y z qw qx qy qz] -> position row
)


def _layout_key(view: SessionView) -> tuple:
    """What determines the viewer layout: the mode family + the active task.

    A queued switch (``view.pending``) jumps straight to the TARGET phase's
    layout, so the whole transition (stopping → saving → idle gap → new run)
    flips the viewer exactly once — the session passes through ``viewing`` for
    a second or two between the runs, and keying on the raw state would flash
    the viewing layout in the middle. ``saving`` keeps its run's layout
    (``view.phase`` stays set); ``down`` and ``starting`` share the welcome
    screen.
    """
    if view.pending:
        phase = view.pending.get("phase") or "collection"
        label = (
            view.pending.get("skill") if phase == "skill" else view.pending.get("task")
        )
        return (phase, label)
    if view.state in RUN_STATES:
        return (view.phase or "collection", view.task)
    if view.state == "viewing":
        return ("viewing", None)
    return ("welcome", None)


def _send_layout(view: SessionView) -> None:
    kind, task = _layout_key(view)
    if kind == "welcome":
        rr.send_blueprint(blueprints.welcome_blueprint())
        log_welcome()
        return
    rr.send_blueprint(blueprints.for_phase(kind, task))
    # While a switch is pending the README follows the target layout too — the
    # transient saving/viewing states mid-switch would otherwise caption the
    # new mode's layout with the old mode's text.
    _log_mode_readme(kind if view.pending else view.state, task)


class SessionMirror:
    """One background thread mirroring the live system into the metrics viewer.

    All real data, read-only, all from shared memory: arm proprio + status
    (published by the session's idle or controlling arms), FACTR leaders from
    their ``factr/<side>`` streams (the SAME samples the control loop converts
    into setpoints — the ghost always shows exactly what teleop would command),
    the 3D robot scene (solid = measured ``<side>/q``, translucent =
    commanded-teleop ghost, purple = eval horizon prediction from
    ``eval/<side>/q_horizon`` — a posed ghost — or ``eval/<side>/eef_horizon`` —
    a predicted-EEF trace). A stream nobody publishes leaves its row empty and
    its arm still — the viewer never shows motion the system isn't making. It
    also switches the viewer layout + README whenever the session's mode
    changes, so VIEWING/COLLECTION/EVAL each get their blueprint without any
    per-run emitter threads.
    """

    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(self._stop,), name="dfc-session-mirror", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stop is not None:
                self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._stop, self._thread = None, None

    def restart(self) -> None:
        """Fresh thread: re-reads the composed config (rig change) + re-caches the
        Rerun recording handles (Reset services)."""
        self.stop()
        self.start()

    def _run(self, stop: threading.Event) -> None:
        try:
            self._run_inner(stop)
        except Exception:  # noqa: BLE001 - a broken mirror must not look like a broken run
            log.exception("session mirror died (the session itself is unaffected)")
            _log_event("live view stopped unexpectedly — the session continues; see dashboard logs")

    def _run_inner(self, stop: threading.Event) -> None:
        from .arms import discover_conventions
        from .arms import read_live_horizon_eef
        from .arms import read_live_horizon_q
        from .arms import read_live_stream

        dt = 1.0 / _MIRROR_HZ
        _style_eef_pos_series(blueprints.SIDES)
        _style_policy_comm_series()
        factr_sides = _factr_leader_sides()
        factr_errored: set[str] = set()
        conventions = discover_conventions()
        robot_rec = robot_view.robot_recording()
        live_q: dict = {}      # last-known real measured q per side
        live_eef: dict = {}    # last-known measured TCP position [x y z] per side
        horizon_q: dict = {}   # eval horizon-end q target per side (eval runs only)
        horizon_eef: dict = {} # eval horizon-end TCP position per side (cartesian kinds)
        # Policy-server comm mirroring (eval runs only): next ring seq to surface
        # + the cumulative packet counters plotted below the robot metrics.
        comm_state = {"next_seq": 0, "sent": 0, "received": 0, "errors": 0}
        plot_every = max(1, round(_MIRROR_HZ / 5.0))   # mirror plots ~5 Hz (each read attaches shm)
        state_every = max(1, round(_MIRROR_HZ))        # poll session.json ~1 Hz
        layout_key: tuple | None = None
        step = 0
        # Anchor the viewer timeline to the wall clock (monotonic), NOT step*dt: an
        # iteration that overruns dt must show up as sparser samples at their true
        # times, never as the timeline lagging further and further behind reality.
        t0_ns = time.monotonic_ns()
        slow_warned = 0.0  # last time (s since t0) a slow-tick warning was logged
        while not stop.is_set():
            t = (time.monotonic_ns() - t0_ns) / 1e9
            rr.set_time("elapsed", duration=t)

            if step % state_every == 0:  # follow the session's mode (layout + README)
                view = self._manager.view()
                key = _layout_key(view)
                if key != layout_key:
                    layout_key = key
                    _send_layout(view)

            if step % plot_every == 0:
                for side in blueprints.SIDES:
                    for suffix, row, take in _VIEW_STREAMS:
                        v = read_live_stream(f"{side}/{suffix}")
                        if v is None:
                            if suffix == "q":
                                live_q.pop(side, None)  # producer gone → freeze this arm
                            elif suffix == "eef":
                                live_eef.pop(side, None)
                            continue
                        vec = np.asarray(v, dtype=float)
                        if take is not None:
                            vec = vec[:take]
                        if suffix == "q" and len(vec) >= 7:
                            live_q[side] = vec[:7]
                        elif suffix == "eef" and len(vec) >= 3:
                            live_eef[side] = vec[:3]  # base-frame TCP position
                        rr.log(blueprints.proprio_path(row, side), rr.Scalars(vec.tolist()))
                    h = read_live_horizon_q(side)
                    if h is not None and len(h) >= 7:
                        horizon_q[side] = np.asarray(h, dtype=float)[:7]
                    else:
                        horizon_q.pop(side, None)
                    he = read_live_horizon_eef(side)
                    if he is not None and len(he) >= 3:
                        horizon_eef[side] = np.asarray(he, dtype=float)[:3]
                    else:
                        horizon_eef.pop(side, None)
                _log_policy_comm(comm_state, t0_ns, t)

            leader_samples = (
                _log_factr_leaders(factr_sides, factr_errored) if factr_sides else {}
            )
            if robot_rec is not None:
                real_q = dict(live_q)
                ghost_q = _ghost_configs(leader_samples, conventions)
                robot_view.update_poses(robot_rec, real_q, ghost_q, t)
                if horizon_q or horizon_eef:
                    robot_view.update_horizon_targets(
                        robot_rec, horizon_q, real_q, t,
                        target_eef=horizon_eef, real_eef=live_eef,
                    )
                else:
                    robot_view.clear_horizon_targets(robot_rec)
            step += 1
            # Pace to the tick boundary: sleep only the remainder of dt. When an
            # iteration overruns, surface it (throttled) — a slow mirror otherwise
            # just looks like "the viewer is slow" with nothing in the logs.
            tick_s = (time.monotonic_ns() - t0_ns) / 1e9 - t
            if tick_s > 4 * dt and t - slow_warned > 30.0:
                slow_warned = t
                log.warning(
                    "session mirror tick took %.2fs (target %.3fs) — live view is "
                    "updating slower than %.0f Hz",
                    tick_s, dt, _MIRROR_HZ,
                )
            stop.wait(max(0.0, dt - tick_s))
