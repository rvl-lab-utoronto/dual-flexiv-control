"""Drive the session daemon from the dashboard and feed the embedded Rerun viewer.

The dashboard owns one long-lived **session daemon** (see
:mod:`dual_flexiv_control.session`) via :class:`~.session.SessionManager`: the
daemon holds the arms + cameras for the dashboard's whole lifespan and runs a
VIEWING ↔ COLLECTION ↔ EVAL state machine. This module is the glue:

* :class:`RunRegistry` — the UI-facing facade: launch/stop are JSON commands to
  the daemon; run state, outcomes, and alerts come from its ``session.json``.
* :class:`SessionMirror` — ONE persistent background thread that mirrors the live
  shared-memory streams (arm proprio, FACTR leaders, the 3D robot scene, eval
  horizon ghosts) into the metrics Rerun recording, in every mode. In VIEWING the
  streams are published by the idle (read-only) arms, so the viewer is live
  *before* any run starts. It never opens a robot connection — shared memory and
  the FACTR HTTP endpoint only.

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

PHASES = ("collection", "eval")

#: Mirror loop rate (matches the recording rate; FACTR is polled per tick).
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

    def launch(self, task: TaskInfo, phase: str, rig: str = "bimanual") -> None:
        """Ask the daemon to start ``phase`` for ``task``; raises on a refusal.

        The daemon composes ``task=<name>`` itself (validating it), hands the
        phase's coefficients to the arms, and spawns the real consumer — the same
        node the CLI runs. ``rig`` is the *session's* rig; a mismatch means the
        session needs a restart (surfaced, not silently absorbed).
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (expected one of {PHASES})")
        view = self.manager.view()
        if view.run_active:
            raise RuntimeError(
                "a run is still active (recording or saving its episode) — stop it "
                "and wait for it to finish first"
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
        if not self.manager.start_run(phase, task.name):
            raise RuntimeError("could not reach the session daemon — see the Logs tab")
        _log_event(f"launch {phase} · task={task.name} · rig={view.rig or rig}")

    def stop_active(self) -> None:
        """Ask the daemon to stop the active run (non-blocking; episode saves)."""
        if self.manager.stop_run():
            _log_event("stop requested — saving episode…")

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


def _open_factr_client():
    """Build the FACTR leader client from config, or ``None`` if unavailable.

    Honours ``runtime.sim`` (the client fabricates leader positions offline). Any
    failure here (missing config, import error, no servers) disables leader
    logging without disturbing the follower proprio stream.
    """
    from ..interfaces.factr.client import FactrClient
    from .arms import discover_factr
    from .arms import runtime_is_sim

    try:
        client = FactrClient.from_config(discover_factr(), sim=runtime_is_sim())
    except Exception as exc:  # noqa: BLE001 - no/invalid FACTR config -> skip leaders
        _log_event(f"FACTR leaders unavailable: {exc}")
        return None
    return client if client.sides else None


def _log_factr_leaders(client, errored: set[str]) -> dict:
    """Query each configured leader, log arm joints (``q``) + gripper (``grip``), return raw samples.

    Per-side error tolerant: an unreachable leader (e.g. the right arm before it
    is plugged in / served) is logged once and skipped, leaving its row empty; it
    is picked back up automatically once it starts responding. The server returns
    a flat ``DoF+1`` ``joint_pos`` — arm joints then a trailing gripper scalar.

    Returns ``{side: joint_pos}`` for every leader that responded this tick, so the
    caller can reuse the samples (e.g. to render the commanded-teleop ghost) without
    a second HTTP round-trip.
    """
    from ..interfaces.factr.client import FactrError

    fetched: dict = {}
    for side in client.sides:
        try:
            jp = client.get_joint_positions_for(side)
        except FactrError as exc:
            if side not in errored:
                errored.add(side)
                _log_event(f"FACTR {side} leader not reachable: {exc}")
            continue
        if side in errored:
            errored.discard(side)
            _log_event(f"FACTR {side} leader reconnected")
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

    ``saving`` keeps its run's layout (``view.phase`` stays set); ``down`` and
    ``starting`` share the welcome screen.
    """
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
    _log_mode_readme(view.state, task)


class SessionMirror:
    """One background thread mirroring the live system into the metrics viewer.

    All real data, read-only: arm proprio + status from shared memory (published
    by the session's idle or controlling arms), FACTR leaders over HTTP, the 3D
    robot scene (solid = measured ``<side>/q``, translucent = commanded-teleop
    ghost, purple = eval horizon target from ``eval/<side>/q_horizon``). A stream
    nobody publishes leaves its row empty and its arm still — the viewer never
    shows motion the system isn't making. It also switches the viewer layout +
    README whenever the session's mode changes, so VIEWING/COLLECTION/EVAL each
    get their blueprint without any per-run emitter threads.
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
        from .arms import read_live_horizon_q
        from .arms import read_live_stream

        dt = 1.0 / _MIRROR_HZ
        _style_eef_pos_series(blueprints.SIDES)
        factr_client = _open_factr_client()
        factr_errored: set[str] = set()
        conventions = discover_conventions()
        robot_rec = robot_view.robot_recording()
        live_q: dict = {}      # last-known real measured q per side
        horizon_q: dict = {}   # eval horizon-end q target per side (eval runs only)
        plot_every = max(1, round(_MIRROR_HZ / 5.0))   # mirror plots ~5 Hz (each read attaches shm)
        state_every = max(1, round(_MIRROR_HZ))        # poll session.json ~1 Hz
        layout_key: tuple | None = None
        step = 0
        while not stop.is_set():
            t = step * dt
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
                            continue
                        vec = np.asarray(v, dtype=float)
                        if take is not None:
                            vec = vec[:take]
                        if suffix == "q" and len(vec) >= 7:
                            live_q[side] = vec[:7]
                        rr.log(blueprints.proprio_path(row, side), rr.Scalars(vec.tolist()))
                    h = read_live_horizon_q(side)
                    if h is not None and len(h) >= 7:
                        horizon_q[side] = np.asarray(h, dtype=float)[:7]
                    else:
                        horizon_q.pop(side, None)

            leader_samples = (
                _log_factr_leaders(factr_client, factr_errored) if factr_client is not None else {}
            )
            if robot_rec is not None:
                real_q = dict(live_q)
                ghost_q = _ghost_configs(leader_samples, conventions)
                robot_view.update_poses(robot_rec, real_q, ghost_q, t)
                if horizon_q:
                    robot_view.update_horizon_targets(robot_rec, horizon_q, real_q, t)
                else:
                    robot_view.clear_horizon_targets(robot_rec)
            step += 1
            stop.wait(dt)
        if factr_client is not None:
            factr_client.close()
