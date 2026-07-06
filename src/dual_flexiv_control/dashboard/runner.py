"""Launch experiments from the dashboard and feed the embedded Rerun viewer.

Pressing *Collection* spawns the **real** recording system
(``dual-flexiv-control runtime.phase=collection``) as a subprocess, mirrors its
live shared-memory proprio + FACTR leaders into the metrics viewer
(:func:`_emit_collection_view`), and monitors its exit so the outcome (episode
saved / crashed, with a log tail) surfaces in the UI as a popup. Stopping is
**asynchronous**: SIGINT tells the system to save the in-progress episode (a long
video finalize — see ``RuntimeCfg.save_grace_s``); a background thread waits it
out and only force-kills on overrun, so the page never blocks.

Pressing *Eval* runs a small **real** check: it opens a read-only ``flexivrdk``
connection per arm and streams live joint velocity (``dq``) and end-effector
position (``eef_pos``) into the viewer **with no motion** — a connectivity probe
for the read → Rerun path.

A run is **one episode** (collection = one teleop demo, eval = one policy
rollout); launch one at a time, matching the single bimanual rig.

Everything logs to the process-global Rerun recording created by
:func:`~.viewer.start_servers`, so the emitter threads show up in the same
embedded viewer.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from uuid import uuid4

import numpy as np
import rerun as rr

from . import blueprints
from . import robot_view
from .tasks import TaskInfo

log = logging.getLogger(__name__)

PHASES = ("collection", "eval")

#: Live-view emitter loop rate.
_EMIT_HZ = 30.0

#: Dashboard stop budget: how long to let the recording subprocess finalize its
#: episode video (after SIGINT) before force-killing it. Must exceed the
#: orchestrator's own save window (``RuntimeCfg.save_grace_s``, default 300s) so the
#: dashboard never SIGKILLs a subprocess that is mid-save. The wait runs in a
#: background thread, so this never blocks the Streamlit UI.
_STOP_GRACE_S = 330.0
#: Trailing subprocess-log lines surfaced in the UI when a collection run errors out.
_ERROR_TAIL_LINES = 25


@dataclass
class CollectionStatus:
    """Snapshot of the active collection subprocess for the status panel."""

    state: str      # "running" | "saving" | "exited-ok" | "exited-error"
    detail: str     # human-readable one-liner
    error_tail: str | None = None  # trailing subprocess log when state == "exited-error"


@dataclass
class RunRecord:
    """A launched run, as surfaced in the dashboard's status panel."""

    run_id: str
    task: str
    phase: str
    instruction: str
    started_wall: str
    started_monotonic: float
    status: str = "running"  # running | stopping | stopped
    _stop: threading.Event | None = field(default=None, repr=False, compare=False)
    _thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    #: For collection: the real recording system (``dual-flexiv-control``) subprocess.
    _proc: subprocess.Popen | None = field(default=None, repr=False, compare=False)
    #: For collection: combined stdout/stderr log (tailed to surface errors).
    _log_path: str | None = field(default=None, repr=False, compare=False)
    #: For collection: the live-view emitter (real robot scene + FACTR) + its stop.
    _view_stop: threading.Event | None = field(default=None, repr=False, compare=False)
    _view_thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    #: For collection: the thread that waits on the subprocess + harvests its outcome.
    _monitor: threading.Thread | None = field(default=None, repr=False, compare=False)
    #: Set by the monitor when the subprocess exits: saved | finished | crashed | stopped-error.
    outcome: str | None = field(default=None, compare=False)


class RunRegistry:
    """Owns the single active run and the run history; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: RunRecord | None = None
        self._history: list[RunRecord] = []
        #: One-shot alert (a run finished/crashed) for the UI to surface once.
        self._alert: dict | None = None

    def active(self) -> RunRecord | None:
        with self._lock:
            return self._active

    def history(self) -> list[RunRecord]:
        with self._lock:
            return list(self._history)

    def collection_status(self) -> CollectionStatus | None:
        """Live state of the active collection subprocess, or ``None`` if not applicable.

        ``running`` while it records; ``saving`` after Stop while it finalizes the
        episode video; ``exited-ok`` if it finished cleanly (hit ``num_episodes``);
        ``exited-error`` (with a log tail) if it died on its own. Terminal states are
        transient here — the monitor thread moves the run to history and raises a
        one-shot alert (see :meth:`take_alert`); this is the inline status panel view.
        """
        with self._lock:
            run = self._active
            if run is None or run._proc is None:
                return None
            if run.status == "stopping":
                return CollectionStatus(
                    "saving", "saving episode — finalizing the video encode (can take a while)…"
                )
            code = run._proc.poll()
            if code is None:
                beat = _last_heartbeat(run._log_path)
                return CollectionStatus(
                    "running",
                    beat or "recording — episodes save when you press Stop.",
                )
            if code == 0:
                return CollectionStatus(
                    "exited-ok", "recording process finished (target episodes reached)."
                )
            return CollectionStatus(
                "exited-error",
                f"recording process exited with code {code} — it stopped on its own "
                "(a source likely failed to start, e.g. a camera).",
                error_tail=_log_tail(run._log_path),
            )

    def take_alert(self) -> dict | None:
        """Return and clear the pending one-shot alert (a run finished/crashed), or ``None``.

        The dashboard polls this each rerun and surfaces it as a popup. Keyed by
        ``run_id`` so the UI shows it exactly once.
        """
        with self._lock:
            alert, self._alert = self._alert, None
            return alert

    def launch(self, task: TaskInfo, phase: str) -> RunRecord:
        """Start a run for ``task`` in ``phase`` ("collection" | "eval").

        Single-rig invariant: an active **eval** probe is stopped inline first; an
        active **collection** run refuses the launch (its subprocess may still be
        recording or saving — overlapping it would fight over the cameras and the
        dataset). The UI disables launch while a run is active, so the raise is a
        defensive backstop.
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (expected one of {PHASES})")
        with self._lock:
            if self._active is not None:
                if self._active._proc is not None:
                    raise RuntimeError(
                        "a collection run is still active (recording or saving its "
                        "episode) — stop it and wait for it to finish first"
                    )
                self._stop_eval_locked(self._active)

            run = RunRecord(
                run_id=uuid4().hex[:8],
                task=task.name,
                phase=phase,
                instruction=task.language_instruction,
                started_wall=time.strftime("%Y-%m-%d %H:%M:%S"),
                started_monotonic=time.monotonic(),
                _stop=threading.Event(),
            )

            # --- visualization: switch the viewer to this phase's layout -------
            #     Eval gets a focused dq-only view (the read-only probe logs just
            #     dq); collection keeps the full proprio grid.
            if phase == "eval":
                rr.send_blueprint(blueprints.eval_probe_blueprint(task.name))
            else:
                rr.send_blueprint(blueprints.for_phase(phase, task.name))
            _log_readme(task, phase)
            _log_event(f"launch {phase} · task={task.name} · run={run.run_id}")

            # --- LAUNCH ---------------------------------------------------------
            # Collection spawns the REAL recording system (`dual-flexiv-control
            # runtime.phase=collection`): it teleoperates, commands the arms, and
            # records LeRobot episodes to `datasets/` — which then appear in the
            # Storage tab. Stopping the run sends SIGINT, so the collection loop
            # saves the in-progress episode on its way down (single-episode-per-run;
            # no tty in the subprocess). Eval still runs the read-only dq probe.
            if phase == "collection":
                try:
                    run._proc, run._log_path = _launch_collection(task, run.run_id)
                    _log_event(
                        f"recording started (pid {run._proc.pid}) — episodes will "
                        "appear in the Storage tab after you stop the run"
                    )
                    # A live view of the running system (real robot scene + FACTR
                    # leaders read from shared memory) so the Metrics tab isn't blank
                    # while the subprocess records; and a monitor that harvests the
                    # subprocess's exit outcome (saved / crashed) for the UI.
                    run._view_stop = threading.Event()
                    run._view_thread = threading.Thread(
                        target=_emit_collection_view,
                        args=(run._view_stop, task),
                        name=f"dfc-view-{run.run_id}",
                        daemon=True,
                    )
                    run._view_thread.start()
                    run._monitor = threading.Thread(
                        target=self._monitor_collection,
                        args=(run,),
                        name=f"dfc-mon-{run.run_id}",
                        daemon=True,
                    )
                    run._monitor.start()
                except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
                    log.exception("failed to launch collection system")
                    _log_event(f"collection launch FAILED: {exc}")
                    self._alert = {
                        "kind": "error",
                        "run_id": run.run_id,
                        "detail": f"Collection failed to launch: {exc}",
                        "tail": None,
                    }
                    run.status = "stopped"
                    self._history.append(run)
                    return run
            else:
                run._thread = threading.Thread(
                    target=_emit_eval_qvel,
                    args=(run._stop, task),
                    name=f"dfc-emit-{run.run_id}",
                    daemon=True,
                )
                run._thread.start()

            self._active = run
            return run

    def stop_active(self) -> None:
        """Begin stopping the active run (if any). Non-blocking for collection.

        Collection: SIGINT the subprocess so its loop saves the in-progress episode,
        then a background thread waits out the (generous) save window and force-kills
        only if it overruns. The monitor thread moves the run to history once it
        exits. Eval: cooperative stop inline (its emitter unwinds in <2s).
        """
        with self._lock:
            self._begin_stop_locked(self._active)

    def reset(self) -> None:
        """Stop the active run and clear history — a clean slate.

        For eval this joins the emitter thread (closing its live ``FlexivSource`` /
        FACTR-leader connections) — the "restart connections" half of *Reset
        services*. For collection it begins the async save-and-stop; the run finalizes
        in the background.
        """
        with self._lock:
            self._begin_stop_locked(self._active)
            self._history.clear()

    # -- internal stop paths (call under self._lock) --------------------------

    def _begin_stop_locked(self, run: RunRecord | None) -> None:
        if run is None or run.status != "running":
            return
        if run._proc is not None:
            self._begin_stop_collection_locked(run)
        else:
            self._stop_eval_locked(run)

    def _begin_stop_collection_locked(self, run: RunRecord) -> None:
        run.status = "stopping"
        if run._view_stop is not None:
            run._view_stop.set()  # recording is over — stop feeding the live view
        _log_event(f"stopping {run.phase} · run={run.run_id} — saving episode…")
        _interrupt(run._proc)  # SIGINT -> orchestrator saves the episode on the way down
        # Wait out the save in the background so the UI never blocks; force-kill only
        # if the save overruns the grace (which exceeds the orchestrator's own).
        threading.Thread(
            target=_escalate_after_grace,
            args=(run._proc,),
            name=f"dfc-stop-{run.run_id}",
            daemon=True,
        ).start()

    def _stop_eval_locked(self, run: RunRecord) -> None:
        if run._stop is not None:
            run._stop.set()
        if run._thread is not None:
            run._thread.join(timeout=2.0)
        run.status = "stopped"
        _log_event(f"stop {run.phase} · run={run.run_id}")
        self._history.append(run)
        if self._active is run:
            self._active = None

    def _monitor_collection(self, run: RunRecord) -> None:
        """Wait on the collection subprocess, then harvest its outcome + alert the UI.

        Runs in a daemon thread for the run's whole life. When the subprocess exits
        — whether from a user Stop (SIGINT -> save -> exit), a clean self-finish
        (``num_episodes`` reached), or a crash — this records the outcome, raises a
        one-shot alert, and moves the run to history.
        """
        proc = run._proc
        try:
            code = proc.wait()
        except Exception:  # noqa: BLE001 - never let the monitor die silently
            log.exception("collection monitor wait failed")
            code = proc.poll()
        if run._view_stop is not None:
            run._view_stop.set()  # the system is gone — stop the live view
        with self._lock:
            if run.status == "stopping":  # user pressed Stop
                if code == 0:
                    run.outcome = "saved"
                    alert = {"kind": "info", "run_id": run.run_id,
                             "detail": "Collection stopped — episode saved to the dataset."}
                else:
                    run.outcome = "stopped-error"
                    alert = {"kind": "error", "run_id": run.run_id,
                             "detail": f"Collection stopped but exited abnormally (code {code}); "
                                       "the episode may not have saved.",
                             "tail": _log_tail(run._log_path)}
            elif code == 0:  # finished on its own (target episodes reached)
                run.outcome = "finished"
                alert = {"kind": "info", "run_id": run.run_id,
                         "detail": "Collection finished (target episodes reached)."}
            else:  # crashed on its own
                run.outcome = "crashed"
                alert = {"kind": "error", "run_id": run.run_id,
                         "detail": f"Collection stopped on its own (exit code {code}) — a source "
                                   "likely failed to start (e.g. a camera or the FACTR server).",
                         "tail": _log_tail(run._log_path)}
            run.status = "stopped"
            self._alert = alert
            if self._active is run:
                self._history.append(run)
                self._active = None
        _log_event(f"collection run {run.run_id} ended: {run.outcome}")


# ---------------------------------------------------------------------------
# Real collection system: spawn + graceful stop
# ---------------------------------------------------------------------------


def _launch_collection(task: TaskInfo, run_id: str) -> tuple[subprocess.Popen, str]:
    """Spawn ``dual-flexiv-control runtime.phase=collection`` for ``task``.

    Inherits this process's cwd, so the ``datasets/`` root it records into is the
    same one the Storage tab scans. Honours ``runtime.sim`` (from ``conf``), so the
    dashboard's sim/real toggle drives the recording too. stdout+stderr are captured
    to a per-run log file (returned) so an error can be surfaced as a UI popup rather
    than lost to a detached terminal. Returns ``(proc, log_path)``.
    """
    from .arms import runtime_is_sim

    if task.config_name is not None:
        # A run profile (e.g. `test`): its own config picks arms/cameras/sim, so we
        # only set the phase — no `task=`/`runtime.sim` override (the profile decides).
        args = ["--config-name", task.config_name, "runtime.phase=collection"]
    else:
        sim = "true" if runtime_is_sim() else "false"
        args = [f"task={task.name}", "runtime.phase=collection", f"runtime.sim={sim}"]
    exe = shutil.which("dual-flexiv-control")
    if exe:
        cmd = [exe, *args]
    else:  # not on PATH (e.g. odd launch) -> run the module with the same interpreter
        import sys

        cmd = [sys.executable, "-m", "dual_flexiv_control.system", *args]
    log.info("launching collection system: %s", " ".join(cmd))
    # Capture combined stdout/stderr to a file so a startup crash (e.g. a camera that
    # cannot open) is surfaceable in the UI; the child owns its dup'd fd after Popen.
    fd, log_path = tempfile.mkstemp(prefix=f"dfc-collection-{run_id}-", suffix=".log")
    logf = os.fdopen(fd, "w")
    try:
        # start_new_session so a later SIGINT can target the whole process group.
        proc = subprocess.Popen(
            cmd, start_new_session=True, stdout=logf, stderr=subprocess.STDOUT
        )
    finally:
        logf.close()  # Popen inherited its own fd; the parent's copy is done with
    return proc, log_path


def _interrupt(proc: subprocess.Popen) -> None:
    """SIGINT the collection subprocess so it saves the in-progress episode + unwinds."""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass


def _escalate_after_grace(proc: subprocess.Popen, grace_s: float = _STOP_GRACE_S) -> None:
    """Wait out the episode save, then SIGTERM → SIGKILL only if it overruns ``grace_s``.

    Called in a background thread after :func:`_interrupt`, so the UI never blocks on
    the (potentially long) video finalize. ``grace_s`` exceeds the orchestrator's own
    save window, so a clean save always completes before we escalate.
    """
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        log.warning("collection system did not exit in %.0fs; terminating", grace_s)
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        log.error("collection system unresponsive; killing")
    proc.kill()


def _log_tail(path: str | None, n: int = _ERROR_TAIL_LINES) -> str | None:
    """Last ``n`` lines of a subprocess log (for surfacing an error), or ``None``."""
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
    "NO frames recorded …" warning naming the stuck stream. Surfacing it verbatim
    tells the operator whether frames are actually flowing.
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
# Rerun helpers
# ---------------------------------------------------------------------------


def log_welcome() -> None:
    """Idle README shown before the first launch (paired with the welcome blueprint)."""
    md = (
        "# dual-flexiv experiments\n\n"
        "Pick a **task** on the left, then launch **Collection** (one teleop demo) "
        "or **Eval** (one policy rollout). Each launch runs a **single episode**.\n\n"
        "**Collection** runs the real recording system and mirrors its live "
        "proprioception, FACTR leaders, and 3D robot scene here. **Eval** streams a "
        "read-only hardware probe (no motion)."
    )
    rr.log(blueprints.README, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)


def reset_viewer() -> None:
    """Return the metrics viewer to its idle welcome state (blueprint + README).

    The Rerun servers themselves keep their bound ports (they are process
    singletons); only the recording's blueprint + README are reset, so the
    embedded viewer drops back to the pre-launch welcome layout.
    """
    rr.send_blueprint(blueprints.welcome_blueprint())
    log_welcome()
    _log_event("services reset — connections restarted, viewers returned to idle")


def _log_readme(task: TaskInfo, phase: str) -> None:
    live = (
        "_Live mirror of the recording system: real proprio + FACTR leaders._"
        if phase == "collection"
        else "_Read-only hardware probe: live dq + TCP position, no motion._"
    )
    md = "\n\n".join(
        [
            f"# {task.name}",
            f"**Phase:** {phase} · single episode",
            f"**Instruction:** {task.language_instruction}",
            live,
        ]
    )
    rr.log(
        blueprints.README,
        rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN),
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


#: Proprio streams the collection view mirrors into the metrics plots. Maps the
#: published stream suffix (``<side>/<suffix>``) to the blueprint row; ``eef`` is
#: a 7-vector pose whose position part feeds the ``eef_pos`` row.
_VIEW_STREAMS = (
    ("q", "q", None),
    ("dq", "dq", None),
    ("tau", "tau", None),
    ("wrench", "wrench", None),
    ("eef_vel", "eef_vel", None),
    ("eef", "eef_pos", 3),  # pose [x y z qw qx qy qz] -> position row
)


def _emit_collection_view(stop: threading.Event, task: TaskInfo) -> None:
    """Mirror the RUNNING collection system into the metrics viewer — all real data.

    The collection subprocess publishes every arm's proprio to shared memory; this
    thread reads those streams **read-only** (never touching the robots) and logs
    them to the proprio rows, so the Metrics tab shows the actual recording, not
    placeholders. The **FACTR leaders** are polled live (``factr/{q,grip}/{side}``)
    — the same signal the recording stores as the action. The **3D robot scene**
    is driven in lock-step: solid arms at the real measured ``<side>/q``,
    translucent ghost at the commanded teleop config. A stream nobody publishes
    (e.g. the right arm in a left-only profile) leaves its row empty and its arm
    still — the viewer never shows motion the system isn't making.
    """
    try:
        _emit_collection_view_inner(stop, task)
    except Exception:  # noqa: BLE001 - a broken view must not look like a broken run
        log.exception("collection live view died (the recording itself is unaffected)")
        _log_event("live view stopped unexpectedly — the recording continues; see dashboard logs")


def _emit_collection_view_inner(stop: threading.Event, task: TaskInfo) -> None:
    from .arms import discover_conventions
    from .arms import read_live_stream

    hz = 15.0  # match the recording rate; FACTR is polled per step (like the recorder)
    dt = 1.0 / hz
    _style_eef_pos_series(blueprints.SIDES)
    factr_client = _open_factr_client()
    factr_errored: set[str] = set()
    conventions = discover_conventions()
    robot_rec = robot_view.robot_recording()
    live_q: dict = {}  # last-known real measured q per side
    plot_every = max(1, round(hz / 5.0))  # mirror plots ~5 Hz (each read attaches shm)
    step = 0
    while not stop.is_set():
        t = step * dt
        rr.set_time("elapsed", duration=t)
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
        leader_samples = (
            _log_factr_leaders(factr_client, factr_errored) if factr_client is not None else {}
        )
        if robot_rec is not None:
            ghost_q = _ghost_configs(leader_samples, conventions)
            robot_view.update_poses(robot_rec, dict(live_q), ghost_q, t)
        step += 1
        stop.wait(dt)
    if factr_client is not None:
        factr_client.close()


# ---------------------------------------------------------------------------
# Eval probe (REAL, read-only dq -> Rerun; NO MOTION)
# ---------------------------------------------------------------------------


def _emit_eval_qvel(stop: threading.Event, task: TaskInfo) -> None:
    """Eval launch (for now): stream each arm's live joint velocity ``dq`` + TCP position.

    A no-motion hardware check. Open a **read-only** ``flexivrdk`` connection per
    arm — ``require_operational`` stays false, so brakes are never released and the
    arm is never commanded — read ``RobotStates`` each tick, and log ``dq`` to
    ``proprio/dq/{side}`` and the end-effector position (``tcp_pose[:3]``) to
    ``proprio/eef_pos/{side}`` (the paths the eval-probe views watch). A robot at
    rest reads dq ~0 and a constant TCP position; the point is to confirm the real
    read → Rerun path end to end before the full eval runner lands. Honours
    ``runtime.sim`` (a :class:`FakeFlexivSource` stands in with no hardware), and
    never raises out — an unreachable arm is reported to the event log and skipped.

    The **3D robot scene** (in this same metrics recording) follows a real eval system
    when one is running (launched from the CLI): solid arms at the measured
    ``<side>/q`` (read-only shared memory, so it never conflicts with the system that
    owns the arm), a **purple ghost** at the policy's horizon-END joint target
    (``eval/<side>/q_horizon``, refreshed once per inference), and a purple **trace**
    from the current EEF to the horizon EEF. Ghost + trace disappear when the eval
    run ends.
    """
    from ..interfaces.flexiv.source import FakeFlexivSource
    from ..interfaces.flexiv.source import FlexivSource
    from .arms import discover_arms
    from .arms import read_live_horizon_q
    from .arms import read_live_joint_positions
    from .arms import runtime_is_sim

    sim = runtime_is_sim()
    arms = discover_arms()
    if not arms:
        _log_event("eval qvel probe: no arms configured")
        return

    sources: dict[str, object] = {}
    for arm in arms:
        try:
            if sim:
                src: object = FakeFlexivSource(arm.serial, dof=arm.dof)
            else:
                # require_operational stays false -> connect & read only, no Enable().
                src = FlexivSource(arm.serial, dof=arm.dof, connect_timeout_s=8.0)
            src.open()
        except Exception as exc:  # noqa: BLE001 - unreachable arm -> report, skip
            _log_event(f"eval qvel probe: {arm.name} ({arm.serial}) connect failed: {exc}")
            continue
        sources[arm.side] = src
        _log_event(
            f"eval probe: {arm.name} ({arm.serial}) connected — streaming dq + TCP position"
        )

    if not sources:
        _log_event("eval probe: no arms reachable; nothing to stream")
        return

    _style_eef_pos_series(sources.keys())
    dt = 1.0 / _EMIT_HZ
    robot_rec = robot_view.robot_recording()
    live_q: dict = {}     # measured q per side from a running eval system's streams
    horizon_q: dict = {}  # policy horizon-end q target per side (eval runs only)
    live_every = max(1, round(_EMIT_HZ * 0.2))  # refresh shared-memory reads ~5 Hz
    step = 0
    try:
        while not stop.is_set():
            t = step * dt
            rr.set_time("elapsed", duration=t)
            for side, src in sources.items():
                try:
                    rs = src.read()
                    dq = np.asarray(rs.dq, dtype=np.float64)
                    eef_pos = np.asarray(rs.tcp_pose[:3], dtype=np.float64)  # [x, y, z] of the TCP pose
                except Exception as exc:  # noqa: BLE001 - transient read error -> skip tick
                    _log_event(f"eval probe: {side} read error: {exc}")
                    continue
                rr.log(blueprints.proprio_path("dq", side), rr.Scalars(dq.tolist()))
                rr.log(blueprints.proprio_path("eef_pos", side), rr.Scalars(eef_pos.tolist()))
            if step % live_every == 0:  # follow the running eval system (throttled)
                for s in blueprints.SIDES:
                    v = read_live_joint_positions(s)
                    if v is not None and len(v) >= 7:
                        live_q[s] = np.asarray(v, dtype=float)[:7]
                    else:
                        live_q.pop(s, None)
                    h = read_live_horizon_q(s)
                    if h is not None and len(h) >= 7:
                        horizon_q[s] = np.asarray(h, dtype=float)[:7]
                    else:
                        horizon_q.pop(s, None)
            if robot_rec is not None:
                real_q = dict(live_q)
                robot_view.update_poses(robot_rec, real_q, {}, t)
                if horizon_q:
                    robot_view.update_horizon_targets(robot_rec, horizon_q, real_q, t)
                else:
                    robot_view.clear_horizon_targets(robot_rec)
            step += 1
            stop.wait(dt)
    finally:
        robot_view.clear_horizon_targets(robot_rec)
        for src in sources.values():
            try:
                src.close()  # read-only handle: close just drops it (no Stop needed)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
