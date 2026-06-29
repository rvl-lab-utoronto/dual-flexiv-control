"""Launch experiments from the dashboard and feed the embedded Rerun viewer.

**Partial launch.** Pressing *Collection* still streams synthetic
**proprioception** (the full teleop/policy runners land with the control path).
Pressing *Eval* now runs a small **real** check: it opens a read-only
``flexivrdk`` connection per arm and streams live joint velocity (``dq``) into the
viewer **with no motion** — a connectivity probe for the read → Rerun path.
:meth:`RunRegistry.launch` records the run, switches the viewer to the phase
blueprint, and starts the matching emitter thread.

A run is **one episode** (collection = one teleop demo, eval = one policy
rollout); launch one at a time, matching the single bimanual rig.

Everything logs to the process-global Rerun recording created by
:func:`~.viewer.start_servers`, so a background emitter thread (and, later, a
real run process connecting back over gRPC) shows up in the same embedded viewer.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from uuid import uuid4

import numpy as np
import rerun as rr

from . import blueprints
from .tasks import TaskInfo

PHASES = ("collection", "eval")

#: Placeholder emitter loop rate.
_EMIT_HZ = 30.0


@dataclass
class RunRecord:
    """A launched run, as surfaced in the dashboard's status panel."""

    run_id: str
    task: str
    phase: str
    instruction: str
    started_wall: str
    started_monotonic: float
    status: str = "running"  # running | stopped
    _stop: threading.Event | None = field(default=None, repr=False, compare=False)
    _thread: threading.Thread | None = field(default=None, repr=False, compare=False)


class RunRegistry:
    """Owns the single active run and the run history; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: RunRecord | None = None
        self._history: list[RunRecord] = []

    def active(self) -> RunRecord | None:
        with self._lock:
            return self._active

    def history(self) -> list[RunRecord]:
        with self._lock:
            return list(self._history)

    def launch(self, task: TaskInfo, phase: str) -> RunRecord:
        """Start a run for ``task`` in ``phase`` ("collection" | "eval").

        Any currently-active run is stopped first (single-rig invariant).
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r} (expected one of {PHASES})")
        with self._lock:
            if self._active is not None:
                self._stop_locked()

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
                rr.send_blueprint(blueprints.qvel_test_blueprint(task.name))
            else:
                rr.send_blueprint(blueprints.for_phase(phase, task.name))
            _log_readme(task, phase)
            _log_event(f"launch {phase} · task={task.name} · run={run.run_id}")

            # --- LAUNCH (partial) ----------------------------------------------
            # Collection still streams synthetic proprio; eval runs a real,
            # read-only dq probe (see _emit_eval_qvel) — no motion. The full
            # eval/collection runners land with the control path; they will spawn
            # the real system, e.g.:
            #
            #   subprocess.Popen([
            #       "dual-flexiv-control",
            #       f"task={task.name}", f"runtime.phase={phase}",
            #       f"runtime.rerun_uri={servers.grpc_uri}",   # run connects back
            #   ])
            #
            # That process would `rr.init(...); rr.connect_grpc(grpc_uri)` and log
            # real per-arm proprio to the same entity paths used here (see
            # dashboard.blueprints), so the embedded viewer needs no change.
            target = _emit_eval_qvel if phase == "eval" else _emit_proprio
            run._thread = threading.Thread(
                target=target,
                args=(run._stop, task),
                name=f"dfc-emit-{run.run_id}",
                daemon=True,
            )
            run._thread.start()

            self._active = run
            return run

    def stop_active(self) -> None:
        """Stop the active run (if any) and move it to history."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        run = self._active
        if run is None:
            return
        if run._stop is not None:
            run._stop.set()
        if run._thread is not None:
            run._thread.join(timeout=2.0)
        run.status = "stopped"
        _log_event(f"stop {run.phase} · run={run.run_id}")
        self._history.append(run)
        self._active = None


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------


def log_welcome() -> None:
    """Idle README shown before the first launch (paired with the welcome blueprint)."""
    md = (
        "# dual-flexiv experiments\n\n"
        "Pick a **task** on the left, then launch **Collection** (one teleop demo) "
        "or **Eval** (one policy rollout). Each launch runs a **single episode**.\n\n"
        "Metrics stream into this viewer live. Launching is a stub for now — the "
        "placeholder metrics are synthetic **proprioception** until the real "
        "eval/collection runners land with the control path."
    )
    rr.log(blueprints.README, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)


def _log_readme(task: TaskInfo, phase: str) -> None:
    md = "\n\n".join(
        [
            f"# {task.name}",
            f"**Phase:** {phase} · single episode",
            f"**Instruction:** {task.language_instruction}",
            "_Placeholder metrics — synthetic proprioception. Real telemetry lands "
            "with the control path._",
        ]
    )
    rr.log(
        blueprints.README,
        rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def _log_event(message: str) -> None:
    rr.log(blueprints.EVENTS, rr.TextLog(message, level="INFO"))


# ---------------------------------------------------------------------------
# Placeholder emitter (SYNTHETIC PROPRIO — swap for real telemetry; see launch())
# ---------------------------------------------------------------------------

#: Per-arm home posture (rad) the synthetic joints oscillate around.
_HOME_Q = {
    "left": np.array([0.0, -0.70, 0.0, 1.55, 0.0, 0.80, 0.0]),
    "right": np.array([0.0, -0.70, 0.0, 1.55, 0.0, 0.80, 0.0]),
}
_EEF_CENTER = {
    "left": np.array([0.45, 0.20, 0.35]),
    "right": np.array([0.45, -0.20, 0.35]),
}
_EEF_COLOR = {"left": [80, 160, 255], "right": [255, 140, 80]}


def _emit_proprio(stop: threading.Event, task: TaskInfo) -> None:
    """Stream synthetic per-arm proprioception for one episode until stopped.

    Mirrors the real proprio streams (``q``, ``dq``, ``tau``, ``wrench``,
    ``eef``, ``eef_vel``): joint signals oscillate around a home posture, the TCP
    pose traces a small path (shown in 3D), and the rest are smooth placeholders.
    No reward/success/episode-count — just proprio.
    """
    dt = 1.0 / _EMIT_HZ
    trails = {s: deque(maxlen=256) for s in blueprints.SIDES}
    joints = np.arange(7)
    step = 0
    while not stop.is_set():
        t = step * dt
        rr.set_time("elapsed", duration=t)
        for side in blueprints.SIDES:
            ph = 0.0 if side == "left" else 1.5
            sgn = 1.0 if side == "left" else -1.0

            q = _HOME_Q[side] + 0.15 * np.sin(0.5 * t + 0.6 * joints + ph)
            dq = 0.15 * 0.5 * np.cos(0.5 * t + 0.6 * joints + ph)
            tau = 3.0 * np.sin(0.3 * t + 0.5 * joints + ph)
            wrench = np.concatenate(
                [
                    4.0 * np.sin(0.4 * t + np.arange(3) + ph),  # force (N)
                    0.6 * np.sin(0.4 * t + np.arange(3) + ph + 1.0),  # torque (Nm)
                ]
            )
            pos = _EEF_CENTER[side] + np.array(
                [
                    0.10 * np.sin(0.7 * t),
                    0.08 * sgn * np.cos(0.9 * t),
                    0.06 * np.sin(1.3 * t + ph),
                ]
            )
            eef_vel = np.concatenate(
                [
                    np.array(
                        [
                            0.070 * np.cos(0.7 * t),
                            -0.072 * sgn * np.sin(0.9 * t),
                            0.078 * np.cos(1.3 * t + ph),
                        ]
                    ),  # linear (m/s)
                    0.2 * np.sin(0.5 * t + np.arange(3) + ph),  # angular (rad/s)
                ]
            )

            rr.log(blueprints.proprio_path("q", side), rr.Scalars(q.tolist()))
            rr.log(blueprints.proprio_path("dq", side), rr.Scalars(dq.tolist()))
            rr.log(blueprints.proprio_path("tau", side), rr.Scalars(tau.tolist()))
            rr.log(blueprints.proprio_path("wrench", side), rr.Scalars(wrench.tolist()))
            rr.log(blueprints.proprio_path("eef_vel", side), rr.Scalars(eef_vel.tolist()))

            trails[side].append(pos.tolist())
            rr.log(
                blueprints.eef_tip_path(side),
                rr.Points3D([pos.tolist()], colors=[_EEF_COLOR[side]], radii=0.012),
            )
            if len(trails[side]) >= 2:
                rr.log(
                    blueprints.eef_path_path(side),
                    rr.LineStrips3D([list(trails[side])], colors=[_EEF_COLOR[side]]),
                )
        step += 1
        stop.wait(dt)


# ---------------------------------------------------------------------------
# Eval probe (REAL, read-only dq -> Rerun; NO MOTION)
# ---------------------------------------------------------------------------


def _emit_eval_qvel(stop: threading.Event, task: TaskInfo) -> None:
    """Eval launch (for now): stream each arm's live joint velocity ``dq``.

    A no-motion hardware check. Open a **read-only** ``flexivrdk`` connection per
    arm — ``require_operational`` stays false, so brakes are never released and the
    arm is never commanded — read ``RobotStates.dq`` each tick, and log it to
    ``proprio/dq/{side}`` (the path the dq time-series view watches). A robot at
    rest reads ~0; the point is to confirm the real read → Rerun path end to end
    before the full eval runner lands. Honours ``runtime.sim`` (a
    :class:`FakeFlexivSource` stands in with no hardware), and never raises out —
    an unreachable arm is reported to the event log and skipped.
    """
    from ..interfaces.flexiv.source import FakeFlexivSource
    from ..interfaces.flexiv.source import FlexivSource
    from .arms import discover_arms
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
        _log_event(f"eval qvel probe: {arm.name} ({arm.serial}) connected — streaming dq")

    if not sources:
        _log_event("eval qvel probe: no arms reachable; nothing to stream")
        return

    dt = 1.0 / _EMIT_HZ
    step = 0
    try:
        while not stop.is_set():
            rr.set_time("elapsed", duration=step * dt)
            for side, src in sources.items():
                try:
                    dq = np.asarray(src.read().dq, dtype=np.float64)
                except Exception as exc:  # noqa: BLE001 - transient read error -> skip tick
                    _log_event(f"eval qvel probe: {side} read error: {exc}")
                    continue
                rr.log(blueprints.proprio_path("dq", side), rr.Scalars(dq.tolist()))
            step += 1
            stop.wait(dt)
    finally:
        for src in sources.values():
            try:
                src.close()  # read-only handle: close just drops it (no Stop needed)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
