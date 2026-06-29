"""Launch experiments from the dashboard and feed the embedded Rerun viewer.

**Stub launch.** Pressing *Collection* / *Eval* does not yet spawn the real
system (eval/collection land with the control path). Instead :meth:`RunRegistry.launch`
records the run, switches the viewer to the phase's blueprint, and starts a
placeholder emitter thread that streams synthetic metrics so the dashboard is
live end-to-end. The seam for the real launch is marked in :meth:`RunRegistry.launch`.

Everything logs to the process-global Rerun recording created by
:func:`~.viewer.start_servers`, so a background emitter thread (and, later, a
real run process connecting back over gRPC) shows up in the same embedded viewer.
One run is active at a time, matching the single bimanual rig.
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
            rr.send_blueprint(blueprints.for_phase(phase, task.name))
            _log_readme(task, phase)
            _log_event(f"launch {phase} · task={task.name} · run={run.run_id}")

            # --- STUB LAUNCH ---------------------------------------------------
            # Real eval/collection lands with the control path. Replace the
            # placeholder emitter below with a spawn of the actual system, e.g.:
            #
            #   subprocess.Popen([
            #       "dual-flexiv-control",
            #       f"task={task.name}", f"runtime.phase={phase}",
            #       f"runtime.rerun_uri={servers.grpc_uri}",   # run connects back
            #   ])
            #
            # The run process would `rr.init(...); rr.connect_grpc(grpc_uri)` and
            # log real per-arm EEF poses / eval metrics to the same entity paths
            # this emitter uses (see dashboard.blueprints), so the embedded viewer
            # needs no change. Until then we stream synthetic data:
            emitter = _EMITTERS[phase]
            run._thread = threading.Thread(
                target=emitter,
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
        "Pick a **task** on the left, then launch **Collection** (teleop demos) "
        "or **Eval** (policy rollouts).\n\n"
        "Metrics for the active run stream into this viewer live. Launching is a "
        "stub for now — synthetic placeholder metrics until the eval/collection "
        "runners land with the control path."
    )
    rr.log(blueprints.README, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)


def _log_readme(task: TaskInfo, phase: str) -> None:
    lines = [
        f"# {task.name}",
        f"**Phase:** {phase}",
        f"**Instruction:** {task.language_instruction}",
    ]
    if phase == "collection" and task.num_episodes is not None:
        lines.append(f"Target: **{task.num_episodes}** demonstration episodes.")
    if phase == "eval" and task.num_timesteps is not None:
        lines.append(f"Rollout horizon: **{task.num_timesteps}** timesteps.")
    lines.append(
        "_Placeholder metrics — real eval/collection telemetry lands with the control path._"
    )
    rr.log(
        blueprints.README,
        rr.TextDocument("\n\n".join(lines), media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def _log_event(message: str) -> None:
    rr.log(blueprints.EVENTS, rr.TextLog(message, level="INFO"))


# ---------------------------------------------------------------------------
# Placeholder emitters (SYNTHETIC — swap for real telemetry; see launch())
# ---------------------------------------------------------------------------

_EEF_CENTER = {
    "left": np.array([0.45, 0.20, 0.35]),
    "right": np.array([0.45, -0.20, 0.35]),
}
_EEF_COLOR = {"left": [80, 160, 255], "right": [255, 140, 80]}


def _emit_eval(stop: threading.Event, task: TaskInfo) -> None:
    """Stream synthetic per-arm EEF tracks + placeholder eval scalars."""
    dt = 1.0 / _EMIT_HZ
    trails = {s: deque(maxlen=256) for s in blueprints.SIDES}
    step = 0
    while not stop.is_set():
        t = step * dt
        rr.set_time("elapsed", duration=t)
        for side in blueprints.SIDES:
            phase = 0.0 if side == "left" else 1.5
            sign = 1.0 if side == "left" else -1.0
            pos = _EEF_CENTER[side] + np.array(
                [
                    0.10 * np.sin(0.7 * t),
                    0.08 * sign * np.cos(0.9 * t),
                    0.06 * np.sin(1.3 * t + phase),
                ]
            )
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
            root = blueprints.eef_axis_root(side)
            rr.log(f"{root}/x", rr.Scalars(float(pos[0])))
            rr.log(f"{root}/y", rr.Scalars(float(pos[1])))
            rr.log(f"{root}/z", rr.Scalars(float(pos[2])))

        rr.log(
            f"{blueprints.EVAL_METRICS}/reward",
            rr.Scalars(float(np.tanh(0.05 * t) + 0.05 * np.sin(2.0 * t))),
        )
        rr.log(
            f"{blueprints.EVAL_METRICS}/success_rate",
            rr.Scalars(float(min(1.0, 0.02 * t))),
        )
        step += 1
        stop.wait(dt)


def _emit_collection(stop: threading.Event, task: TaskInfo) -> None:
    """Stream synthetic teleop-collection progress (episodes / samples)."""
    dt = 0.25
    target = task.num_episodes or 0
    episodes = 0
    samples = 0
    step = 0
    while not stop.is_set():
        rr.set_time("elapsed", duration=step * dt)
        if step > 0 and step % 12 == 0 and (target == 0 or episodes < target):
            episodes += 1
            suffix = f"/{target}" if target else ""
            _log_event(f"recorded episode {episodes}{suffix}")
        samples += 7
        rr.log(
            f"{blueprints.COLLECTION_METRICS}/episodes_recorded",
            rr.Scalars(float(episodes)),
        )
        rr.log(f"{blueprints.COLLECTION_METRICS}/samples", rr.Scalars(float(samples)))
        step += 1
        stop.wait(dt)


_EMITTERS = {"eval": _emit_eval, "collection": _emit_collection}
