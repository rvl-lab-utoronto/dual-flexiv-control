"""Rerun blueprints: the metric layout shown for a run.

A run (collection or eval) is **one episode** and its placeholder metrics are
**proprioception** — the same per-arm signals the Flexiv interfaces stream
(`q`, `dq`, `tau`, `wrench`, `eef`, `eef_vel`). The **3D panel is the robot scene**
(both arms on the Vention pedestal, from :mod:`~.robot_view`): solid arms at the
measured joint state, a translucent ghost at the commanded teleop config, and —
during eval — the policy horizon target in purple. It replaces the old end-effector
trace, so this single viewer shows the arms beside the metrics. The end-effector
position is still readable over time as an x/y/z time series (`eef_pos`, the
position slice of `eef`). The time series are laid out as **two per-arm columns
(left | right)** rather than overlaying both arms in each panel. The emitter in
:mod:`~.runner` (and, later, the real run) logs to exactly these paths, so the
layout never drifts from what's produced.

Entity-path scheme (kept in one place):

* ``robot/*``                    — the 3D robot scene (arms + pedestal), owned by
                                   :mod:`~.robot_view`, logged into this recording.
* ``proprio/{signal}/{side}``    — one time-series entity per **follower** signal, per arm.
* ``factr/{signal}/{side}``      — one time-series entity per **FACTR leader** signal, per arm.
* ``events``                     — text-log of run lifecycle events.
* ``readme``                     — markdown panel describing the active run.
"""

from __future__ import annotations

import rerun.blueprint as rrb

SIDES = ("left", "right")

# -- entity paths (the contract with the emitter) ---------------------------

#: The merged 3D panel: the robot scene (arms on the pedestal) that replaced the
#: old end-effector trace. Its entities (``/robot/*``) are logged into the metrics
#: recording by :mod:`~.robot_view` (attached at startup) and animated live by the
#: emitter, so this one viewer shows the arms beside the time series.
ROBOT_ORIGIN = "/robot"
ROBOT_VIEW_NAME = "Robot — measured (solid) · teleop cmd (ghost) · policy horizon (purple)"
PROPRIO_ROOT = "proprio"
FACTR_ROOT = "factr"
EVENTS = "events"
README = "readme"

#: Time-series signals. The full ``eef`` *pose* (with quaternion) drives the 3D
#: robot scene rather than a series, but its position (``eef_pos`` = ``eef[:3]``) is
#: plotted here — first, so the requested TCP-position metric is the top-left panel.
PROPRIO_SERIES: tuple[str, ...] = ("eef_pos", "q", "dq", "tau", "wrench", "eef_vel")

#: Human titles + dimensionality for each time-series signal (matches the RDK mapping;
#: ``eef_pos`` is the 3-vector position slice of the 7-vector ``eef`` pose).
PROPRIO_TITLES = {
    "eef_pos": "TCP position (m)",
    "q": "Joint position q (rad)",
    "dq": "Joint velocity dq (rad/s)",
    "tau": "Joint torque τ (Nm)",
    "wrench": "TCP wrench (N, Nm)",
    "eef": "TCP pose",
    "eef_vel": "TCP twist (m/s, rad/s)",
}
PROPRIO_DIMS = {"eef_pos": 3, "q": 7, "dq": 7, "tau": 7, "wrench": 6, "eef": 7, "eef_vel": 6}

#: ``eef_pos``'s three scalar series are the world X/Y/Z axes; naming + colouring
#: them (logged statically by the emitter) makes the plot legend read x/y/z in the
#: classic axis colours, so the position panel reads as world X/Y/Z at a glance.
EEF_POS_COMPONENTS: tuple[str, ...] = ("x", "y", "z")
EEF_POS_COLORS = [[220, 70, 70], [70, 200, 70], [70, 120, 240]]


def robot_view(name: str = ROBOT_VIEW_NAME) -> rrb.Spatial3DView:
    """The robot 3D scene view (shared by the run + welcome layouts).

    Points at ``/robot`` — the arms-on-the-pedestal scene :mod:`~.robot_view` logs
    into this recording — so every layout renders it from one place.
    """
    return rrb.Spatial3DView(origin=ROBOT_ORIGIN, name=name)


def proprio_path(signal: str, side: str) -> str:
    """Time-series entity for one arm's signal, e.g. ``proprio/tau/left``."""
    return f"{PROPRIO_ROOT}/{signal}/{side}"


def proprio_group(signal: str) -> str:
    """View origin spanning both arms for a signal, e.g. ``proprio/tau``."""
    return f"{PROPRIO_ROOT}/{signal}"


# -- FACTR leader arms (teleop input) ---------------------------------------

#: FACTR leader time-series signals. The server returns one flat ``joint_pos``
#: vector of ``DoF+1`` values per leader — the arm joints followed by a trailing
#: gripper/trigger value — so the emitter splits it into ``q`` (arm joints) and
#: ``grip`` (the trailing scalar) and logs each to its own row.
FACTR_SERIES: tuple[str, ...] = ("q", "grip")
FACTR_TITLES = {
    "q": "Leader joint q (rad)",
    "grip": "Leader gripper (rad)",
}


def factr_path(signal: str, side: str) -> str:
    """Time-series entity for one FACTR leader's signal, e.g. ``factr/q/left``."""
    return f"{FACTR_ROOT}/{signal}/{side}"


def factr_group(signal: str) -> str:
    """View origin spanning both leaders for a signal, e.g. ``factr/q``."""
    return f"{FACTR_ROOT}/{signal}"


# -- layouts ----------------------------------------------------------------


def for_phase(phase: str, task_name: str | None = None) -> rrb.Blueprint:
    """Blueprint for a single-episode run; both phases show proprio."""
    if phase not in ("eval", "collection"):
        raise ValueError(f"unknown phase {phase!r} (expected 'eval' or 'collection')")
    return _proprio_blueprint(phase, task_name)


def _arm_column(side: str, signals: tuple[str, ...]) -> rrb.Vertical:
    """One arm's vertical stack of per-signal time series (origin ``proprio/<sig>/<side>``).

    Splitting by side — one column per arm — instead of overlaying both arms in each
    panel matches the bimanual rig. Each view name carries a ``· <side>`` tag because
    layout containers have no visible header in the viewport, so the tag is what keeps
    the left and right columns unambiguous.
    """
    return rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/{proprio_path(sig, side)}", name=f"{PROPRIO_TITLES[sig]} · {side}"
            )
            for sig in signals
        ],
        name=f"{side.capitalize()} arm",
    )


def _factr_column(side: str) -> rrb.Vertical:
    """One FACTR leader's stack of time series (origin ``factr/<sig>/<side>``).

    A row per leader arm mirroring :func:`_arm_column` for the followers. Both
    sides always get a column so the layout is stable even when only one leader
    is plugged in — the absent side's views simply stay empty (the emitter logs
    nothing for a leader it cannot reach).
    """
    return rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/{factr_path(sig, side)}", name=f"{FACTR_TITLES[sig]} · {side}"
            )
            for sig in FACTR_SERIES
        ],
        name=f"{side.capitalize()} leader (FACTR)",
    )


def _proprio_blueprint(phase: str, task_name: str | None) -> rrb.Blueprint:
    """Robot 3D scene beside per-arm columns: FACTR leaders over followers (left | right)."""
    title = ROBOT_VIEW_NAME
    if task_name:
        title = f"{title} — {task_name} · {phase}"
    return rrb.Blueprint(
        rrb.Horizontal(
            # 3D shows both arms on the pedestal (measured/ghost); time series split per arm.
            robot_view(title),
            rrb.Vertical(
                # FACTR leaders (teleop input) on top, follower proprio below.
                rrb.Horizontal(
                    _factr_column("left"), _factr_column("right"), name="FACTR leaders"
                ),
                rrb.Horizontal(
                    _arm_column("left", PROPRIO_SERIES),
                    _arm_column("right", PROPRIO_SERIES),
                ),
                rrb.TextLogView(origin=f"/{EVENTS}", name="Events"),
                row_shares=[3, 8, 1],
            ),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def eval_probe_blueprint(task_name: str | None = None) -> rrb.Blueprint:
    """Eval no-motion probe view: robot 3D scene beside per-arm TCP position + ``dq``.

    The eval launch runs a read-only hardware probe (see :mod:`~.runner`) that logs
    ``proprio/eef_pos/{side}`` and ``proprio/dq/{side}`` with no motion, and drives
    the robot 3D scene (solid arms at the measured ``q``, plus the policy horizon
    target in purple when a real eval system is running). The 3D scene sits beside
    per-arm columns (TCP position over dq) so it is immediately obvious whether live
    telemetry is arriving from *each* robot, rather than overlaying both arms.
    """
    signals = ("eef_pos", "dq")
    name = "Eval no-motion probe" + (f" — {task_name}" if task_name else "")
    return rrb.Blueprint(
        rrb.Horizontal(
            robot_view(),
            rrb.Vertical(
                rrb.Horizontal(
                    _arm_column("left", signals), _arm_column("right", signals), name=name
                ),
                rrb.TextLogView(origin=f"/{EVENTS}", name="Events"),
                row_shares=[6, 1],
            ),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def welcome_blueprint() -> rrb.Blueprint:
    """Idle layout shown before any run is launched: the robot scene beside the README."""
    return rrb.Blueprint(
        rrb.Horizontal(
            robot_view(),
            rrb.TextDocumentView(origin=f"/{README}", name="Dashboard"),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )
