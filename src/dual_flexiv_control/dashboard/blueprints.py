"""Rerun blueprints: the metric layout shown for a run.

A run (collection or eval) is **one episode** and its placeholder metrics are
**proprioception** — the same per-arm signals the Flexiv interfaces stream
(`q`, `dq`, `tau`, `tau_ext`, `wrench`, `eef`, `eef_vel`). The end-effector
position is readable over time as an x/y/z time series (`eef_pos`, the position
slice of `eef`). The time series are laid out as **two per-arm columns
(left | right)** rather than overlaying both arms in each panel. The emitter in
:mod:`~.runner` logs to exactly these paths, so the layout never drifts from
what's produced.

Entity-path scheme (kept in one place):

* ``proprio/{signal}/{side}``    — one time-series entity per **follower** signal, per arm.
* ``factr/{signal}/{side}``      — one time-series entity per **FACTR leader** signal, per arm.
* ``events``                     — text-log of run lifecycle events.
* ``readme``                     — markdown panel describing the active run.
"""

from __future__ import annotations

import os

import rerun.blueprint as rrb
from rerun.blueprint.components import PlayState

SIDES = ("left", "right")

#: Default trailing window (seconds) the live plots render, tunable with
#: ``DFC_DASHBOARD_PLOT_WINDOW_S`` (``0`` or negative = show the full history).
DEFAULT_PLOT_WINDOW_S = 30.0


def _plot_window_s() -> float:
    """Trailing plot window in seconds (env-overridable; falls back on garbage)."""
    try:
        return float(os.environ.get("DFC_DASHBOARD_PLOT_WINDOW_S", DEFAULT_PLOT_WINDOW_S))
    except (TypeError, ValueError):
        return DEFAULT_PLOT_WINDOW_S


def _live_window() -> "rrb.VisibleTimeRange | None":
    """Scrolling last-N-seconds visible range for a live time-series view.

    Anchored to the play cursor (which tracks the live edge while streaming), so
    each plot draws only the trailing window instead of the whole accumulated
    ``elapsed`` history — keeping render cost flat no matter how long the
    dashboard has been streaming. Returns ``None`` (view keeps its default
    full-history range) when the window is disabled.
    """
    window = _plot_window_s()
    if window <= 0:
        return None
    return rrb.VisibleTimeRange(
        "elapsed",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-window),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=0.0),
    )


def _live_time_panel() -> rrb.TimePanel:
    """Pin the viewer to the ``elapsed`` timeline (and follow the live edge).

    The mirror stamps every sample on ``elapsed``, but ``rr.log`` also auto-adds
    ``log_time`` / ``log_tick``. Without pinning the active timeline the viewer is
    free to plot against ``log_time`` — on which the ``elapsed`` visible-time-range
    window (see :func:`_live_window`) has no effect, so the scroll never clips.
    Forcing ``elapsed`` makes the window bite and the time axis read in seconds.
    While windowed we also start in *Following* so the trailing range tracks the
    live edge; with the window off we leave the play state alone (free scrubbing
    over the full history).
    """
    return rrb.TimePanel(
        timeline="elapsed",
        play_state=PlayState.Following if _plot_window_s() > 0 else None,
    )

# -- entity paths (the contract with the emitter) ---------------------------

PROPRIO_ROOT = "proprio"
FACTR_ROOT = "factr"
EVENTS = "events"
README = "readme"

#: Time-series signals. The full ``eef`` pose includes a quaternion; its position
#: (``eef_pos`` = ``eef[:3]``) is plotted separately as the first metric.
PROPRIO_SERIES: tuple[str, ...] = (
    "eef_pos", "q", "dq", "tau", "tau_ext", "wrench", "eef_vel",
)

#: Human titles + dimensionality for each time-series signal (matches the RDK mapping;
#: ``eef_pos`` is the 3-vector position slice of the 7-vector ``eef`` pose).
PROPRIO_TITLES = {
    "eef_pos": "TCP position (m)",
    "q": "Joint position q (rad)",
    "dq": "Joint velocity dq (rad/s)",
    "tau": "Measured joint torque τ (Nm)",
    "tau_ext": "External joint torque τ_ext (Nm)",
    "wrench": "TCP wrench (N, Nm)",
    "eef": "TCP pose",
    "eef_vel": "TCP twist (m/s, rad/s)",
}
PROPRIO_DIMS = {
    "eef_pos": 3, "q": 7, "dq": 7, "tau": 7, "tau_ext": 7,
    "wrench": 6, "eef": 7, "eef_vel": 6,
}

#: ``eef_pos``'s three scalar series are the world X/Y/Z axes; naming + colouring
#: them (logged statically by the emitter) makes the plot legend read x/y/z in the
#: classic axis colours, so the position panel reads as world X/Y/Z at a glance.
EEF_POS_COMPONENTS: tuple[str, ...] = ("x", "y", "z")
EEF_POS_COLORS = [[220, 70, 70], [70, 200, 70], [70, 120, 240]]

# -- policy-server comms (eval runs) -----------------------------------------

#: Policy-server communication entities (eval only), fed by the mirror from the
#: run's ``eval/policy_comm`` stream. ``packets`` holds the cumulative
#: sent/received/error counters plus the 0/1 in-flight square wave (each step is
#: one packet event at its true time); ``latency`` the per-request round trip.
POLICY_COMM_ROOT = "policy/comm"
POLICY_LATENCY_PATH = "policy/latency"
POLICY_COMM_SERIES: tuple[str, ...] = ("sent", "received", "errors", "in_flight")
POLICY_COMM_COLORS = [[70, 120, 240], [70, 200, 70], [220, 70, 70], [160, 160, 160]]


def policy_comm_path(series: str) -> str:
    """Entity for one comm counter, e.g. ``policy/comm/sent``."""
    return f"{POLICY_COMM_ROOT}/{series}"



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
    "q": "Leader joint q — DFC/Rizon convention (rad)",
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
    """Blueprint for a session mode; runs and idle viewing all show live proprio.

    ``viewing`` is the session's idle mode: the arms publish read-only telemetry
    with no run active, so the same proprio grid applies (its rows simply follow
    whatever the hardware is doing).
    """
    if phase not in ("eval", "collection", "skill", "viewing"):
        raise ValueError(
            f"unknown phase {phase!r} (expected 'eval', 'collection', 'skill' "
            "or 'viewing')"
        )
    return _proprio_blueprint(phase, task_name)


def _arm_column(side: str, signals: tuple[str, ...]) -> rrb.Vertical:
    """One arm's vertical stack of per-signal time series (origin ``proprio/<sig>/<side>``).

    Splitting by side — one column per arm — instead of overlaying both arms in each
    panel matches the bimanual rig. Each view name carries a ``· <side>`` tag because
    layout containers have no visible header in the viewport, so the tag is what keeps
    the left and right columns unambiguous.
    """
    win = _live_window()
    return rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/{proprio_path(sig, side)}",
                name=f"{PROPRIO_TITLES[sig]} · {side}",
                time_ranges=win,
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
    win = _live_window()
    return rrb.Vertical(
        *[
            rrb.TimeSeriesView(
                origin=f"/{factr_path(sig, side)}",
                name=f"{FACTR_TITLES[sig]} · {side}",
                time_ranges=win,
            )
            for sig in FACTR_SERIES
        ],
        name=f"{side.capitalize()} leader (FACTR)",
    )


def _policy_comm_row() -> rrb.Horizontal:
    """Policy-server comm panels (eval): packet activity beside round-trip latency.

    The packets panel steps its counters at each send/receive (with a 0/1
    in-flight wave marking the gap between them); the latency panel plots one
    round-trip point per completed request.
    """
    win = _live_window()
    return rrb.Horizontal(
        rrb.TimeSeriesView(
            origin=f"/{POLICY_COMM_ROOT}",
            name="Policy server — packets (sent · received · errors · in-flight)",
            time_ranges=win,
        ),
        rrb.TimeSeriesView(
            origin=f"/{POLICY_LATENCY_PATH}",
            name="Policy server — round-trip latency (ms)",
            time_ranges=win,
        ),
        name="Policy server comms",
    )


def _proprio_blueprint(phase: str, task_name: str | None) -> rrb.Blueprint:
    """Per-arm metrics: FACTR leaders over followers (left | right).

    Eval adds a policy-server comms row (packet activity + round-trip latency)
    below the arm metrics.
    """
    rows = [
        rrb.Horizontal(
            _factr_column("left"), _factr_column("right"), name="FACTR leaders"
        ),
        rrb.Horizontal(
            _arm_column("left", PROPRIO_SERIES),
            _arm_column("right", PROPRIO_SERIES),
        ),
    ]
    row_shares = [3, 8]
    if phase == "eval":
        rows.append(_policy_comm_row())
        row_shares.append(2)
    rows.append(rrb.TextLogView(origin=f"/{EVENTS}", name="Events"))
    row_shares.append(1)
    return rrb.Blueprint(
        rrb.Vertical(*rows, row_shares=row_shares),
        _live_time_panel(),
        collapse_panels=True,
    )


def eval_probe_blueprint(task_name: str | None = None) -> rrb.Blueprint:
    """Eval no-motion probe view: per-arm TCP position + ``dq``.

    The eval launch runs a read-only hardware probe (see :mod:`~.runner`) that logs
    ``proprio/eef_pos/{side}`` and ``proprio/dq/{side}`` with no motion. Separate
    per-arm columns make it immediately obvious whether live telemetry is arriving
    from each robot rather than overlaying both arms.
    """
    signals = ("eef_pos", "dq")
    name = "Eval no-motion probe" + (f" — {task_name}" if task_name else "")
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                _arm_column("left", signals), _arm_column("right", signals), name=name
            ),
            rrb.TextLogView(origin=f"/{EVENTS}", name="Events"),
            row_shares=[6, 1],
        ),
        _live_time_panel(),
        collapse_panels=True,
    )


def welcome_blueprint() -> rrb.Blueprint:
    """Idle layout shown before any run is launched."""
    return rrb.Blueprint(
        rrb.TextDocumentView(origin=f"/{README}", name="Dashboard"),
        _live_time_panel(),
        collapse_panels=True,
    )
