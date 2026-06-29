"""Rerun blueprints: the metric layout shown for a run.

A run (collection or eval) is **one episode** and its placeholder metrics are
**proprioception** — the same per-arm signals the Flexiv interfaces stream
(`q`, `dq`, `tau`, `wrench`, `eef`, `eef_vel`). The end-effector position (from
`eef`) is shown as a 3D track; the rest are per-signal time series with both
arms overlaid. The emitter in :mod:`~.runner` (and, later, the real run) logs to
exactly these paths, so the layout never drifts from what's produced.

Entity-path scheme (kept in one place):

* ``eef/{side}/{tip,path}``      — end-effector point + trailing path, per arm (3D).
* ``proprio/{signal}/{side}``    — one time-series entity per signal, per arm.
* ``events``                     — text-log of run lifecycle events.
* ``readme``                     — markdown panel describing the active run.
"""

from __future__ import annotations

import rerun.blueprint as rrb

SIDES = ("left", "right")

# -- entity paths (the contract with the emitter) ---------------------------

EEF_ROOT = "eef"
PROPRIO_ROOT = "proprio"
EVENTS = "events"
README = "readme"

#: Proprio signals shown as time series (eef is shown in 3D, not as a series).
PROPRIO_SERIES: tuple[str, ...] = ("q", "dq", "tau", "wrench", "eef_vel")

#: Human titles + dimensionality for each proprio signal (matches the RDK mapping).
PROPRIO_TITLES = {
    "q": "Joint position q (rad)",
    "dq": "Joint velocity dq (rad/s)",
    "tau": "Joint torque τ (Nm)",
    "wrench": "TCP wrench (N, Nm)",
    "eef": "TCP pose",
    "eef_vel": "TCP twist (m/s, rad/s)",
}
PROPRIO_DIMS = {"q": 7, "dq": 7, "tau": 7, "wrench": 6, "eef": 7, "eef_vel": 6}


def eef_tip_path(side: str) -> str:
    """3D point of one arm's current end-effector position."""
    return f"{EEF_ROOT}/{side}/tip"


def eef_path_path(side: str) -> str:
    """Trailing 3D line-strip of one arm's recent end-effector path."""
    return f"{EEF_ROOT}/{side}/path"


def proprio_path(signal: str, side: str) -> str:
    """Time-series entity for one arm's signal, e.g. ``proprio/tau/left``."""
    return f"{PROPRIO_ROOT}/{signal}/{side}"


def proprio_group(signal: str) -> str:
    """View origin spanning both arms for a signal, e.g. ``proprio/tau``."""
    return f"{PROPRIO_ROOT}/{signal}"


# -- layouts ----------------------------------------------------------------


def for_phase(phase: str, task_name: str | None = None) -> rrb.Blueprint:
    """Blueprint for a single-episode run; both phases show proprio."""
    if phase not in ("eval", "collection"):
        raise ValueError(f"unknown phase {phase!r} (expected 'eval' or 'collection')")
    return _proprio_blueprint(phase, task_name)


def _proprio_blueprint(phase: str, task_name: str | None) -> rrb.Blueprint:
    """3D end-effector track beside per-signal proprio time series (both arms)."""
    title = "End-effector position (3D)"
    if task_name:
        title = f"{title} — {task_name} · {phase}"
    series = [
        rrb.TimeSeriesView(origin=f"/{proprio_group(sig)}", name=PROPRIO_TITLES[sig])
        for sig in PROPRIO_SERIES
    ]
    series.append(rrb.TextLogView(origin=f"/{EVENTS}", name="Events"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"/{EEF_ROOT}", name=title),
            rrb.Grid(*series, grid_columns=2),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def welcome_blueprint() -> rrb.Blueprint:
    """Idle layout shown before any run is launched."""
    return rrb.Blueprint(
        rrb.TextDocumentView(origin=f"/{README}", name="Dashboard"),
        collapse_panels=True,
    )
