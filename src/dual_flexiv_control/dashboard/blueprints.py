"""Rerun blueprints: the metric layout shown for each experiment phase.

The blueprint is the right-hand side of the dashboard — a declarative layout of
Rerun views over a fixed set of entity paths. The placeholder emitter in
:mod:`~.runner` (and, later, the real eval/collection run) logs to exactly these
paths, so adding a metric is: log it under one of the roots below, then drop a
view here.

Entity-path scheme (kept in one place so emitter and layout never drift):

* ``eef/{side}/{tip,path}``  — end-effector point + trailing path, per arm (3D).
* ``metrics/eef_{side}/{x,y,z}`` — per-axis EEF position scalars (time series).
* ``metrics/eval/*``        — eval episode scalars (reward, success, length …).
* ``metrics/collection/*``  — collection progress scalars.
* ``events``                — text-log of run lifecycle events.
* ``readme``                — a markdown panel describing the active task.

``eval`` centers on 3D EEF-position tracking (a Spatial3DView of both arms) with
the per-axis series beside it; ``collection`` is a progress placeholder for now.
"""

from __future__ import annotations

import rerun.blueprint as rrb

SIDES = ("left", "right")

# -- entity paths (the contract with the emitter) ---------------------------

EEF_ROOT = "eef"
METRICS_ROOT = "metrics"
EVAL_METRICS = f"{METRICS_ROOT}/eval"
COLLECTION_METRICS = f"{METRICS_ROOT}/collection"
EVENTS = "events"
README = "readme"


def eef_tip_path(side: str) -> str:
    """3D point of one arm's current end-effector position."""
    return f"{EEF_ROOT}/{side}/tip"


def eef_path_path(side: str) -> str:
    """Trailing 3D line-strip of one arm's recent end-effector path."""
    return f"{EEF_ROOT}/{side}/path"


def eef_axis_root(side: str) -> str:
    """Time-series root for one arm's per-axis (x/y/z) EEF position scalars."""
    return f"{METRICS_ROOT}/eef_{side}"


# -- layouts ----------------------------------------------------------------


def for_phase(phase: str, task_name: str | None = None) -> rrb.Blueprint:
    """The blueprint for an experiment ``phase`` ("eval" | "collection")."""
    if phase == "eval":
        return _eval_blueprint(task_name)
    if phase == "collection":
        return _collection_blueprint(task_name)
    raise ValueError(f"unknown phase {phase!r} (expected 'eval' or 'collection')")


def _eval_blueprint(task_name: str | None) -> rrb.Blueprint:
    """Eval: 3D end-effector tracking front-and-center, per-axis series beside it."""
    title = "End-effector tracking (3D)"
    if task_name:
        title = f"{title} — {task_name}"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"/{EEF_ROOT}", name=title),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin=f"/{eef_axis_root('left')}", name="Left EEF position (x/y/z)"
                ),
                rrb.TimeSeriesView(
                    origin=f"/{eef_axis_root('right')}", name="Right EEF position (x/y/z)"
                ),
                rrb.TimeSeriesView(
                    origin=f"/{EVAL_METRICS}", name="Episode metrics (placeholder)"
                ),
                rrb.TextLogView(origin=f"/{EVENTS}", name="Events"),
                row_shares=[3, 3, 3, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def _collection_blueprint(task_name: str | None) -> rrb.Blueprint:
    """Collection: teleop-demo progress placeholder + the task description."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin=f"/{COLLECTION_METRICS}",
                    name="Collection progress (placeholder)",
                ),
                rrb.TextLogView(origin=f"/{EVENTS}", name="Events"),
                row_shares=[3, 1],
            ),
            rrb.TextDocumentView(origin=f"/{README}", name="Task"),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def welcome_blueprint() -> rrb.Blueprint:
    """Idle layout shown before any run is launched."""
    return rrb.Blueprint(
        rrb.TextDocumentView(origin=f"/{README}", name="Dashboard"),
        collapse_panels=True,
    )
