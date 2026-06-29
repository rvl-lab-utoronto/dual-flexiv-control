"""Rerun-backed experiment dashboard for the bimanual Flexiv setup.

A small Streamlit app whose left column drives experiments — pick a task from the
``conf/task`` group, then launch **collection** (teleop demos) or **eval** (policy
rollouts) — and whose right side embeds a live `Rerun <https://rerun.io>`_ web
viewer holding the metrics for the active run.

Rerun's viewer is a visualization layer: it cannot host the dropdown/launch
buttons itself, so Streamlit hosts the controls and serves the version-matched
Rerun web viewer (``serve_grpc`` + ``serve_web_viewer``) to embed alongside.

Layering (kept import-light so :mod:`~.tasks` works without Rerun/Streamlit):

* :mod:`~.tasks`      — enumerate launchable tasks from ``conf/task/*.yaml``.
* :mod:`~.blueprints` — the per-phase metric layouts (Rerun blueprints).
* :mod:`~.viewer`     — start the Rerun gRPC + web-viewer servers once.
* :mod:`~.runner`     — stub launch + placeholder metric emitter (clean seam for
                        the real eval/collection run, which lands with control).
* :mod:`~.app`        — the Streamlit page wiring it together.

Run it with the ``dfc-dashboard`` console script (``streamlit run`` under the
hood); see :mod:`~.launch`.
"""

from __future__ import annotations

__all__ = ["tasks", "blueprints", "viewer", "runner", "cameras", "editor"]
