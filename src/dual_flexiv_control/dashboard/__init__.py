"""Streamlit controls with a Viser stream viewer for the bimanual Flexiv setup.

A small Streamlit app whose left column drives experiments — pick a task from the
``conf/task`` group, then launch **collection** (teleop demos) or **eval** (policy
rollouts) — and whose right side embeds Viser for 3D plus Plotly Dash for live
plots. Both consume the system's shared-memory streams at 3 Hz.

Visualization is deliberately non-authoritative: Streamlit hosts controls,
while isolated ``ProcessNode`` consumers discover producer streams and own the
scene and plots respectively.

Layering (kept import-light so :mod:`~.tasks` works without Rerun/Streamlit):

* :mod:`~.tasks`      — enumerate launchable tasks from ``conf/task/*.yaml``.
* :mod:`~.runner`     — session launch/status facade (no telemetry reads).
* :mod:`~.app`        — the Streamlit page wiring it together.

The deprecated Rerun backend lives behind :mod:`dual_flexiv_control.rerun` and
is never imported or started by the dashboard.

Run it with the ``dfc-dashboard`` console script (``streamlit run`` under the
hood); see :mod:`~.launch`.
"""

from __future__ import annotations

__all__ = ["tasks", "runner", "cameras", "editor", "arms"]
