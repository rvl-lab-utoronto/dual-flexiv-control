"""Plot definitions, incremental buffers, and the Plotly Dash application."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..interfaces.factr.interface import TELEMETRY_SCALAR_FIELDS
from ..interfaces.factr.interface import TELEMETRY_VECTOR_FIELDS
from ..visualization import schema

HISTORY_LENGTH = 1024
# Increment when a live Dash process must be rebuilt rather than merely picking
# up an asset file.  The dashboard uses this to roll only the plot consumer.
PLOT_VIEW_REVISION = 1
PLOT_HEIGHT_PX = 270
COLORS = (
    "#4ea1ff", "#ff8c50", "#55c97a", "#d783ff", "#ffd15c",
    "#49d6cf", "#ff6685", "#aab5c4",
)
FOLLOWER_GRID_SIGNALS = ("eef", "q", "dq", "tau", "tau_ext", "wrench", "eef_vel")
LEADER_GRID_SIGNALS = tuple(schema.FACTR_TITLES)
POLICY_GRID_SIGNALS = ("packets", "latency_ms")
POLICY_COUNTER_LABELS = ("sent", "received", "errors", "in flight")
PROPRIO_DIMS = {
    "q": 7, "dq": 7, "tau": 7, "tau_ext": 7,
    "wrench": 6, "eef": 3, "eef_vel": 6,
}


@dataclass(frozen=True, slots=True)
class PlotSpec:
    key: str
    stream_name: str
    tab: str
    route: schema.StreamRoute
    dim: int
    title: str
    order: int

    @property
    def graph_id(self) -> str:
        return "plot-" + self.key.replace("/", "-").replace("_", "-")


def component_labels(route: schema.StreamRoute, dim: int) -> tuple[str, ...]:
    if route.kind == "policy" and route.signal == "packets":
        labels = POLICY_COUNTER_LABELS
    elif route.kind == "policy" and route.signal == "latency_ms":
        labels = ("ms",)
    elif (
        route.signal in ("q", "dq", "tau", "tau_ext", "leader")
        or "torque" in route.signal
    ):
        labels = tuple(f"J{i + 1}" for i in range(dim))
    elif route.signal == "wrench":
        labels = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
    elif route.signal == "eef":
        labels = ("x", "y", "z")
    elif route.signal == "eef_vel":
        labels = ("vx", "vy", "vz", "wx", "wy", "wz")
    else:
        labels = ()
    return tuple(labels[i] if i < len(labels) else f"[{i}]" for i in range(dim))


def plot_grid_order(route: schema.StreamRoute, signal: str) -> int:
    """Stable row-major order: each signal is ``left | right``."""
    if route.kind == "proprio":
        rows = FOLLOWER_GRID_SIGNALS
    elif route.kind == "factr":
        rows = LEADER_GRID_SIGNALS
    else:
        rows = POLICY_GRID_SIGNALS
    try:
        row = rows.index(signal)
    except ValueError:
        row = len(rows)
    if route.kind == "policy":
        return row
    try:
        column = schema.SIDES.index(route.side)
    except ValueError:
        column = 0
    return 2 * row + column


def _plot_specs() -> tuple[PlotSpec, ...]:
    specs = []
    for signal in FOLLOWER_GRID_SIGNALS:
        for side in schema.SIDES:
            route = schema.StreamRoute("proprio", side, signal)
            specs.append(PlotSpec(
                key=f"follower/{side}/{signal}",
                stream_name=f"{side}/{signal}",
                tab="follower",
                route=route,
                dim=PROPRIO_DIMS[signal],
                title=f"{side.title()} · {schema.PROPRIO_TITLES[signal]}",
                order=plot_grid_order(route, signal),
            ))
    for signal in LEADER_GRID_SIGNALS:
        for side in schema.SIDES:
            route = schema.StreamRoute("factr", side, signal)
            if signal in ("q", "grip"):
                stream_name = f"factr/{side}"
                dim = 7 if signal == "q" else 1
            elif signal == "raw":
                stream_name, dim = f"factr/raw/{side}", 8
            elif signal in TELEMETRY_VECTOR_FIELDS:
                stream_name = f"factr/telemetry/{side}/{signal}"
                dim = 7
            elif signal in TELEMETRY_SCALAR_FIELDS:
                stream_name = f"factr/telemetry/{side}/{signal}"
                dim = 1
            else:  # guarded by FACTR_TITLES; fail loudly if contracts diverge
                raise ValueError(f"no Plotly stream mapping for FACTR signal {signal!r}")
            specs.append(PlotSpec(
                key=f"leader/{side}/{signal}",
                stream_name=stream_name,
                tab="leader",
                route=route,
                dim=dim,
                title=f"{side.title()} · {schema.FACTR_TITLES[signal]}",
                order=plot_grid_order(route, signal),
            ))
    policy_titles = {
        "packets": "Policy server · packets",
        "latency_ms": "Policy server · round-trip latency",
    }
    for signal, dim in (("packets", 4), ("latency_ms", 1)):
        route = schema.StreamRoute("policy", None, signal)
        specs.append(PlotSpec(
            key=f"policy/{signal}",
            stream_name="eval/policy_comm",
            tab="policy",
            route=route,
            dim=dim,
            title=policy_titles[signal],
            order=plot_grid_order(route, signal),
        ))
    return tuple(specs)


PLOT_SPECS = _plot_specs()
SPEC_BY_KEY = {spec.key: spec for spec in PLOT_SPECS}
SPECS_BY_STREAM: dict[str, tuple[PlotSpec, ...]] = {
    name: tuple(spec for spec in PLOT_SPECS if spec.stream_name == name)
    for name in {spec.stream_name for spec in PLOT_SPECS}
}


@dataclass
class _Series:
    points: deque[tuple[int, float, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LENGTH)
    )
    version: int = 0

    def append(self, timestamp: float, value: np.ndarray) -> None:
        self.version += 1
        self.points.append((self.version, float(timestamp), value.copy()))


class PlotStore:
    """Thread-safe, process-local bridge from stream polling to Dash callbacks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._series = {spec.key: _Series() for spec in PLOT_SPECS}
        self._epoch = 0
        self._run_id: str | None = None
        self._origin_t_ns: int | None = None

    def attach(self, run_id: str) -> None:
        with self._lock:
            if self._run_id == run_id:
                return
            self._reset(run_id)

    def detach(self) -> None:
        with self._lock:
            if self._run_id is not None:
                self._reset(None)

    def _reset(self, run_id: str | None) -> None:
        self._epoch += 1
        self._run_id = run_id
        self._origin_t_ns = None
        self._series = {spec.key: _Series() for spec in PLOT_SPECS}

    def append(self, key: str, value, t_ns: int) -> bool:
        spec = SPEC_BY_KEY[key]
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size != spec.dim or not np.isfinite(vector).all():
            return False
        with self._lock:
            if self._origin_t_ns is None:
                self._origin_t_ns = int(t_ns)
            timestamp = (int(t_ns) - self._origin_t_ns) / 1e9
            self._series[key].append(timestamp, vector)
        return True

    def snapshot(self, cursors: dict[str, int] | None = None) -> dict:
        cursors = cursors or {}
        with self._lock:
            series = {}
            for key, buffer in self._series.items():
                cursor = int(cursors.get(key, 0))
                fresh = [point for point in buffer.points if point[0] > cursor]
                if not fresh:
                    continue
                series[key] = {
                    "version": fresh[-1][0],
                    "x": [point[1] for point in fresh],
                    "y": np.stack([point[2] for point in fresh]),
                }
            return {
                "epoch": self._epoch,
                "run_id": self._run_id,
                "series": series,
            }


def empty_figure(spec: PlotSpec) -> dict:
    labels = component_labels(spec.route, spec.dim)
    return {
        "data": [
            {
                # One page owns 56 small plots. SVG line traces avoid exhausting
                # the browser's limited WebGL contexts; incremental extension
                # keeps their 1024-point windows inexpensive at 3 Hz.
                "type": "scatter",
                "mode": "lines",
                "name": label,
                "x": [],
                "y": [],
                "line": {"color": COLORS[i % len(COLORS)], "width": 1.5},
                "hovertemplate": "%{x:.2f}s · %{y:.4g}<extra>" + label + "</extra>",
            }
            for i, label in enumerate(labels)
        ],
        "layout": {
            "title": {"text": spec.title, "font": {"size": 14}, "x": 0.02},
            "template": "plotly_dark",
            "paper_bgcolor": "#111722",
            "plot_bgcolor": "#111722",
            "margin": {"l": 48, "r": 16, "t": 42, "b": 38},
            "height": 270,
            "hovermode": "x unified",
            "uirevision": spec.key,
            "legend": {
                "orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "right", "x": 1.0, "font": {"size": 10},
            },
            "xaxis": {"title": {"text": "elapsed (s)"}, "showgrid": True},
            "yaxis": {"showgrid": True, "zeroline": False},
        },
    }


def create_dash_app(store: PlotStore, rate_hz: float):
    """Build one single-process Dash app over the consumer-owned store."""
    from pathlib import Path

    from dash import Dash
    from dash import Input
    from dash import Output
    from dash import State
    from dash import dcc
    from dash import html
    from dash import no_update

    assets = Path(__file__).resolve().parent / "assets"
    app = Dash(
        __name__, assets_folder=str(assets), title="dual-flexiv live plots",
        update_title=None,
    )

    def grid(tab: str):
        return html.Div([
            dcc.Graph(
                id=spec.graph_id,
                figure=empty_figure(spec),
                responsive=True,
                style={"height": f"{PLOT_HEIGHT_PX}px"},
                config={"displaylogo": False, "scrollZoom": True},
                className="plot-card",
            )
            for spec in sorted(
                (candidate for candidate in PLOT_SPECS if candidate.tab == tab),
                key=lambda candidate: candidate.order,
            )
        ], className="plot-grid")

    interval_ms = max(1, round(1000.0 / float(rate_hz)))
    app.layout = html.Div([
        dcc.Store(id="plot-cursors", storage_type="memory"),
        dcc.Interval(id="plot-tick", interval=interval_ms, n_intervals=0),
        html.Div("waiting for streams", id="stream-status", className="stream-status"),
        dcc.Tabs(
            id="plot-tabs", value="leader", persistence=True,
            parent_className="plot-tabs-parent", className="plot-tabs",
            children=[
                dcc.Tab(
                    label="Leader", value="leader", children=grid("leader"),
                    className="plot-tab", selected_className="plot-tab selected",
                ),
                dcc.Tab(
                    label="Follower", value="follower", children=grid("follower"),
                    className="plot-tab", selected_className="plot-tab selected",
                ),
                dcc.Tab(
                    label="Policy", value="policy", children=grid("policy"),
                    className="plot-tab", selected_className="plot-tab selected",
                ),
            ],
        ),
    ], className="plot-app")

    figure_outputs = [Output(spec.graph_id, "figure") for spec in PLOT_SPECS]
    extend_outputs = [Output(spec.graph_id, "extendData") for spec in PLOT_SPECS]

    @app.callback(
        *figure_outputs,
        *extend_outputs,
        Output("plot-cursors", "data"),
        Output("stream-status", "children"),
        Input("plot-tick", "n_intervals"),
        State("plot-cursors", "data"),
    )
    def update_plots(_tick, cursor_state):
        cursor_state = cursor_state or {"epoch": None, "versions": {}}
        snapshot = store.snapshot(cursor_state.get("versions"))
        status = (
            f"stream {snapshot['run_id']} · {rate_hz:g} Hz · incremental"
            if snapshot["run_id"] else f"waiting for streams · {rate_hz:g} Hz"
        )
        if cursor_state.get("epoch") != snapshot["epoch"]:
            figures = [empty_figure(spec) for spec in PLOT_SPECS]
            return [
                *figures,
                *([no_update] * len(PLOT_SPECS)),
                {"epoch": snapshot["epoch"], "versions": {}},
                status,
            ]

        versions = dict(cursor_state.get("versions") or {})
        extensions = []
        for spec in PLOT_SPECS:
            fresh = snapshot["series"].get(spec.key)
            if fresh is None:
                extensions.append(no_update)
                continue
            matrix = fresh["y"]
            x_values = fresh["x"]
            extensions.append([
                {
                    "x": [x_values for _ in range(spec.dim)],
                    "y": [matrix[:, i].tolist() for i in range(spec.dim)],
                },
                list(range(spec.dim)),
                HISTORY_LENGTH,
            ])
            versions[spec.key] = fresh["version"]
        return [
            *([no_update] * len(PLOT_SPECS)),
            *extensions,
            {"epoch": snapshot["epoch"], "versions": versions},
            status,
        ]

    return app
