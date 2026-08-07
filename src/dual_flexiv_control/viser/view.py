"""Viser GUI and scene driven exclusively by typed stream samples."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..visualization import schema
from .scene import RobotScene

HISTORY_LENGTH = 1024
COLORS = (
    "#4ea1ff", "#ff8c50", "#55c97a", "#d783ff", "#ffd15c",
    "#49d6cf", "#ff6685", "#aab5c4",
)


@dataclass
class SeriesBuffer:
    maxlen: int = HISTORY_LENGTH
    times: deque[float] = field(init=False)
    values: deque[np.ndarray] = field(init=False)
    dim: int | None = None

    def __post_init__(self) -> None:
        self.times, self.values = deque(maxlen=self.maxlen), deque(maxlen=self.maxlen)

    def append(self, timestamp: float, values) -> bool:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        changed = self.dim is not None and self.dim != vector.size
        if changed:
            self.times.clear()
            self.values.clear()
        self.dim = int(vector.size)
        self.times.append(float(timestamp))
        self.values.append(vector.copy())
        return changed

    def clear(self) -> None:
        self.times.clear()
        self.values.clear()
        self.dim = None

    def data(self):
        x = np.asarray(self.times, dtype=np.float64)
        if not self.values:
            return (x, np.empty(0))
        matrix = np.stack(self.values)
        return (x, *(matrix[:, i] for i in range(matrix.shape[1])))


def component_labels(route: schema.StreamRoute, dim: int):
    if route.signal in ("q", "dq", "tau", "tau_ext", "leader") or "torque" in route.signal:
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


class ViserLiveView:
    """Own Viser handles; the consumer supplies newest samples at 3 Hz."""

    def __init__(self, server, rate_hz: float, factr_urdfs=None, cameras=None) -> None:
        import viser.uplot

        self.server = server
        self.uplot = viser.uplot
        self.rate_hz = rate_hz
        self.scene = RobotScene(server, factr_urdfs)
        self.cameras = cameras or {}
        for name, cam in self.cameras.items():
            self.scene.add_camera_frustum(name, cam)
        self.attached_run = None
        self._origin_t_ns = None
        self._events = deque(maxlen=64)
        self._buffers = {}
        self._plots = {}
        self._plot_folders = {}
        self._phase_key = None

        server.gui.configure_theme(
            control_layout="fixed", control_width="large", dark_mode=True,
            show_logo=False, show_share_button=False, brand_color=(80, 160, 255),
        )
        with server.gui.add_folder("Session", expand_by_default=True, order=0.0):
            self._status = server.gui.add_markdown("")
            self._event_log = server.gui.add_markdown("")
        self._follower_folder = server.gui.add_folder(
            "Follower proprioception", expand_by_default=True, order=1.0
        )
        self._leader_folder = server.gui.add_folder(
            "FACTR leaders", expand_by_default=False, order=2.0
        )
        self._telemetry_folder = server.gui.add_folder(
            "FACTR full telemetry", expand_by_default=False, order=3.0
        )
        self._policy_folder = server.gui.add_folder(
            "Policy communication", expand_by_default=False, order=4.0
        )
        self.add_event("Viser stream consumer started; waiting for producers")
        self.update_session(None)

    def add_event(self, message: str) -> None:
        self._events.append(f"- `{time.strftime('%H:%M:%S')}` {message}")
        self._event_log.content = "### Events\n\n" + "\n".join(self._events)

    def attach_run(self, run_id: str) -> None:
        if self.attached_run == run_id:
            return
        self.attached_run = run_id
        self._origin_t_ns = None
        for buffer in self._buffers.values():
            buffer.clear()
        self.add_event(f"attached shared-memory run `{run_id}`")

    def detach_run(self) -> None:
        if self.attached_run is not None:
            self.add_event(f"stream run `{self.attached_run}` detached")
        self.attached_run = None
        self.scene.update_followers({})
        self.scene.update_commands({})
        self.scene.update_factr({})
        self.scene.clear_targets()
        self.scene.update_depth(None, None)

    def update_session(self, state: dict | None) -> None:
        state = state or {}
        phase = str(state.get("state") or "waiting")
        task = state.get("task")
        key = (phase, task, state.get("run_seq"), state.get("pending"))
        if self._phase_key is not None and key[:2] != self._phase_key[:2]:
            self.add_event(f"session mode: {phase}" + (f" · {task}" if task else ""))
        self._phase_key = key
        pending = state.get("pending")
        pending_text = f"  \n**Pending:** `{pending}`" if pending else ""
        down = [*(state.get("arms_down") or ()), *(state.get("cameras_down") or ())]
        health = ", ".join(down) if down else "all configured producers healthy"
        self._status.content = (
            "# dual-flexiv live view\n\n"
            f"**Mode:** `{phase}`" + (f" · **{task}**" if task else "") + "  \n"
            f"**Stream run:** `{self.attached_run or 'waiting'}`  \n"
            f"**Health:** {health}{pending_text}\n\n"
            f"Newest-value polling at **{self.rate_hz:g} Hz**. This viewer is lossy "
            "and non-authoritative; producer rings retain native-rate history."
        )
        if phase not in ("eval", "skill", "saving"):
            self.scene.clear_targets()

    def update_stream(
        self, name: str, route: schema.StreamRoute, values: np.ndarray, t_ns: int
    ) -> None:
        if route.kind in ("horizon", "policy", "camera"):
            return
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if route.kind == "proprio" and route.signal == "eef":
            vector = vector[:3]
        elif route.kind == "factr" and route.signal == "leader":
            self._update_plot(name + "/q", route, vector[:-1], t_ns, title_signal="q")
            self._update_plot(name + "/grip", route, vector[-1:], t_ns, title_signal="grip")
            return
        self._update_plot(name, route, vector, t_ns)

    def _update_plot(self, key, route, values, t_ns, title_signal=None) -> None:
        if self._origin_t_ns is None:
            self._origin_t_ns = int(t_ns)
        timestamp = (int(t_ns) - self._origin_t_ns) / 1e9
        buffer = self._buffers.setdefault(key, SeriesBuffer())
        changed = buffer.append(timestamp, values)
        if changed and key in self._plots:
            self._plots.pop(key).remove()
        handle = self._plots.get(key)
        if handle is None:
            handle = self._add_plot(key, route, buffer, title_signal)
            self._plots[key] = handle
        else:
            handle.data = buffer.data()
            handle.visible = True

    def _add_plot(self, key, route, buffer, title_signal=None):
        assert buffer.dim is not None
        if route.kind == "proprio":
            folder = self._follower_folder
        elif route.signal in ("leader", "raw", "model_q_rad"):
            folder = self._leader_folder
        elif route.kind == "policy":
            folder = self._policy_folder
        else:
            folder = self._telemetry_folder
        signal = title_signal or route.signal
        title_route = schema.StreamRoute(route.kind, route.side, signal)
        labels = component_labels(title_route, buffer.dim)
        title = schema.FACTR_TITLES.get(
            signal,
            schema.PROPRIO_TITLES.get(signal, signal.replace("_", " ")),
        )
        with folder:
            return self.server.gui.add_uplot(
                data=buffer.data(),
                series=(
                    self.uplot.Series(label="time"),
                    *(self.uplot.Series(
                        label=label, stroke=COLORS[i % len(COLORS)], width=1.5
                    ) for i, label in enumerate(labels)),
                ),
                title=f"{(route.side or 'policy').title()} · {title}",
                scales={"x": self.uplot.Scale(time=False, auto=True),
                        "y": self.uplot.Scale(auto=True)},
                legend=self.uplot.Legend(show=True, live=True),
                aspect=2.4,
            )

    def update_policy_series(self, name: str, value: float, t_ns: int) -> None:
        route = schema.StreamRoute("policy", None, name)
        self._update_plot(f"policy/{name}", route, np.asarray([value]), t_ns)

    def apply_command(self, command: dict) -> None:
        kind = command.get("kind")
        if kind == "event":
            self.add_event(str(command.get("message", "")))
        elif kind == "show_calibration":
            self.scene.show_calibration(str(command["side"]), command["q"])
        elif kind == "clear_calibration":
            self.scene.clear_calibration()
        elif kind == "reset":
            self.scene.clear_calibration()
            self.scene.clear_targets()
            self.add_event("viewer reset")

    def close(self) -> None:
        self.server.stop()
