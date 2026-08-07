"""Viser GUI and scene driven exclusively by typed stream samples."""

from __future__ import annotations

import time
from collections import deque

from .client import HIDDEN_PANEL_LABEL
from .scene import RobotScene


class ViserLiveView:
    """Own Viser handles; the consumer supplies newest samples at 3 Hz."""

    def __init__(self, server, rate_hz: float, factr_urdfs=None, cameras=None) -> None:
        self.server = server
        self.rate_hz = rate_hz
        self.scene = RobotScene(server, factr_urdfs)
        self.cameras = cameras or {}
        for name, cam in self.cameras.items():
            self.scene.add_camera_frustum(name, cam)
        self.attached_run = None
        self._events = deque(maxlen=64)
        self._phase_key = None

        server.gui.configure_theme(
            control_layout="fixed", control_width="small", dark_mode=True,
            show_logo=False, show_share_button=False, brand_color=(80, 160, 255),
        )
        server.gui.set_panel_label(HIDDEN_PANEL_LABEL)
        with server.gui.add_folder("Session", expand_by_default=False, order=-1.0):
            self._status = server.gui.add_markdown("")
            self._event_log = server.gui.add_markdown("")
        self.add_event("Viser 3D stream consumer started; waiting for producers")
        self.update_session(None)

    def add_event(self, message: str) -> None:
        self._events.append(f"- `{time.strftime('%H:%M:%S')}` {message}")
        self._event_log.content = "### Events\n\n" + "\n".join(self._events)

    def attach_run(self, run_id: str) -> None:
        if self.attached_run == run_id:
            return
        self.attached_run = run_id
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
            f"**Mode:** `{phase}`" + (f" · **{task}**" if task else "") + "  \n"
            f"**Stream run:** `{self.attached_run or 'waiting'}`  \n"
            f"**Health:** {health}{pending_text}\n\n"
            f"Newest-value polling at **{self.rate_hz:g} Hz**. This viewer is lossy "
            "and non-authoritative; producer rings retain native-rate history."
        )
        if phase not in ("eval", "skill", "saving"):
            self.scene.clear_targets()

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
