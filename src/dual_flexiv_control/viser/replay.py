"""Viser episode replay: 3D state/action, cameras, plots, and a scrubber."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .scene import RobotScene

log = logging.getLogger(__name__)

DEFAULT_REPLAY_PORT = 9093
_IMAGE_PREFIX = "observation.images."


def _state_q_index(state_names: list[str]) -> dict[str, list[int]]:
    q_index = {}
    for i, name in enumerate(state_names):
        parts = str(name).split(".")
        if len(parts) >= 3 and parts[1] == "q":
            q_index.setdefault(parts[0], []).append(i)
    return q_index


def _action_layout(action_names: list[str]) -> dict[str, dict]:
    blocks = {}
    for i, name in enumerate(action_names):
        parts = str(name).split(".")
        side = parts[0]
        block = blocks.setdefault(side, {"field": None, "cmd": [], "gripper": None})
        if len(parts) == 2 and parts[1] == "gripper":
            block["gripper"] = i
        elif len(parts) >= 3:
            block["field"] = parts[1]
            block["cmd"].append(i)
    return blocks


def _names(feature) -> list[str]:
    names = (feature or {}).get("names")
    return list(names) if isinstance(names, (list, tuple)) else []


def _to_hwc_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if float(array.max(initial=0.0)) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def read_episode(ds_info, episode_index: int):
    """Read one local LeRobot episode using its stored column-name layout."""
    from ..dashboard.storage import _hub_offline

    with _hub_offline():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(ds_info.repo_id, root=ds_info.path)
        row = dataset.meta.episodes[int(episode_index)]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        features = dataset.meta.features
        q_index = _state_q_index(_names(features.get("observation.state")))
        action_blocks = _action_layout(_names(features.get("action")))
        camera_keys = [key for key in features if key.startswith(_IMAGE_PREFIX)]
        camera_names = [key[len(_IMAGE_PREFIX):] for key in camera_keys]
        state_dim = int(np.asarray(dataset[lo]["observation.state"]).size)
        pose_sides = [side for side, idx in q_index.items() if idx and max(idx) < state_dim]

        frames = []
        for i in range(lo, hi):
            item = dataset[i]
            state = np.asarray(item["observation.state"], dtype=np.float64).ravel()
            action = np.asarray(item["action"], dtype=np.float64).ravel()
            real_q, ghost_q, blocks = {}, {}, {}
            for side in pose_sides:
                real_q[side] = state[q_index[side]]
            for side, block in action_blocks.items():
                if block["cmd"] and max(block["cmd"]) < action.size:
                    command = action[block["cmd"]]
                    grip = (
                        float(action[block["gripper"]])
                        if block["gripper"] is not None and block["gripper"] < action.size
                        else 0.0
                    )
                    blocks[side] = {"cmd": command.tolist(), "gripper": grip}
                    if block["field"] == "q_d" and side in real_q:
                        ghost_q[side] = command
            images = {
                name: _to_hwc_uint8(item[key])
                for name, key in zip(camera_names, camera_keys)
            }
            frames.append({
                "t": float(np.asarray(item["timestamp"]).item()),
                "real_q": real_q,
                "ghost_q": ghost_q,
                "images": images,
                "action": blocks,
            })
    return frames, camera_names


class ReplayView:
    def __init__(self, server) -> None:
        self.server = server
        self.frames = []
        self.scene = None
        self.slider = None
        self.playing = False
        self._generation = 0
        self._lock = threading.RLock()

    def load(self, frames, camera_names) -> None:
        with self._lock:
            self.playing = False
            self._generation += 1
            self.frames = frames
            self.server.scene.reset()
            self.server.gui.reset()
            self.server.scene.set_up_direction("+z")
            self.scene = RobotScene(self.server)
            self.server.gui.configure_theme(
                control_layout="fixed", control_width="large", dark_mode=True,
                show_logo=False, show_share_button=False,
            )
            with self.server.gui.add_folder("Replay", expand_by_default=True):
                duration = frames[-1]["t"] - frames[0]["t"] if frames else 0.0
                self._status = self.server.gui.add_markdown(
                    f"# Episode replay\n\n{len(frames)} frames · {duration:.2f} s"
                )
                self.slider = self.server.gui.add_slider(
                    "Frame", min=0, max=max(0, len(frames) - 1), step=1,
                    initial_value=0, disabled=not frames,
                )
                play = self.server.gui.add_button("▶ Play", disabled=not frames)
                pause = self.server.gui.add_button("⏸ Pause", disabled=not frames)

            @self.slider.on_update
            def _on_frame(event):
                self.show(int(event.target.value))

            @play.on_click
            def _on_play(_event):
                self.playing = True
                generation = self._generation
                threading.Thread(
                    target=self._play, args=(generation,),
                    name="dfc-viser-replay", daemon=True,
                ).start()

            @pause.on_click
            def _on_pause(_event):
                self.playing = False

            self._image_handles = {}
            for index, name in enumerate(camera_names):
                if not frames or name not in frames[0]["images"]:
                    continue
                image = frames[0]["images"][name]
                aspect = float(image.shape[1]) / float(image.shape[0])
                self._image_handles[name] = self.server.scene.add_camera_frustum(
                    f"/replay/camera/{name}", fov=np.deg2rad(55.0), aspect=aspect,
                    scale=0.7, image=image, color=(180, 200, 220),
                    position=(1.3, (index - (len(camera_names) - 1) / 2) * 0.8, 1.2),
                    wxyz=(0.7071068, 0.0, 0.7071068, 0.0),
                )
                self.server.scene.add_label(
                    f"/replay/camera/{name}/label", name,
                    position=(0.0, 0.0, 0.05),
                )
            self._add_plots()
            if frames:
                self.show(0)

    def _add_plots(self) -> None:
        if not self.frames:
            return
        import viser.uplot

        times = np.asarray([frame["t"] for frame in self.frames], dtype=float)
        times -= times[0]
        with self.server.gui.add_folder("Recorded state and action", expand_by_default=True):
            for side in ("left", "right"):
                q_rows = [frame["real_q"].get(side) for frame in self.frames]
                if q_rows and all(row is not None for row in q_rows):
                    matrix = np.stack(q_rows)
                    self.server.gui.add_uplot(
                        data=(times, *(matrix[:, i] for i in range(matrix.shape[1]))),
                        series=(viser.uplot.Series(label="time"), *(
                            viser.uplot.Series(label=f"J{i + 1}")
                            for i in range(matrix.shape[1])
                        )),
                        title=f"{side.title()} recorded q", aspect=2.4,
                    )
                commands = [frame["action"].get(side, {}).get("cmd") for frame in self.frames]
                if commands and all(row is not None for row in commands):
                    matrix = np.asarray(commands, dtype=float)
                    self.server.gui.add_uplot(
                        data=(times, *(matrix[:, i] for i in range(matrix.shape[1]))),
                        series=(viser.uplot.Series(label="time"), *(
                            viser.uplot.Series(label=f"[{i}]")
                            for i in range(matrix.shape[1])
                        )),
                        title=f"{side.title()} action", aspect=2.4,
                    )
                grippers = [
                    frame["action"].get(side, {}).get("gripper")
                    for frame in self.frames
                ]
                if grippers and all(value is not None for value in grippers):
                    self.server.gui.add_uplot(
                        data=(times, np.asarray(grippers, dtype=float)),
                        series=(
                            viser.uplot.Series(label="time"),
                            viser.uplot.Series(label="gripper"),
                        ),
                        title=f"{side.title()} gripper action", aspect=2.4,
                    )

    def show(self, index: int) -> None:
        with self._lock:
            if not self.frames or self.scene is None:
                return
            index = max(0, min(int(index), len(self.frames) - 1))
            frame = self.frames[index]
            self.scene.update_followers(frame["real_q"])
            self.scene.update_commands(frame["ghost_q"])
            for name, image in frame["images"].items():
                if name in self._image_handles:
                    self._image_handles[name].image = image
            self._status.content = (
                f"# Episode replay\n\nFrame **{index + 1}/{len(self.frames)}** "
                f"· t = **{frame['t'] - self.frames[0]['t']:.3f} s**\n\n"
                "Solid = recorded state · translucent = joint-position action target."
            )

    def _play(self, generation: int) -> None:
        while self.playing and generation == self._generation and self.frames:
            index = int(self.slider.value)
            if index >= len(self.frames) - 1:
                self.playing = False
                return
            delay = max(0.001, self.frames[index + 1]["t"] - self.frames[index]["t"])
            time.sleep(delay)
            if self.playing and generation == self._generation:
                self.slider.value = index + 1
                self.show(index + 1)


@dataclass(frozen=True)
class ReplayViewer:
    port: int

    @property
    def web_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


_LOCK = threading.Lock()
_SERVER = None
_VIEW: ReplayView | None = None
_VIEWER: ReplayViewer | None = None


def start_replay_viewer(port: int | None = None) -> ReplayViewer:
    global _SERVER, _VIEW, _VIEWER
    with _LOCK:
        if _VIEWER is not None:
            return _VIEWER
        import viser

        port = int(port or os.environ.get("DFC_VISER_REPLAY_PORT", DEFAULT_REPLAY_PORT))
        _SERVER = viser.ViserServer(
            host="0.0.0.0", port=port, label="dual-flexiv episode replay", verbose=False
        )
        _VIEW = ReplayView(_SERVER)
        _VIEWER = ReplayViewer(port)
        return _VIEWER


def log_episode(ds_info, episode_index: int) -> int:
    frames, camera_names = read_episode(ds_info, episode_index)
    viewer = start_replay_viewer()
    del viewer
    _VIEW.load(frames, camera_names)
    return len(frames)


def reset() -> None:
    global _SERVER, _VIEW, _VIEWER
    with _LOCK:
        if _VIEW is not None:
            _VIEW.playing = False
        if _SERVER is not None:
            _SERVER.stop()
        _SERVER = _VIEW = _VIEWER = None
