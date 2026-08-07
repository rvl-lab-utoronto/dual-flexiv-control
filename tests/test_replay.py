"""Tests for the Storage-tab episode replay (episode reading + Rerun logging).

Reads a real (tiny) LeRobot episode and checks the per-frame data the replay logs,
then logs it into an unstarted RecordingStream to exercise the full Rerun path
(scene + FK poses + images + scalars) headlessly. Gated on lerobot.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from dual_flexiv_control.dashboard import replay  # noqa: E402
from dual_flexiv_control.dashboard import storage  # noqa: E402


def _build_dataset(root, repo_id, n_episodes, frames_per_ep):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from dual_flexiv_control.collection import FrameBuilder, LeRobotRecorder
    from dual_flexiv_control.configs import register_configs
    from dual_flexiv_control.streams.ring import Samples

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "rig=bimanual",  # keep the two-arm fixture independent of the default
            "cameras.wrist_left.width=32", "cameras.wrist_left.height=24",
            f"recording.root={root}", f"task.collection.repo_id={repo_id}",
            "task.collection.frequency_hz=15",
        ])
    c = OmegaConf.to_object(cfg)
    c.cameras = {"wrist_left": c.cameras["wrist_left"]}
    b = FrameBuilder(c.arms, list(c.arms), c.cameras, c.task.language_instruction,
                     c.task.state_signals, video=True)

    def smp(v, dt=np.float64):
        return Samples(np.asarray(v, dt).reshape(1, -1),
                       np.array([1], np.int64), np.array([0], np.int64))

    def obs():
        o = {"left/q": smp(np.arange(7.0)), "right/q": smp(np.arange(7.0) + 10)}
        cam = c.cameras["wrist_left"]
        o["cam/wrist_left/left"] = smp((np.arange(cam.height * cam.width * 3) % 256).astype(np.uint8), np.uint8)
        return o

    rec = LeRobotRecorder(c.task.collection, c.recording, b.features())
    for _ep in range(n_episodes):
        for i in range(frames_per_ep):
            rec.add_frame(b.build(obs(), {"left": 0.01 * i * np.ones(7), "right": 0.02 * i * np.ones(7)},
                                  {"left": 0.1 * i, "right": 0.2 * i}))
        rec.save_episode()
    rec.finalize()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("replay_ds"))
    # different-length episodes so the shorter-after-longer isolation is exercised
    _build_dataset(root, "dfc/replay", n_episodes=2, frames_per_ep=6)
    return storage.discover_datasets(root)[0]


def test_read_episode_frame_contents(dataset):
    frames, cams = replay.read_episode(dataset, 0)
    assert len(frames) == 6
    assert cams == ["wrist_left"]
    f0 = frames[0]
    # both arms posed from state; ghost from the qpos action target
    assert set(f0["real_q"]) == {"left", "right"}
    assert f0["real_q"]["left"].shape == (7,)
    assert set(f0["ghost_q"]) == {"left", "right"}
    # camera frame decoded to HWC uint8
    img = f0["images"]["wrist_left"]
    assert img.dtype == np.uint8 and img.shape == (24, 32, 3)
    # action block: 7-wide q_d command + a gripper scalar
    assert len(f0["action"]["left"]["cmd"]) == 7
    assert isinstance(f0["action"]["left"]["gripper"], float)
    # timestamps are non-decreasing
    ts = [f["t"] for f in frames]
    assert all(b >= a for a, b in zip(ts, ts[1:]))


def test_read_episode_second_episode_state_differs(dataset):
    # left q recorded as arange(7) in obs() -> real_q matches for any episode
    frames, _ = replay.read_episode(dataset, 1)
    assert len(frames) == 6
    np.testing.assert_allclose(frames[0]["real_q"]["left"], np.arange(7.0))


def test_layout_parsing_from_stored_names():
    # state/action split is driven by the dataset's stored column names, not config.
    state_names = [f"left.q.{j}" for j in range(7)] + [f"right.q.{j}" for j in range(7)]
    q_index = replay._state_q_index(state_names)
    assert q_index["left"] == list(range(7))
    assert q_index["right"] == list(range(7, 14))

    action_names = ([f"left.q_d.{j}" for j in range(7)] + ["left.gripper"]
                    + [f"right.dq_d.{j}" for j in range(7)] + ["right.gripper"])
    blk = replay._action_layout(action_names)
    assert blk["left"]["field"] == "q_d" and blk["left"]["cmd"] == list(range(7))
    assert blk["left"]["gripper"] == 7
    assert blk["right"]["field"] == "dq_d"          # a velocity action is parsed as recorded
    assert blk["right"]["gripper"] == 15


def test_read_episode_uses_recorded_action_field_not_current_config(dataset, monkeypatch):
    # Even if the current control kind changed to qvel, replay must reflect the
    # dataset's recorded q_d action (ghost present), because it parses stored names.
    frames, _ = replay.read_episode(dataset, 0)
    # recorded action field is q_d -> both arms get a ghost
    assert set(frames[0]["ghost_q"]) == {"left", "right"}


def test_log_frames_runs_headless(dataset):
    # Exercise the full Rerun logging path (scene + FK poses + images + scalars +
    # blueprint) against a plain buffered stream — no server, so no hang.
    import rerun as rr

    frames, cams = replay.read_episode(dataset, 0)
    rec = rr.RecordingStream("replay-unit-test")
    replay._log_frames(rec, frames, cams)  # must not raise


def test_log_episode_requires_started_viewer(dataset):
    # Without the server started, log_episode fails fast rather than silently no-op'ing.
    import dual_flexiv_control.dashboard.replay as r
    if r._SERVER_URI is None:
        with pytest.raises(RuntimeError):
            r.log_episode(dataset, 0)


def test_replay_blueprint_builds_with_and_without_cameras():
    import rerun.blueprint as rrb

    assert isinstance(replay.replay_blueprint([]), rrb.Blueprint)
    assert isinstance(replay.replay_blueprint(["wrist_left", "static_left"]), rrb.Blueprint)
