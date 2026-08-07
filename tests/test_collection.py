"""Hardware-free tests for the teleoperated collection loop.

The FrameBuilder (schema + frame packing) is pure. The loop is exercised with a
fake Brain and a fake Recorder, so no shared memory, no FACTR server, no arms,
and no ``lerobot`` dependency are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.collection import CollectionLoop
from dual_flexiv_control.collection import FrameBuilder
from dual_flexiv_control.collection import keyboard
from dual_flexiv_control.configs import register_configs
from dual_flexiv_control.streams.ring import Samples


def _config(*overrides: str):
    # Pin the full two-arm setup explicitly so this fixture is independent of the
    # shipped default.
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=["rig=bimanual", *overrides])
    return OmegaConf.to_object(cfg)


def _samples(vec, dtype=np.float64) -> Samples:
    data = np.asarray(vec, dtype=dtype).reshape(1, -1)
    return Samples(data=data, t_ns=np.array([1], np.int64), seq=np.array([0], np.int64))


def _empty_samples(dtype=np.float64) -> Samples:
    """A never-produced stream: n == 0, so ``newest`` is None."""
    return Samples(
        data=np.empty((0, 0), dtype=dtype),
        t_ns=np.empty((0,), np.int64),
        seq=np.empty((0,), np.int64),
    )


def _builder(cfg, teleop_sides=("left",)):
    return FrameBuilder(
        cfg.arms, list(teleop_sides), cfg.cameras,
        cfg.task.language_instruction, cfg.task.state_signals,
        video=cfg.recording.video,
    )


# --------------------------------------------------------------------------- #
# FrameBuilder: schema
# --------------------------------------------------------------------------- #


def test_features_state_and_action_dims():
    cfg = _config()
    b = _builder(cfg, teleop_sides=("left", "right"))
    feats = b.features()
    # state = both arms' q (7 each); action = both arms' q_d(7)+gripper(1)
    assert feats["observation.state"]["shape"] == (14,)
    assert feats["action"]["shape"] == (16,)
    assert feats["observation.state"]["dtype"] == "float32"
    assert b.state_names[:2] == ["left.q.0", "left.q.1"]
    assert "left.gripper" in b.action_names and "right.gripper" in b.action_names


def test_features_image_keys_per_camera_view():
    cfg = _config()
    b = _builder(cfg)
    feats = b.features()
    # wrist cams publish only `left` (single view -> bare key); static publishes
    # left+right (view-suffixed keys).
    assert "observation.images.wrist_left" in feats
    assert "observation.images.wrist_right" in feats
    assert "observation.images.static_left" in feats
    assert "observation.images.static_right" in feats
    static_left = feats["observation.images.static_left"]
    assert static_left["dtype"] == "video"
    assert static_left["shape"] == (720, 1280, 3)


def test_stream_names_cover_state_and_cameras():
    cfg = _config()
    b = _builder(cfg)
    names = set(b.stream_names)
    assert {"left/q", "right/q"} <= names
    assert {"cam/wrist_left/left", "cam/static/right"} <= names


# --------------------------------------------------------------------------- #
# FrameBuilder: per-tick assembly
# --------------------------------------------------------------------------- #


def test_actions_from_leaders_uses_ingestion_converted_pose_and_extracts_gripper():
    cfg = _config()
    b = _builder(cfg)
    conv = cfg.arms["left"].convention
    leader = np.linspace(0.1, 0.8, 8)  # dof+1: 7 joints + trailing gripper
    acts = b.actions_from_leaders({"left": leader}, {"left": conv})
    np.testing.assert_allclose(acts["q_d"]["left"], leader[:7])
    # uncalibrated convention (gripper_open/closed unset) -> raw radian value
    assert acts["gripper"]["left"] == pytest.approx(leader[-1])


def test_actions_from_leaders_normalizes_gripper_when_calibrated():
    # With gripper_open/closed set, the trailing gripper is mapped to a 0..1 fraction.
    cfg = _config("arms.left.convention.gripper_open=0.0",
                  "arms.left.convention.gripper_closed=2.0")
    b = _builder(cfg)
    conv = cfg.arms["left"].convention
    leader = np.concatenate([np.zeros(7), [1.0]])  # trailing gripper = 1.0 rad -> halfway
    acts = b.actions_from_leaders({"left": leader}, {"left": conv})
    assert acts["gripper"]["left"] == pytest.approx(0.5)
    # past the closed endpoint clips to 1.0 (not >1)
    leader[-1] = 3.0
    acts = b.actions_from_leaders({"left": leader}, {"left": conv})
    assert acts["gripper"]["left"] == pytest.approx(1.0)


def _full_observation(cfg, b):
    """A complete observation snapshot: q for both arms + every camera frame."""
    obs = {"left/q": _samples(np.zeros(7)), "right/q": _samples(np.ones(7))}
    for side in b.action_sides:
        obs[f"factr/{side}"] = _samples(np.zeros(8))
        obs[f"factr/raw/{side}"] = _samples(np.zeros(8))
    for name, cam in cfg.cameras.items():
        for view in cam.views:
            dim = cam.height * cam.width * (3 if view in ("left", "right") else 1)
            obs[f"cam/{name}/{view}"] = _samples(np.zeros(dim, np.uint8), dtype=np.uint8)
    return obs


def test_build_frame_shapes_and_task():
    cfg = _config()
    b = _builder(cfg)
    obs = _full_observation(cfg, b)
    frame = b.build(obs, {"left": np.arange(7.0)}, {"left": 0.5})
    assert frame is not None
    assert frame["observation.state"].shape == (14,)
    assert frame["action"].shape == (8,)  # left only teleop: q_d(7)+gripper(1)
    assert frame["action"][-1] == pytest.approx(0.5)
    assert frame["observation.factr.left"].shape == (8,)
    assert frame["observation.factr_raw.left"].shape == (8,)
    assert frame["task"] == cfg.task.language_instruction
    assert frame["observation.images.static_left"].shape == (720, 1280, 3)


def test_build_returns_none_when_camera_frame_missing():
    cfg = _config()
    b = _builder(cfg)
    obs = _full_observation(cfg, b)
    obs["cam/wrist_left/left"] = _empty_samples(np.uint8)  # camera hasn't produced yet
    assert b.build(obs, {"left": np.arange(7.0)}, {"left": 0.5}) is None


def test_build_returns_none_when_action_missing():
    cfg = _config()
    b = _builder(cfg)
    obs = _full_observation(cfg, b)
    assert b.build(obs, {}, {}) is None  # no q_d for teleop side 'left'


# --------------------------------------------------------------------------- #
# CollectionLoop: end-to-end with fakes
# --------------------------------------------------------------------------- #


class _FakeBrain:
    def __init__(self, cfg, builder):
        self._cfg = cfg
        self._builder = builder
        self.commands = []
        self.gripper_commands = []

    def factr_joint_positions(self):
        return {"left": np.linspace(0.1, 0.8, 8)}

    def observe(self):
        return _full_observation(self._cfg, self._builder)

    def command(self, side, setpoint):
        self.commands.append((side, np.asarray(setpoint)))

    def command_gripper(self, side, value):
        self.gripper_commands.append((side, float(value)))


class _FakeRecorder:
    def __init__(self):
        self.frames = 0
        self.pending = 0
        self.episodes = []
        self.discarded = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames += 1
        self.pending += 1

    def save_episode(self):
        self.episodes.append(self.pending)
        self.pending = 0

    def discard_episode(self):
        self.discarded += 1
        self.pending = 0

    def finalize(self):
        self.finalized = True


class _StopAfter:
    """Stop event that trips after N `is_set` checks (bounds the loop)."""

    def __init__(self, n):
        self._n = n
        self._forced = False

    def is_set(self):
        if self._forced:
            return True
        self._n -= 1
        return self._n < 0

    def set(self):
        self._forced = True


def _loop(cfg, brain, recorder, events=None, num_episodes=1, leaders=None):
    return CollectionLoop(
        brain, brain._builder, recorder,
        command_arms={"left": cfg.arms["left"]},
        conventions={"left": cfg.arms["left"].convention},
        leaders=leaders if leaders is not None else brain.factr_joint_positions,
        frequency_hz=1000.0, num_episodes=num_episodes, events=events,
    )


def test_loop_commands_arm_and_records_and_saves_on_stop():
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    loop = _loop(cfg, brain, rec)
    loop.run(_StopAfter(5))
    assert rec.frames >= 5
    assert brain.commands and brain.commands[0][0] == "left"
    assert brain.commands[0][1].shape == (14,)  # qpos: q_d(7)+dq_d(7)
    # single-episode fallback saves the accumulated frames on stop
    assert rec.episodes and sum(rec.episodes) == rec.frames
    assert loop.episodes_done == 1
    assert rec.finalized


def test_loop_posts_gripper_command_from_leader():
    """Each tick posts the leader's normalized gripper on the gripper mailbox, so the
    follower's gripper tracks the leader alongside the joint setpoints."""
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    loop = _loop(cfg, brain, rec)
    loop.run(_StopAfter(3))
    # leader sample is linspace(0.1, 0.8, 8): trailing gripper raw = 0.8, passed
    # through unchanged by an uncalibrated convention.
    assert brain.gripper_commands and brain.gripper_commands[0][0] == "left"
    assert brain.gripper_commands[0][1] == pytest.approx(0.8)


def test_loop_saves_and_discards_on_keyboard_events():
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    scripted = [[], [keyboard.END], [], [keyboard.DISCARD], [keyboard.STOP]]
    events = lambda: scripted.pop(0) if scripted else []  # noqa: E731
    loop = _loop(cfg, brain, rec, events=events, num_episodes=5)
    loop.run(_StopAfter(50))
    assert loop.episodes_done == 1        # one END saved an episode
    assert rec.episodes == [2]            # 2 frames accrued before END
    # DISCARD dropped an in-progress episode; the tick recorded before STOP is
    # also dropped (operator-paced runs never half-save on stop) -> 2 discards.
    assert rec.discarded == 2
    assert rec.finalized


def test_loop_never_commands_before_first_real_leader_sample():
    """A side with no leader data yet is recorded (zeros fill) but NEVER commanded:
    a fabricated all-zeros q_d is the straight-up home pose — the arm must stay
    parked in its first-setpoint wait until real teleop data arrives."""
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    loop = _loop(cfg, brain, rec, leaders=lambda: {})  # leaders never fresh
    loop.run(_StopAfter(5))
    assert brain.commands == []          # no setpoint ever posted
    assert brain.gripper_commands == []  # no gripper target either
    assert rec.frames >= 5               # frames still record (zero-filled action)


def test_loop_holds_last_real_target_through_a_dropout():
    """After real leader data has flowed, a dropout holds the last REAL target
    (commands keep riding it) instead of stalling or fabricating."""
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    reads = {"n": 0}

    def flaky_leaders():
        reads["n"] += 1
        # One real sample, then a dropout for the rest of the run.
        if reads["n"] == 1:
            return {"left": np.linspace(0.1, 0.8, 8)}
        return {}

    loop = _loop(cfg, brain, rec, leaders=flaky_leaders)
    loop.run(_StopAfter(4))
    assert len(brain.commands) >= 2  # kept commanding through the dropout
    held = brain.commands[0][1]
    for _, setpoint in brain.commands[1:]:
        np.testing.assert_allclose(setpoint, held)  # the held target, unchanged


def test_loop_stops_after_target_episode_count():
    cfg = _config()
    b = _builder(cfg)
    brain = _FakeBrain(cfg, b)
    rec = _FakeRecorder()
    scripted = [[keyboard.END], [keyboard.END]]
    events = lambda: scripted.pop(0) if scripted else []  # noqa: E731
    loop = _loop(cfg, brain, rec, events=events, num_episodes=2)
    loop.run(_StopAfter(1000))
    assert loop.episodes_done == 2  # reached num_episodes and exited
