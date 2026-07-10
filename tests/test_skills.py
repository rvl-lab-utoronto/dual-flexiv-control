"""Teach-and-repeat skills: file format, episode extraction, and the replay loop.

The storage layer (save/load/discover/delete) and :class:`SkillLoop` run against
plain temp dirs and a scripted fake brain — no hardware, no processes. Teaching
from an episode is exercised against a real (tiny) LeRobot dataset, gated on
lerobot like ``test_replay``. The full skill *run* (daemon → EnterControl →
SkillNode → sim arm) is covered end-to-end in ``test_session.py``.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from dual_flexiv_control import skills
from dual_flexiv_control.configs import ArmCfg
from dual_flexiv_control.configs import ControlCfg
from dual_flexiv_control.skills import Skill
from dual_flexiv_control.skills import SkillLoop
from dual_flexiv_control.streams.ring import Samples


def _skill(name="demo", frames=4, dof=7, fps=15.0, sides=("left",)):
    q = {
        side: np.linspace(0.0, 0.3, frames)[:, None] * np.ones(dof) + i
        for i, side in enumerate(sides)
    }
    grip = {side: np.linspace(0.0, 1.0, frames) for side in sides}
    return Skill(name=name, fps=fps, q=q, gripper=grip, source={"type": "test"})


# --------------------------------------------------------------------------- #
# Storage: save / load / discover / delete
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip(tmp_path):
    skill = _skill(sides=("left", "right"))
    path = skills.save_skill(skill, tmp_path)
    assert path == tmp_path / "demo.json"

    loaded = skills.load_skill(tmp_path, "demo")
    assert loaded.name == "demo"
    assert loaded.fps == 15.0
    assert loaded.sides == ["left", "right"]
    assert loaded.frames == 4
    np.testing.assert_allclose(loaded.q["left"], skill.q["left"])
    np.testing.assert_allclose(loaded.q["right"], skill.q["right"])
    np.testing.assert_allclose(loaded.gripper["left"], skill.gripper["left"])
    assert loaded.source == {"type": "test"}
    assert loaded.created  # stamped at save time


def test_save_overwrites_same_name(tmp_path):
    skills.save_skill(_skill(frames=4), tmp_path)
    skills.save_skill(_skill(frames=9), tmp_path)
    assert skills.load_skill(tmp_path, "demo").frames == 9
    assert len(skills.discover_skills(tmp_path)) == 1


def test_discover_lists_and_skips_corrupt(tmp_path):
    skills.save_skill(_skill(name="a"), tmp_path)
    skills.save_skill(_skill(name="b", frames=6), tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    infos = skills.discover_skills(tmp_path)
    assert [i.name for i in infos] == ["a", "b"]
    b = infos[1]
    assert b.frames == 6 and b.sides == ("left",) and b.fps == 15.0
    assert b.duration_s == pytest.approx(6 / 15.0)


def test_discover_missing_root_is_empty(tmp_path):
    assert skills.discover_skills(tmp_path / "nope") == []


def test_delete_skill(tmp_path):
    skills.save_skill(_skill(), tmp_path)
    skills.delete_skill(tmp_path, "demo")
    assert skills.discover_skills(tmp_path) == []
    skills.delete_skill(tmp_path, "demo")  # idempotent


def test_rename_skill(tmp_path):
    skills.save_skill(_skill(name="old"), tmp_path)
    before = skills.load_skill(tmp_path, "old")
    path = skills.rename_skill(tmp_path, "old", "new")
    assert path == tmp_path / "new.json"
    assert [i.name for i in skills.discover_skills(tmp_path)] == ["new"]
    loaded = skills.load_skill(tmp_path, "new")
    assert loaded.name == "new"                    # embedded name follows the file
    assert loaded.created == before.created       # provenance preserved
    assert loaded.source == before.source
    np.testing.assert_allclose(loaded.q["left"], before.q["left"])
    # Same-name rename is a no-op; a missing source or taken target refuses.
    assert skills.rename_skill(tmp_path, "new", "new") == tmp_path / "new.json"
    skills.save_skill(_skill(name="other"), tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        skills.rename_skill(tmp_path, "new", "other")
    with pytest.raises(ValueError, match="not found"):
        skills.rename_skill(tmp_path, "ghost", "x")
    with pytest.raises(ValueError, match="invalid skill name"):
        skills.rename_skill(tmp_path, "new", "bad/name")
    assert sorted(i.name for i in skills.discover_skills(tmp_path)) == ["new", "other"]


def test_bad_names_and_bad_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid skill name"):
        skills.save_skill(_skill(name="../evil"), tmp_path)
    with pytest.raises(ValueError, match="invalid skill name"):
        skills.load_skill(tmp_path, "a/b")
    with pytest.raises(ValueError, match="not found"):
        skills.load_skill(tmp_path, "ghost")
    with pytest.raises(ValueError, match="no skill selected"):
        skills.load_skill(tmp_path, None)
    # A future format version must refuse loudly, not half-load.
    payload = {"version": skills.FORMAT_VERSION + 1, "name": "future",
               "fps": 15.0, "q": {"left": [[0.0] * 7]}}
    (tmp_path / "future.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format version"):
        skills.load_skill(tmp_path, "future")


def test_skill_validate_rejects_empty_and_bad_shapes():
    with pytest.raises(ValueError, match="no joint trajectory"):
        Skill(name="x", fps=15.0, q={}).validate()
    with pytest.raises(ValueError, match="fps"):
        Skill(name="x", fps=0.0, q={"left": np.zeros((2, 7))}).validate()
    with pytest.raises(ValueError, match="frames, dof"):
        Skill(name="x", fps=15.0, q={"left": np.zeros(7)}).validate()


# --------------------------------------------------------------------------- #
# Teach type 1: from a recorded episode
# --------------------------------------------------------------------------- #


def test_layout_parsing_from_stored_names():
    state_names = [f"left.q.{j}" for j in range(7)] + [f"right.q.{j}" for j in range(7)]
    q_index = skills._state_q_index(state_names)
    assert q_index["left"] == list(range(7))
    assert q_index["right"] == list(range(7, 14))
    action_names = [f"left.q_d.{j}" for j in range(7)] + ["left.gripper", "right.gripper"]
    assert skills._gripper_index(action_names) == {"left": 7, "right": 8}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    pytest.importorskip("lerobot")
    from test_replay import _build_dataset

    from dual_flexiv_control.dashboard import storage

    root = str(tmp_path_factory.mktemp("skills_ds"))
    _build_dataset(root, "dfc/teach", n_episodes=2, frames_per_ep=6)
    return storage.discover_datasets(root)[0]


def test_skill_from_episode_extracts_measured_q_and_gripper(dataset):
    skill = skills.skill_from_episode(dataset.path, dataset.repo_id, 0, "taught")
    assert skill.name == "taught"
    assert skill.fps == 15.0
    assert sorted(skill.sides) == ["left", "right"]
    assert skill.frames == 6
    # The fixture records left q = arange(7), right q = arange(7)+10 every frame.
    np.testing.assert_allclose(skill.q["left"][0], np.arange(7.0))
    np.testing.assert_allclose(skill.q["right"][-1], np.arange(7.0) + 10)
    # Gripper action was 0.1*i (left) / 0.2*i (right).
    np.testing.assert_allclose(skill.gripper["left"], 0.1 * np.arange(6.0), atol=1e-6)
    np.testing.assert_allclose(skill.gripper["right"], 0.2 * np.arange(6.0), atol=1e-6)
    assert skill.source["type"] == "episode"
    assert skill.source["repo_id"] == dataset.repo_id
    assert skill.source["episode_index"] == 0


def test_skill_from_episode_roundtrips_through_disk(dataset, tmp_path):
    skill = skills.skill_from_episode(dataset.path, dataset.repo_id, 1, "taught2")
    skills.save_skill(skill, tmp_path)
    loaded = skills.load_skill(tmp_path, "taught2")
    np.testing.assert_allclose(loaded.q["left"], skill.q["left"])
    assert loaded.frames == 6 and loaded.fps == 15.0


# --------------------------------------------------------------------------- #
# The replay loop (fake brain — no processes, no hardware)
# --------------------------------------------------------------------------- #


def _qpos_arm() -> ArmCfg:
    ctrl = ControlCfg(
        kind="qpos",
        mode="NRT_JOINT_POSITION",
        send_fn="SendJointPosition",
        command={"q_d": 7, "dq_d": 7, "dq_max": 7, "ddq_max": 7},
        streamed=["q_d", "dq_d"],
    )
    return ArmCfg(serial="sim", streams={}, control=ctrl, control_enabled=True)


class FakeBrain:
    """Scripted measured-q source + setpoint sink (the Brain surface SkillLoop uses)."""

    def __init__(self, q: dict[str, np.ndarray], track: bool = True):
        self.q = dict(q)          # side -> measured q returned by latest()
        self.track = track        # measured q instantly follows the last command
        self.commands: list[tuple[str, np.ndarray]] = []

    def latest(self, name: str) -> Samples:
        side = name.split("/")[0]
        q = self.q.get(side)
        if q is None:
            return Samples(np.zeros((0, 7)), np.zeros(0, np.int64), np.zeros(0, np.int64))
        return Samples(
            np.asarray(q, dtype=np.float64)[None, :],
            np.array([1], np.int64),
            np.array([0], np.int64),
        )

    def command(self, side: str, setpoint: np.ndarray) -> None:
        self.commands.append((side, np.asarray(setpoint, dtype=np.float64)))
        if self.track:  # like the sim arm: snap onto the commanded q_d
            self.q[side] = np.asarray(setpoint, dtype=np.float64)[:7]


def test_skill_loop_replays_to_completion():
    skill = _skill(frames=5, fps=200.0)
    brain = FakeBrain({"left": skill.q["left"][0].copy()})
    loop = SkillLoop(brain, skill, {"left": _qpos_arm()}, settle_s=0.0)
    assert loop.run(threading.Event()) is True
    # Every posted setpoint is a full [q_d, dq_d] vector with zero feedforward.
    assert all(sp.shape == (14,) for _side, sp in brain.commands)
    assert all(np.all(sp[7:] == 0.0) for _side, sp in brain.commands)
    # The final frame's target was reached.
    np.testing.assert_allclose(brain.commands[-1][1][:7], skill.q["left"][-1])
    assert loop.frames_posted >= skill.frames - 1


def test_skill_loop_holds_first_target_until_converged():
    skill = _skill(frames=3, fps=200.0)
    # Start far away, and do NOT track commands: convergence must gate replay.
    brain = FakeBrain({"left": skill.q["left"][0] + 1.0}, track=False)
    loop = SkillLoop(
        brain, skill, {"left": _qpos_arm()}, start_tolerance=0.05,
        start_timeout_s=0.5, settle_s=0.0,
    )
    with pytest.raises(RuntimeError, match="start pose"):
        loop.run(threading.Event())
    # Only the first frame's target was ever posted while (not) converging.
    first = skill.q["left"][0]
    assert brain.commands
    for _side, sp in brain.commands:
        np.testing.assert_allclose(sp[:7], first)


def test_skill_loop_missing_measured_q_times_out():
    skill = _skill(frames=3, fps=200.0)
    brain = FakeBrain({}, track=False)  # the q stream never produces
    loop = SkillLoop(
        brain, skill, {"left": _qpos_arm()}, start_timeout_s=0.3, settle_s=0.0
    )
    with pytest.raises(RuntimeError, match="start pose"):
        loop.run(threading.Event())


def test_skill_loop_stop_event_aborts():
    skill = _skill(frames=3, fps=200.0)
    brain = FakeBrain({"left": skill.q["left"][0].copy()})
    stop = threading.Event()
    stop.set()
    loop = SkillLoop(brain, skill, {"left": _qpos_arm()}, settle_s=0.0)
    assert loop.run(stop) is False


def test_skill_loop_drives_only_matching_sides():
    skill = _skill(frames=3, fps=200.0, sides=("left", "right"))
    brain = FakeBrain({"left": skill.q["left"][0].copy()})
    # Only the left arm is offered — the right side of the skill is ignored.
    loop = SkillLoop(brain, skill, {"left": _qpos_arm()}, settle_s=0.0)
    assert loop.run(threading.Event()) is True
    assert {side for side, _sp in brain.commands} == {"left"}
    with pytest.raises(ValueError, match="no driveable arm"):
        SkillLoop(brain, _skill(sides=("right",)), {"left": _qpos_arm()})


def test_skill_loop_low_fps_reposts_for_the_deadman():
    # 3 frames at 5 fps = 0.4 s of trajectory; posts must come at >= 20 Hz so the
    # arm's staleness deadman (soft 100 ms) never trips between frames.
    skill = _skill(frames=3, fps=5.0)
    brain = FakeBrain({"left": skill.q["left"][0].copy()})
    loop = SkillLoop(brain, skill, {"left": _qpos_arm()}, settle_s=0.0)
    assert loop.run(threading.Event()) is True
    assert len(brain.commands) > skill.frames  # re-posted between frames


# --------------------------------------------------------------------------- #
# Phase wiring: coeffs + consumer construction
# --------------------------------------------------------------------------- #


def _compose(overrides):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_object(cfg)


def test_skill_phase_coeffs_and_consumer(tmp_path):
    from dual_flexiv_control.skills import SkillNode
    from dual_flexiv_control.system import active_coeffs
    from dual_flexiv_control.system import build_consumer

    skills.save_skill(_skill(), tmp_path)
    config = _compose([
        "rig=bimanual", "runtime.phase=skill",
        f"skill.root={tmp_path}", "skill.name=demo",
        "arms.left.control_enabled=true",
    ])
    assert active_coeffs(config) is config.skill.coeffs
    consumer = build_consumer(config, "rid")
    assert isinstance(consumer, SkillNode)
    assert consumer.name == "skill"
    # Only the taught side that is control-enabled with a qpos kind is driven.
    assert consumer.drive_sides == ["left"]


def test_skill_phase_requires_an_existing_skill(tmp_path):
    from dual_flexiv_control.system import build_consumer

    config = _compose([
        "runtime.phase=skill", f"skill.root={tmp_path}", "skill.name=ghost",
    ])
    with pytest.raises(ValueError, match="not found"):
        build_consumer(config, "rid")
    config = _compose(["runtime.phase=skill", f"skill.root={tmp_path}"])
    with pytest.raises(ValueError, match="no skill selected"):
        build_consumer(config, "rid")
