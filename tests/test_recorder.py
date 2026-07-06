"""Tests for LeRobotRecorder resume/create handling.

``_resumable`` is pure (file checks, no lerobot). The recreate-on-corrupt and
resume-and-append round-trip build a real dataset, so they are gated on lerobot.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dual_flexiv_control.collection.recorder import _resumable


# --------------------------------------------------------------------------- #
# _resumable (pure)
# --------------------------------------------------------------------------- #


def _make_meta(tmp_path, *, episodes, with_tasks=True):
    d = tmp_path / "ds"
    (d / "meta").mkdir(parents=True, exist_ok=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes}))
    if with_tasks:
        (d / "meta" / "tasks.parquet").write_bytes(b"\0")
    return str(d)


def test_resumable_true_only_for_complete_nonempty_dataset(tmp_path):
    assert _resumable(_make_meta(tmp_path, episodes=3)) is True


def test_resumable_false_for_zero_episode_leftover(tmp_path):
    # a crashed create: meta present, tasks present, but nothing committed
    assert _resumable(_make_meta(tmp_path, episodes=0)) is False


def test_resumable_false_when_tasks_parquet_missing(tmp_path):
    # exactly the corrupt state that made collection 401 on the Hub
    assert _resumable(_make_meta(tmp_path, episodes=2, with_tasks=False)) is False


def test_resumable_false_when_absent(tmp_path):
    assert _resumable(str(tmp_path / "nope")) is False


# --------------------------------------------------------------------------- #
# create / resume round-trip (needs lerobot)
# --------------------------------------------------------------------------- #

pytest.importorskip("lerobot")


def _cfg_and_features(root, repo_id):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from dual_flexiv_control.collection import FrameBuilder
    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = OmegaConf.to_object(compose(config_name="config", overrides=[
            "rig=bimanual",  # wrist_left exists only on the full rig
            "cameras.wrist_left.width=32", "cameras.wrist_left.height=24",
            f"recording.root={root}", f"task.collection.repo_id={repo_id}",
        ]))
    cfg.cameras = {"wrist_left": cfg.cameras["wrist_left"]}
    b = FrameBuilder(cfg.arms, list(cfg.arms), cfg.cameras, cfg.task.language_instruction,
                     cfg.task.state_signals, video=True)
    return cfg, b


def _record_episode(rec, builder, cfg, ep):
    from dual_flexiv_control.streams.ring import Samples

    def smp(v, dt=np.float64):
        return Samples(np.asarray(v, dt).reshape(1, -1), np.array([1], np.int64), np.array([0], np.int64))

    cam = cfg.cameras["wrist_left"]
    for i in range(5):
        obs = {"left/q": smp(np.zeros(7)), "right/q": smp(np.ones(7)),
               "cam/wrist_left/left": smp((np.arange(cam.height * cam.width * 3) % 256).astype(np.uint8), np.uint8)}
        rec.add_frame(builder.build(obs, {"left": 0.01 * ep * np.ones(7), "right": 0.02 * ep * np.ones(7)},
                                    {"left": 0.1 * ep, "right": 0.2 * ep}))
    rec.save_episode()


def test_recorder_recreates_corrupt_leftover_then_resumes(tmp_path):
    from dual_flexiv_control.collection import LeRobotRecorder

    root = str(tmp_path)
    cfg, builder = _cfg_and_features(root, "dual_flexiv/collected")
    coll = cfg.task.collection
    feats = builder.features()

    # Seed exactly the corrupt state from the field: meta/info.json (0 episodes),
    # no tasks.parquet — this previously 401'd against the Hub on resume.
    dd = os.path.join(root, "dual_flexiv", "collected", "meta")
    os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, "info.json"), "w") as f:
        json.dump({"total_episodes": 0, "total_frames": 0, "fps": 15, "video_keys": []}, f)

    # create over the corrupt leftover (must not touch the Hub), record one episode
    rec = LeRobotRecorder(coll, cfg.recording, feats)
    _record_episode(rec, builder, cfg, ep=0)
    rec.finalize()
    assert rec.dataset.meta.total_episodes == 1

    # resume the now-valid dataset and append a second episode (the path that used
    # to crash on the missing start_image_writer)
    rec2 = LeRobotRecorder(coll, cfg.recording, feats)
    assert rec2.dataset.meta.total_episodes == 1
    _record_episode(rec2, builder, cfg, ep=1)
    rec2.finalize()
    assert rec2.dataset.meta.total_episodes == 2
