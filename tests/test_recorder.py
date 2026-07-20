"""Tests for LeRobotRecorder resume/create handling.

``_resumable`` and ``_schema_mismatches`` are pure (file/dict checks, no
lerobot). The recreate-on-corrupt and resume-and-append round-trip build a real
dataset, so they are gated on lerobot.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dual_flexiv_control.collection.recorder import _resumable
from dual_flexiv_control.collection.recorder import _schema_mismatches


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
# _schema_mismatches (pure)
# --------------------------------------------------------------------------- #

_STATE_14 = {
    "observation.state": {"dtype": "float32", "shape": (14,),
                          "names": [f"{s}.q.{i}" for s in ("left", "right") for i in range(7)]},
    "action": {"dtype": "float32", "shape": (16,), "names": ["..."] * 16},
}


def test_schema_match_ignores_bookkeeping_features():
    disk = {k: {**v, "shape": list(v["shape"])} for k, v in _STATE_14.items()}
    disk["timestamp"] = {"dtype": "float32", "shape": [1]}
    disk["episode_index"] = {"dtype": "int64", "shape": [1]}
    assert _schema_mismatches(disk, _STATE_14) == []


def test_schema_mismatch_on_shape():
    # the field failure: a single-arm dataset (state 7) resumed by a bimanual run
    disk = {"observation.state": {"dtype": "float32", "shape": [7],
                                  "names": [f"left.q.{i}" for i in range(7)]},
            "action": {"dtype": "float32", "shape": [8], "names": ["..."] * 8}}
    diffs = _schema_mismatches(disk, _STATE_14)
    assert any("observation.state" in d and "[7]" in d and "[14]" in d for d in diffs)
    assert any("action" in d for d in diffs)


def test_schema_mismatch_on_names_at_same_shape():
    disk = {k: {**v, "shape": list(v["shape"])} for k, v in _STATE_14.items()}
    disk["observation.state"]["names"] = [f"{s}.dq.{i}" for s in ("left", "right") for i in range(7)]
    assert any("layout" in d for d in _schema_mismatches(disk, _STATE_14))


def test_schema_mismatch_on_camera_missing_from_run():
    disk = {k: {**v, "shape": list(v["shape"])} for k, v in _STATE_14.items()}
    disk["observation.images.static"] = {"dtype": "video", "shape": [720, 1280, 3]}
    assert any("observation.images.static" in d for d in _schema_mismatches(disk, _STATE_14))


def test_schema_mismatch_on_feature_missing_from_disk():
    assert any("not in the existing dataset" in d for d in _schema_mismatches({}, _STATE_14))


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
            "rig=bimanual",
            # the test brings its own tiny camera so it doesn't depend on which
            # cameras the rig currently enables
            "+camera@cameras.testcam=zedx_wrist",
            "cameras.testcam.placement=static",
            "cameras.testcam.width=32", "cameras.testcam.height=24",
            f"recording.root={root}", f"task.collection.repo_id={repo_id}",
        ]))
    cfg.cameras = {"testcam": cfg.cameras["testcam"]}
    b = FrameBuilder(cfg.arms, list(cfg.arms), cfg.cameras, cfg.task.language_instruction,
                     cfg.task.state_signals, video=True)
    return cfg, b


def _record_episode(rec, builder, cfg, ep):
    from dual_flexiv_control.streams.ring import Samples

    def smp(v, dt=np.float64):
        return Samples(np.asarray(v, dt).reshape(1, -1), np.array([1], np.int64), np.array([0], np.int64))

    cam = cfg.cameras["testcam"]
    for i in range(5):
        obs = {"left/q": smp(np.zeros(7)), "right/q": smp(np.ones(7)),
               "cam/testcam/left": smp((np.arange(cam.height * cam.width * 3) % 256).astype(np.uint8), np.uint8)}
        rec.add_frame(builder.build(obs, {"left": 0.01 * ep * np.ones(7), "right": 0.02 * ep * np.ones(7)},
                                    {"left": 0.1 * ep, "right": 0.2 * ep}))
    rec.save_episode()


def _seed_complete_dataset(root, repo_id, features, episodes=5):
    """A minimal on-disk dataset that passes _resumable: meta/info.json with
    committed episodes + the given features, and a tasks.parquet stub."""
    dd = os.path.join(root, repo_id, "meta")
    os.makedirs(dd, exist_ok=True)
    feats = {k: {**v, "shape": list(v["shape"])} for k, v in features.items()}
    with open(os.path.join(dd, "info.json"), "w") as f:
        json.dump({"total_episodes": episodes, "fps": 15, "features": feats}, f)
    with open(os.path.join(dd, "tasks.parquet"), "wb") as f:
        f.write(b"\0")
    return os.path.join(root, repo_id)


def test_recorder_refuses_resume_onto_mismatched_schema(tmp_path):
    from dual_flexiv_control.collection import LeRobotRecorder
    from dual_flexiv_control.collection.recorder import RecorderUnavailable

    root = str(tmp_path)
    cfg, builder = _cfg_and_features(root, "dual_flexiv/mismatched")
    feats = builder.features()

    # the field failure: the dataset on disk was recorded single-arm (state 7)
    old = dict(feats)
    old["observation.state"] = {"dtype": "float32", "shape": (7,),
                                "names": [f"left.q.{i}" for i in range(7)]}
    old["action"] = {"dtype": "float32", "shape": (8,),
                     "names": [f"left.q_d.{j}" for j in range(7)] + ["left.gripper"]}
    dataset_dir = _seed_complete_dataset(root, "dual_flexiv/mismatched", old)

    with pytest.raises(RecorderUnavailable, match="different schema"):
        LeRobotRecorder(cfg.task.collection, cfg.recording, feats)
    # the mismatched dataset is left untouched, never recreated
    assert os.path.isfile(os.path.join(dataset_dir, "meta", "info.json"))


def test_recorder_never_deletes_complete_dataset_when_resume_off(tmp_path):
    from dual_flexiv_control.collection import LeRobotRecorder
    from dual_flexiv_control.collection.recorder import RecorderUnavailable

    root = str(tmp_path)
    cfg, builder = _cfg_and_features(root, "dual_flexiv/noresume")
    feats = builder.features()
    dataset_dir = _seed_complete_dataset(root, "dual_flexiv/noresume", feats)

    cfg.recording.resume = False
    with pytest.raises(RecorderUnavailable, match="refusing to delete"):
        LeRobotRecorder(cfg.task.collection, cfg.recording, feats)
    assert os.path.isfile(os.path.join(dataset_dir, "meta", "info.json"))


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
