"""Tests for the dashboard storage backend (dataset discovery + episode deletion).

The pure discovery/formatting paths run with just a hand-written ``meta/info.json``
(no lerobot). The list + delete round-trip is gated on ``lerobot`` being installed,
building a tiny real dataset and exercising re-index + folder-removal.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dual_flexiv_control.dashboard import storage


# --------------------------------------------------------------------------- #
# Pure paths (no lerobot)
# --------------------------------------------------------------------------- #


def test_human_size_units():
    assert storage.human_size(0) == "0 B"
    assert storage.human_size(1536) == "1.5 KB"
    assert storage.human_size(3 * 1024**3) == "3.0 GB"


def _write_fake_dataset(root, repo_id, *, episodes, frames, fps=15, videos=()):
    ds_dir = os.path.join(root, *repo_id.split("/"))
    os.makedirs(os.path.join(ds_dir, "meta"), exist_ok=True)
    info = {
        "total_episodes": episodes, "total_frames": frames,
        "fps": fps, "video_keys": list(videos),
    }
    with open(os.path.join(ds_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f)
    # a payload file so size is non-zero
    with open(os.path.join(ds_dir, "data.bin"), "wb") as f:
        f.write(b"\0" * 2048)
    return ds_dir


def test_discover_datasets_reads_info(tmp_path):
    _write_fake_dataset(tmp_path, "dfc/pick", episodes=3, frames=18, videos=["observation.images.cam"])
    _write_fake_dataset(tmp_path, "handover", episodes=1, frames=5)
    found = {d.repo_id: d for d in storage.discover_datasets(str(tmp_path))}
    assert set(found) == {"dfc/pick", "handover"}
    pick = found["dfc/pick"]
    assert pick.num_episodes == 3 and pick.num_frames == 18 and pick.fps == 15
    assert pick.video_keys == ("observation.images.cam",)
    assert pick.size_bytes >= 2048


def test_discover_empty_and_missing_root(tmp_path):
    assert storage.discover_datasets(str(tmp_path)) == []
    assert storage.discover_datasets(str(tmp_path / "nope")) == []


def test_delete_rejects_empty_and_invalid(tmp_path):
    _write_fake_dataset(tmp_path, "d", episodes=2, frames=4)
    ds = storage.discover_datasets(str(tmp_path))[0]
    with pytest.raises(ValueError):
        storage.delete_episodes(ds, [])
    with pytest.raises(ValueError):
        storage.delete_episodes(ds, [5])


def test_delete_all_removes_folder_without_lerobot(tmp_path):
    # Deleting every episode is a plain rmtree — no lerobot import on this path.
    ds_dir = _write_fake_dataset(tmp_path, "d", episodes=2, frames=4)
    ds = storage.discover_datasets(str(tmp_path))[0]
    assert storage.delete_episodes(ds, [0, 1]) == "dataset-removed"
    assert not os.path.exists(ds_dir)


# --------------------------------------------------------------------------- #
# Real dataset round-trip (needs lerobot)
# --------------------------------------------------------------------------- #

pytest.importorskip("lerobot")


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
            "cameras.wrist_left.width=32", "cameras.wrist_left.height=24",
            f"task.collection.root={root}", f"task.collection.repo_id={repo_id}",
            "task.collection.frequency_hz=15",
        ])
    c = OmegaConf.to_object(cfg)
    c.cameras = {"wrist_left": c.cameras["wrist_left"]}  # single tiny cam
    b = FrameBuilder(c.arms, ["left"], c.cameras, c.task.language_instruction,
                     c.task.collection.state_signals, video=True)

    def smp(v, dt=np.float64):
        return Samples(data=np.asarray(v, dt).reshape(1, -1),
                       t_ns=np.array([1], np.int64), seq=np.array([0], np.int64))

    def obs():
        o = {"left/q": smp(np.zeros(7)), "right/q": smp(np.ones(7))}
        cam = c.cameras["wrist_left"]
        dim = cam.height * cam.width * 3
        o["cam/wrist_left/left"] = smp((np.arange(dim) % 256).astype(np.uint8), np.uint8)
        return o

    rec = LeRobotRecorder(c.task.collection, b.features())
    for ep in range(n_episodes):
        for i in range(frames_per_ep):
            rec.add_frame(b.build(obs(), {"left": 0.01 * i * np.ones(7)}, {"left": 0.1 * ep}))
        rec.save_episode()
    rec.finalize()


def test_list_and_partial_delete_reindexes(tmp_path):
    root = str(tmp_path)
    _build_dataset(root, "dfc/pick", n_episodes=3, frames_per_ep=6)

    ds = storage.discover_datasets(root)[0]
    eps = storage.list_episodes(ds)
    assert [e.index for e in eps] == [0, 1, 2]
    assert all(e.length == 6 for e in eps)
    assert eps[0].duration_s == pytest.approx(6 / 15)
    assert eps[0].tasks and eps[0].tasks[0]

    assert storage.delete_episodes(ds, [1]) == "reindexed"

    ds2 = storage.discover_datasets(root)[0]
    eps2 = storage.list_episodes(ds2)
    assert ds2.num_episodes == 2
    assert [e.index for e in eps2] == [0, 1]  # contiguous re-index

    # the rebuilt dataset must still load cleanly
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    reloaded = LeRobotDataset(ds2.repo_id, root=ds2.path)
    assert reloaded.num_episodes == 2 and reloaded.num_frames == 12
