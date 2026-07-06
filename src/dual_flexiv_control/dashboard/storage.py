"""Recorded-episode storage: discover LeRobot datasets and delete episodes.

Backs the dashboard's **Storage** tab. Collection writes demonstrations as
LeRobot datasets under ``task.collection.root`` (one folder per ``repo_id``,
possibly namespaced, e.g. ``dfc/pick``). This module:

* finds every dataset under that root (any folder with a ``meta/info.json``),
* lists each dataset's episodes (index, frame count, duration, task), and
* deletes episodes — single or in bulk.

Deletion detail: LeRobot's :func:`dataset_tools.delete_episodes` is an *immutable*
op (it builds a fresh, re-indexed dataset and refuses to empty one). So a partial
delete rebuilds into a temp dir and atomically swaps it into place; deleting every
episode just removes the dataset folder. Both leave the on-disk dataset
self-consistent (contiguous ``episode_index`` / frame index, updated meta + stats).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _hub_offline():
    """Force lerobot/HF calls fully local for the duration of the block.

    Datasets recorded here are local-only (their ``repo_id`` is not a real Hub
    repo). Without this, a failed *local* metadata load makes lerobot fall back to
    downloading ``meta/`` from the Hub — which 401s on the bogus repo id and hides
    the real (local) problem. Offline turns that into a clean local error instead.
    """
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

_LOCK = threading.Lock()
_ROOT: str | None = None


@dataclass(frozen=True)
class EpisodeInfo:
    """One recorded episode within a dataset."""

    index: int
    length: int              # frames
    duration_s: float        # length / fps
    tasks: tuple[str, ...]   # language instruction(s)


@dataclass(frozen=True)
class DatasetInfo:
    """A LeRobot dataset on disk under the collection root."""

    repo_id: str             # path relative to the root, e.g. "dfc/pick"
    path: str                # absolute dataset directory
    num_episodes: int
    num_frames: int
    fps: float
    video_keys: tuple[str, ...]
    size_bytes: int


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def collection_root() -> str:
    """Absolute ``recording.root`` (where datasets live); composed once, cached."""
    global _ROOT
    with _LOCK:
        if _ROOT is None:
            _ROOT = _compose_root()
        return _ROOT


def reset() -> None:
    """Drop the cached root so the next call re-reads config (Reset services)."""
    global _ROOT
    with _LOCK:
        _ROOT = None


def _compose_root() -> str:
    from hydra import compose
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config")
    root = str(cfg.recording.root)
    return os.path.abspath(root)


def discover_datasets(root: str | None = None) -> list[DatasetInfo]:
    """Every LeRobot dataset under ``root`` (a folder with ``meta/info.json``).

    Datasets may be namespaced (nested ``<ns>/<name>``); ``repo_id`` is the path
    relative to the root. Ordered by ``repo_id``. Unreadable/corrupt datasets are
    skipped (logged), never raised — one bad folder must not hide the rest.
    """
    root = root or collection_root()
    if not os.path.isdir(root):
        return []
    out: list[DatasetInfo] = []
    for info_path in Path(root).rglob("meta/info.json"):
        ds_dir = info_path.parent.parent
        repo_id = os.path.relpath(ds_dir, root).replace(os.sep, "/")
        try:
            out.append(_read_dataset(repo_id, str(ds_dir), info_path))
        except Exception:  # noqa: BLE001 - a broken dataset must not hide the others
            log.exception("skipping unreadable dataset at %s", ds_dir)
    out.sort(key=lambda d: d.repo_id)
    return out


def _read_dataset(repo_id: str, ds_dir: str, info_path: Path) -> DatasetInfo:
    info = json.loads(info_path.read_text())
    return DatasetInfo(
        repo_id=repo_id,
        path=ds_dir,
        num_episodes=int(info.get("total_episodes", 0)),
        num_frames=int(info.get("total_frames", 0)),
        fps=float(info.get("fps", 0) or 0),
        video_keys=tuple(info.get("video_keys", []) or []),
        size_bytes=_dir_size(ds_dir),
    )


def list_episodes(ds: DatasetInfo) -> list[EpisodeInfo]:
    """Per-episode index / frame count / duration / task, oldest → newest.

    Purely local (never contacts the Hub). An empty/partial dataset (no committed
    episodes — e.g. a crashed run) returns ``[]``. A genuinely unreadable dataset
    raises :class:`ValueError` with the underlying local cause (the caller offers
    to delete it) rather than a misleading Hub 401.
    """
    if ds.num_episodes <= 0:
        return []  # nothing committed yet; skip the metadata load entirely
    try:
        with _hub_offline():
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            meta = LeRobotDatasetMetadata(ds.repo_id, root=ds.path)
            fps = float(meta.fps) or 1.0
            rows = list(meta.episodes)
    except Exception as exc:  # noqa: BLE001 - surface a clean local error
        raise ValueError(
            f"dataset metadata is unreadable (likely an incomplete/corrupt "
            f"recording): {type(exc).__name__}: {exc}"
        ) from exc

    episodes = []
    for row in rows:
        length = int(row["length"])
        tasks = row.get("tasks") or []
        episodes.append(
            EpisodeInfo(
                index=int(row["episode_index"]),
                length=length,
                duration_s=length / fps,
                tasks=tuple(str(t) for t in tasks),
            )
        )
    episodes.sort(key=lambda e: e.index)
    return episodes


def delete_dataset(ds: DatasetInfo) -> None:
    """Remove an entire dataset folder — works even when it's corrupt/unreadable."""
    shutil.rmtree(ds.path, ignore_errors=True)
    log.info("removed dataset %s at %s", ds.repo_id, ds.path)


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #


def delete_episodes(ds: DatasetInfo, indices) -> str:
    """Delete ``indices`` from the dataset in place. Returns a short status string.

    Deleting every episode removes the dataset folder ("dataset-removed"); a
    partial delete rebuilds a re-indexed dataset in a sibling temp dir and swaps
    it in atomically ("reindexed"). Raises ``ValueError`` on an empty/invalid set.
    """
    indices = sorted({int(i) for i in indices})
    if not indices:
        raise ValueError("no episodes selected")
    total = ds.num_episodes
    invalid = [i for i in indices if i < 0 or i >= total]
    if invalid:
        raise ValueError(f"invalid episode indices: {invalid}")

    if len(indices) >= total:  # deleting all -> just remove the folder
        shutil.rmtree(ds.path, ignore_errors=True)
        log.info("removed dataset %s (all %d episodes)", ds.repo_id, total)
        return "dataset-removed"

    tmp = ds.path + ".__editing__"
    backup = ds.path + ".__old__"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        with _hub_offline():
            from lerobot.datasets import dataset_tools
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            source = LeRobotDataset(ds.repo_id, root=ds.path)
            dataset_tools.delete_episodes(
                source, indices, output_dir=tmp, repo_id=ds.repo_id
            )
        # Swap in the rebuilt dataset: move the old aside, promote the temp, then
        # drop the old. Rename within one directory is atomic on POSIX.
        os.replace(ds.path, backup)
        os.replace(tmp, ds.path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
    log.info("deleted %d episode(s) from %s; reindexed", len(indices), ds.repo_id)
    return "reindexed"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def human_size(n: int) -> str:
    """Bytes -> human string (e.g. ``1.4 GB``)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
