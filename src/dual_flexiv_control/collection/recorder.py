"""LeRobot dataset recorder — the collection loop's sink (lazy ``lerobot`` import).

The loop is decoupled from LeRobot behind the tiny :class:`Recorder` protocol
(``add_frame`` / ``save_episode`` / ``discard_episode`` / ``finalize``), so it is
testable with a fake and the heavy ``lerobot`` dependency is imported only when a
real recording actually runs.

:class:`LeRobotRecorder` wraps ``LeRobotDataset`` and turns on its **async image
writer** (background threads/processes) plus **streaming video encoding** so disk
I/O and video encoding never stall the fixed-rate control loop — the real-time
capability LeRobot provides for on-robot recording. If ``lerobot`` is not
installed it raises an actionable :class:`RecorderUnavailable` at construction.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from typing import Protocol

log = logging.getLogger(__name__)


class RecorderUnavailable(RuntimeError):
    """Raised when a real LeRobot recorder cannot be constructed (missing dep)."""


@contextlib.contextmanager
def _hub_offline():
    """Force LeRobot/HF fully local while loading a local dataset.

    Datasets recorded here are local-only (``repo_id`` is not a real Hub repo). Without
    this, a failed *local* load (e.g. an incomplete dataset from a crashed run) makes
    LeRobot fall back to downloading from the Hub — which 401s on the bogus repo id and
    crashes collection. Offline turns that into a clean local error we can act on.
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


def _resumable(dataset_dir: str) -> bool:
    """Whether ``dataset_dir`` is a complete dataset worth resuming (>=1 episode).

    A crashed prior run can leave a half-written dataset (``meta/`` present but
    missing ``tasks.parquet`` / episode data, ``total_episodes == 0``). Resuming that
    fails; such leftovers are recreated fresh instead.
    """
    info = os.path.join(dataset_dir, "meta", "info.json")
    tasks = os.path.join(dataset_dir, "meta", "tasks.parquet")
    if not (os.path.isfile(info) and os.path.isfile(tasks)):
        return False
    try:
        return int(json.loads(open(info).read()).get("total_episodes", 0) or 0) > 0
    except (OSError, ValueError):
        return False


class Recorder(Protocol):
    """What the collection loop needs from a recording sink."""

    def add_frame(self, frame: dict) -> None: ...
    def save_episode(self) -> None: ...
    def discard_episode(self) -> None: ...
    def finalize(self) -> None: ...


class LeRobotRecorder:
    """Real sink: appends frames/episodes to a LeRobot dataset on disk.

    ``cfg`` (:class:`~dual_flexiv_control.configs.CollectionCfg`) carries the
    dataset *identity* (``repo_id``, ``frequency_hz`` → fps); ``recording``
    (:class:`~dual_flexiv_control.configs.RecordingCfg`) the export *machinery*
    (root, encoders, writer threading). ``features`` is the spec from
    :meth:`FrameBuilder.features`. The dataset lives at ``root/repo_id``; when
    ``resume`` and that folder already exists it is opened for append, otherwise
    it is created.
    """

    def __init__(self, cfg, recording, features: dict, robot_type: str = "dual_flexiv") -> None:
        LeRobotDataset = _import_lerobot()
        self._cfg = cfg
        self._recording = recording
        root = os.path.abspath(recording.root)
        dataset_dir = os.path.join(root, cfg.repo_id)
        fps = int(round(cfg.frequency_hz))

        create_kwargs = dict(
            image_writer_processes=int(recording.image_writer_processes),
            image_writer_threads=int(recording.image_writer_threads),
        )
        # streaming_encoding is a newer LeRobot option; pass it only if supported.
        if _accepts(LeRobotDataset.create, "streaming_encoding"):
            create_kwargs["streaming_encoding"] = bool(recording.streaming_encoding)

        has_meta = os.path.isdir(os.path.join(dataset_dir, "meta"))
        if recording.resume and has_meta and _resumable(dataset_dir):
            log.info("resuming LeRobot dataset at %s", dataset_dir)
            # resume() (not the read-only constructor) opens the dataset in WRITE mode
            # with the async image writer + streaming encoder — same kwargs as create().
            with _hub_offline():   # local-only; never fall back to the Hub (401)
                self._ds = LeRobotDataset.resume(cfg.repo_id, root=dataset_dir, **create_kwargs)
        else:
            # A leftover half-written dataset (meta/ present but not resumable — e.g. a
            # prior crashed run) would make create() fail on the existing dir; clear it.
            if has_meta:
                log.warning(
                    "existing dataset at %s is incomplete/not resumable — recreating fresh",
                    dataset_dir,
                )
                shutil.rmtree(dataset_dir, ignore_errors=True)
            log.info("creating LeRobot dataset at %s (fps=%d)", dataset_dir, fps)
            with _hub_offline():   # create is local; keep HF off the path entirely
                self._ds = LeRobotDataset.create(
                    repo_id=cfg.repo_id,
                    fps=fps,
                    features=features,
                    root=dataset_dir,
                    robot_type=robot_type,
                    use_videos=bool(recording.video),
                    **create_kwargs,
                )
        self._episodes = 0

    @property
    def dataset(self):
        return self._ds

    @property
    def episodes_saved(self) -> int:
        return self._episodes

    def add_frame(self, frame: dict) -> None:
        self._ds.add_frame(frame)

    def save_episode(self) -> None:
        self._ds.save_episode()
        self._episodes += 1
        if getattr(self._recording, "push_to_hub", False):
            try:
                self._ds.push_to_hub()
            except Exception:  # noqa: BLE001 - a push failure must not lose the local data
                log.exception("push_to_hub failed (episode kept locally)")

    def discard_episode(self) -> None:
        """Drop the in-progress episode buffer without writing it (re-record)."""
        clear = getattr(self._ds, "clear_episode_buffer", None)
        if clear is not None:
            clear()
        else:  # older API: reset the buffer explicitly
            self._ds.episode_buffer = self._ds.create_episode_buffer()

    def finalize(self) -> None:
        for method in ("finalize", "stop_image_writer"):
            fn = getattr(self._ds, method, None)
            if fn is not None:
                try:
                    fn()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("LeRobot %s failed", method)


def _import_lerobot():
    """Return ``LeRobotDataset`` or raise :class:`RecorderUnavailable`."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        return LeRobotDataset
    except ImportError:
        pass
    try:  # older module path
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        return LeRobotDataset
    except ImportError as exc:
        raise RecorderUnavailable(
            "collection requires the 'lerobot' package to write datasets. "
            "Install it with:  pip install 'dual-flexiv-control[collection]'  "
            "(or  pip install lerobot)."
        ) from exc


def _accepts(fn, param: str) -> bool:
    import inspect

    try:
        return param in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
