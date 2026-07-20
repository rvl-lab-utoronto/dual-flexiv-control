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
    """Raised when a real LeRobot recorder cannot be constructed — the ``lerobot``
    dependency is missing, or an existing dataset at the destination cannot
    safely be appended to (schema mismatch / resume disabled)."""


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


def _disk_features(dataset_dir: str) -> dict:
    """The ``features`` dict recorded in an existing dataset's ``meta/info.json``."""
    try:
        info = json.loads(open(os.path.join(dataset_dir, "meta", "info.json")).read())
    except (OSError, ValueError):
        return {}
    feats = info.get("features")
    return feats if isinstance(feats, dict) else {}


def _schema_mismatches(existing: dict, expected: dict) -> list[str]:
    """Differences between an existing dataset's features and this run's (empty =
    safe to append).

    Only the keys this writer produces are compared — LeRobot's bookkeeping
    features (timestamp, indices) are its own. ``names`` are compared too, when
    both sides carry them: the same state dim can hide a different per-side
    layout. The reverse direction matters as well: a feature the dataset has but
    this run no longer produces (a removed camera, a dropped arm) would fail
    ``add_frame`` just the same.
    """
    diffs: list[str] = []
    for key, spec in expected.items():
        have = existing.get(key)
        if have is None:
            diffs.append(f"{key}: not in the existing dataset")
            continue
        want_shape = [int(x) for x in spec.get("shape", ())]
        have_shape = [int(x) for x in have.get("shape", ())]
        if have_shape != want_shape:
            diffs.append(f"{key}: shape {have_shape} on disk vs {want_shape} in this run")
        elif (have.get("names") and spec.get("names")
              and list(have["names"]) != list(spec["names"])):
            diffs.append(f"{key}: same shape but a different layout (names differ)")
    for key in existing:
        if key.startswith("observation.") and key not in expected:
            diffs.append(f"{key}: in the existing dataset but not produced by this run")
    return diffs


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
    it is created. Appending is gated on the on-disk features matching this
    run's: a dataset recorded under a different rig raises
    :class:`RecorderUnavailable` (and is never overwritten).
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
        resumable = has_meta and _resumable(dataset_dir)
        if recording.resume and resumable:
            # Appending is only safe when the on-disk features match what this run
            # records. A dataset recorded under a different rig (single-arm vs
            # bimanual, different cameras or state signals) would otherwise resume
            # fine and then explode mid-run at the first add_frame — after the arms
            # are already in control. Refuse up front, and never touch the data.
            diffs = _schema_mismatches(_disk_features(dataset_dir), features)
            if diffs:
                raise RecorderUnavailable(
                    f"existing dataset at {dataset_dir} was recorded with a "
                    "different schema than this run produces — appending would "
                    "corrupt it, so it is left untouched:\n  - "
                    + "\n  - ".join(diffs)
                    + "\nThis usually means the rig / cameras / state signals "
                    "changed since it was recorded (e.g. single-arm vs bimanual). "
                    "Record to a fresh dataset with e.g. "
                    "task.collection.repo_id=<namespace>/<name>, or move the "
                    "existing folder away."
                )
            log.info("resuming LeRobot dataset at %s", dataset_dir)
            # resume() (not the read-only constructor) opens the dataset in WRITE mode
            # with the async image writer + streaming encoder — same kwargs as create().
            with _hub_offline():   # local-only; never fall back to the Hub (401)
                self._ds = LeRobotDataset.resume(cfg.repo_id, root=dataset_dir, **create_kwargs)
        else:
            # A COMPLETE dataset only lands here when resume is disabled; recorded
            # episodes are never deleted on the recorder's own initiative.
            if resumable:
                raise RecorderUnavailable(
                    f"a complete dataset already exists at {dataset_dir} and "
                    "recording.resume is disabled — refusing to delete recorded "
                    "episodes. Re-enable resume to append, record elsewhere via "
                    "task.collection.repo_id=<namespace>/<name>, or move the "
                    "existing folder away."
                )
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
