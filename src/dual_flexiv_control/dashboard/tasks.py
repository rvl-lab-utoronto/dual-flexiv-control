"""Discover the tasks and rigs the dashboard can launch.

A **task** is a Hydra config file in the ``conf/task`` group (selected at runtime
with ``task=<name>``): the natural-language goal, its dataset, and per-phase
counts. A **rig** is a file in the ``conf/rig`` group (selected with
``rig=<name>``): which hardware exists — arms/cameras/FACTR servers + serials.
The dashboard shows one dropdown per axis and launches
``dual-flexiv-control task=<t> rig=<r> …``.

We read the YAML directly with OmegaConf (already a project dependency; cheap and
cwd-independent) rather than composing through Hydra. That is sufficient because
task files carry their fields inline and a rig's dropdown entry only needs its
name + description. If tasks ever grow ``defaults`` that must be merged to
resolve these values, swap :func:`discover_tasks` to use ``hydra.compose``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

import dual_flexiv_control

#: The ``conf`` root and its groups, shipped inside the installed package.
CONF_DIR = Path(dual_flexiv_control.__file__).resolve().parent / "conf"
TASK_GROUP_DIR = CONF_DIR / "task"
RIG_GROUP_DIR = CONF_DIR / "rig"


@dataclass(frozen=True)
class TaskInfo:
    """One task in the dashboard's Task dropdown (a ``conf/task`` file)."""

    name: str
    """Config stem, passed as ``task=<name>``."""

    language_instruction: str
    """Natural-language goal, shared by collection and eval."""

    num_episodes: int | None
    """``collection.num_episodes`` (demonstrations to teleoperate), if declared."""

    num_timesteps: int | None
    """``eval.num_timesteps`` (rollout horizon), if declared."""

    path: Path
    """The source YAML, for display / click-through."""


@dataclass(frozen=True)
class RigInfo:
    """One rig in the dashboard's Rig dropdown (a ``conf/rig`` file)."""

    name: str
    """Config stem, passed as ``rig=<name>``."""

    description: str
    """First descriptive comment line of the file (what hardware this is)."""

    path: Path
    """The source YAML, for display / click-through."""


def discover_tasks(task_dir: Path = TASK_GROUP_DIR) -> list[TaskInfo]:
    """Every task in the ``conf/task`` group, sorted by name.

    A file qualifies only if it declares a ``language_instruction`` (the one
    field every task shares), which keeps the dropdown to real tasks and skips
    any non-task YAML that might land in the group.
    """
    tasks: list[TaskInfo] = []
    for path in sorted(task_dir.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        instruction = _maybe_str(cfg, "language_instruction")
        if instruction is None:
            continue
        collection = cfg.get("collection") or {}
        evaluation = cfg.get("eval") or {}
        tasks.append(
            TaskInfo(
                name=path.stem,
                language_instruction=instruction,
                num_episodes=_maybe_int(collection, "num_episodes"),
                num_timesteps=_maybe_int(evaluation, "num_timesteps"),
                path=path,
            )
        )
    return tasks


def discover_rigs(rig_dir: Path = RIG_GROUP_DIR) -> list[RigInfo]:
    """Every rig in the ``conf/rig`` group, sorted by name.

    The description is the file's first descriptive comment line (skipping the
    ``@package`` directive) — rig files lead with a one-line summary of the
    hardware by convention.
    """
    return [
        RigInfo(name=path.stem, description=_head_comment(path), path=path)
        for path in sorted(rig_dir.glob("*.yaml"))
    ]


def _head_comment(path: Path) -> str:
    """First descriptive ``#`` comment line of ``path`` ('' if none)."""
    try:
        for line in path.read_text().splitlines():
            text = line.strip()
            if not text.startswith("#"):
                break  # past the header comment block
            text = text.lstrip("#").strip()
            if text and not text.startswith("@package"):
                return text
    except OSError:
        pass
    return ""


def _maybe_str(cfg, key: str) -> str | None:
    """``cfg[key]`` as ``str``, or ``None`` if absent / unresolved (MISSING)."""
    if not OmegaConf.is_missing(cfg, key) and key in cfg:
        value = cfg.get(key)
        if value is not None:
            return str(value)
    return None


def _maybe_int(cfg, key: str) -> int | None:
    """``cfg[key]`` as ``int``, or ``None`` if absent / non-numeric / MISSING."""
    try:
        if OmegaConf.is_missing(cfg, key):
            return None
        value = cfg.get(key)
    except Exception:  # noqa: BLE001 - plain dict or anything without OmegaConf API
        value = cfg.get(key) if hasattr(cfg, "get") else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
