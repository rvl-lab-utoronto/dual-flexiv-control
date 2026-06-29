"""Discover the manipulation tasks the dashboard can launch.

A "task" is a Hydra config file in the ``conf/task`` group (the same files you
select at runtime with ``task=<name>``). The dropdown enumerates them, and for
each we surface the shared ``language_instruction`` plus the per-phase counts —
``collection.num_episodes`` and ``eval.num_timesteps`` — so the operator sees
what they are about to launch.

We read the YAML directly with OmegaConf (already a project dependency; cheap and
cwd-independent) rather than composing through Hydra. That is sufficient because
task files carry their fields inline. If tasks ever grow ``defaults`` that must
be merged to resolve these values, swap :func:`discover_tasks` to use
``hydra.compose`` — the rest of the dashboard only depends on :class:`TaskInfo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

import dual_flexiv_control

#: The ``conf/task`` group shipped inside the installed package.
TASK_GROUP_DIR = Path(dual_flexiv_control.__file__).resolve().parent / "conf" / "task"


@dataclass(frozen=True)
class TaskInfo:
    """One launchable task, as shown in the dashboard dropdown."""

    name: str
    """Config stem — pass to Hydra as ``task=<name>``."""

    language_instruction: str
    """Natural-language goal, shared by collection and eval."""

    num_episodes: int | None
    """``collection.num_episodes`` (demonstrations to teleoperate), if declared."""

    num_timesteps: int | None
    """``eval.num_timesteps`` (rollout horizon), if declared."""

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
