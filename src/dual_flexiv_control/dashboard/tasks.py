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

#: The ``conf`` root and its ``task`` group, shipped inside the installed package.
CONF_DIR = Path(dual_flexiv_control.__file__).resolve().parent / "conf"
TASK_GROUP_DIR = CONF_DIR / "task"


@dataclass(frozen=True)
class TaskInfo:
    """One launchable entry in the dashboard dropdown — a task or a run profile.

    A **task** (``config_name is None``) is a ``conf/task`` file selected on the
    default config with ``task=<name>``. A **run profile** (``config_name`` set) is
    a top-level ``conf/<name>.yaml`` that reconfigures the whole run — different
    arms/cameras/factr — launched with ``--config-name <name>`` (e.g. the ``test``
    profile: a dummy arm + one real camera). Both share this shape so the dropdown
    and launcher treat them uniformly.
    """

    name: str
    """Config stem — a task passed as ``task=<name>``, or a profile's config name."""

    language_instruction: str
    """Natural-language goal, shared by collection and eval."""

    num_episodes: int | None
    """``collection.num_episodes`` (demonstrations to teleoperate), if declared."""

    num_timesteps: int | None
    """``eval.num_timesteps`` (rollout horizon), if declared."""

    path: Path
    """The source YAML, for display / click-through."""

    config_name: str | None = None
    """If set, launch with ``--config-name <config_name>`` (a whole-run profile)
    rather than ``task=<name>`` (a task on the default config)."""


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


def discover_profiles(conf_dir: Path = CONF_DIR) -> list[TaskInfo]:
    """Top-level run *profiles*: ``conf/<name>.yaml`` (except ``config``) that carry a
    full config (``defaults`` including ``base_config``).

    Unlike tasks, a profile reconfigures the whole run (arms/cameras/factr), so its
    task fields come via ``defaults`` — we Hydra-compose each to resolve them. A
    broken profile is skipped (never breaks the dropdown). Sorted by name.
    """
    candidates = [
        p.stem for p in sorted(conf_dir.glob("*.yaml"))
        if p.stem != "config" and _is_profile(p)
    ]
    if not candidates:
        return []

    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    profiles: list[TaskInfo] = []
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        for name in candidates:
            try:
                cfg = compose(config_name=name)
                task = cfg.get("task") or {}
                profiles.append(
                    TaskInfo(
                        name=name,
                        language_instruction=str(task.get("language_instruction", name)),
                        num_episodes=_maybe_int(task.get("collection") or {}, "num_episodes"),
                        num_timesteps=_maybe_int(task.get("eval") or {}, "num_timesteps"),
                        path=conf_dir / f"{name}.yaml",
                        config_name=name,
                    )
                )
            except Exception:  # noqa: BLE001 - a broken profile must not hide the rest
                continue
    return profiles


def discover_launchables() -> list[TaskInfo]:
    """Every dropdown entry: tasks (default config) followed by run profiles."""
    return discover_tasks() + discover_profiles()


def _is_profile(path: Path) -> bool:
    """A top-level config file (its ``defaults`` list pulls in ``base_config``)."""
    try:
        cfg = OmegaConf.load(path)
    except Exception:  # noqa: BLE001
        return False
    defaults = cfg.get("defaults") if hasattr(cfg, "get") else None
    return bool(defaults) and any(
        d == "base_config" or (isinstance(d, str) and d.strip() == "base_config")
        for d in defaults
    )


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
