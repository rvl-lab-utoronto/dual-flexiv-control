"""Taught skills for the dashboard: Teach from a replayed episode, list, delete.

Backs the Viewer tab's **teach-and-repeat bar** and the Storage tab's 🎓 Teach
button. Thin dashboard glue over :mod:`dual_flexiv_control.skills` (the format,
episode extraction, and the replay node live there): this module resolves the
skills root from the composed config (``skill.root``, same cached-compose
pattern as :mod:`.storage`'s recording root) and re-exports the operations the
UI needs. Repeating a skill goes through :meth:`~.runner.RunRegistry.launch_skill`,
not this module — the daemon owns the arms.
"""

from __future__ import annotations

import os
import re
import threading

from dual_flexiv_control import skills as _core
from dual_flexiv_control.skills import SkillInfo

_LOCK = threading.Lock()
_ROOT: str | None = None


def skills_root() -> str:
    """Absolute ``skill.root`` (where taught skills live); composed once, cached."""
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

    from .arms import compose_overrides

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config", overrides=compose_overrides())
    return os.path.abspath(str(cfg.skill.root))


def discover() -> list[SkillInfo]:
    """Every taught skill under the skills root, sorted by name."""
    return _core.discover_skills(skills_root())


def teach_from_episode(ds, episode_index: int, name: str) -> SkillInfo:
    """Save one replayed episode's measured trajectory as skill ``name``.

    ``ds`` is a :class:`~.storage.DatasetInfo`. Reads only the proprio columns
    (no video decode), so teaching is quick even for long episodes. Overwrites
    an existing skill of the same name.
    """
    skill = _core.skill_from_episode(ds.path, ds.repo_id, int(episode_index), name)
    path = _core.save_skill(skill, skills_root())
    return SkillInfo(
        name=skill.name,
        path=str(path),
        sides=tuple(skill.sides),
        frames=skill.frames,
        fps=skill.fps,
        duration_s=skill.duration_s,
        created=skill.created,
        source=skill.source,
    )


def delete(name: str) -> None:
    _core.delete_skill(skills_root(), name)


def rename(old: str, new: str) -> None:
    """Rename a taught skill; refuses to clobber an existing name."""
    _core.rename_skill(skills_root(), old, new)


def default_name(repo_id: str, episode_index: int) -> str:
    """A suggested skill name for one episode, e.g. ``default-ep3``."""
    base = repo_id.rstrip("/").rsplit("/", 1)[-1] or "skill"
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.") or "skill"
    return f"{base}-ep{int(episode_index)}"
