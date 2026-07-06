"""Discover and read the flexiv control system's log files.

Backs the dashboard's **Logs** tab. Every ``dual-flexiv-control`` run — whether
launched from the dashboard (Collection/Eval) or straight from the CLI — is a
Hydra job that writes a ``system.log`` under its run dir (``hydra.run.dir``,
``outputs/<timestamp>/`` by default; see ``conf/config.yaml``). This module:

* finds those log files under the outputs root, newest first, and
* reads the tail of one (bounded, so a huge log never blows up the page).

The outputs root is resolved against the dashboard's cwd (the same cwd the
spawned system inherits — see :func:`~.runner._launch_collection`), so the logs
the tab lists are exactly the ones the launched runs write. Override the root
with ``DFC_OUTPUTS_DIR`` if Hydra's ``run.dir`` is customised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogFile:
    """One flexiv control run's log file (a Hydra job's ``system.log``)."""

    name: str
    """Display label — the run dir's name (e.g. ``2026-07-06_15-09-30``)."""

    path: Path
    """The log file on disk."""

    size_bytes: int
    """Current size, shown so a growing (live) run is visible at a glance."""

    mtime: float
    """Last-modified epoch seconds, used to sort newest-first."""


def outputs_root() -> Path:
    """The directory Hydra runs write under (``outputs/`` by default).

    Resolved against the current cwd so it matches where launched runs land;
    overridable with ``DFC_OUTPUTS_DIR`` when ``hydra.run.dir`` is customised.
    """
    return Path(os.environ.get("DFC_OUTPUTS_DIR", "outputs")).resolve()


def discover_logs(root: Path | None = None) -> list[LogFile]:
    """Every run's ``*.log`` under ``root``, newest (most recently written) first.

    Each Hydra job dir holds a single log named after the job (``system.log``);
    we glob ``*/*.log`` so a renamed job is still picked up. A run dir with no
    log yet (just spawned) is simply absent until its log appears.
    """
    root = root if root is not None else outputs_root()
    if not root.is_dir():
        return []
    files: list[LogFile] = []
    for path in root.glob("*/*.log"):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            LogFile(
                name=path.parent.name,
                path=path,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    files.sort(key=lambda f: f.mtime, reverse=True)
    return files


#: Cap the tail read so an enormous log never loads wholesale into the page. The
#: full window is shown (all its lines) — the Logs tab scrolls it in a fixed box.
_MAX_TAIL_BYTES = 256 * 1024


def read_tail(path: Path, max_bytes: int = _MAX_TAIL_BYTES) -> str:
    """Return the last ``max_bytes`` of ``path`` as text (decoded leniently).

    Reads from the end so a large, actively-growing log stays cheap, and returns
    every line in that window (no line cap — the caller scrolls it). When the file
    is longer than ``max_bytes`` the leading partial line is dropped and a
    truncation marker is prepended, so the operator knows earlier lines exist.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        # Drop the leading partial line, then flag the truncation.
        _, _, rest = text.partition("\n")
        text = f"… (showing last {max_bytes // 1024} KB of {size // 1024} KB)\n{rest}"
    return text


def human_size(n: int) -> str:
    """Compact human-readable byte size (matches the Storage tab's formatting)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
