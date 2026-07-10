"""Discover and read the flexiv control system's log files.

Backs the dashboard's **Logs** tab, which shows two kinds of log:

* A run launched straight from the CLI (``dual-flexiv-control``) goes through
  ``@hydra.main`` and writes a ``system.log`` under its run dir
  (``hydra.run.dir``, ``outputs/<timestamp>/`` by default; see
  ``conf/config.yaml``). :func:`discover_logs` finds those, newest first.
* A run launched from the dashboard goes through the persistent session daemon
  (:mod:`~.session`), which composes each run via the Hydra *compose* API and
  spawns only the consumer node — it never runs a fresh ``@hydra.main`` job, so
  there is no per-run ``system.log``. Instead all of a daemon's dashboard-run
  output accumulates in one ``session.log`` for the daemon's lifetime, written
  under a ``session_<timestamp>/`` dir in the outputs root (see
  :class:`~.session.SessionManager`) so :func:`discover_logs` lists it once the
  daemon exits; :func:`live_daemon_log` additionally flags the currently-live
  one so the tab can follow it while it grows.

Either way, :func:`read_tail` reads the tail of the selected file (bounded, so
a huge log never blows up the page).

The outputs root is resolved against the dashboard's cwd, so
:func:`discover_logs` lists exactly what a direct CLI launch writes there.
Override the root with ``DFC_OUTPUTS_DIR`` if Hydra's ``run.dir`` is customised.
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


def live_daemon_log(path: str | None) -> LogFile | None:
    """Wrap the session daemon's own log file as a :class:`LogFile`, if live.

    ``path`` is :attr:`~.session.SessionView.log_path` — the daemon most
    recently spawned by this dashboard process, if any (even a dead one, for
    post-mortem reading). ``None`` if no daemon has been spawned yet, or its
    log file is gone.
    """
    if not path:
        return None
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return None
    return LogFile(name="live session daemon", path=p, size_bytes=stat.st_size, mtime=stat.st_mtime)


def human_size(n: int) -> str:
    """Compact human-readable byte size (matches the Storage tab's formatting)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
