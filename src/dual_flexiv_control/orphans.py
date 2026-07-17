"""Reap orphaned dfc processes and their stale shared memory / run dirs.

Hardware nodes are multiprocessing *spawn* children. When their supervisor (the
session daemon or the one-shot orchestrator) dies hard — SIGKILL, OOM, a closed
terminal — the children can survive, reparented to init **or a desktop
subreaper** (observed on this machine: ppid of a real orphan was a session
manager, not 1 — so ppid is deliberately no part of any heuristic here). Such
orphans keep ``/dev/shm/dfc_*`` segments mapped and ZED/RDK device handles
open, which is exactly what wedges the next session ("camera detects but won't
open").

Everything here anchors on a **provably dead supervisor**; a live pid always
protects its processes and segments, so other live checkouts sharing this
machine are never touched, and unreadable (other-uid) processes are skipped
outright. Two detectors:

* **tag-based** (primary): every dfc supervisor stamps
  ``DFC_SUPERVISOR_PID=<its pid>`` into its environment before spawning
  anything (:func:`tag_supervisor`); spawn children inherit it, fixed at exec
  in their ``/proc/<pid>/environ``. A same-uid process whose tag names a dead
  pid is an orphan. Cheap and precise — used at boot *and* periodically.
* **legacy heuristic** (pre-tag debris): a process whose cmdline is an mp
  ``spawn_main`` child AND whose ``/proc/<pid>/maps`` maps only
  ``/dev/shm/dfc_<pid>_<hex8>_*`` segments with dead embedded creator pids
  (:func:`~dual_flexiv_control.system.make_run_id` embeds the supervisor pid).
  Sharper edge, so it runs at boot only.

The stale-*data* sweeps are independent of the process ones: any
``/dev/shm/dfc_*`` segment or ``<runtime_dir>/<run_id>`` manifest dir whose
embedded creator pid is dead (and isn't explicitly kept) is reclaimed.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Environment tag naming a process's dfc supervisor (see module doc).
ENV_TAG = "DFC_SUPERVISOR_PID"

_SHM_DIR = "/dev/shm"
#: A run's shm segments: ``dfc_<supervisor pid>_<uuid hex8>_<stream>``.
_SHM_RE = re.compile(r"^dfc_(\d+)_([0-9a-f]{8})_")
#: A run's manifest dir under the runtime dir: ``<supervisor pid>_<uuid hex8>``.
_RUN_DIR_RE = re.compile(r"^(\d+)_([0-9a-f]{8})$")
#: dfc shm segments in a /proc/<pid>/maps line.
_MAPS_RE = re.compile(r"/dev/shm/dfc_(\d+)_[0-9a-f]{8}_")

_TERM_GRACE_S = 5.0


def tag_supervisor() -> None:
    """Stamp this process as a dfc supervisor (call before spawning anything).

    Children inherit the tag through their (exec-time) environment, which makes
    any future orphans of THIS supervisor provable: tag pid dead == orphan.
    """
    os.environ[ENV_TAG] = str(os.getpid())


def pid_alive(pid) -> bool:
    """Is ``pid`` a live process? (signal-0 probe; EPERM counts as alive)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, not ours
    return True


# ---------------------------------------------------------------------------
# Detection (pure classification over plain records, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcRecord:
    """The facts about one candidate process the detectors consume."""

    pid: int
    tag: str | None                 # DFC_SUPERVISOR_PID from its environ, if any
    cmdline: tuple[str, ...]
    mapped_creators: tuple[int, ...]  # creator pids of the dfc shm it maps


def classify_orphan(rec: ProcRecord, *, self_pid: int, alive=pid_alive) -> str | None:
    """Why ``rec`` is an orphan (``"tagged"`` / ``"legacy"``), or None.

    A tag naming a LIVE supervisor protects the process outright — even from
    the legacy heuristic. Untagged processes qualify only via the legacy path:
    an mp ``spawn_main`` child mapping dfc shm segments whose creators are ALL
    dead (a single live creator protects it: better to leak than to kill a
    live run's node).
    """
    if rec.pid == self_pid:
        return None
    if rec.tag is not None:
        if not rec.tag.isdigit():
            return None  # malformed tag: unknown provenance — leave it alone
        return "tagged" if not alive(int(rec.tag)) else None
    if not any("spawn_main" in arg for arg in rec.cmdline):
        return None
    if not rec.mapped_creators:
        return None
    if any(alive(pid) for pid in rec.mapped_creators):
        return None
    return "legacy"


def _read_proc_record(pid: int) -> ProcRecord | None:
    """Build a :class:`ProcRecord` from /proc, or None (gone / not ours)."""
    base = f"/proc/{pid}"
    try:
        if os.stat(base).st_uid != os.getuid():
            return None
        with open(f"{base}/environ", "rb") as f:
            environ = f.read()
        with open(f"{base}/cmdline", "rb") as f:
            cmdline_raw = f.read()
        with open(f"{base}/maps") as f:
            maps = f.read()
    except (OSError, PermissionError):
        return None  # vanished mid-scan, kernel thread, or not readable: skip
    tag = None
    for entry in environ.split(b"\0"):
        if entry.startswith(ENV_TAG.encode() + b"="):
            tag = entry.split(b"=", 1)[1].decode(errors="replace")
            break
    cmdline = tuple(
        a.decode(errors="replace") for a in cmdline_raw.split(b"\0") if a
    )
    creators = tuple(sorted({int(m) for m in _MAPS_RE.findall(maps)}))
    return ProcRecord(pid=pid, tag=tag, cmdline=cmdline, mapped_creators=creators)


def find_orphans(include_legacy: bool = False) -> list[tuple[ProcRecord, str]]:
    """Scan /proc for provable dfc orphans; ``(record, reason)`` per hit.

    Same-uid processes only; anything unreadable is skipped (never touched).
    ``include_legacy`` adds the pre-tag heuristic — boot-time only by policy.
    """
    self_pid = os.getpid()
    out: list[tuple[ProcRecord, str]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        rec = _read_proc_record(int(entry))
        if rec is None:
            continue
        reason = classify_orphan(rec, self_pid=self_pid)
        if reason is None or (reason == "legacy" and not include_legacy):
            continue
        out.append((rec, reason))
    return out


# ---------------------------------------------------------------------------
# Reaping
# ---------------------------------------------------------------------------


def kill_orphans(pids: list[int], term_grace_s: float = _TERM_GRACE_S) -> list[int]:
    """SIGTERM the pids, wait out a grace, SIGKILL survivors. Returns the killed."""
    killed: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            continue  # already gone (or not ours after all): leave it be
    deadline = time.monotonic() + term_grace_s
    remaining = list(killed)
    while remaining and time.monotonic() < deadline:
        remaining = [pid for pid in remaining if pid_alive(pid)]
        if remaining:
            time.sleep(0.1)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return killed


def sweep_stale_shm(keep_run_ids: tuple = (), shm_dir: str = _SHM_DIR) -> int:
    """Unlink ``dfc_*`` shm segments whose embedded creator pid is dead.

    ``keep_run_ids`` protects named runs regardless (belt and braces beside the
    live-pid check). Returns the number of segments unlinked. Driven purely by
    the segment NAME, so it reclaims debris from any checkout on this machine
    — a live creator always protects its run's segments.
    """
    keep = set(keep_run_ids)
    removed = 0
    try:
        names = os.listdir(shm_dir)
    except OSError:
        return 0
    for name in names:
        m = _SHM_RE.match(name)
        if m is None:
            continue
        run_id = f"{m.group(1)}_{m.group(2)}"
        if run_id in keep or pid_alive(int(m.group(1))):
            continue
        try:
            os.unlink(os.path.join(shm_dir, name))
            removed += 1
        except OSError:
            pass
    return removed


def sweep_stale_run_dirs(runtime_dir, keep_run_ids: tuple = ()) -> int:
    """``cleanup_run`` every dead-creator run dir under ``runtime_dir``.

    Removes the manifest tree AND any shm segments its manifests still name
    (a second net beside :func:`sweep_stale_shm` for oddly-named segments).
    Returns the number of run dirs cleaned.
    """
    from .streams.registry import cleanup_run

    keep = set(keep_run_ids)
    cleaned = 0
    try:
        entries = os.listdir(runtime_dir)
    except OSError:
        return 0
    for entry in entries:
        m = _RUN_DIR_RE.match(entry)
        if m is None or entry in keep or pid_alive(int(m.group(1))):
            continue
        if not os.path.isdir(os.path.join(str(runtime_dir), entry)):
            continue
        cleanup_run(runtime_dir, entry)
        cleaned += 1
    return cleaned


def sweep(
    runtime_dir,
    keep_run_ids: tuple = (),
    include_legacy: bool = False,
) -> None:
    """Kill provable orphans, then reclaim dead-creator shm + run dirs.

    ``include_legacy=True`` is the boot-time mode (a fresh session is exactly
    when wedged ZEDs must be reclaimable); the periodic mode keeps the sharper
    pre-tag heuristic off. Never raises — a sweep failure must not take the
    session with it.
    """
    try:
        orphans = find_orphans(include_legacy=include_legacy)
        for rec, reason in orphans:
            log.warning(
                "orphaned dfc process pid=%d (%s: supervisor dead) — killing: %s",
                rec.pid, reason, " ".join(rec.cmdline)[:160],
            )
        if orphans:
            kill_orphans([rec.pid for rec, _ in orphans])
        n_shm = sweep_stale_shm(keep_run_ids)
        n_dirs = sweep_stale_run_dirs(runtime_dir, keep_run_ids)
        if orphans or n_shm or n_dirs:
            log.info(
                "orphan sweep: killed %d process(es), unlinked %d shm segment(s), "
                "cleaned %d stale run dir(s)",
                len(orphans), n_shm, n_dirs,
            )
    except Exception:  # noqa: BLE001 - reaping is best-effort housekeeping
        log.exception("orphan sweep failed (continuing)")
