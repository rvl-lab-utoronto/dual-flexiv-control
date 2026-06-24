"""Cross-process stream discovery.

Producers and consumers live in different processes (spawned, not forked), so a
consumer cannot be handed Python objects describing the streams — it must
*discover* them. The registry is a directory of tiny JSON manifest files, one
per stream, under ``<runtime_dir>/<run_id>/streams/``.

Why one file per stream (instead of a single shared manifest): each producer
writes only its own files, so there is never write contention and never a need
for locking. A consumer discovers streams by listing the directory. Writes are
atomic (temp file + ``os.replace``) so a reader never sees a half-written entry.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import StreamSpec

_SAFE = re.compile(r"[^A-Za-z0-9_]+")


class AttachAborted(Exception):
    """Raised by :meth:`StreamRegistry.wait_for` when a stop is requested mid-wait."""


def sanitize(name: str) -> str:
    """Map a logical stream name to a filesystem/POSIX-shm-safe token."""
    return _SAFE.sub("_", name)


def shm_name_for(run_id: str, stream_name: str) -> str:
    """Deterministic POSIX shared-memory segment name for a stream in a run.

    POSIX shm names must not contain ``/`` and are length-limited; the logical
    name (e.g. ``right/tau``) is sanitised to ``right_tau``.
    """
    return f"dfc_{sanitize(run_id)}_{sanitize(stream_name)}"


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A discovered, attachable stream."""

    name: str
    shm_name: str
    dim: int
    capacity: int
    dtype: str
    rate_hz: float | None
    pid: int

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "shm_name": self.shm_name,
            "dim": self.dim,
            "capacity": self.capacity,
            "dtype": self.dtype,
            "rate_hz": self.rate_hz,
            "pid": self.pid,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RegistryEntry":
        return cls(
            name=d["name"],
            shm_name=d["shm_name"],
            dim=int(d["dim"]),
            capacity=int(d["capacity"]),
            dtype=d["dtype"],
            rate_hz=d["rate_hz"],
            pid=int(d["pid"]),
        )


class StreamRegistry:
    """Publish/discover stream manifests for a single run."""

    def __init__(self, runtime_dir: str | os.PathLike[str], run_id: str) -> None:
        self.run_id = run_id
        self.root = Path(runtime_dir) / run_id
        self.dir = self.root / "streams"

    # -- producer side --------------------------------------------------------

    def publish(self, spec: StreamSpec, shm_name: str) -> RegistryEntry:
        """Record that ``spec`` is now live at shared-memory segment ``shm_name``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = RegistryEntry(
            name=spec.name,
            shm_name=shm_name,
            dim=spec.dim,
            capacity=spec.capacity,
            dtype=spec.dtype,
            rate_hz=spec.rate_hz,
            pid=os.getpid(),
        )
        path = self.dir / f"{sanitize(spec.name)}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry.to_json()))
        os.replace(tmp, path)  # atomic publish
        return entry

    def remove(self, name: str) -> None:
        """Remove a stream's manifest entry (on producer teardown)."""
        path = self.dir / f"{sanitize(name)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # -- consumer side --------------------------------------------------------

    def discover(self) -> dict[str, RegistryEntry]:
        """Return every currently-published stream, keyed by logical name."""
        out: dict[str, RegistryEntry] = {}
        if not self.dir.is_dir():
            return out
        for path in self.dir.glob("*.json"):
            try:
                entry = RegistryEntry.from_json(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # racing with a writer / partial; skip this tick
            out[entry.name] = entry
        return out

    def get(self, name: str) -> RegistryEntry | None:
        return self.discover().get(name)

    def wait_for(
        self,
        names: list[str],
        timeout_s: float,
        poll_s: float = 0.02,
        stop_event=None,
    ) -> dict[str, RegistryEntry]:
        """Block until all ``names`` are published.

        Raises ``TimeoutError`` if they do not appear within ``timeout_s``, or
        :class:`AttachAborted` if ``stop_event`` is set while waiting (so a
        shutdown requested mid-attach unwinds promptly instead of blocking for the
        full timeout). Uses :func:`time.monotonic`; safe from any consumer process.
        """
        import time

        deadline = time.monotonic() + timeout_s
        wanted = set(names)
        while True:
            found = self.discover()
            if wanted.issubset(found):
                return {n: found[n] for n in names}
            if stop_event is not None and stop_event.is_set():
                raise AttachAborted("stream attach aborted: stop requested")
            if time.monotonic() >= deadline:
                missing = sorted(wanted - set(found))
                raise TimeoutError(
                    f"streams not published within {timeout_s}s: {missing}"
                )
            time.sleep(poll_s)


def cleanup_run(runtime_dir: str | os.PathLike[str], run_id: str) -> int:
    """Unlink every shared-memory segment for a run and delete its manifest dir.

    Returns the number of shared-memory segments unlinked. Safe to call after a
    crash to reclaim leaked ``/dev/shm`` segments. Driven by the manifest so it
    only touches this run's segments.
    """
    from multiprocessing import shared_memory

    registry = StreamRegistry(runtime_dir, run_id)
    unlinked = 0
    for entry in registry.discover().values():
        try:
            shm = shared_memory.SharedMemory(name=entry.shm_name, create=False)
            shm.close()
            shm.unlink()
            unlinked += 1
        except FileNotFoundError:
            pass
    # Remove the manifest directory tree.
    root = registry.root
    if root.is_dir():
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                path.unlink() if path.is_file() else path.rmdir()
            except OSError:
                pass
        try:
            root.rmdir()
        except OSError:
            pass
    return unlinked


def cleanup_cli() -> None:
    """Console entry point: ``dfc-cleanup-shm <runtime_dir> <run_id>``."""
    import sys

    if len(sys.argv) != 3:
        print("usage: dfc-cleanup-shm <runtime_dir> <run_id>", file=sys.stderr)
        raise SystemExit(2)
    n = cleanup_run(sys.argv[1], sys.argv[2])
    print(f"unlinked {n} shared-memory segment(s) for run {sys.argv[2]!r}")
