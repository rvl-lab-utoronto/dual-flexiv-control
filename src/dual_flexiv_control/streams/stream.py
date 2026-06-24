"""Role-typed wrappers around :class:`SharedRingBuffer`.

A stream has exactly one :class:`StreamWriter` (the producing interface) and any
number of :class:`StreamReader` views (the brain and other consumers). Splitting
the roles keeps the producer/consumer contract explicit: a reader has no
``write`` method and never unlinks shared memory.
"""

from __future__ import annotations

import time

import numpy as np

from .registry import RegistryEntry
from .registry import StreamRegistry
from .registry import shm_name_for
from .ring import Samples
from .ring import SharedRingBuffer
from .spec import StreamSpec


class StreamWriter:
    """The single producer end of a stream. Owns the shared-memory segment."""

    def __init__(self, ring: SharedRingBuffer, spec: StreamSpec) -> None:
        self._ring = ring
        self.spec = spec

    @classmethod
    def create(
        cls,
        spec: StreamSpec,
        run_id: str,
        registry: StreamRegistry,
    ) -> "StreamWriter":
        """Allocate the segment, publish it to the registry, return the writer."""
        shm_name = shm_name_for(run_id, spec.name)
        ring = SharedRingBuffer.create(shm_name, spec.capacity, spec.dim, spec.dtype)
        try:
            registry.publish(spec, shm_name)
        except Exception:
            # Publish failed (e.g. read-only/full runtime_dir): don't orphan the
            # segment we just allocated — release it before propagating.
            ring.close()
            ring.unlink()
            raise
        return cls(ring, spec)

    @property
    def name(self) -> str:
        return self.spec.name

    def write(self, vec: np.ndarray, t_ns: int | None = None) -> int:
        """Publish one sample. If ``t_ns`` is omitted, stamps with the monotonic clock.

        Prefer passing a single ``t_ns`` captured once per control tick so that all
        of an arm's per-signal streams share one coherent timestamp.
        """
        if t_ns is None:
            t_ns = time.monotonic_ns()
        return self._ring.append(vec, t_ns)

    def close(self) -> None:
        self._ring.close()

    def unlink(self) -> None:
        self._ring.unlink()


class StreamReader:
    """A read-only consumer view of a stream in another process's memory."""

    def __init__(self, ring: SharedRingBuffer, entry: RegistryEntry) -> None:
        self._ring = ring
        self.entry = entry

    @classmethod
    def attach(cls, entry: RegistryEntry) -> "StreamReader":
        ring = SharedRingBuffer.attach(entry.shm_name)
        return cls(ring, entry)

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def dim(self) -> int:
        return self._ring.dim

    @property
    def capacity(self) -> int:
        return self._ring.capacity

    def last(self, k: int) -> Samples:
        """Return up to the newest ``k`` samples, oldest -> newest."""
        return self._ring.last(k)

    def latest(self) -> Samples:
        """Return the single newest sample (``n in {0, 1}``)."""
        return self._ring.latest()

    def close(self) -> None:
        self._ring.close()
