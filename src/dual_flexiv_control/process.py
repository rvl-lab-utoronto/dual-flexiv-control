"""Process lifecycle: the run loop, rate limiting, and graceful shutdown.

Each component (each Flexiv arm, FACTR, the brain) runs as its own OS process.
Processes are **spawned**, never forked — flexivrdk starts background C++
threads and network services, and forking a process that has already loaded the
SDK would duplicate that state into a broken child. Spawn gives every node a
clean interpreter that constructs its own robot connection.

A :class:`StreamProducerNode` is the common shape for the interfaces: declare
streams, open a data source, then poll -> write at a fixed rate until asked to
stop. The brain is a consumer and implements :class:`ProcessNode` directly.
"""

from __future__ import annotations

import abc
import logging
import signal
import time

import numpy as np

from .streams.registry import StreamRegistry
from .streams.spec import StreamSpec
from .streams.stream import StreamWriter

log = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-rate pacing with drift correction and optional terminal spin.

    ``sleep()`` returns at successive multiples of the period measured from the
    first call. If a tick overruns, the schedule resyncs to *now* rather than
    trying to "catch up" with a burst (which would be wrong for sampling).

    ``spin_margin_s > 0`` busy-waits the final fraction of each period for
    sub-millisecond precision at high rates, at the cost of a hot core. Default
    0 (pure sleep) — good enough for reading; raise it for hard real-time.
    """

    def __init__(self, hz: float, spin_margin_s: float = 0.0) -> None:
        if hz <= 0:
            raise ValueError(f"rate must be positive, got {hz}")
        self.period = 1.0 / float(hz)
        self.spin_margin = spin_margin_s
        self._next: float | None = None

    def reset(self) -> None:
        self._next = time.perf_counter() + self.period

    def sleep(self) -> None:
        if self._next is None:
            self.reset()
        target = self._next
        remaining = target - time.perf_counter()
        if remaining > self.spin_margin:
            time.sleep(remaining - self.spin_margin)
        if self.spin_margin > 0:
            while time.perf_counter() < target:
                pass
        self._next = target + self.period
        now = time.perf_counter()
        if self._next < now:  # overran: resync, don't spiral into catch-up bursts
            self._next = now + self.period


class ProcessNode(abc.ABC):
    """A unit of work that owns one process's main loop."""

    name: str

    @abc.abstractmethod
    def run(self, stop_event) -> None:
        """Run until ``stop_event`` is set. Called inside the child process."""

    def cleanup(self) -> None:
        """Best-effort teardown hook, always called after :meth:`run` returns."""


class StreamProducerNode(ProcessNode):
    """Base for a producer: declare streams, open a source, poll -> write."""

    def __init__(
        self,
        name: str,
        runtime_dir: str,
        run_id: str,
        rate_hz: float,
    ) -> None:
        self.name = name
        self.runtime_dir = runtime_dir
        self.run_id = run_id
        self.rate_hz = rate_hz
        # Populated inside run() (the child process), never before spawn.
        self._writers: dict[str, StreamWriter] = {}

    # -- subclass contract ----------------------------------------------------

    @abc.abstractmethod
    def declare_streams(self) -> list[StreamSpec]:
        """The streams this producer publishes (called in the child)."""

    @abc.abstractmethod
    def open_source(self) -> None:
        """Connect to the underlying data source (robot, FACTR backend)."""

    @abc.abstractmethod
    def poll(self) -> dict[str, np.ndarray] | None:
        """Read one sample set. Map of ``stream_name -> vector``; ``None`` to skip a tick."""

    def close_source(self) -> None:
        """Disconnect the data source. Override if needed."""

    # -- run loop -------------------------------------------------------------

    def run(self, stop_event) -> None:
        registry = StreamRegistry(self.runtime_dir, self.run_id)
        try:
            for spec in self.declare_streams():
                self._writers[spec.name] = StreamWriter.create(spec, self.run_id, registry)
            log.info("[%s] published %d streams", self.name, len(self._writers))

            self.open_source()
            log.info("[%s] source open; producing at %.1f Hz", self.name, self.rate_hz)

            rate = RateLimiter(self.rate_hz)
            rate.reset()
            while not stop_event.is_set():
                sample = self.poll()
                if sample:
                    t_ns = time.monotonic_ns()  # one coherent stamp per tick
                    for stream_name, vec in sample.items():
                        self._writers[stream_name].write(vec, t_ns)
                rate.sleep()
        finally:
            self._teardown(registry)

    def _teardown(self, registry: StreamRegistry) -> None:
        try:
            self.close_source()
        except Exception:  # noqa: BLE001 - teardown must not raise
            log.exception("[%s] error closing source", self.name)
        for stream_name, writer in self._writers.items():
            try:
                writer.close()
                writer.unlink()
            except Exception:  # noqa: BLE001
                log.exception("[%s] error releasing stream %s", self.name, stream_name)
            registry.remove(stream_name)
        log.info("[%s] stopped", self.name)


def run_node(node: ProcessNode, stop_event) -> None:
    """Process entry point: install cooperative signal handlers, run, clean up.

    Used as the ``target`` of every spawned :class:`multiprocessing.Process`.
    SIGINT/SIGTERM just set the shared ``stop_event`` so the loop unwinds through
    its normal teardown path (releasing shared memory) instead of dying abruptly.
    """

    # Spawned children start with no logging handlers; give them one so their
    # diagnostics surface on stderr alongside the parent's.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        )

    def _handle(signum, _frame):  # noqa: ANN001
        log.info("[%s] signal %s -> stopping", getattr(node, "name", "?"), signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    try:
        node.run(stop_event)
    except Exception:  # noqa: BLE001
        log.exception("[%s] crashed", getattr(node, "name", "?"))
        stop_event.set()  # bring the rest of the system down with us
    finally:
        node.cleanup()
