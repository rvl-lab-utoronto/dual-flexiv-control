"""The brain: the main processing pipeline.

The brain subscribes to streams (by attaching read-only views of the producers'
shared memory) and observes them — pulling the latest sample or the last ``k``
samples of any signal. It owns no streams of its own; it is a pure consumer.

:class:`Brain` is the reusable API (also usable in-process, e.g. tests).
:class:`BrainNode` wraps it in the standard process loop for the live system.
"""

from __future__ import annotations

import logging

from ..configs import BrainCfg
from ..configs import FactrCfg
from ..configs import RuntimeCfg
from ..interfaces.factr import FactrClient
from ..process import ProcessNode
from ..process import RateLimiter
from ..streams.registry import AttachAborted
from ..streams.registry import StreamRegistry
from ..streams.ring import Samples
from ..streams.stream import StreamReader

log = logging.getLogger(__name__)


def default_stream_names(arms) -> list[str]:
    """Every stream the brain observes by default: each arm's proprio signals.

    ``arms`` maps side -> ``ArmCfg`` (with ``.streams``); names follow the
    ``"<side>/<signal>"`` paths. FACTR is queried on request, not streamed.
    """
    names: list[str] = []
    for side, arm in arms.items():
        names.extend(f"{side}/{sig}" for sig in arm.streams)
    return names


class Brain:
    """Subscribes to streams and observes them; gets the last ``k`` elements."""

    def __init__(
        self,
        registry: StreamRegistry,
        stream_names: list[str],
        attach_timeout_s: float = 10.0,
        factr: FactrClient | None = None,
    ) -> None:
        self._registry = registry
        self._stream_names = list(stream_names)
        self._attach_timeout_s = attach_timeout_s
        self._readers: dict[str, StreamReader] = {}
        #: On-request FACTR client; query with :meth:`factr_joint_positions`.
        self.factr = factr

    def attach(self, stop_event=None) -> None:
        """Block until all subscribed streams exist, then attach read-only views.

        If ``stop_event`` is provided and set during the wait, raises
        :class:`AttachAborted` so a shutdown requested mid-attach unwinds promptly.
        """
        entries = self._registry.wait_for(
            self._stream_names, self._attach_timeout_s, stop_event=stop_event
        )
        for name in self._stream_names:
            self._readers[name] = StreamReader.attach(entries[name])
        log.info("brain attached to %d streams", len(self._readers))

    @property
    def stream_names(self) -> list[str]:
        return list(self._readers) or list(self._stream_names)

    def last(self, name: str, k: int) -> Samples:
        """Last ``k`` samples of one stream, oldest -> newest."""
        return self._readers[name].last(k)

    def latest(self, name: str) -> Samples:
        """The single newest sample of one stream."""
        return self._readers[name].latest()

    def observe(self) -> dict[str, Samples]:
        """Snapshot the newest sample of every subscribed stream."""
        return {name: reader.latest() for name, reader in self._readers.items()}

    def observe_last(self, k: int) -> dict[str, Samples]:
        """Snapshot the last ``k`` samples of every subscribed stream."""
        return {name: reader.last(k) for name, reader in self._readers.items()}

    def factr_joint_positions(self):
        """Fetch the FACTR leaders' joint positions on request: ``{side: array}``.

        Raises if no FACTR client was configured, or :class:`FactrError` if the
        server is unreachable / returns a bad response.
        """
        if self.factr is None:
            raise RuntimeError("brain has no FACTR client configured")
        return self.factr.get_joint_positions()

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        if self.factr is not None:
            self.factr.close()


class BrainNode(ProcessNode):
    """Runs the brain in its own process: attach, then observe at a fixed rate."""

    def __init__(
        self,
        brain: BrainCfg,
        runtime: RuntimeCfg,
        factr: FactrCfg,
        run_id: str,
        stream_names: list[str],
    ) -> None:
        self.name = "brain"
        self.cfg = brain
        self.runtime = runtime
        self.factr_cfg = factr
        self.run_id = run_id
        self.stream_names = stream_names
        self._brain: Brain | None = None

    def run(self, stop_event) -> None:
        registry = StreamRegistry(self.runtime.runtime_dir, self.run_id)
        factr = FactrClient.from_config(self.factr_cfg, sim=self.runtime.sim)
        brain = Brain(registry, self.stream_names, self.cfg.attach_timeout_s, factr=factr)
        try:
            brain.attach(stop_event=stop_event)
        except AttachAborted:
            log.info("brain attach aborted by shutdown")
            brain.close()
            return
        self._brain = brain

        rate = RateLimiter(self.cfg.rate_hz)
        rate.reset()
        ticks = 0
        heartbeat_every = max(1, int(self.cfg.rate_hz))  # ~1 s
        try:
            while not stop_event.is_set():
                observation = brain.observe()
                self.process(observation)
                ticks += 1
                if ticks % heartbeat_every == 0:
                    self._heartbeat(observation)
                rate.sleep()
        finally:
            brain.close()
            self._brain = None

    def process(self, observation: dict[str, Samples]) -> None:
        """Hook for the downstream control/policy logic. No-op for now.

        Override (or replace the brain loop) with the real processing pipeline.
        ``observation`` maps stream name -> newest :class:`Samples`; use
        ``self._brain.last(name, k)`` for windowed history.
        """

    def _heartbeat(self, observation: dict[str, Samples]) -> None:
        fresh = sum(1 for s in observation.values() if s.n > 0)
        log.info("brain heartbeat: %d/%d streams fresh", fresh, len(observation))
