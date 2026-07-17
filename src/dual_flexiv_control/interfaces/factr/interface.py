"""The FACTR interface node: ONE process polling the leader servers, publishing streams.

FACTR leader data used to be fetched over HTTP independently by every consumer
(the collection loop, the brain node, the dashboard mirror, the status probe) —
four keep-alive connections that could disagree about reachability and payload
(the viewer showing live motion while control read the server's zeros
placeholder). This node makes FACTR a first-class stream producer like the arms
and cameras: it is the **only** HTTP reader, and everyone else attaches
read-only to the ``factr/<side>`` shared-memory streams it publishes — so the
viewer and the control path see the *same samples* by construction.

Stream contract: one ``factr/<side>`` stream per configured leader, dim =
``server.dof`` (the raw payload: arm joints + the trailing gripper value, in
radians), float64, at :attr:`FactrCfg.rate_hz`.

Outage behavior: per-side tolerant, never crashes. An unreachable leader is
logged once (and once on recovery) and simply publishes nothing — its stream
goes stale, which readers detect by sample age (:attr:`FactrCfg.max_age_s`).
The stream segments survive the outage, so attached readers resume seamlessly
when the server comes back; consumers hold their last real target meanwhile.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ...configs import FactrCfg
from ...configs import RuntimeCfg
from ...process import StreamProducerNode
from ...streams.spec import StreamSpec
from .client import FactrClient
from .client import FactrError

log = logging.getLogger(__name__)


def factr_stream_name(side: str) -> str:
    """The canonical stream name for one FACTR leader's raw payload."""
    return f"factr/{side}"


def leader_stream_names(cfg: FactrCfg) -> list[str]:
    """The ``factr/<side>`` stream names for every configured leader."""
    return [factr_stream_name(side) for side in cfg.servers]


def fresh_leader_positions(source, cfg: FactrCfg) -> dict[str, np.ndarray]:
    """The leaders' newest *fresh* joint positions from their streams: ``{side: array}``.

    ``source`` is any attached stream consumer with ``latest(name) -> Samples``
    (a :class:`~dual_flexiv_control.brain.Brain`); the FACTR semantics — which
    streams are leaders, and how old a sample may be (``cfg.max_age_s``) — live
    here, not in the consumer. A side whose stream is unsubscribed, empty, or
    stale (leader dropout) is simply omitted: callers hold their last real
    target for missing sides, never fabricate one.
    """
    max_age_ns = int(float(cfg.max_age_s) * 1e9)
    now_ns = time.monotonic_ns()
    out: dict[str, np.ndarray] = {}
    for side in cfg.servers:
        try:
            s = source.latest(factr_stream_name(side))
        except KeyError:
            continue
        if s.n == 0 or (now_ns - s.newest_t_ns) > max_age_ns:
            continue
        out[side] = np.asarray(s.newest, dtype=np.float64)
    return out


def wait_leaders_fresh(source, cfg: FactrCfg, timeout_s: float, stop_event=None) -> None:
    """Block until EVERY configured leader stream carries a fresh sample.

    The launch-time teleop preflight, judged on the streams the run actually
    reads: the producer publishes ``factr/<side>`` the moment it starts (so a
    successful attach alone proves nothing about data), and this waits for real
    samples. Raises :class:`FactrError` naming the missing side(s) on timeout —
    the caller surfaces it as a failed run. Returns quietly if ``stop_event``
    fires (shutdown unwinds normally). No-op with no configured leaders.
    """
    if not cfg.servers:
        return
    deadline = time.monotonic() + timeout_s
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        missing = set(cfg.servers) - set(fresh_leader_positions(source, cfg))
        if not missing:
            return
        if time.monotonic() > deadline:
            raise FactrError(
                f"FACTR teleop preflight failed — no fresh leader data on "
                f"stream(s): {sorted(missing)} (is the FACTR server up and its "
                "leader publisher running?)"
            )
        time.sleep(0.05)


class FactrInterface(StreamProducerNode):
    """Polls every configured FACTR leader server; publishes ``factr/<side>``.

    One node covers all leaders (they are HTTP endpoints, not exclusive devices),
    so a rig spawns exactly one ``factr`` process. ``runtime.sim`` propagates to
    the client, which then fabricates positions without a network.
    """

    def __init__(self, cfg: FactrCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(
            name="factr",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=cfg.rate_hz,
        )
        self.cfg = cfg
        self.sim = runtime.sim
        self._client: FactrClient | None = None
        #: sides currently failing, so outages log once (and once on recovery).
        self._errored: set[str] = set()

    @property
    def sides(self) -> list[str]:
        return list(self.cfg.servers)

    def declare_streams(self) -> list[StreamSpec]:
        return [
            StreamSpec(
                name=factr_stream_name(side),
                dim=server.dof,
                capacity=4096,
                dtype="float64",
                rate_hz=self.cfg.rate_hz,
            )
            for side, server in self.cfg.servers.items()
        ]

    def open_source(self) -> None:
        self._client = FactrClient.from_config(self.cfg, sim=self.sim)

    def poll(self) -> dict[str, np.ndarray] | None:
        sample: dict[str, np.ndarray] = {}
        for side in self._client.sides:
            try:
                jp = self._client.get_joint_positions_for(side)
            except FactrError as exc:
                if side not in self._errored:
                    self._errored.add(side)
                    log.warning(
                        "[factr] %s leader not reachable (stream goes stale until "
                        "it recovers): %s", side, exc,
                    )
                continue
            if side in self._errored:
                self._errored.discard(side)
                log.info("[factr] %s leader recovered", side)
            sample[factr_stream_name(side)] = jp
        return sample or None

    def close_source(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
