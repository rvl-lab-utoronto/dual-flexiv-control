r"""System orchestrator (Hydra entry point): spawn and supervise every node.

Topology for the bimanual setup::

    FlexivInterface(left)  --\  shared-memory streams   /-- BrainNode (reads all)
    FlexivInterface(right) ---/                         \-- (+ on-request FACTR client)

FACTR is not a spawned node: the brain holds a ``FactrClient`` and queries the
FACTR server's joint-position endpoint on demand.

All nodes run as **spawned** processes sharing a single stop ``Event``. The
parent supervises: if any node dies, it signals the rest to unwind, joins them,
escalates to SIGKILL for anything wedged, then unlinks the run's shm segments.

Configuration is composed by Hydra from ``conf/`` (validated against the
structured schema in :mod:`dual_flexiv_control.configs`). Override anything from
the CLI, e.g. ``runtime.sim=true runtime.duration_s=10 control@arms.left.control=force``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
import uuid

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .brain import BrainNode
from .brain import default_stream_names
from .configs import Config
from .configs import register_configs
from .interfaces.flexiv import FlexivInterface
from .process import ProcessNode
from .process import run_node
from .streams.registry import cleanup_run

log = logging.getLogger(__name__)

# Register structured-config schemas with Hydra's ConfigStore at import time so
# they are available when @hydra.main composes (and in spawned children that
# re-import this module).
register_configs()


def make_run_id() -> str:
    """Short, collision-resistant id namespacing this run's shm segments."""
    return f"{os.getpid()}_{uuid.uuid4().hex[:8]}"


def build_nodes(config: Config, run_id: str) -> list[ProcessNode]:
    """The set of spawned nodes for a run: one process per arm + the brain.

    FACTR is not a node — it is an on-request HTTP client the brain holds.
    """
    nodes: list[ProcessNode] = [
        FlexivInterface(side, arm, config.runtime, run_id)
        for side, arm in config.arms.items()
    ]
    stream_names = config.brain.subscribe or default_stream_names(config.arms)
    nodes.append(
        BrainNode(config.brain, config.runtime, config.factr, run_id, stream_names)
    )
    return nodes


def run_system(config: Config, run_id: str | None = None) -> None:
    """Launch every node, supervise, and tear everything down cleanly."""
    ctx = mp.get_context("spawn")  # never fork: flexivrdk has live threads/services
    run_id = run_id or make_run_id()
    duration_s = config.runtime.duration_s
    log.info(
        "starting run_id=%s sim=%s runtime_dir=%s",
        run_id,
        config.runtime.sim,
        config.runtime.runtime_dir,
    )

    stop_event = ctx.Event()
    nodes = build_nodes(config, run_id)
    procs = [
        ctx.Process(target=run_node, args=(node, stop_event), name=node.name)
        for node in nodes
    ]

    def _handle(signum, _frame):  # noqa: ANN001
        log.info("orchestrator received signal %s -> stopping", signum)
        stop_event.set()

    prev_int = signal.signal(signal.SIGINT, _handle)
    prev_term = signal.signal(signal.SIGTERM, _handle)

    deadline = None if duration_s is None else time.monotonic() + duration_s
    try:
        for proc in procs:
            proc.start()
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                log.info("duration %.1fs elapsed -> stopping", duration_s)
                break
            for proc in procs:
                if not proc.is_alive():
                    log.warning(
                        "node %s exited early (code %s); stopping system",
                        proc.name,
                        proc.exitcode,
                    )
                    stop_event.set()
                    break
            time.sleep(0.05)
    finally:
        _shutdown(procs, stop_event)
        n = cleanup_run(config.runtime.runtime_dir, run_id)
        log.info("shutdown complete; unlinked %d shm segment(s)", n)
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def _shutdown(procs, stop_event) -> None:
    """Cooperative stop -> SIGTERM -> SIGKILL escalation. Leaves no orphans."""
    stop_event.set()
    # 1. Cooperative: let nodes unwind through their normal teardown.
    for proc in procs:
        proc.join(timeout=5.0)
    # 2. SIGTERM for stragglers (children treat it cooperatively).
    for proc in procs:
        if proc.is_alive():
            log.warning("node %s still alive; sending SIGTERM", proc.name)
            proc.terminate()
    for proc in procs:
        if proc.is_alive():
            proc.join(timeout=2.0)
    # 3. SIGKILL anything genuinely wedged (e.g. blocked in a hung RDK call):
    #    cooperative handlers cannot intercept SIGKILL, so no orphan survives —
    #    critical so a wedged child never keeps a live robot connection.
    for proc in procs:
        if proc.is_alive():
            log.error("node %s unresponsive; sending SIGKILL", proc.name)
            proc.kill()
            proc.join(timeout=2.0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert the validated DictConfig into plain, typed, picklable dataclasses
    # (so nodes survive the spawn boundary unchanged), then resolve the runtime
    # dir to an absolute path against the launch cwd (hydra.job.chdir=false).
    config: Config = OmegaConf.to_object(cfg)
    if not os.path.isabs(config.runtime.runtime_dir):
        config.runtime.runtime_dir = os.path.abspath(config.runtime.runtime_dir)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    run_system(config)


if __name__ == "__main__":
    main()
