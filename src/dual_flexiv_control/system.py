r"""System orchestrator (Hydra entry point): spawn and supervise every node.

Topology for the bimanual setup::

    FlexivInterface(left)   --\
    FlexivInterface(right)  ---\  shared-memory streams   /-- CollectionNode (teleop+record)
    ZedInterface(wrist_left) --->                         \-- or EvalNode (policy rollout)
    ZedInterface(wrist_right) -/
    ZedInterface(static)    --/

One process per arm (proprio) and one per ZED camera (frames), plus one
consumer selected by ``runtime.phase``: collection (FACTR teleop -> LeRobot
recording) or eval (policy-server client -> setpoints). FACTR is not a spawned
node: the collection brain holds a ``FactrClient`` and queries the FACTR
server's joint-position endpoint on demand.

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
import threading
import time
import uuid

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .collection import CollectionNode
from .configs import Config
from .configs import register_configs
from .interfaces.flexiv import FlexivInterface
from .interfaces.zed import ZedInterface
from .policy import EvalNode
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
    """The set of spawned nodes for a run: one process per arm, one per camera, + the brain.

    FACTR is not a node — it is an on-request HTTP client the brain holds. Camera
    streams are produced unconditionally but are not in the brain's default
    subscription (proprio only); subscribe to them via ``brain.subscribe`` (see
    :func:`dual_flexiv_control.cameras.camera_stream_names`).
    """
    # Active-phase controller coefficients (collection vs eval) applied by every
    # control-enabled arm; the brain posts setpoints, the arms apply these coeffs.
    if config.runtime.phase not in ("collection", "eval"):
        raise ValueError(
            f"runtime.phase must be 'collection' or 'eval', got {config.runtime.phase!r}"
        )
    active_coeffs = getattr(config.task, config.runtime.phase).coeffs

    nodes: list[ProcessNode] = [
        FlexivInterface(side, arm, config.runtime, run_id, coeffs=active_coeffs)
        for side, arm in config.arms.items()
    ]
    nodes += [
        ZedInterface(name, cam, config.runtime, run_id)
        for name, cam in config.cameras.items()
    ]
    # The phase selects the consumer: collection runs the recording teleop loop
    # (reads FACTR + proprio, commands the arms at the collection frequency, samples
    # all cameras software-synchronised, and exports LeRobot demos); eval runs the
    # policy rollout (same observation schema, actions from the policy server).
    if config.runtime.phase == "collection":
        nodes.append(
            CollectionNode(
                config.task,
                config.runtime,
                config.factr,
                config.brain,
                config.recording,
                run_id,
                config.arms,
                config.cameras,
            )
        )
    else:
        nodes.append(
            EvalNode(
                config.task,
                config.runtime,
                config.policy,
                config.brain,
                run_id,
                config.arms,
                config.cameras,
            )
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

    # Repeated signals escalate: the 1st stops cooperatively (the recording node
    # then gets runtime.save_grace_s to finalize its episode video); the 3rd sets
    # ``force`` so _shutdown abandons that wait — the operator's escape hatch from
    # a genuinely wedged save.
    force = threading.Event()
    signal_count = [0]

    def _handle(signum, _frame):  # noqa: ANN001
        signal_count[0] += 1
        if signal_count[0] == 1:
            log.info("orchestrator received signal %s -> stopping "
                     "(saving may take a while; Ctrl-C twice more to abandon it)", signum)
        elif signal_count[0] == 2:
            log.warning("stop already in progress — Ctrl-C once more to abandon the episode save")
        else:
            log.warning("repeated signals -> abandoning the episode save")
            force.set()
        stop_event.set()

    prev_int = signal.signal(signal.SIGINT, _handle)
    prev_term = signal.signal(signal.SIGTERM, _handle)

    deadline = None if duration_s is None else time.monotonic() + duration_s
    crashed_node = None  # (name, exitcode) of the first node to die abnormally, if any
    try:
        for proc in procs:
            proc.start()
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                log.info("duration %.1fs elapsed -> stopping", duration_s)
                break
            for proc in procs:
                if not proc.is_alive():
                    # exitcode 0 == a node finished cleanly (e.g. collection hit its
                    # episode target); anything else is a crash we must surface.
                    if proc.exitcode not in (0, None):
                        log.warning(
                            "node %s crashed (exit code %s); stopping system",
                            proc.name,
                            proc.exitcode,
                        )
                        crashed_node = (proc.name, proc.exitcode)
                    else:
                        log.info(
                            "node %s finished (exit code %s); stopping system",
                            proc.name,
                            proc.exitcode,
                        )
                    stop_event.set()
                    break
            time.sleep(0.05)
    finally:
        _shutdown(procs, stop_event, save_grace_s=config.runtime.save_grace_s, force=force)
        n = cleanup_run(config.runtime.runtime_dir, run_id)
        log.info("shutdown complete; unlinked %d shm segment(s)", n)
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
    # Propagate a node crash as a non-zero process exit so the dashboard (which keys
    # its error popup off the exit code) surfaces it instead of reporting "finished".
    if crashed_node is not None:
        name, code = crashed_node
        raise SystemExit(f"node {name!r} crashed (exit code {code}); see log above")


#: The consumer node that finalizes an episode (a long video encode) on shutdown.
#: It gets a far larger cooperative window than the hardware nodes.
_RECORDING_NODE = "collection"
#: Prompt cooperative windows for hardware nodes (close the RDK/ZED handle fast).
_HARDWARE_GRACE_S = 5.0
_TERM_GRACE_S = 2.0


def _shutdown(procs, stop_event, save_grace_s: float = 300.0, force=None) -> None:
    """Cooperative stop -> SIGTERM -> SIGKILL escalation. Leaves no orphans.

    The recording consumer ("collection") may be draining a multi-thousand-frame
    video encode as it saves the in-progress episode on shutdown — that legitimately
    takes far longer than a hardware node's teardown. It gets a generous
    ``save_grace_s`` cooperative window so the episode actually commits; the hardware
    nodes keep the short window so a wedged RDK call is force-killed promptly and
    never keeps a live robot connection. The waits return the instant a proc exits,
    so a healthy node never waits out its whole window. Setting ``force`` (a
    ``threading.Event``, from repeated operator signals) abandons the cooperative
    waits and escalates immediately.
    """
    stop_event.set()
    saver = next((p for p in procs if p.name == _RECORDING_NODE), None)
    hardware = [p for p in procs if p is not saver]
    # Hardware/eval nodes first: prompt cooperative unwind then escalation, so a
    # wedged arm is SIGKILLed within seconds regardless of how long the save runs.
    _escalate(hardware, cooperative_s=_HARDWARE_GRACE_S, force=force)
    # Recording consumer: wait out the episode-save video finalize (SIGKILL backstop).
    # Its frames are already buffered, so tearing the hardware down first is safe.
    if saver is not None:
        _escalate([saver], cooperative_s=save_grace_s, force=force)


def _escalate(procs, cooperative_s: float, force=None) -> None:
    """Join cooperatively, SIGTERM stragglers, then SIGKILL the genuinely wedged.

    ``cooperative_s`` bounds the wait for a clean unwind (cut short if ``force`` is
    set); SIGKILL (uncatchable, so no cooperative handler can block it) is the final
    backstop — no orphan survives, so a wedged child never keeps a live robot
    connection.
    """
    deadline = time.monotonic() + cooperative_s
    for proc in procs:
        # Slice the join so repeated operator signals (force) cut the wait short.
        while proc.is_alive() and time.monotonic() < deadline:
            if force is not None and force.is_set():
                break
            proc.join(timeout=0.25)
    for proc in procs:
        if proc.is_alive():
            log.warning("node %s still alive; sending SIGTERM", proc.name)
            proc.terminate()
    for proc in procs:
        if proc.is_alive():
            proc.join(timeout=_TERM_GRACE_S)
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
