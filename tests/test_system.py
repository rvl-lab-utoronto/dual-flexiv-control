"""Cross-process integration tests using the real spawn machinery.

These spawn actual processes (sim sources, no hardware) so they exercise the
full shared-memory + registry + lifecycle path exactly as production does. The
config is composed by Hydra just like the live system.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import time

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.brain import BrainNode
from dual_flexiv_control.brain import default_stream_names
from dual_flexiv_control.configs import Config
from dual_flexiv_control.configs import register_configs
from dual_flexiv_control.interfaces.flexiv import FlexivInterface
from dual_flexiv_control.process import run_node
from dual_flexiv_control.proprio import POSE_SIZE
from dual_flexiv_control.streams.registry import StreamRegistry
from dual_flexiv_control.streams.registry import cleanup_run
from dual_flexiv_control.streams.stream import StreamReader
from dual_flexiv_control.system import make_run_id
from dual_flexiv_control.system import run_system


def _make_config(tmp_path, *overrides: str) -> Config:
    register_configs()
    base = [
        "runtime.sim=true",
        f"runtime.runtime_dir={tmp_path}",
    ]
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=base + list(overrides))
    return OmegaConf.to_object(cfg)


def test_flexiv_sim_streams_cross_process(tmp_path):
    """A spawned sim Flexiv interface streams proprio that a parent reader sees."""
    config = _make_config(tmp_path, "arms.left.rate_hz=500")
    arm = config.arms["left"]
    run_id = make_run_id()

    ctx = mp.get_context("spawn")
    node = FlexivInterface("left", arm, config.runtime, run_id)
    stop = ctx.Event()
    proc = ctx.Process(target=run_node, args=(node, stop), name=node.name)
    proc.start()
    try:
        reg = StreamRegistry(tmp_path, run_id)
        names = [f"left/{sig}" for sig in arm.streams]
        entries = reg.wait_for(names, timeout_s=10.0)
        readers = {n: StreamReader.attach(entries[n]) for n in names}

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and readers["left/q"].last(1).n == 0:
            time.sleep(0.02)

        assert readers["left/q"].dim == arm.dof
        assert readers["left/q"].last(1).n > 0, "no data produced by sim Flexiv"
        assert readers["left/eef"].latest().newest.shape == (POSE_SIZE,)
        assert readers["left/wrench"].dim == 6
        win = readers["left/q"].last(10)
        assert np.all(np.diff(win.t_ns) >= 0)
        for r in readers.values():
            r.close()
    finally:
        stop.set()
        proc.join(timeout=10.0)
        assert proc.exitcode == 0
        cleanup_run(tmp_path, run_id)


def test_brain_attach_aborts_promptly_on_stop(tmp_path):
    """Fix #4: a stop requested while the brain waits for (never-published) streams
    must abort the attach promptly, not block for the full attach_timeout_s."""
    config = _make_config(tmp_path, "brain.attach_timeout_s=30")
    run_id = make_run_id()
    node = BrainNode(config.brain, config.runtime, config.factr, run_id, ["never/published"])

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    proc = ctx.Process(target=run_node, args=(node, stop), name="brain")
    t0 = time.monotonic()
    proc.start()
    time.sleep(0.5)
    stop.set()
    proc.join(timeout=10.0)
    elapsed = time.monotonic() - t0
    try:
        assert not proc.is_alive()
        assert proc.exitcode == 0
        assert elapsed < 8.0, f"attach did not abort on stop (took {elapsed:.1f}s)"
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        cleanup_run(tmp_path, run_id)


@pytest.mark.slow
def test_full_system_sim_clean_shutdown(tmp_path):
    """The whole 4-node system runs in sim and leaves no shared memory behind."""
    before = set(glob.glob("/dev/shm/dfc_*"))
    config = _make_config(
        tmp_path,
        "runtime.duration_s=2.0",
        "arms.left.rate_hz=200",
        "arms.right.rate_hz=200",
        "brain.rate_hz=50",
    )
    # The brain's default subscription covers both arms' proprio (FACTR is
    # on-request, not streamed).
    assert set(default_stream_names(config.arms)) == {
        f"{side}/{sig}" for side in ("left", "right") for sig in config.arms[side].streams
    }

    run_system(config)

    after = set(glob.glob("/dev/shm/dfc_*"))
    assert before == after, f"leaked shared-memory segments: {after - before}"
