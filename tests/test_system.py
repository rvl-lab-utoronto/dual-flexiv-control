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
from dual_flexiv_control.cameras import camera_stream_name
from dual_flexiv_control.cameras import reshape_frame
from dual_flexiv_control.configs import Config
from dual_flexiv_control.configs import register_configs
from dual_flexiv_control.interfaces.flexiv import FlexivInterface
from dual_flexiv_control.interfaces.zed import ZedInterface
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


def test_build_nodes_wires_per_phase_coeffs(tmp_path):
    """build_nodes selects runtime.phase's coeffs and injects them into every arm node."""
    from dual_flexiv_control.system import build_nodes

    cfg_c = _make_config(tmp_path, "runtime.phase=collection")
    arms_c = [n for n in build_nodes(cfg_c, make_run_id()) if isinstance(n, FlexivInterface)]
    assert arms_c, "expected FlexivInterface nodes"
    for n in arms_c:  # collection -> compliant
        assert n.coeffs.max_joint_vel == cfg_c.task.collection.coeffs.max_joint_vel
        assert (
            n.coeffs.cartesian_impedance.K_x[0]
            == cfg_c.task.collection.coeffs.cartesian_impedance.K_x[0]
        )

    cfg_e = _make_config(tmp_path, "runtime.phase=eval")
    arms_e = [n for n in build_nodes(cfg_e, make_run_id()) if isinstance(n, FlexivInterface)]
    for n in arms_e:  # eval -> stiff
        assert n.coeffs.max_joint_vel == cfg_e.task.eval.coeffs.max_joint_vel
    assert arms_c[0].coeffs.max_joint_vel != arms_e[0].coeffs.max_joint_vel  # phases differ


def test_build_nodes_rejects_bad_phase(tmp_path):
    from dual_flexiv_control.system import build_nodes

    cfg = _make_config(tmp_path, "runtime.phase=bogus")
    with pytest.raises(ValueError):
        build_nodes(cfg, make_run_id())


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


def test_zed_sim_camera_streams_cross_process(tmp_path):
    """A spawned sim ZED interface streams frames a parent reader sees and reshapes."""
    config = _make_config(
        tmp_path,
        # Shrink the static cam so the test is light/fast (sim ignores `resolution`).
        "cameras.static.width=64",
        "cameras.static.height=48",
        "cameras.static.capacity=8",
    )
    cam_name = "static"
    cam = config.cameras[cam_name]
    run_id = make_run_id()

    ctx = mp.get_context("spawn")
    node = ZedInterface(cam_name, cam, config.runtime, run_id)
    stop = ctx.Event()
    proc = ctx.Process(target=run_node, args=(node, stop), name=node.name)
    proc.start()
    try:
        reg = StreamRegistry(tmp_path, run_id)
        names = [camera_stream_name(cam_name, v) for v in cam.views]
        assert names == ["cam/static/left", "cam/static/right"]
        entries = reg.wait_for(names, timeout_s=10.0)
        readers = {n: StreamReader.attach(entries[n]) for n in names}

        left = camera_stream_name(cam_name, "left")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and readers[left].last(1).n == 0:
            time.sleep(0.02)

        s = readers[left].latest()
        assert s.n > 0, "no frames produced by sim ZED"
        assert s.data.dtype == np.uint8
        assert readers[left].dim == cam.width * cam.height * 3
        frame = reshape_frame(s.newest, cam, "left")
        assert frame.shape == (cam.height, cam.width, 3)
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
    node = BrainNode(config.brain, config.runtime, config.factr, run_id, ["never/published"], {})

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
    """The whole system (arms + cameras + brain) runs in sim, leaks no shared memory."""
    before = set(glob.glob("/dev/shm/dfc_*"))
    config = _make_config(
        tmp_path,
        "runtime.duration_s=2.0",
        "arms.left.rate_hz=200",
        "arms.right.rate_hz=200",
        "brain.rate_hz=50",
        # Shrink the camera frames so the spawned sim cameras stay light here.
        "cameras.wrist_left.width=64", "cameras.wrist_left.height=48",
        "cameras.wrist_right.width=64", "cameras.wrist_right.height=48",
        "cameras.static.width=64", "cameras.static.height=48",
    )
    # The brain's default subscription covers both arms' proprio (cameras are
    # produced but not auto-subscribed; FACTR is on-request, not streamed).
    assert set(default_stream_names(config.arms)) == {
        f"{side}/{sig}" for side in ("left", "right") for sig in config.arms[side].streams
    }
    assert set(config.cameras) == {"wrist_left", "wrist_right", "static"}

    run_system(config)

    after = set(glob.glob("/dev/shm/dfc_*"))
    assert before == after, f"leaked shared-memory segments: {after - before}"
