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
        "rig=bimanual",  # exercise the full two-arm system (shipped default is now left_only)
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
        assert n.arm.control == cfg_c.policy.control

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


def test_control_session_actuates_gripper_from_mailbox(tmp_path):
    """End-to-end (in-process, sim): a control session discovers the gripper mailbox,
    sets the gripper up during bootstrap, and drives it from posted normalized targets.

    Runs the interface's real ``_control_session`` in a thread (not a spawned process)
    so its ``FakeFlexivSource`` stays directly inspectable, while the test plays the
    brain — opening setpoint/command/gripper channels and posting targets.
    """
    import threading

    from dual_flexiv_control.configs import ControlCoeffsCfg
    from dual_flexiv_control.control import COMMAND
    from dual_flexiv_control.control import SETPOINT
    from dual_flexiv_control.control import control_specs
    from dual_flexiv_control.control import gripper_spec
    from dual_flexiv_control.control import pack_streamed
    from dual_flexiv_control.streams.stream import StreamWriter

    config = _make_config(
        tmp_path,
        "arms.left.control_enabled=true",
        "arms.left.control_rate_hz=200",
        "arms.left.gripper.enabled=true",
        "arms.left.gripper.name=G",
    )
    arm = config.arms["left"]
    run_id = make_run_id()

    node = FlexivInterface("left", arm, config.runtime, run_id)
    # Set up exactly like run() before a session: telemetry writers + sim source.
    registry = StreamRegistry(tmp_path, run_id)
    for spec in node.declare_streams():
        node._writers[spec.name] = StreamWriter.create(spec, run_id, registry)
    node.open_source()
    source = node._source  # keep a handle (teardown nulls node._source)

    stop = threading.Event()
    th = threading.Thread(
        target=node._control_session, args=(stop, ControlCoeffsCfg()), daemon=True
    )
    th.start()

    creg = StreamRegistry(tmp_path, run_id, sub="control")
    specs = control_specs("left", arm.control)
    # Gripper channel FIRST (as brain.open_control does) so it exists once the arm
    # sees setpoint+command and looks it up.
    gw = StreamWriter.create(gripper_spec("left", arm.control.channel), run_id, creg)
    sw = StreamWriter.create(specs[SETPOINT], run_id, creg)
    cw = StreamWriter.create(specs[COMMAND], run_id, creg)
    try:
        q_d = np.linspace(0.0, 0.3, 7)
        setpoint = pack_streamed(arm.control, {"q_d": q_d, "dq_d": np.zeros(7)})
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and source.last_gripper is None:
            sw.write(setpoint)                      # keep the setpoint fresh (deadman)
            gw.write(np.array([0.7], dtype=np.float64))
            time.sleep(0.02)
        assert source.last_gripper == pytest.approx(0.7), "gripper mailbox never actuated"
    finally:
        stop.set()
        th.join(timeout=5.0)
        for w in (gw, sw, cw):
            w.close()
            w.unlink()
        node._teardown(registry)
        cleanup_run(tmp_path, run_id)


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
        # Proprio streams (from config) + the dashboard status stream (declared by
        # the interface, not part of arm.streams) — the arm must publish both.
        names = [f"left/{sig}" for sig in arm.streams] + ["left/status"]
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

        # Status: operational state, E-stop, control, servo state, actual RDK mode.
        # The sim reports READY (1) with the E-stop clear (0) — what makes the
        # dashboard show "connected" rather than "disconnected" — and this
        # producer enters its configured control session (control_active 1).
        status = readers["left/status"]
        assert status.dim == 5
        assert status.last(1).n > 0, "sim Flexiv published no status"
        newest = status.latest().newest
        np.testing.assert_array_equal(newest, [1.0, 0.0, 1.0, 1.0, 1.0])
        for r in readers.values():
            r.close()
    finally:
        stop.set()
        proc.join(timeout=10.0)
        assert proc.exitcode == 0
        cleanup_run(tmp_path, run_id)


def test_dashboard_reads_arm_as_connected_while_running(tmp_path):
    """The dashboard's own consumer reports a running sim arm as connected, not disconnected.

    This is the end-to-end regression for the bug: proprio streamed fine but the
    dashboard always read "disconnected" because no ``<side>/status`` stream existed.
    """
    from dual_flexiv_control.dashboard.arms import ArmInfo
    from dual_flexiv_control.dashboard.arms import read_arm_status

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
        reg.wait_for(["left/status"], timeout_s=10.0)

        info = ArmInfo(side="left", name="Lauer", serial=arm.serial, dof=arm.dof)
        deadline = time.monotonic() + 5.0
        status = read_arm_status(info, runtime_dir=str(tmp_path))
        while time.monotonic() < deadline and status.source != "live":
            time.sleep(0.05)
            status = read_arm_status(info, runtime_dir=str(tmp_path))

        assert status.source == "live", "dashboard still reads the running arm as disconnected"
        assert status.mode == "Idle"
        assert status.operational_status == "Auto (Remote)"
        assert status.servo_enabled is True
        assert status.estop_pressed is False  # sim E-stop clear
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
        assert names == ["cam/static/left", "cam/static/right", "cam/static/depth"]
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

        depth_name = camera_stream_name(cam_name, "depth")
        d = readers[depth_name].latest()
        assert d.n > 0, "no depth frames produced by sim ZED"
        assert d.data.dtype == np.float32
        depth_frame = reshape_frame(d.newest, cam, "depth")
        assert depth_frame.shape == (cam.height, cam.width)
        assert np.isfinite(depth_frame).all() and (depth_frame > 0).all()  # metres
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
    # produced but not auto-subscribed; FACTR is published by its own producer).
    assert set(default_stream_names(config.arms)) == {
        f"{side}/{sig}" for side in ("left", "right") for sig in config.arms[side].streams
    }
    assert set(config.cameras) == {"wrist_left", "wrist_right", "static"}

    # Pin the run id so the leak check covers exactly THIS run's segments —
    # other sessions on the host (e.g. a live dashboard) create and destroy
    # their own dfc_* segments concurrently, so a global before/after is flaky.
    from dual_flexiv_control.streams.registry import sanitize

    run_id = make_run_id()
    run_system(config, run_id=run_id)

    leaked = glob.glob(f"/dev/shm/dfc_{sanitize(run_id)}_*")
    assert not leaked, f"leaked shared-memory segments: {leaked}"


# ---------------------------------------------------------------------------
# Shutdown grace: the recording node ("collection") gets the long save window
# ---------------------------------------------------------------------------

def _exit_now():  # spawn target: a "hardware" node that unwinds immediately
    pass


def _exit_after(seconds: float):  # spawn target: a "saver" mid video-finalize
    time.sleep(seconds)


def test_shutdown_lets_the_recording_node_finish_saving(monkeypatch, tmp_path):
    """The collection node outlives the hardware escalation window while saving.

    Regression: the old _shutdown gave EVERY node the same short cooperative
    window, so a Stop during a long episode save SIGKILLed the save mid-encode
    and the episode never committed. Now hardware nodes keep the short window,
    while the node named "collection" gets runtime.save_grace_s.
    """
    import threading as _threading

    from dual_flexiv_control import system as system_mod

    monkeypatch.setattr(system_mod, "_HARDWARE_GRACE_S", 0.3)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    # The "save" (1.5s) far exceeds the hardware window (0.3s + 2s SIGTERM grace
    # would previously have killed it at ~2.3s only if > that; use a saver that
    # would NOT survive the old uniform window scaled down, but does survive now).
    saver = ctx.Process(target=_exit_after, args=(1.5,), name="collection")
    hw = ctx.Process(target=_exit_now, name="flexiv:left")
    saver.start(); hw.start()
    # Let the immediately-exiting hardware node finish booting first: under a
    # loaded machine, spawn latency alone can exceed the 0.3s window and the
    # SIGTERM escalation would race the interpreter start, flaking the test.
    hw.join(timeout=30.0)
    t0 = time.monotonic()
    system_mod._shutdown([hw, saver], stop, save_grace_s=10.0)
    elapsed = time.monotonic() - t0
    assert saver.exitcode == 0, "the saving node must exit cleanly, not be killed"
    assert hw.exitcode == 0
    assert elapsed < 8.0  # returned as soon as the save finished, not after grace

    # Force (repeated operator signals) abandons the wait: a wedged saver is
    # escalated promptly instead of holding shutdown for the whole grace.
    stop2 = ctx.Event()
    wedged = ctx.Process(target=_exit_after, args=(30.0,), name="collection")
    wedged.start()
    time.sleep(0.2)  # let it boot so the kill has a live target
    force = _threading.Event()
    force.set()
    t0 = time.monotonic()
    system_mod._shutdown([wedged], stop2, save_grace_s=30.0, force=force)
    assert time.monotonic() - t0 < 10.0, "force must cut the save wait short"
    assert wedged.exitcode != 0, "the wedged saver is terminated, not waited out"


def test_shutdown_still_kills_a_wedged_saver_after_grace(tmp_path):
    """SIGKILL remains the backstop: overrun the save grace -> the node dies."""
    from dual_flexiv_control import system as system_mod

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    wedged = ctx.Process(target=_exit_after, args=(30.0,), name="collection")
    wedged.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    system_mod._shutdown([wedged], stop, save_grace_s=0.5)
    assert time.monotonic() - t0 < 10.0
    assert wedged.exitcode != 0  # SIGTERM/SIGKILL, never left orphaned


# ---------------------------------------------------------------------------
# Arm idle ↔ control state machine (session hosting)
# ---------------------------------------------------------------------------


def _wait_until(cond, timeout_s: float = 10.0, msg: str = "condition not met in time"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(msg)


def test_arm_session_idle_control_idle(tmp_path):
    """A session-hosted arm idles read-only, runs a control session per EnterControl,
    and returns to IDLE (telemetry + connection intact) — repeatably.

    Runs the node in-process (a thread + a plain queue stand in for the spawned
    process + mp.Queue) with the sim source, and plays the brain's side by hand:
    create the control channels, feed setpoints (the deadman needs them), send
    STOP, and watch ``<side>/status``'s ``control_active`` flag flip 0 → 1 → 0.
    """
    import queue
    import threading

    from dual_flexiv_control.control import COMMAND
    from dual_flexiv_control.control import SETPOINT
    from dual_flexiv_control.control import CommandKind
    from dual_flexiv_control.control import ControlCommand
    from dual_flexiv_control.control import control_specs
    from dual_flexiv_control.control import pack_streamed
    from dual_flexiv_control.interfaces.flexiv.interface import EnterControl
    from dual_flexiv_control.streams.stream import StreamWriter

    config = _make_config(
        tmp_path,
        "arms.left.control_enabled=true",
        "arms.left.control_rate_hz=100",
        "arms.left.rate_hz=100",
    )
    arm = config.arms["left"]
    run_id = make_run_id()
    session_q: queue.Queue = queue.Queue()
    node = FlexivInterface(
        "left", arm, config.runtime, run_id, session_q=session_q
    )
    stop = threading.Event()
    thread = threading.Thread(target=node.run, args=(stop,), daemon=True)
    thread.start()

    feed_stop = threading.Event()
    feeder: threading.Thread | None = None
    try:
        reg = StreamRegistry(tmp_path, run_id)
        entries = reg.wait_for(["left/status", "left/q"], timeout_s=10.0)
        status = StreamReader.attach(entries["left/status"])
        _wait_until(lambda: status.latest().n > 0, msg="no status published")
        assert status.latest().newest[2] == 0.0, "arm must boot into IDLE"

        # -- the brain's side of one control session ---------------------------
        control_reg = StreamRegistry(tmp_path, run_id, sub="control")
        specs = control_specs("left", arm.control)
        sp_writer = StreamWriter.create(specs[SETPOINT], run_id, control_reg)
        cmd_writer = StreamWriter.create(specs[COMMAND], run_id, control_reg)
        setpoint = pack_streamed(
            arm.control,
            {"q_d": np.zeros(arm.dof), "dq_d": np.zeros(arm.dof)},
        )

        def _feed():  # keep the deadman fed for the session's whole life
            while not feed_stop.is_set():
                sp_writer.write(setpoint)
                time.sleep(0.02)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()

        session_q.put(EnterControl(
            coeffs=config.task.collection.coeffs,
            control=config.policy.control,
            phase="collection",
        ))
        _wait_until(
            lambda: status.latest().newest[2] == 1.0,
            msg="arm never entered the control session",
        )

        cmd_writer.write(ControlCommand(CommandKind.STOP).encode(cmd_writer.spec.dim))
        _wait_until(
            lambda: status.latest().newest[2] == 0.0,
            msg="arm never returned to IDLE after STOP",
        )

        # Telemetry survives the session end (connection + streams intact).
        q_reader = StreamReader.attach(entries["left/q"])
        t0 = q_reader.latest().newest_t_ns
        _wait_until(
            lambda: q_reader.latest().newest_t_ns > t0,
            msg="telemetry stalled after the control session",
        )
        q_reader.close()

        # -- a second session proves re-entry works ----------------------------
        session_q.put(EnterControl(
            coeffs=config.task.eval.coeffs,
            control=config.policy.control,
            phase="eval",
        ))
        _wait_until(
            lambda: status.latest().newest[2] == 1.0,
            msg="arm never entered the SECOND control session",
        )
        cmd_writer.write(ControlCommand(CommandKind.STOP).encode(cmd_writer.spec.dim))
        _wait_until(
            lambda: status.latest().newest[2] == 0.0,
            msg="arm never returned to IDLE after the second STOP",
        )

        status.close()
        feed_stop.set()
        feeder.join(timeout=2.0)
        for writer in (sp_writer, cmd_writer):
            writer.close()
            writer.unlink()
    finally:
        feed_stop.set()
        if feeder is not None and feeder.is_alive():
            feeder.join(timeout=2.0)
        stop.set()
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "arm node did not exit on stop_event"
        cleanup_run(tmp_path, run_id)
