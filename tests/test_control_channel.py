"""Tests for the brain→arm control channel: specs, packing, mailbox, command cursor,
and an in-process brain→fake-arm setpoint round-trip."""

from __future__ import annotations

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.configs import ControlCoeffsCfg
from dual_flexiv_control.configs import register_configs
from dual_flexiv_control.control import COMMAND
from dual_flexiv_control.control import SETPOINT
from dual_flexiv_control.control import CommandCursor
from dual_flexiv_control.control import CommandKind
from dual_flexiv_control.control import ControlCommand
from dual_flexiv_control.control import control_channel_name
from dual_flexiv_control.control import control_specs
from dual_flexiv_control.control import gripper_channel_name
from dual_flexiv_control.control import gripper_spec
from dual_flexiv_control.control import pack_streamed
from dual_flexiv_control.control import setpoint_dim
from dual_flexiv_control.control import slice_streamed
from dual_flexiv_control.streams.registry import StreamRegistry
from dual_flexiv_control.streams.stream import StreamReader
from dual_flexiv_control.streams.stream import StreamWriter


def _ctrl(kind: str):
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=[f"control@policy.control={kind}"])
    return OmegaConf.to_object(cfg).policy.control


@pytest.mark.parametrize(
    "kind,expected",
    [("qpos", 14), ("qvel", 7), ("end_effector", 13), ("eef_vel", 6), ("force", 13)],
)
def test_setpoint_dim_per_kind(kind, expected):
    assert setpoint_dim(_ctrl(kind)) == expected


def test_pack_slice_round_trip():
    ctrl = _ctrl("qpos")
    q = np.arange(7.0)
    dq = np.arange(7.0) + 100
    v = pack_streamed(ctrl, {"q_d": q, "dq_d": dq})
    assert v.shape == (14,)
    fields = slice_streamed(ctrl, v)
    np.testing.assert_allclose(fields["q_d"], q)
    np.testing.assert_allclose(fields["dq_d"], dq)


@pytest.mark.parametrize(
    "kind,field,dim",
    [("qpos", "q_d", 7), ("qvel", "dq_d", 7), ("end_effector", "pose_d", 7),
     ("eef_vel", "twist_d", 6), ("force", "wrench_d", 6)],
)
def test_action_field_and_dim_per_kind(kind, field, dim):
    from dual_flexiv_control.control import action_dim
    from dual_flexiv_control.control import action_field

    ctrl = _ctrl(kind)
    assert action_field(ctrl) == field
    assert action_dim(ctrl) == dim


@pytest.mark.parametrize("kind", ["qpos", "qvel", "end_effector", "eef_vel"])
def test_pack_action_fills_primary_and_zeros_feedforward(kind):
    """A policy's primary field lands in the setpoint; feedforward fields are zeroed."""
    from dual_flexiv_control.control import action_dim
    from dual_flexiv_control.control import action_field
    from dual_flexiv_control.control import pack_action

    ctrl = _ctrl(kind)
    primary = np.arange(1.0, action_dim(ctrl) + 1.0)
    setpoint = pack_action(ctrl, primary)
    assert setpoint.shape == (setpoint_dim(ctrl),)
    fields = slice_streamed(ctrl, setpoint)
    np.testing.assert_allclose(fields[action_field(ctrl)], primary)
    for f, vec in fields.items():
        if f != action_field(ctrl):
            np.testing.assert_allclose(vec, 0.0)  # non-primary streamed fields zeroed


def test_pack_action_force_holds_measured_pose():
    """force streams wrench_d (policy) + pose_d (held at the measured TCP pose)."""
    from dual_flexiv_control.control import action_hold_fields
    from dual_flexiv_control.control import pack_action

    ctrl = _ctrl("force")
    assert action_hold_fields(ctrl) == ["pose_d"]
    wrench = np.arange(1.0, 7.0)
    pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    setpoint = pack_action(ctrl, wrench, held={"pose_d": pose})
    fields = slice_streamed(ctrl, setpoint)
    np.testing.assert_allclose(fields["wrench_d"], wrench)
    np.testing.assert_allclose(fields["pose_d"], pose)
    # without the measured hold it must refuse rather than command a zero pose
    with pytest.raises(ValueError):
        pack_action(ctrl, wrench)


@pytest.mark.parametrize(
    "kind,hk,signals",
    [("qpos", "q", []), ("qvel", "q", ["q"]), ("end_effector", "eef", []),
     ("eef_vel", "eef", ["eef"]), ("force", None, [])],
)
def test_horizon_kind_and_signals_per_kind(kind, hk, signals):
    from dual_flexiv_control.control import horizon_kind
    from dual_flexiv_control.control import horizon_signals

    ctrl = _ctrl(kind)
    assert horizon_kind(ctrl) == hk
    assert horizon_signals(ctrl) == signals


def test_estimate_chunk_end_per_kind():
    """The chunk-end viz estimate: absolute kinds take the last action; velocity
    kinds Euler-integrate from the measured baseline; force predicts nothing."""
    from dual_flexiv_control.control import estimate_chunk_end

    dt = 0.1
    q0 = np.arange(7.0)
    pose0 = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

    hk, v = estimate_chunk_end(_ctrl("qpos"), np.tile(np.arange(7.0), (4, 1)), dt)
    assert hk == "q"
    np.testing.assert_allclose(v, np.arange(7.0))

    dq_chunk = np.ones((5, 7))
    hk, v = estimate_chunk_end(_ctrl("qvel"), dq_chunk, dt, {"q": q0})
    assert hk == "q"
    np.testing.assert_allclose(v, q0 + 5 * dt)  # q0 + Σ dq·dt

    pose_chunk = np.tile(np.array([0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]), (3, 1))
    hk, v = estimate_chunk_end(_ctrl("end_effector"), pose_chunk, dt)
    assert hk == "eef"
    np.testing.assert_allclose(v, [0.4, 0.5, 0.6])  # position part of the last pose_d

    twist_chunk = np.tile(np.array([1.0, 2.0, 3.0, 9.0, 9.0, 9.0]), (2, 1))
    hk, v = estimate_chunk_end(_ctrl("eef_vel"), twist_chunk, dt, {"eef": pose0})
    assert hk == "eef"
    np.testing.assert_allclose(v, pose0[:3] + np.array([1.0, 2.0, 3.0]) * 2 * dt)

    assert estimate_chunk_end(_ctrl("force"), np.ones((3, 6)), dt) is None


def test_estimate_chunk_trajectory_integrates_every_qvel_step():
    from dual_flexiv_control.control import estimate_chunk_trajectory

    q0 = np.arange(7.0)
    dq = np.vstack([
        np.ones(7),
        np.arange(7.0),
        -np.ones(7),
    ])
    kind, path = estimate_chunk_trajectory(_ctrl("qvel"), dq, 0.1, {"q": q0})

    assert kind == "q"
    assert path.shape == (3, 7)
    np.testing.assert_allclose(path, q0 + np.cumsum(dq, axis=0) * 0.1)

    q_targets = np.vstack([q0 + 1, q0 + 2])
    kind, absolute_path = estimate_chunk_trajectory(_ctrl("qpos"), q_targets, 0.1)
    assert kind == "q"
    np.testing.assert_allclose(absolute_path, q_targets)


def test_estimate_chunk_end_needs_measured_baseline_for_velocity_kinds():
    """A velocity chunk is relative: with no measured baseline there is nothing to
    integrate from, so the estimate is None (side skipped, not fabricated)."""
    from dual_flexiv_control.control import estimate_chunk_end

    assert estimate_chunk_end(_ctrl("qvel"), np.ones((3, 7)), 0.1) is None
    assert estimate_chunk_end(_ctrl("eef_vel"), np.ones((3, 6)), 0.1, {}) is None


def test_normalize_gripper_passthrough_when_uncalibrated():
    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control import normalize_gripper

    conv = JointConventionCfg()  # gripper_open/closed default None
    assert normalize_gripper(0.73, conv) == pytest.approx(0.73)   # raw radians unchanged
    assert normalize_gripper(-1.5, conv) == pytest.approx(-1.5)


def test_normalize_gripper_maps_and_clips_to_unit_interval():
    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control import normalize_gripper

    conv = JointConventionCfg(gripper_open=0.2, gripper_closed=1.2)  # 1.0 rad span
    assert normalize_gripper(0.2, conv) == pytest.approx(0.0)     # open endpoint
    assert normalize_gripper(1.2, conv) == pytest.approx(1.0)     # closed endpoint
    assert normalize_gripper(0.7, conv) == pytest.approx(0.5)     # midpoint
    assert normalize_gripper(-1.0, conv) == pytest.approx(0.0)    # below open -> clip
    assert normalize_gripper(9.0, conv) == pytest.approx(1.0)     # above closed -> clip


def test_normalize_gripper_handles_reversed_endpoints():
    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control import normalize_gripper

    # closed reading below the open reading (sign-flipped leader) still maps correctly
    conv = JointConventionCfg(gripper_open=1.0, gripper_closed=0.0)
    assert normalize_gripper(1.0, conv) == pytest.approx(0.0)
    assert normalize_gripper(0.0, conv) == pytest.approx(1.0)
    assert normalize_gripper(0.5, conv) == pytest.approx(0.5)


def test_gripper_spec_dim_and_name():
    """The gripper mailbox is a latest-wins scalar, named cmd/<side>/gripper, and is
    orthogonal to the joint control kind (adding it never widens the setpoint vector)."""
    ctrl = _ctrl("qpos")
    spec = gripper_spec("left", ctrl.channel)
    assert spec.name == gripper_channel_name("left") == "cmd/left/gripper"
    assert spec.dim == 1
    assert spec.capacity == ctrl.channel.setpoint_capacity
    assert setpoint_dim(ctrl) == 14  # gripper does NOT ride the joint setpoint vector


def test_gripper_mailbox_round_trip_drives_fake_gripper(tmp_path):
    """A normalized gripper target posted on the mailbox actuates the fake source."""
    from dual_flexiv_control.configs import GripperCfg
    from dual_flexiv_control.interfaces.flexiv import FakeFlexivSource

    ctrl = _ctrl("qpos")
    reg = StreamRegistry(tmp_path, "rid", sub="control")
    spec = gripper_spec("left", ctrl.channel)
    writer = StreamWriter.create(spec, "rid", reg)
    try:
        reader = StreamReader.attach(reg.get(spec.name))
        src = FakeFlexivSource("sim", dof=7)
        src.open()
        assert src.setup_gripper(GripperCfg(enabled=True, name="G")) is True

        writer.write(np.array([0.25], dtype=np.float64))
        writer.write(np.array([0.80], dtype=np.float64))  # latest-wins
        g = reader.latest()
        src.send_gripper(float(g.newest[0]))
        assert src.last_gripper == pytest.approx(0.80)
        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_fake_source_reports_actual_mode_through_control_lifecycle():
    """The sim status mode mirrors the state machine like a real ``robot.mode()``:
    IDLE -> the control mode switched into by start_control -> IDLE on stop."""
    from types import SimpleNamespace

    from dual_flexiv_control.interfaces.flexiv import FakeFlexivSource
    from dual_flexiv_control.interfaces.flexiv.source import _rdk_mode_code

    src = FakeFlexivSource("sim", dof=7)
    src.open()
    assert src.read_status()[3] == 1.0  # IDLE

    # Joint kind: the configured RDK mode (NRT_JOINT_POSITION for default qpos).
    ctrl = _ctrl("qpos")
    rs = SimpleNamespace(q=np.zeros(7), tcp_pose=np.zeros(7))
    assert src.start_control(ctrl, None, {"q_d": np.zeros(7)}, rs) is True
    assert src.read_status()[3] == _rdk_mode_code(ctrl.mode) == 6.0
    src.stop()
    assert src.read_status()[3] == 1.0  # back to IDLE

    # Cartesian kind: always NRT_CARTESIAN_MOTION_FORCE, like the real source.
    ctrl = _ctrl("end_effector")
    assert src.start_control(ctrl, None, {"pose_d": np.zeros(7)}, rs) is True
    assert src.read_status()[3] == 10.0
    src.stop()
    assert src.read_status()[3] == 1.0


def test_fake_source_gripper_setup_declines_when_disabled_or_unnamed():
    from dual_flexiv_control.configs import GripperCfg
    from dual_flexiv_control.interfaces.flexiv import FakeFlexivSource

    src = FakeFlexivSource("sim", dof=7)
    assert src.setup_gripper(GripperCfg(enabled=False, name="G")) is False
    assert src.setup_gripper(GripperCfg(enabled=True, name="")) is False
    assert src.gripper_ready is False
    src.send_gripper(0.5)  # not ready -> ignored
    assert src.last_gripper is None


def test_flexiv_source_setup_gripper_reads_params_and_clamps():
    """setup_gripper enables/homes once, maps widths from params(), clamps vel/force."""
    pytest.importorskip("flexivrdk")
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import flexivrdk

    from dual_flexiv_control.configs import GripperCfg
    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    params = SimpleNamespace(
        min_width=0.01, max_width=0.11, min_vel=0.02, max_vel=0.15,
        min_force=5.0, max_force=40.0, name="G",
    )
    handle = MagicMock()
    handle.params.return_value = params
    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()
    with patch.object(flexivrdk, "Gripper", return_value=handle) as GripperCls:
        ok = src.setup_gripper(GripperCfg(enabled=True, name="G", velocity=999.0, force=1.0))
        assert ok is True
        assert src.setup_gripper(GripperCfg(enabled=True, name="G")) is True  # idempotent
    GripperCls.assert_called_once_with(src._robot)
    handle.Enable.assert_called_once_with("G")
    handle.Init.assert_called_once()  # homed (init_on_start default True)
    assert (src._gripper_open_w, src._gripper_closed_w) == (0.11, 0.01)  # open=max, closed=min
    assert src._gripper_vel == pytest.approx(0.15)  # clamped down to max_vel
    assert src._gripper_force == pytest.approx(5.0)  # clamped up to min_force


def test_flexiv_source_setup_gripper_degrades_on_bad_name():
    """A wrong/unconfigured gripper name must NOT crash the arm: Enable raising leaves
    the gripper off (returns False, no handle, send_gripper a no-op)."""
    pytest.importorskip("flexivrdk")
    from unittest.mock import MagicMock, patch

    import flexivrdk

    from dual_flexiv_control.configs import GripperCfg
    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    handle = MagicMock()
    handle.Enable.side_effect = RuntimeError("no gripper named 'Bogus'")
    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()
    with patch.object(flexivrdk, "Gripper", return_value=handle):
        assert src.setup_gripper(GripperCfg(enabled=True, name="Bogus")) is False
    assert src.gripper_ready is False
    src.send_gripper(0.8)  # not set up -> no Move, no raise
    handle.Move.assert_not_called()


def test_flexiv_source_send_gripper_throttles_deadbands_and_maps():
    """send_gripper: first send always goes; then throttled to move_rate_hz + deadbanded,
    and the normalized value maps linearly open_w..closed_w (clipped to [0,1])."""
    pytest.importorskip("flexivrdk")
    from unittest.mock import MagicMock

    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    src = FlexivSource("sim", dof=7)
    handle = MagicMock()
    src._gripper = handle
    src._gripper_open_w, src._gripper_closed_w = 0.10, 0.00
    src._gripper_vel, src._gripper_force = 0.05, 20.0
    period = int(1e9 / 15)
    src._gripper_min_period_ns = period
    src._gripper_deadband = 0.02

    src.send_gripper(0.0, now_ns=0)  # first send unconditional -> width = open
    assert handle.Move.call_count == 1
    assert handle.Move.call_args[0] == pytest.approx((0.10, 0.05, 20.0))
    src.send_gripper(1.0, now_ns=1000)  # within min_period -> throttled
    assert handle.Move.call_count == 1
    src.send_gripper(0.005, now_ns=period + 1)  # change < deadband -> skipped
    assert handle.Move.call_count == 1
    src.send_gripper(1.0, now_ns=2 * period + 2)  # ok -> width = closed
    assert handle.Move.call_count == 2
    assert handle.Move.call_args[0][0] == pytest.approx(0.00)
    src.send_gripper(9.0, now_ns=3 * period + 3)  # clips to 1.0 (still closed width)
    assert handle.Move.call_args[0][0] == pytest.approx(0.00)


def test_control_specs_names_and_dims():
    ctrl = _ctrl("qpos")
    specs = control_specs("left", ctrl)
    assert specs[SETPOINT].name == control_channel_name("left", "setpoint") == "cmd/left/setpoint"
    assert specs[SETPOINT].dim == 14
    assert specs[COMMAND].name == "cmd/left/command"
    assert specs[COMMAND].dim == ctrl.channel.command_dim


def test_setpoint_mailbox_is_latest_wins(tmp_path):
    ctrl = _ctrl("qpos")
    reg = StreamRegistry(tmp_path, "rid", sub="control")
    spec = control_specs("left", ctrl)[SETPOINT]
    writer = StreamWriter.create(spec, "rid", reg)
    try:
        reader = StreamReader.attach(reg.get(spec.name))
        for i in range(5):
            writer.write(pack_streamed(ctrl, {"q_d": np.full(7, float(i)), "dq_d": np.zeros(7)}))
        fields = slice_streamed(ctrl, reader.latest().newest)
        np.testing.assert_allclose(fields["q_d"], np.full(7, 4.0))  # freshest wins
        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_command_cursor_baselines_and_drains_in_order(tmp_path):
    ctrl = _ctrl("qpos")
    reg = StreamRegistry(tmp_path, "rid", sub="control")
    spec = control_specs("left", ctrl)[COMMAND]
    writer = StreamWriter.create(spec, "rid", reg)
    try:
        # Pre-attach backlog: must be ignored (startup-race fix).
        writer.write(ControlCommand(CommandKind.HOME).encode(spec.dim))
        cursor = CommandCursor(StreamReader.attach(reg.get(spec.name)))
        assert cursor.drain_new() == []  # the prior HOME is before the baseline

        writer.write(ControlCommand(CommandKind.STOP, (3.0,)).encode(spec.dim))
        writer.write(ControlCommand(CommandKind.SERVO_ON).encode(spec.dim))
        rows = cursor.drain_new()
        kinds = [ControlCommand.decode(r).kind for r in rows]
        assert kinds == [CommandKind.STOP, CommandKind.SERVO_ON]  # in order
        assert ControlCommand.decode(rows[0]).args[0] == pytest.approx(3.0)
        assert cursor.drain_new() == []  # nothing new on a re-drain
        cursor.close()
    finally:
        writer.close()
        writer.unlink()


def test_brain_setpoint_drives_fake_arm(tmp_path):
    """Post a qpos setpoint through real shared memory; the fake arm tracks it."""
    from dual_flexiv_control.interfaces.flexiv import FakeFlexivSource

    ctrl = _ctrl("qpos")
    coeffs = ControlCoeffsCfg()
    reg = StreamRegistry(tmp_path, "rid", sub="control")
    spec = control_specs("left", ctrl)[SETPOINT]
    writer = StreamWriter.create(spec, "rid", reg)
    try:
        reader = StreamReader.attach(reg.get(spec.name))
        src = FakeFlexivSource("sim", dof=7)
        src.open()

        q_target = np.linspace(0.1, 0.7, 7)
        writer.write(pack_streamed(ctrl, {"q_d": q_target, "dq_d": np.zeros(7)}))

        first = slice_streamed(ctrl, reader.latest().newest)
        src.enter_control()
        src.start_control(ctrl, coeffs, first, src.read())
        # One arm tick: read the freshest setpoint and actuate.
        fields = slice_streamed(ctrl, reader.latest().newest)
        src.send_control(ctrl, coeffs, fields, src.read(), 1.0 / ctrl.channel.rate_hz)

        np.testing.assert_allclose(np.asarray(src.read().q), q_target, atol=1e-3)
        assert src.last_command[0] == "qpos"
        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_send_control_safety_halt_covers_qpos_and_qvel():
    """The L-inf gate trips for both joint kinds and does NOT advance qvel's integrator."""
    pytest.importorskip("flexivrdk")
    from types import SimpleNamespace

    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource
    from dual_flexiv_control.interfaces.flexiv.source import SafetyHalt

    coeffs = ControlCoeffsCfg()
    rs = SimpleNamespace(q=np.zeros(7))
    src = FlexivSource("sim", dof=7)  # not opened: the gate runs before any robot call

    with pytest.raises(SafetyHalt):  # qpos: commanded q_d far from measured q
        src.send_control(
            _ctrl("qpos"), coeffs, {"q_d": np.full(7, 5.0), "dq_d": np.zeros(7)}, rs, 0.005,
            safety_check=True, tolerance=0.5,
        )

    src._control_target = np.zeros(7)
    with pytest.raises(SafetyHalt):  # qvel: integrated target would jump far from q
        src.send_control(
            _ctrl("qvel"), coeffs, {"dq_d": np.full(7, 1000.0)}, rs, 0.005,
            safety_check=True, tolerance=0.5,
        )
    np.testing.assert_array_equal(src._control_target, np.zeros(7))  # integrator NOT committed


def test_start_control_switches_to_the_configured_mode_not_a_hardcoded_one():
    """`qpos_overdamped` must enter NRT_JOINT_IMPEDANCE, not plain qpos mode."""
    pytest.importorskip("flexivrdk")
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import flexivrdk

    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()
    rs = SimpleNamespace(q=np.zeros(7), tcp_pose=np.zeros(7))

    with patch.object(FlexivSource, "_bootstrap_movej", return_value=True):
        ok = src.start_control(
            _ctrl("qpos_overdamped"), ControlCoeffsCfg(), {"q_d": np.zeros(7)}, rs,
        )
    assert ok is True
    src._robot.SwitchMode.assert_called_once_with(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)


def test_apply_coeffs_resolves_K_q_fraction_against_live_nominal_stiffness():
    """Control impedance fractions scale the robot's own K_q_nom."""
    pytest.importorskip("flexivrdk")
    from unittest.mock import MagicMock

    from dual_flexiv_control.configs import JointImpedanceCfg
    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()
    src._robot.info.return_value.K_q_nom = [1000.0] * 7

    src._apply_coeffs(_ctrl("qpos_overdamped"), ControlCoeffsCfg())

    src._robot.SetJointImpedance.assert_called_once_with([400.0] * 7, [0.8] * 7)


def test_apply_coeffs_scales_joint_stiffness_for_collection():
    """Collection applies one third of the selected controller's base stiffness."""
    pytest.importorskip("flexivrdk")
    from unittest.mock import MagicMock

    from dual_flexiv_control.configs import ControlCoeffsCfg
    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()
    src._robot.info.return_value.K_q_nom = [900.0] * 7
    coeffs = ControlCoeffsCfg(joint_stiffness_scale=1.0 / 3.0)

    src._apply_coeffs(_ctrl("qpos_impedance"), coeffs)

    src._robot.SetJointImpedance.assert_called_once_with([300.0] * 7, [0.8] * 7)


def test_apply_coeffs_falls_back_to_absolute_K_q_without_a_fraction():
    pytest.importorskip("flexivrdk")
    from unittest.mock import MagicMock

    from dual_flexiv_control.configs import JointImpedanceCfg
    from dual_flexiv_control.interfaces.flexiv.source import FlexivSource

    src = FlexivSource("sim", dof=7)
    src._robot = MagicMock()

    ctrl = _ctrl("qpos_overdamped")
    ctrl.joint_impedance = JointImpedanceCfg(K_q=[50.0] * 7, Z_q=[0.7] * 7)
    src._apply_coeffs(ctrl, ControlCoeffsCfg())

    src._robot.info.assert_not_called()  # no live query needed for an absolute K_q
    src._robot.SetJointImpedance.assert_called_once_with([50.0] * 7, [0.7] * 7)


def test_deadman_config_rejects_inverted_or_nonpositive_thresholds():
    from dual_flexiv_control.configs import ControlChannelCfg

    ControlChannelCfg(deadman_ms=100.0, deadman_hard_ms=500.0)  # ok
    with pytest.raises(ValueError):
        ControlChannelCfg(deadman_ms=600.0, deadman_hard_ms=500.0)  # inverted
    with pytest.raises(ValueError):
        ControlChannelCfg(deadman_ms=0.0)  # non-positive
