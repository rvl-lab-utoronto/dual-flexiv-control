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
from dual_flexiv_control.control import pack_streamed
from dual_flexiv_control.control import setpoint_dim
from dual_flexiv_control.control import slice_streamed
from dual_flexiv_control.streams.registry import StreamRegistry
from dual_flexiv_control.streams.stream import StreamReader
from dual_flexiv_control.streams.stream import StreamWriter


def _ctrl(kind: str):
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=[f"control@arms.left.control={kind}"])
    return OmegaConf.to_object(cfg).arms["left"].control


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


def test_deadman_config_rejects_inverted_or_nonpositive_thresholds():
    from dual_flexiv_control.configs import ControlChannelCfg

    ControlChannelCfg(deadman_ms=100.0, deadman_hard_ms=500.0)  # ok
    with pytest.raises(ValueError):
        ControlChannelCfg(deadman_ms=600.0, deadman_hard_ms=500.0)  # inverted
    with pytest.raises(ValueError):
        ControlChannelCfg(deadman_ms=0.0)  # non-positive
