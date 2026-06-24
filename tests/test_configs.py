"""Tests for the Hydra structured config: composition, schema, overrides."""

from __future__ import annotations

import pickle

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.configs import ArmCfg
from dual_flexiv_control.configs import Config
from dual_flexiv_control.configs import ControlCfg
from dual_flexiv_control.configs import StreamCfg
from dual_flexiv_control.configs import register_configs


def _compose(*overrides: str):
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_composes_to_typed_objects_and_pickles():
    cfg = _compose()
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, Config)
    assert set(obj.arms) == {"left", "right"}
    assert isinstance(obj.arms["left"], ArmCfg)
    assert isinstance(obj.arms["left"].streams["q"], StreamCfg)
    assert isinstance(obj.arms["left"].control, ControlCfg)  # control nested per arm
    pickle.loads(pickle.dumps(obj))  # must cross the spawn boundary


def test_proprio_stream_schema_matches_rdk_dims():
    cfg = _compose()
    s = cfg.arms.right.streams
    assert (s.q.dim, s.dq.dim, s.tau.dim) == (7, 7, 7)
    assert (s.wrench.dim, s.eef.dim, s.eef_vel.dim) == (6, 7, 6)
    assert cfg.arms.left.serial != cfg.arms.right.serial  # per-arm override applied


def test_all_four_control_schemas_present_and_shaped():
    # Each arm carries one ControlCfg, composed from the `control` group. Compose
    # each kind onto an arm and check its command schema is correctly shaped.
    def ctrl(kind: str):
        return _compose(f"control@arms.left.control={kind}").arms.left.control

    # qpos -> RtJointPositionCmd{q_d, dq_d, ddq_d}, each DoF
    qpos = ctrl("qpos")
    assert qpos.mode == "RT_JOINT_POSITION"
    assert qpos.cmd_struct == "RtJointPositionCmd"
    assert dict(qpos.command) == {"q_d": 7, "dq_d": 7, "ddq_d": 7}

    # qvel -> velocity is the primary commanded quantity
    assert "dq_d" in ctrl("qvel").command

    # end_effector -> RtCartesianCmd shapes (pose 7, twist/acc/wrench 6)
    eef = ctrl("end_effector")
    assert eef.mode == "RT_CARTESIAN_MOTION_FORCE"
    assert dict(eef.command) == {
        "pose_d": 7, "twist_d": 6, "acc_d": 6, "wrench_d": 6,
    }
    assert list(eef.force_control_axes) == [False] * 6  # pure motion

    # force -> wrench primary + force axes + frame + limits
    force = ctrl("force")
    assert force.command["wrench_d"] == 6
    assert len(force.force_control_axes) == 6
    assert force.force_control_frame.root_coord in ("WORLD", "TCP")
    assert len(force.max_contact_wrench) == 6
    assert len(force.max_linear_vel) == 3


def test_cli_style_overrides():
    cfg = _compose(
        "runtime.sim=true",
        "control@arms.left.control=force",
        "arms.left.serial=Rizon4-AAA",
        "brain.rate_hz=250",
        "arms.right.streams.tau.capacity=8192",
    )
    assert cfg.runtime.sim is True
    assert cfg.arms.left.control.kind == "force"
    assert cfg.arms.left.serial == "Rizon4-AAA"
    assert cfg.brain.rate_hz == 250
    assert cfg.arms.right.streams.tau.capacity == 8192


def test_schema_rejects_unknown_key():
    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigCompositionException, Exception)):
        _compose("arms.left.bogus_field=1")
