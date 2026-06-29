"""Tests for the Hydra structured config: composition, schema, overrides."""

from __future__ import annotations

import pickle

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.cameras import camera_stream_name
from dual_flexiv_control.cameras import camera_streams_to_specs
from dual_flexiv_control.configs import ArmCfg
from dual_flexiv_control.configs import CameraCfg
from dual_flexiv_control.configs import Config
from dual_flexiv_control.configs import ControlCfg
from dual_flexiv_control.configs import StreamCfg
from dual_flexiv_control.configs import TaskCfg
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


def test_all_control_schemas_present_and_shaped():
    # Each arm carries one ControlCfg, composed from the `control` group. Verified
    # against flexivrdk 2.1.0: all NRT, brain-driven over IPC.
    def ctrl(kind: str):
        return _compose(f"control@arms.left.control={kind}").arms.left.control

    # qpos -> NRT_JOINT_POSITION / SendJointPosition / NrtJointPositionCmd
    qpos = ctrl("qpos")
    assert qpos.mode == "NRT_JOINT_POSITION"
    assert qpos.send_fn == "SendJointPosition"
    assert qpos.cmd_struct == "NrtJointPositionCmd"
    assert dict(qpos.command) == {"q_d": 7, "dq_d": 7, "dq_max": 7, "ddq_max": 7}
    assert list(qpos.streamed) == ["q_d", "dq_d"]  # setpoint dim 14

    # qvel -> velocity is the only streamed quantity (arm integrates q_d)
    qvel = ctrl("qvel")
    assert qvel.mode == "NRT_JOINT_POSITION"
    assert list(qvel.streamed) == ["dq_d"]

    # end_effector -> NRT_CARTESIAN_MOTION_FORCE / SendCartesianMotionForce / NrtCartesianCmd
    eef = ctrl("end_effector")
    assert eef.mode == "NRT_CARTESIAN_MOTION_FORCE"
    assert eef.send_fn == "SendCartesianMotionForce"
    assert eef.cmd_struct == "NrtCartesianCmd"
    assert dict(eef.command) == {"pose_d": 7, "twist_d": 6, "wrench_d": 6}
    assert list(eef.streamed) == ["pose_d", "twist_d"]
    assert list(eef.force_control_axes) == [False] * 6  # pure motion

    # eef_vel -> twist is the only streamed quantity (arm integrates pose_d)
    assert list(ctrl("eef_vel").streamed) == ["twist_d"]

    # force -> wrench primary + structural force axes/frame (coeffs hold impedance)
    force = ctrl("force")
    assert force.command["wrench_d"] == 6
    assert list(force.streamed) == ["wrench_d", "pose_d"]
    assert len(force.force_control_axes) == 6
    assert force.force_control_frame.root_coord in ("WORLD", "TCP")
    assert len(force.force_axis_max_linear_vel) == 3


def test_per_phase_control_coeffs_imported_into_task():
    # The control_coeffs group is imported per task phase: compliant for collection
    # (training), stiff for eval. They are independently overridable.
    obj = OmegaConf.to_object(_compose())
    coll = obj.task.collection.coeffs
    ev = obj.task.eval.coeffs
    # compliant < stiff on cartesian stiffness and joint velocity limits
    assert coll.cartesian_impedance.K_x[0] < ev.cartesian_impedance.K_x[0]
    assert coll.max_joint_vel < ev.max_joint_vel
    # joint motion limits feed NrtJointPositionCmd dq_max/ddq_max
    assert coll.max_joint_vel == pytest.approx(1.5)
    assert ev.max_joint_vel == pytest.approx(2.5)


def test_control_coeffs_override_and_phase_selector():
    cfg = _compose(
        "control_coeffs@task.collection.coeffs=stiff",  # swap the whole coeffs group
        "task.eval.coeffs.max_joint_vel=9.0",           # tune one field
        "runtime.phase=eval",
        "arms.left.control_enabled=true",
    )
    assert cfg.task.collection.coeffs.max_joint_vel == pytest.approx(2.5)  # stiff
    assert cfg.task.eval.coeffs.max_joint_vel == pytest.approx(9.0)
    assert cfg.runtime.phase == "eval"
    assert cfg.arms.left.control_enabled is True


def test_cameras_compose_to_typed_objects():
    obj = OmegaConf.to_object(_compose())
    assert set(obj.cameras) == {"wrist_left", "wrist_right", "static"}
    assert isinstance(obj.cameras["static"], CameraCfg)
    # Wrist cams: ZED X Nano, left RGB only; static cam: ZED 2, stereo RGB.
    assert obj.cameras["wrist_left"].model == "zedx_nano"
    assert obj.cameras["wrist_left"].views == ["left"]
    assert obj.cameras["static"].model == "zed2"
    assert obj.cameras["static"].views == ["left", "right"]
    assert obj.cameras["wrist_left"].placement == "wrist_left"


def test_camera_stream_specs_derive_image_dims():
    obj = OmegaConf.to_object(_compose())

    wl = obj.cameras["wrist_left"]
    specs = {s.name: s for s in camera_streams_to_specs("wrist_left", wl)}
    left = camera_stream_name("wrist_left", "left")
    assert left == "cam/wrist_left/left"
    assert specs[left].dim == wl.width * wl.height * 3       # derived, not hand-set
    assert specs[left].dtype == "uint8"
    assert specs[left].rate_hz == wl.fps

    static = obj.cameras["static"]
    assert {s.name for s in camera_streams_to_specs("static", static)} == {
        "cam/static/left", "cam/static/right",
    }


def test_camera_cli_overrides():
    cfg = _compose(
        "cameras.static.resolution=HD1080",
        "cameras.static.width=1920",
        "cameras.static.height=1080",
        "cameras.wrist_left.serial=12345678",
    )
    assert cfg.cameras.static.width == 1920
    assert cfg.cameras.wrist_left.serial == "12345678"


def test_task_shared_and_per_phase_templates():
    cfg = _compose()
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj.task, TaskCfg)
    # shared spec lives on the task node; phase-unique counts on the sub-configs.
    assert isinstance(obj.task.language_instruction, str)
    assert obj.task.collection.num_episodes == 50
    assert obj.task.eval.num_timesteps == 400


def test_task_switch_and_field_overrides():
    cfg = _compose(
        "task=handover",
        "task.collection.num_episodes=10",
        "task.eval.num_timesteps=800",
    )
    assert "hand it to the right arm" in cfg.task.language_instruction
    assert cfg.task.collection.num_episodes == 10
    assert cfg.task.eval.num_timesteps == 800


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
