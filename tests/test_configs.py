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


def _compose(*overrides: str, config_name: str = "config"):
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def test_stream_dummy_flag_defaults_false():
    cfg = _compose()
    assert cfg.arms.left.streams.q.dummy is False  # real hardware by default


def test_runtime_save_grace_default_and_override():
    # The shutdown window the recording node gets to finalize its episode video
    # (see system._shutdown). Generous by default; CLI-tunable like any field.
    assert _compose().runtime.save_grace_s == pytest.approx(300.0)
    assert _compose("runtime.save_grace_s=15").runtime.save_grace_s == pytest.approx(15.0)


def test_bench_rig_is_dummy_arm_plus_real_static_camera():
    cfg = OmegaConf.to_object(_compose("rig=bench"))
    # single dummy arm, single real camera, left-only FACTR
    assert set(cfg.arms) == {"left"}
    assert set(cfg.cameras) == {"static"}
    assert set(cfg.factr.servers) == {"left"}
    assert cfg.arms["left"].control_enabled is False
    assert all(s.dummy for s in cfg.arms["left"].streams.values())  # whole arm fabricated
    assert cfg.cameras["static"].auto_serial is True                # binds the connected ZED
    assert cfg.recording.root == "datasets/bench"                   # output quarantined


def test_left_only_rig_single_real_arm():
    cfg = OmegaConf.to_object(_compose("rig=left_only"))
    assert set(cfg.arms) == {"left"}
    assert set(cfg.cameras) == {"static"}  # no wrist cam plugged in on this rig
    assert set(cfg.factr.servers) == {"left"}
    assert not any(s.dummy for s in cfg.arms["left"].streams.values())  # real arm
    assert cfg.recording.root == "datasets"                              # not quarantined


def test_rig_composes_with_task_and_field_overrides():
    cfg = _compose("rig=bench", "task=handover", "arms.left.serial=Rizon4-XYZ")
    assert cfg.task.collection.repo_id == "dfc/handover"  # task picks the dataset
    assert cfg.recording.root == "datasets/bench"         # rig picks the destination
    assert cfg.arms.left.serial == "Rizon4-XYZ"           # CLI still wins over the rig


def test_flexiv_interface_uses_fake_source_when_streams_dummy(tmp_path):
    from dual_flexiv_control.configs import RuntimeCfg
    from dual_flexiv_control.interfaces.flexiv import FlexivInterface
    from dual_flexiv_control.interfaces.flexiv.source import FakeFlexivSource

    cfg = OmegaConf.to_object(_compose("rig=bench"))
    runtime = RuntimeCfg(runtime_dir=str(tmp_path), sim=False)  # NOT global sim
    node = FlexivInterface("left", cfg.arms["left"], runtime, run_id="t")
    node.open_source()
    try:
        # dummy streams -> fabricated source even though runtime.sim is False
        assert isinstance(node._source, FakeFlexivSource)
    finally:
        node.close_source()


def test_default_rig_is_left_only():
    # The shipped default matches the plugged-in hardware (left arm + left FACTR
    # leader), so a collection run's launch-time teleop preflight is satisfiable.
    obj = OmegaConf.to_object(_compose())
    assert set(obj.arms) == {"left"}
    assert set(obj.factr.servers) == {"left"}


def test_composes_to_typed_objects_and_pickles():
    cfg = _compose("rig=bimanual")
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, Config)
    assert set(obj.arms) == {"left", "right"}
    assert isinstance(obj.arms["left"], ArmCfg)
    assert isinstance(obj.arms["left"].streams["q"], StreamCfg)
    assert isinstance(obj.arms["left"].control, ControlCfg)  # control nested per arm
    pickle.loads(pickle.dumps(obj))  # must cross the spawn boundary


def test_proprio_stream_schema_matches_rdk_dims():
    cfg = _compose("rig=bimanual")
    s = cfg.arms.right.streams
    assert (s.q.dim, s.dq.dim, s.tau.dim) == (7, 7, 7)
    assert (s.wrench.dim, s.eef.dim, s.eef_vel.dim) == (6, 7, 6)
    assert cfg.arms.left.serial != cfg.arms.right.serial  # per-arm override applied


def test_all_control_schemas_present_and_shaped():
    # Each arm carries one ControlCfg, composed from the `control` group. Verified
    # against flexivrdk 1.8.0: all NRT, flat send API, brain-driven over IPC.
    def ctrl(kind: str):
        return _compose(f"control@arms.left.control={kind}").arms.left.control

    # qpos -> NRT_JOINT_POSITION / SendJointPosition(q_d, dq_d, dq_max, ddq_max)
    qpos = ctrl("qpos")
    assert qpos.mode == "NRT_JOINT_POSITION"
    assert qpos.send_fn == "SendJointPosition"
    assert dict(qpos.command) == {"q_d": 7, "dq_d": 7, "dq_max": 7, "ddq_max": 7}
    assert list(qpos.streamed) == ["q_d", "dq_d"]  # setpoint dim 14

    # qvel -> velocity is the only streamed quantity (arm integrates q_d)
    qvel = ctrl("qvel")
    assert qvel.mode == "NRT_JOINT_POSITION"
    assert list(qvel.streamed) == ["dq_d"]

    # qpos_impedance -> same schema as qpos, but NRT_JOINT_IMPEDANCE (low-authority
    # tracking via SetJointImpedance instead of a fixed high-gain position loop)
    qpos_imp = ctrl("qpos_impedance")
    assert qpos_imp.mode == "NRT_JOINT_IMPEDANCE"
    assert qpos_imp.send_fn == "SendJointPosition"
    assert dict(qpos_imp.command) == dict(qpos.command)
    assert list(qpos_imp.streamed) == list(qpos.streamed)

    # end_effector -> NRT_CARTESIAN_MOTION_FORCE / SendCartesianMotionForce(pose, wrench, velocity, ...)
    eef = ctrl("end_effector")
    assert eef.mode == "NRT_CARTESIAN_MOTION_FORCE"
    assert eef.send_fn == "SendCartesianMotionForce"
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


def test_per_phase_control_coeffs_default_compliant_vs_stiff():
    # Schema defaults (configs.py): compliant for collection (training), stiff
    # for eval — no per-task composition boilerplate needed.
    obj = OmegaConf.to_object(_compose())
    coll = obj.task.collection.coeffs
    ev = obj.task.eval.coeffs
    # compliant < stiff on cartesian stiffness and joint velocity limits
    assert coll.cartesian_impedance.K_x[0] < ev.cartesian_impedance.K_x[0]
    assert coll.max_joint_vel < ev.max_joint_vel
    # joint motion limits feed SendJointPosition max_vel/max_acc args
    assert coll.max_joint_vel == pytest.approx(1.5)
    assert ev.max_joint_vel == pytest.approx(2.5)


def test_very_compliant_coeffs_preset_uses_fraction_of_nominal_stiffness():
    # `very_compliant` asks for a small FRACTION of the live robot's own K_q_nom
    # (resolved at apply-time; see FlexivSource._apply_coeffs) rather than a
    # hard-coded absolute K_q, so it stays "insanely low" for any arm model.
    cfg = _compose(
        "control@arms.left.control=qpos_impedance",
        "+control_coeffs@task.eval.coeffs=very_compliant",
    )
    imp = cfg.task.eval.coeffs.joint_impedance
    assert imp is not None
    assert 0.0 < imp.K_q_fraction < 0.3  # a small slice of nominal, not "None"/absolute
    assert not imp.K_q  # no absolute K_q hard-coded alongside the fraction
    assert cfg.task.eval.coeffs.max_joint_vel < cfg.task.eval.coeffs.max_joint_acc


def test_control_coeffs_override_and_phase_selector():
    cfg = _compose(
        # Swap a whole preset (ConfigStore-registered; `+` appends the group entry):
        "+control_coeffs@task.collection.coeffs=stiff",
        "task.eval.coeffs.max_joint_vel=9.0",           # tune one field
        "runtime.phase=eval",
        "arms.left.control_enabled=true",
    )
    assert cfg.task.collection.coeffs.max_joint_vel == pytest.approx(2.5)  # stiff
    assert cfg.task.eval.coeffs.max_joint_vel == pytest.approx(9.0)
    assert cfg.runtime.phase == "eval"
    assert cfg.arms.left.control_enabled is True


def test_recording_group_defaults_and_task_dataset_identity():
    obj = OmegaConf.to_object(_compose())
    # export machinery is run-wide (recording group), dataset identity is per task
    assert obj.recording.root == "datasets"
    assert obj.recording.video is True and obj.recording.resume is True
    assert obj.task.collection.repo_id == "dfc/default"
    assert obj.task.state_signals == ["q"]  # shared by collection + eval
    assert _compose("task=handover").task.collection.repo_id == "dfc/handover"


def test_cameras_compose_to_typed_objects():
    obj = OmegaConf.to_object(_compose("rig=bimanual"))
    assert set(obj.cameras) == {"wrist_left", "wrist_right", "static"}
    assert isinstance(obj.cameras["static"], CameraCfg)
    # Wrist cams: ZED X Nano, left RGB only; static cam: ZED 2, stereo RGB + depth.
    assert obj.cameras["wrist_left"].model == "zedx_nano"
    assert obj.cameras["wrist_left"].views == ["left"]
    assert obj.cameras["static"].model == "zed2"
    assert obj.cameras["static"].views == ["left", "right", "depth"]
    assert obj.cameras["static"].depth_mode == "ULTRA"  # depth view needs != NONE
    assert obj.cameras["wrist_left"].placement == "wrist_left"
    # RGB-D overlay extrinsics: static cam anchored to the left arm's URDF base.
    assert obj.cameras["static"].pose_frame == "mount_left"
    assert obj.cameras["wrist_left"].pose_frame == "world"  # schema default


def test_camera_stream_specs_derive_image_dims():
    obj = OmegaConf.to_object(_compose("rig=bimanual"))  # full camera set

    wl = obj.cameras["wrist_left"]
    specs = {s.name: s for s in camera_streams_to_specs("wrist_left", wl)}
    left = camera_stream_name("wrist_left", "left")
    assert left == "cam/wrist_left/left"
    assert specs[left].dim == wl.width * wl.height * 3       # derived, not hand-set
    assert specs[left].dtype == "uint8"
    assert specs[left].rate_hz == wl.fps

    static = obj.cameras["static"]
    static_specs = {s.name: s for s in camera_streams_to_specs("static", static)}
    assert set(static_specs) == {
        "cam/static/left", "cam/static/right", "cam/static/depth",
    }
    depth = static_specs["cam/static/depth"]
    assert depth.dim == static.width * static.height  # single channel
    assert depth.dtype == "float32"


def test_camera_cli_overrides():
    cfg = _compose(
        "rig=bimanual",  # full camera set (wrist cams exist only on this rig)
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
        "rig=bimanual",
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
