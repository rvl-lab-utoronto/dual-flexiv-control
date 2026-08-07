"""Hardware-free tests for the eval path.

The ObservationBuilder, DFCStateActionLayout, and endpoint adapters are pure. The EvalLoop
is exercised with a fake Brain and a scripted Policy, so no shared memory, no
policy server is needed, and no ``msgpack`` codec is needed (the
msgpack round-trip test skips if ``msgpack`` is absent).
"""

from __future__ import annotations

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from dual_flexiv_control.collection import FrameBuilder
from dual_flexiv_control.configs import register_configs
from dual_flexiv_control.policy import DFCStateActionLayout
from dual_flexiv_control.policy import EvalLoop
from dual_flexiv_control.policy import ObservationBuilder
from dual_flexiv_control.policy import PolicyError
from dual_flexiv_control.policy import build_policy
from dual_flexiv_control.policy import build_adapter
from dual_flexiv_control.streams.ring import Samples


def _config(*overrides: str):
    # Pin the full two-arm setup explicitly so this fixture is independent of the
    # shipped default.
    register_configs()
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=["rig=bimanual", *overrides])
    return OmegaConf.to_object(cfg)


def _samples(vec, dtype=np.float64) -> Samples:
    data = np.asarray(vec, dtype=dtype).reshape(1, -1)
    return Samples(data=data, t_ns=np.array([1], np.int64), seq=np.array([0], np.int64))


def _empty_samples(dtype=np.float64) -> Samples:
    return Samples(
        data=np.empty((0, 0), dtype=dtype),
        t_ns=np.empty((0,), np.int64),
        seq=np.empty((0,), np.int64),
    )


def _observer(cfg) -> ObservationBuilder:
    return ObservationBuilder(
        cfg.arms, cfg.cameras,
        cfg.task.language_instruction, cfg.task.state_signals,
    )


def _layout(cfg, sides) -> DFCStateActionLayout:
    return DFCStateActionLayout.from_arms(cfg.arms, cfg.task.state_signals, sides)


def _full_observation(cfg):
    """A complete snapshot: q for both arms + every camera frame."""
    obs = {"left/q": _samples(np.arange(7.0)), "right/q": _samples(np.arange(7.0) + 10)}
    for name, cam in cfg.cameras.items():
        for view in cam.views:
            dim = cam.height * cam.width * (3 if view in ("left", "right") else 1)
            obs[f"cam/{name}/{view}"] = _samples(np.zeros(dim, np.uint8), dtype=np.uint8)
    return obs


# --------------------------------------------------------------------------- #
# ObservationBuilder: the canonical (training-frame) observation
# --------------------------------------------------------------------------- #


def test_observation_matches_training_frame_schema():
    cfg = _config()
    ob = _observer(cfg)
    obs = ob.build(_full_observation(cfg))
    assert obs is not None
    assert "action" not in obs
    assert obs["observation.state"].shape == (14,)
    assert obs["task"] == cfg.task.language_instruction
    # Image keys are exactly what collection would record (schema parity).
    fb = FrameBuilder(cfg.arms, [], cfg.cameras,
                      cfg.task.language_instruction, cfg.task.state_signals)
    assert ob.image_keys == fb.image_keys
    assert obs["observation.images.static_left"].shape == (720, 1280, 3)


def test_observation_none_until_streams_warm():
    cfg = _config()
    ob = _observer(cfg)
    snapshot = _full_observation(cfg)
    snapshot["cam/static/left"] = _empty_samples(np.uint8)
    assert ob.build(snapshot) is None


def test_observation_state_slices():
    cfg = _config()
    ob = _observer(cfg)
    obs = ob.build(_full_observation(cfg))
    state = obs["observation.state"]
    np.testing.assert_allclose(state[ob.state_slice("left", "q")], np.arange(7.0))
    np.testing.assert_allclose(state[ob.state_slice("right", "q")], np.arange(7.0) + 10)
    with pytest.raises(KeyError):
        ob.state_slice("left", "tau")  # not among state_signals


# --------------------------------------------------------------------------- #
# DFCStateActionLayout: parity with the collection action feature
# --------------------------------------------------------------------------- #


def test_action_layout_matches_collection_action_schema():
    cfg = _config()
    layout = _layout(cfg, ["left", "right"])
    fb = FrameBuilder(cfg.arms, ["left", "right"], cfg.cameras,
                      cfg.task.language_instruction, cfg.task.state_signals)
    assert layout.names == fb.action_names
    assert layout.dim == fb.action_dim == 16


def test_action_layout_split():
    cfg = _config()
    layout = _layout(cfg, ["left", "right"])
    action = np.arange(16.0)
    targets = layout.split(action)
    np.testing.assert_allclose(targets["left"]["q_d"], np.arange(7.0))
    assert targets["left"]["gripper"] == pytest.approx(7.0)
    np.testing.assert_allclose(targets["right"]["q_d"], np.arange(8.0, 15.0))
    assert targets["right"]["gripper"] == pytest.approx(15.0)
    with pytest.raises(ValueError):
        layout.split(np.zeros(5))


# --------------------------------------------------------------------------- #
# OpenPIEndpointAdapter: request mapping + response parsing
# --------------------------------------------------------------------------- #


def test_openpi_adapter_request_keys():
    cfg = _config()
    obs = _observer(cfg).build(_full_observation(cfg))
    req = build_adapter(cfg.policy).encode_request(obs)
    assert req["prompt"] == cfg.task.language_instruction
    assert req["observation/state"].shape == (14,)
    assert req["observation/images/static_left"].shape == (720, 1280, 3)
    assert "observation/images/static_right" in req


def test_openpi_adapter_image_key_overrides_and_drops():
    cfg = _config()
    cfg.policy.openpi.request.image_keys = {
        "static_left": "observation/front_image",
        "static_right": "",  # checkpoint does not use this view: drop it
    }
    obs = _observer(cfg).build(_full_observation(cfg))
    req = build_adapter(cfg.policy).encode_request(obs)
    assert "observation/front_image" in req
    assert "observation/images/static_left" not in req
    assert "observation/images/static_right" not in req
    assert req["prompt"] == cfg.task.language_instruction


def test_openpi_adapter_actions_parsing():
    cfg = _config()
    adapter = build_adapter(cfg.policy)
    chunk = adapter.decode_actions({"actions": np.ones((5, 16))})
    assert chunk.shape == (5, 16)
    single = adapter.decode_actions({"actions": np.zeros(16)})  # single action -> chunk of one
    assert single.shape == (1, 16)
    with pytest.raises(PolicyError):
        adapter.decode_actions({"wrong_key": np.ones((5, 16))})


def test_unknown_adapter_rejected():
    cfg = _config()
    cfg.policy.adapter = "nonsense"
    with pytest.raises(ValueError):
        build_adapter(cfg.policy)


# --------------------------------------------------------------------------- #
# OpenPIEndpointAdapter: stock ALOHA wire contract + YAML-owned Flexiv mapping
# --------------------------------------------------------------------------- #


def test_aloha_adapter_request_drops_zero_based_l2_r2():
    cfg = _config("policy=pi05_aloha")
    obs = _observer(cfg).build(_full_observation(cfg))
    obs["observation.images.static_left"][0, 0] = [1, 2, 3]

    req = build_adapter(cfg.policy).encode_request(obs)

    assert set(req) == {"state", "images", "prompt"}
    assert set(req["images"]) == {"cam_high"}
    np.testing.assert_allclose(
        req["state"],
        [0, 1, 3, 4, 5, 6, 0, 10, 11, 13, 14, 15, 16, 0],
    )
    image = req["images"]["cam_high"]
    assert image.shape == (3, 720, 1280)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    np.testing.assert_array_equal(image[:, 0, 0], [1, 2, 3])
    assert req["prompt"] == cfg.task.language_instruction


def test_aloha_adapter_response_expands_to_16d_and_holds_l2_r2():
    cfg = _config("policy=pi05_aloha")
    obs = _observer(cfg).build(_full_observation(cfg))
    adapter = build_adapter(cfg.policy)
    adapter.encode_request(obs)  # caches the measured L2/R2 values used for the hold

    raw = np.vstack([np.arange(14.0), np.arange(14.0) + 100])
    chunk = adapter.decode_actions({"actions": raw})

    expected = np.array([
        [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 11, 12, 13],
        [100, 101, 2, 102, 103, 104, 105, 106,
         107, 108, 12, 109, 110, 111, 112, 113],
    ])
    np.testing.assert_allclose(chunk, expected)
    assert chunk.shape == (2, 16)


def test_aloha_mapping_requires_observation_before_action_hold():
    cfg = _config("policy=pi05_aloha")
    adapter = build_adapter(cfg.policy)
    with pytest.raises(PolicyError, match="before mapping a DFC observation"):
        adapter.decode_actions({"actions": np.zeros((1, 14))})


# --------------------------------------------------------------------------- #
# AcmeEndpointAdapter: single-arm multipart request mapping + response parsing
# --------------------------------------------------------------------------- #


def test_acme_adapter_request_structure():
    cfg = _config("policy=acme")
    obs = _observer(cfg).build(_full_observation(cfg))
    req = build_adapter(cfg.policy).encode_request(obs)
    assert set(req["images"]) == {
        "exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"
    }
    assert req["images"]["exterior_image_1_left"].shape == (720, 1280, 3)  # static_left
    assert req["images"]["wrist_image_left"].shape == (720, 1280, 3)       # static_left
    # qpos is the LEFT arm's 7 joints sliced out of the 14-dim state.
    np.testing.assert_allclose(req["lowdim"]["qpos"], np.arange(7.0))
    assert req["lowdim"]["qpos"].shape == (7,)
    assert req["form"]["prompt"] == cfg.task.language_instruction
    assert req["form"]["obs_steps"] == 1


def test_acme_schema_qpos_slice_selects_right_arm():
    cfg = _config("policy=acme", "policy.qpos_slice=[7,14]")
    obs = _observer(cfg).build(_full_observation(cfg))
    req = build_adapter(cfg.policy).encode_request(obs)
    np.testing.assert_allclose(req["lowdim"]["qpos"], np.arange(7.0) + 10)


def test_acme_schema_missing_camera_view_raises():
    cfg = _config("policy=acme")
    cfg.policy.acme_image_keys = {"wrist_image_left": "nonexistent_cam"}
    obs = _observer(cfg).build(_full_observation(cfg))
    with pytest.raises(PolicyError):
        build_adapter(cfg.policy).encode_request(obs)


def test_acme_adapter_actions_parsing():
    cfg = _config("policy=acme")
    adapter = build_adapter(cfg.policy)
    # (B, H, 8) batched -> the single item's (H, 8) chunk.
    chunk = adapter.decode_actions({"action": np.ones((1, 10, 8)), "success": True})
    assert chunk.shape == (10, 8)
    # (H, 8) already unbatched passes through.
    assert adapter.decode_actions({"action": np.zeros((4, 8))}).shape == (4, 8)
    with pytest.raises(PolicyError):
        adapter.decode_actions({"action": np.ones((2, 10, 8))})  # B != 1
    with pytest.raises(PolicyError):
        adapter.decode_actions({"wrong_key": np.ones((10, 8))})


def test_acme_transport_encodes_torch_and_npz():
    pytest.importorskip("torch")
    pytest.importorskip("requests")
    from dual_flexiv_control.policy.client import AcmeHttpTransport

    cfg = _config("policy=acme")
    obs = _observer(cfg).build(_full_observation(cfg))
    payload = build_adapter(cfg.policy).encode_request(obs)
    # Encode without constructing (no server contacted): exercises the wire format.
    transport = AcmeHttpTransport.__new__(AcmeHttpTransport)
    import torch

    transport._torch = torch
    files = transport._encode_files(payload)
    assert set(files) == {
        "exterior_image_1_left", "exterior_image_2_left", "wrist_image_left", "lowdim_data"
    }
    # Image part round-trips to a (B, T, C, H, W) uint8 tensor.
    import io

    name, data, _ = files["exterior_image_1_left"]
    assert name.endswith(".pt")
    tensor = torch.load(io.BytesIO(data))
    assert tuple(tensor.shape) == (1, 1, 3, 720, 1280)
    assert tensor.dtype == torch.uint8
    # lowdim npz carries qpos as (B, T, D).
    npz = np.load(io.BytesIO(files["lowdim_data"][1]))
    assert npz["qpos"].shape == (1, 1, 7)


# --------------------------------------------------------------------------- #
# msgpack-numpy wire format (openpi-compatible)
# --------------------------------------------------------------------------- #


def test_msgpack_numpy_roundtrip():
    msgpack = pytest.importorskip("msgpack")
    from dual_flexiv_control.policy.client import pack_array
    from dual_flexiv_control.policy.client import unpack_array

    payload = {
        "observation/state": np.linspace(0, 1, 14, dtype=np.float32),
        "observation/images/wrist_left": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "prompt": "do the task",
    }
    packed = msgpack.packb(payload, default=pack_array)
    out = msgpack.unpackb(packed, object_hook=unpack_array)
    assert out["prompt"] == "do the task"
    for key in ("observation/state", "observation/images/wrist_left"):
        np.testing.assert_array_equal(out[key], payload[key])
        assert out[key].dtype == payload[key].dtype


# --------------------------------------------------------------------------- #
# HoldPolicy: serverless stand-still
# --------------------------------------------------------------------------- #


def test_hold_policy_repeats_measured_q():
    cfg = _config()
    cfg.policy.kind = "hold"
    ob = _observer(cfg)
    layout = _layout(cfg, ["left", "right"])
    policy = build_policy(cfg.policy, layout, ob)
    obs = ob.build(_full_observation(cfg))
    chunk = policy.infer(obs)
    assert chunk.ndim == 2 and chunk.shape[1] == layout.dim
    targets = layout.split(chunk[0])
    np.testing.assert_allclose(targets["left"]["q_d"], np.arange(7.0))
    np.testing.assert_allclose(targets["right"]["q_d"], np.arange(7.0) + 10)
    assert targets["left"]["gripper"] == 0.0


def test_build_policy_rejects_unknown_kind():
    cfg = _config()
    cfg.policy.kind = "nonsense"
    with pytest.raises(ValueError):
        build_policy(cfg.policy, _layout(cfg, ["left"]), _observer(cfg))


class _PassthroughAdapter:
    def encode_request(self, obs):
        return {"obs": obs}

    def decode_actions(self, response):
        return np.zeros((2, 3))


class _FakeTransport:
    def __init__(self):
        self.fail = False

    def infer(self, payload):
        if self.fail:
            raise PolicyError("server down")
        return {"actions": []}

    def close(self):
        pass


def test_remote_policy_emits_comm_events():
    """SENT then RECV per request (matching seq); a failed request emits ERROR."""
    from dual_flexiv_control.policy.client import COMM_ERROR
    from dual_flexiv_control.policy.client import COMM_RECV
    from dual_flexiv_control.policy.client import COMM_SENT
    from dual_flexiv_control.policy.client import RemotePolicy

    events = []
    transport = _FakeTransport()
    policy = RemotePolicy(
        _PassthroughAdapter(), transport,
        on_comm=lambda kind, seq, elapsed: events.append((kind, seq, elapsed)),
    )
    policy.infer({})
    assert [(k, s) for k, s, _ in events] == [(COMM_SENT, 1), (COMM_RECV, 1)]
    assert events[0][2] == 0.0 and events[1][2] >= 0.0

    transport.fail = True
    with pytest.raises(PolicyError):
        policy.infer({})
    assert [(k, s) for k, s, _ in events[2:]] == [(COMM_SENT, 2), (COMM_ERROR, 2)]


def test_remote_policy_comm_hook_failure_is_not_fatal():
    from dual_flexiv_control.policy.client import RemotePolicy

    def broken_hook(kind, seq, elapsed):
        raise RuntimeError("viz exploded")

    policy = RemotePolicy(_PassthroughAdapter(), _FakeTransport(), on_comm=broken_hook)
    chunk = policy.infer({})  # must not raise
    assert chunk.shape == (2, 3)


# --------------------------------------------------------------------------- #
# EvalLoop: end-to-end with fakes
# --------------------------------------------------------------------------- #


class _FakeBrain:
    def __init__(self, cfg):
        self._cfg = cfg
        self.commands = []

    def observe(self):
        return _full_observation(self._cfg)

    def command(self, side, setpoint):
        self.commands.append((side, np.asarray(setpoint)))


class _ScriptedPolicy:
    """Returns a fixed (horizon, dim) chunk; optionally fails the first N calls."""

    def __init__(self, dim, horizon=4, fail_first=0):
        self.dim = dim
        self.horizon = horizon
        self.fail_first = fail_first
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise PolicyError("scripted failure")
        return np.tile(np.arange(self.dim, dtype=np.float64), (self.horizon, 1))

    def close(self):
        pass


class _StopAfter:
    """Stop event that trips after N `is_set` checks (bounds the loop)."""

    def __init__(self, n):
        self._n = n

    def is_set(self):
        self._n -= 1
        return self._n < 0

    def set(self):
        self._n = 0


def _loop(cfg, brain, policy, layout, num_timesteps, replan_steps=0):
    return EvalLoop(
        brain, _observer(cfg), policy, layout,
        control_arms={"left": cfg.arms["left"]},
        frequency_hz=1000.0, num_timesteps=num_timesteps, replan_steps=replan_steps,
    )


def test_loop_executes_chunk_and_replans():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=6, replan_steps=2)
    loop.run(_StopAfter(100))
    assert loop.timesteps_done == 6
    assert policy.calls == 3          # 2 executed actions per inference
    assert len(brain.commands) == 6
    side, setpoint = brain.commands[0]
    assert side == "left"
    assert setpoint.shape == (14,)    # qpos: q_d(7) + dq_d(7)
    np.testing.assert_allclose(setpoint[:7], np.arange(7.0))   # q_d from the action
    np.testing.assert_allclose(setpoint[7:], 0.0)              # dq_d zeroed


def test_action_layout_is_control_kind_aware():
    # Each arm's control kind sets its action space (primary field + width).
    for kind, field, dim in [("qvel", "dq_d", 7), ("eef_vel", "twist_d", 6),
                             ("end_effector", "pose_d", 7), ("force", "wrench_d", 6)]:
        cfg = _config(f"control@policy.control={kind}")
        layout = _layout(cfg, ["left"])
        assert layout.field("left") == field
        assert layout.dim == dim + 1                         # primary + gripper
        assert layout.names[0] == f"left.{field}.0"
        assert layout.names[-1] == "left.gripper"
        split = layout.split(np.arange(float(layout.dim)))
        np.testing.assert_allclose(split["left"][field], np.arange(float(dim)))


def test_loop_executes_qvel_and_zero_fills_nothing():
    cfg = _config("control@policy.control=qvel")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0).run(_StopAfter(100))
    _, setpoint = brain.commands[0]
    assert setpoint.shape == (7,)                             # streamed [dq_d]
    np.testing.assert_allclose(setpoint, np.arange(7.0))      # dq_d = policy primary


def test_loop_executes_end_effector_zeros_feedforward_twist():
    cfg = _config("control@policy.control=end_effector")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0).run(_StopAfter(100))
    _, setpoint = brain.commands[0]
    assert setpoint.shape == (13,)                            # streamed [pose_d, twist_d]
    np.testing.assert_allclose(setpoint[:7], np.arange(7.0))  # pose_d from action
    np.testing.assert_allclose(setpoint[7:], 0.0)             # twist_d feedforward = 0


def test_loop_force_holds_measured_pose_from_eef():
    cfg = _config("control@policy.control=force")
    layout = _layout(cfg, ["left"])
    measured_pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

    class _BrainWithEef(_FakeBrain):
        def observe(self):
            snap = _full_observation(self._cfg)
            snap["left/eef"] = _samples(measured_pose)        # arm's measured TCP pose
            return snap

    brain = _BrainWithEef(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0).run(_StopAfter(100))
    _, setpoint = brain.commands[0]
    assert setpoint.shape == (13,)                            # streamed [wrench_d, pose_d]
    np.testing.assert_allclose(setpoint[:6], np.arange(6.0))  # wrench_d = policy primary
    np.testing.assert_allclose(setpoint[6:], measured_pose)   # pose_d held at measured


def test_loop_force_holds_command_when_pose_unavailable():
    # No eef stream in the snapshot -> can't build a force setpoint -> hold (no command),
    # rather than silently commanding a zero pose.
    cfg = _config("control@policy.control=force")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)                                   # snapshot has no left/eef
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0)
    loop.run(_StopAfter(100))
    assert brain.commands == []


def test_loop_consumes_full_chunk_when_replan_zero():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=8, replan_steps=0)
    loop.run(_StopAfter(100))
    assert loop.timesteps_done == 8
    assert policy.calls == 2


def test_loop_holds_on_policy_error_then_recovers():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4, fail_first=2)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=4)
    loop.run(_StopAfter(100))
    # Failed inferences hold (no command, timestep not counted), then recover.
    assert loop.timesteps_done == 4
    assert len(brain.commands) == 4
    assert policy.calls == 3  # 2 failures + 1 success


def test_loop_surfaces_persistent_policy_error_after_retry_budget():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4, fail_first=10)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=4)

    with pytest.raises(PolicyError, match="3 consecutive times") as error:
        loop.run(_StopAfter(100))

    assert "scripted failure" in str(error.value)
    assert policy.calls == 3
    assert brain.commands == []


def test_loop_policy_error_budget_resets_after_success():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)

    class AlternatingPolicy(_ScriptedPolicy):
        def infer(self, obs):
            self.calls += 1
            if self.calls % 2:
                raise PolicyError("intermittent failure")
            return np.tile(
                np.arange(self.dim, dtype=np.float64), (self.horizon, 1)
            )

    policy = AlternatingPolicy(layout.dim, horizon=1)
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3)
    loop.run(_StopAfter(100))

    assert loop.timesteps_done == 3
    assert policy.calls == 6


def test_loop_announces_horizon_end_target_per_inference():
    """on_chunk gets the FULL chunk's estimated end state (policy intent), once per
    inference — even when replan_steps executes only a prefix of the chunk."""
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4)
    announced = []
    loop = _loop(cfg, brain, policy, layout, num_timesteps=6, replan_steps=2)
    loop._on_chunk = announced.append
    loop.run(_StopAfter(100))
    assert len(announced) == policy.calls == 3
    assert set(announced[0]) == {"left"}
    hk, vec = announced[0]["left"]
    assert hk == "q"
    # scripted chunk rows are all arange(dim): last row's q_d = first 7 values
    np.testing.assert_allclose(vec, np.arange(7.0))


def test_loop_announces_integrated_horizon_for_qvel():
    """qvel has no joint target in the chunk: the horizon estimate integrates the
    velocity actions forward from the measured q (Euler, one step per action)."""
    cfg = _config("control@policy.control=qvel")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    announced = []
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0)
    loop._on_chunk = announced.append
    loop.run(_StopAfter(100))
    hk, vec = announced[0]["left"]
    assert hk == "q"
    # snapshot's left/q = arange(7); every dq_d row = arange(7); dt = 1/1000
    np.testing.assert_allclose(vec, np.arange(7.0) * (1.0 + 3 * 1e-3))


def test_loop_announces_full_integrated_qvel_trajectory():
    cfg = _config("control@policy.control=qvel")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    trajectories = []
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0)
    loop._on_trajectory = trajectories.append
    loop.run(_StopAfter(100))

    kind, path = trajectories[0]["left"]
    assert kind == "q"
    assert path.shape == (3, 7)
    q0 = np.arange(7.0)
    dq = np.tile(np.arange(7.0), (3, 1))
    np.testing.assert_allclose(path, q0 + np.cumsum(dq, axis=0) / 1000.0)


def test_loop_announces_eef_horizon_for_end_effector():
    """Cartesian kinds predict a TCP position, not a joint config: the chunk-end
    pose_d's position is announced tagged \"eef\"."""
    cfg = _config("control@policy.control=end_effector")
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    announced = []
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0)
    loop._on_chunk = announced.append
    loop.run(_StopAfter(100))
    hk, vec = announced[0]["left"]
    assert hk == "eef"
    np.testing.assert_allclose(vec, np.arange(3.0))  # pose_d = arange(7): [x y z ...]


def test_loop_announces_nothing_for_force():
    """A wrench chunk implies no kinematic displacement: no horizon prediction
    (rather than a fabricated one)."""
    cfg = _config("control@policy.control=force")
    layout = _layout(cfg, ["left"])
    measured_pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

    class _BrainWithEef(_FakeBrain):
        def observe(self):
            snap = _full_observation(self._cfg)
            snap["left/eef"] = _samples(measured_pose)
            return snap

    brain = _BrainWithEef(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=3)
    announced = []
    loop = _loop(cfg, brain, policy, layout, num_timesteps=3, replan_steps=0)
    loop._on_chunk = announced.append
    loop.run(_StopAfter(100))
    assert announced == []


def test_loop_predicts_without_driving_when_no_control_arms():
    """Viz is decoupled from actuation: a dry run (no control-enabled arm) still
    announces horizon predictions for every layout side, but posts no setpoint."""
    cfg = _config()
    layout = _layout(cfg, ["left", "right"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4)
    announced = []
    loop = EvalLoop(
        brain, _observer(cfg), policy, layout,
        control_arms={},
        frequency_hz=1000.0, num_timesteps=4, replan_steps=0,
        on_chunk=announced.append,
        horizon_arms={"left": cfg.arms["left"], "right": cfg.arms["right"]},
    )
    loop.run(_StopAfter(100))
    assert brain.commands == []
    assert set(announced[0]) == {"left", "right"}
    hk, vec = announced[0]["right"]
    assert hk == "q"
    np.testing.assert_allclose(vec, np.arange(8.0, 15.0))  # right q_d slice of arange(16)


def test_loop_survives_failing_horizon_hook():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(layout.dim, horizon=4)

    def bad_hook(targets):
        raise RuntimeError("viz exploded")

    loop = _loop(cfg, brain, policy, layout, num_timesteps=4)
    loop._on_chunk = bad_hook
    loop.run(_StopAfter(100))  # must not raise
    assert loop.timesteps_done == 4


def test_loop_rejects_wrong_action_dim():
    cfg = _config()
    layout = _layout(cfg, ["left"])
    brain = _FakeBrain(cfg)
    policy = _ScriptedPolicy(dim=3, horizon=4)  # checkpoint/config mismatch
    loop = _loop(cfg, brain, policy, layout, num_timesteps=4)
    with pytest.raises(ValueError):
        loop.run(_StopAfter(100))


# --------------------------------------------------------------------------- #
# Config composition
# --------------------------------------------------------------------------- #


def test_policy_config_composes():
    cfg = _config()
    assert cfg.policy.kind == "remote"
    assert cfg.policy.adapter == "openpi"
    assert cfg.policy.port == 8000
    assert cfg.task.eval.frequency_hz == pytest.approx(15.0)

    cfg2 = _config("policy.kind=hold", "policy.replan_steps=8", "policy.host=jeju")
    assert (cfg2.policy.kind, cfg2.policy.replan_steps, cfg2.policy.host) == ("hold", 8, "jeju")


def test_pi05_aloha_policy_config_composes():
    cfg = _config("policy=pi05_aloha")
    assert cfg.policy.adapter == "openpi"
    assert cfg.policy.transport == "websocket"
    assert cfg.policy.host == "192.168.2.101"
    copies = cfg.policy.openpi.request.state.slices
    assert copies[0].source == [0, 1, 3, 4, 5, 6]
    assert copies[1].source == [7, 8, 10, 11, 12, 13]
    holds = cfg.policy.openpi.response.actions.state_holds
    assert holds[0].source == [2]
    assert holds[1].source == [9]
    assert cfg.policy.openpi.response.actions.dim == 16


def test_acme_policy_config_composes():
    cfg = _config("policy=acme")
    assert cfg.policy.adapter == "acme"
    assert cfg.policy.transport == "http"
    assert cfg.policy.port == 53805
    assert cfg.policy.qpos_slice == [0, 7]
    assert cfg.policy.acme_image_keys["exterior_image_1_left"] == "static_left"
    # acme.yaml overrides the wrist slot onto the static cam (default rigs have no
    # wrist camera); the merge keeps the exterior slots from the structured default.
    assert cfg.policy.acme_image_keys["wrist_image_left"] == "static_left"
    assert cfg.policy.acme_image_keys["exterior_image_2_left"] == "static_right"
