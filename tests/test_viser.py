"""Stream-boundary and backend-neutral checks for the primary Viser viewer."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def test_primary_viewer_rate_is_three_hz():
    from dual_flexiv_control.viser.consumer import DEFAULT_VIEWER_RATE_HZ

    assert DEFAULT_VIEWER_RATE_HZ == 3.0


def test_visualization_contract_covers_all_follower_and_factr_telemetry():
    from dual_flexiv_control.interfaces.factr.interface import TELEMETRY_SCALAR_FIELDS
    from dual_flexiv_control.interfaces.factr.interface import TELEMETRY_VECTOR_FIELDS
    from dual_flexiv_control.proprio import PROPRIO_SIGNALS
    from dual_flexiv_control.visualization.schema import default_stream_names

    names = set(default_stream_names())
    for side in ("left", "right"):
        assert {f"{side}/{signal}" for signal in PROPRIO_SIGNALS} <= names
        assert f"factr/{side}" in names
        assert {
            f"factr/telemetry/{side}/{field}"
            for field in TELEMETRY_VECTOR_FIELDS + TELEMETRY_SCALAR_FIELDS
        } <= names
        assert f"eval/{side}/q_horizon" in names
        assert f"eval/{side}/eef_horizon" in names
    assert "eval/policy_comm" in names


def test_dashboard_import_does_not_load_rerun_sdk():
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dual_flexiv_control.dashboard.app; "
            "raise SystemExit(int('rerun' in sys.modules))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_run_registry_does_not_own_a_telemetry_mirror():
    from dual_flexiv_control.dashboard.runner import RunRegistry

    class Manager:
        pass

    registry = RunRegistry(manager=Manager())
    assert not hasattr(registry, "_mirror")


def test_neutral_fk_and_mount_are_finite_and_mirrored():
    from dual_flexiv_control.visualization import geometry

    q = np.zeros(7)
    left = geometry.fk_world_eef("left", q)
    right = geometry.fk_world_eef("right", q)
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert right[0] - left[0] > geometry.MOUNT_SEP_M + 0.5
    np.testing.assert_allclose(
        geometry.mount_world_point("left", np.zeros(3)),
        geometry.MOUNTS["left"]["translation"],
    )


def test_viser_model_scale_is_applied_to_meshes_and_kinematics(monkeypatch):
    """Viser frame scale is cosmetic, so model scaling must be explicit."""
    from dual_flexiv_control.visualization import geometry
    from dual_flexiv_control.viser import scene as scene_module

    class Handle:
        def __init__(self, name, **kwargs):
            self.name = name
            self.position = kwargs.get("position", (0.0, 0.0, 0.0))
            self.wxyz = kwargs.get("wxyz", (1.0, 0.0, 0.0, 0.0))
            self.visible = kwargs.get("visible", True)

    class FakeScene:
        def __init__(self):
            self.frames = {}
            self.mesh_scales = {}

        def add_frame(self, name, *args, **kwargs):
            assert "scale" not in kwargs
            handle = Handle(name, **kwargs)
            self.frames[name] = handle
            return handle

        def add_mesh_trimesh(self, name, mesh, *, scale=1.0, **kwargs):
            self.mesh_scales[name] = scale
            return Handle(name, **kwargs)

        def add_mesh_simple(self, name, vertices, faces, *, scale=1.0, **kwargs):
            self.mesh_scales[name] = scale
            return Handle(name, **kwargs)

    fake_mesh = SimpleNamespace(
        vertices=np.zeros((3, 3)),
        faces=np.asarray(((0, 1, 2),)),
        copy=lambda: None,
    )
    fake_mesh.copy = lambda: fake_mesh
    monkeypatch.setattr(scene_module, "_load_mesh", lambda _path: fake_mesh)

    visual = geometry.Visual(
        "ring",
        geometry.URDF_PATH.parent.parent / "meshes/Rizon4s/visual/ring.obj",
        (0.0, 0.0, -0.0025),
        (0.0, 0.0, 0.0),
        (0.062, 0.062, 0.002),
    )
    chain = (
        geometry.Link(
            "flange",
            (0.1, 0.2, 0.3),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            None,
            (visual,),
        ),
    )
    fake_scene = FakeScene()
    model = scene_module.RobotModel(
        SimpleNamespace(scene=fake_scene),
        "/robot/test",
        chain,
        {
            "translation": (1.0, 2.0, 3.0),
            "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
        },
        scale=2.0,
        follower_gripper=True,
    )

    np.testing.assert_allclose(model.frames[0].position, (0.2, 0.4, 0.6))
    np.testing.assert_allclose(
        fake_scene.frames["/robot/test/flange/visual/ring"].position,
        (0.0, 0.0, -0.005),
    )
    assert fake_scene.mesh_scales["/robot/test/flange/visual/ring/mesh"] == (
        0.124,
        0.124,
        0.004,
    )
    assert fake_scene.mesh_scales[
        "/robot/test/flange/visual/grav_gripper/mesh"
    ] == (0.002, 0.002, 0.002)
