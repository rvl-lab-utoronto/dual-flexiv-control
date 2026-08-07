"""Stream-boundary and backend-neutral checks for the primary Viser viewer."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def test_primary_viewer_rate_is_three_hz():
    from dual_flexiv_control.plotly_dash.consumer import DEFAULT_PLOT_RATE_HZ
    from dual_flexiv_control.viser.consumer import DEFAULT_VIEWER_RATE_HZ

    assert DEFAULT_VIEWER_RATE_HZ == 3.0
    assert DEFAULT_PLOT_RATE_HZ == 3.0


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


def test_scene_and_plot_consumers_subscribe_only_to_their_inputs():
    from dual_flexiv_control.visualization.schema import plot_stream_names
    from dual_flexiv_control.visualization.schema import scene_stream_names

    scene = set(scene_stream_names(camera_names=("static",)))
    plots = set(plot_stream_names())
    assert "left/q" in scene & plots
    assert "left/dq" in plots - scene
    assert "eval/policy_comm" in plots - scene
    assert "eval/left/q_horizon" in scene - plots
    assert "cam/static/depth" in scene - plots


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
    monkeypatch.setattr(
        scene_module, "_load_mesh", lambda _path, _decimation=1: fake_mesh
    )

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


def test_factr_mesh_repair_exceeds_50x_and_is_disk_cached(
    tmp_path, monkeypatch
):
    import trimesh

    from dual_flexiv_control.viser import scene as scene_module

    source = tmp_path / "factr.stl"
    # Dense coplanar tessellation resembles the redundant faces in exported
    # CAD while retaining a known box silhouette for bounds validation.
    dense = trimesh.creation.box()
    for _ in range(5):
        dense = dense.subdivide()
    dense.export(source)
    monkeypatch.setenv("DFC_VISER_MESH_CACHE", str(tmp_path / "cache"))
    scene_module._load_mesh.cache_clear()

    original = trimesh.load(source, force="mesh", process=False)
    simplified = scene_module._load_mesh(
        source, scene_module.FACTR_MESH_DECIMATION_FACTOR
    )
    assert len(original.faces) / len(simplified.faces) >= 50.0
    np.testing.assert_allclose(simplified.extents, original.extents, rtol=0.03)
    assert list((tmp_path / "cache").glob("factr-*.npz"))

    # Clear only the in-memory cache; the second load must use the persistent
    # result rather than invoking the decimator again.
    scene_module._load_mesh.cache_clear()
    monkeypatch.setattr(
        trimesh.convex,
        "convex_hull",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disk cache was not used")
        ),
    )
    cached = scene_module._load_mesh(
        source, scene_module.FACTR_MESH_DECIMATION_FACTOR
    )
    assert len(cached.faces) == len(simplified.faces)


def test_plot_grid_restores_left_right_columns_and_policy_row():
    from dual_flexiv_control.visualization.schema import StreamRoute
    from dual_flexiv_control.plotly_dash.view import plot_grid_order

    assert plot_grid_order(StreamRoute("proprio", "left", "eef"), "eef") == 0
    assert plot_grid_order(StreamRoute("proprio", "right", "eef"), "eef") == 1
    assert plot_grid_order(StreamRoute("proprio", "left", "q"), "q") == 2
    assert plot_grid_order(StreamRoute("proprio", "right", "q"), "q") == 3
    assert plot_grid_order(StreamRoute("factr", "left", "leader"), "q") == 0
    assert plot_grid_order(StreamRoute("factr", "right", "leader"), "q") == 1
    assert plot_grid_order(StreamRoute("policy", None, "packets"), "packets") == 0
    assert plot_grid_order(
        StreamRoute("policy", None, "latency_ms"), "latency_ms"
    ) == 1


def test_plot_cards_have_an_inline_height_for_responsive_plotly():
    from dual_flexiv_control.plotly_dash.view import PLOT_HEIGHT_PX
    from dual_flexiv_control.plotly_dash.view import PLOT_SPECS
    from dual_flexiv_control.plotly_dash.view import PlotStore
    from dual_flexiv_control.plotly_dash.view import create_dash_app

    app = create_dash_app(PlotStore(), 3.0)
    graphs = []

    def visit(component):
        if getattr(component, "id", None) in {spec.graph_id for spec in PLOT_SPECS}:
            graphs.append(component)
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)
        elif children is not None:
            visit(children)

    visit(app.layout)
    assert len(graphs) == len(PLOT_SPECS)
    assert all(graph.style == {"height": f"{PLOT_HEIGHT_PX}px"} for graph in graphs)


def test_plot_store_returns_only_incremental_samples_and_resets_per_run():
    from dual_flexiv_control.plotly_dash.view import PlotStore

    store = PlotStore()
    store.attach("run-a")
    assert store.append("follower/left/q", np.arange(7), 1_000_000_000)
    first = store.snapshot()
    point = first["series"]["follower/left/q"]
    assert point["version"] == 1
    np.testing.assert_array_equal(point["y"], np.arange(7)[None, :])
    assert "follower/left/q" not in store.snapshot({"follower/left/q": 1})["series"]

    store.attach("run-b")
    reset = store.snapshot()
    assert reset["epoch"] != first["epoch"]
    assert reset["run_id"] == "run-b"
    assert reset["series"] == {}


def test_one_factr_ring_feeds_leader_q_and_gripper_plots():
    from dual_flexiv_control.plotly_dash.consumer import PlotlyDashConsumer
    from dual_flexiv_control.plotly_dash.view import PlotStore

    store = PlotStore()
    store.attach("run")
    PlotlyDashConsumer._append_stream(
        "factr/left", np.arange(8, dtype=float), 10, store
    )
    series = store.snapshot()["series"]
    np.testing.assert_array_equal(
        series["leader/left/q"]["y"], np.arange(7, dtype=float)[None, :]
    )
    np.testing.assert_array_equal(series["leader/left/grip"]["y"], [[7.0]])
