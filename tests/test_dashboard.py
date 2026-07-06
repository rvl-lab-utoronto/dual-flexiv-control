"""Dashboard logic: task discovery and per-phase blueprint construction.

These are pure, hardware-free checks. Anything that binds ports or starts the
Rerun servers is out of scope here (covered by the manual smoke run).
"""

from __future__ import annotations

import importlib.util

import pytest

from dual_flexiv_control.dashboard.tasks import TaskInfo
from dual_flexiv_control.dashboard.tasks import discover_tasks

_HAVE_RERUN = importlib.util.find_spec("rerun") is not None
_needs_rerun = pytest.mark.skipif(not _HAVE_RERUN, reason="rerun-sdk not installed")


def test_discover_tasks_finds_shipped_tasks():
    tasks = {t.name: t for t in discover_tasks()}
    assert {"default", "handover"} <= set(tasks)
    assert all(isinstance(t, TaskInfo) for t in tasks.values())


def test_discover_tasks_reads_phase_fields():
    handover = next(t for t in discover_tasks() if t.name == "handover")
    assert handover.language_instruction.startswith("Pick up")
    assert handover.num_episodes == 100
    assert handover.num_timesteps == 600


def test_discover_tasks_every_entry_has_an_instruction():
    # The dropdown filters to real tasks: each must carry a language_instruction.
    assert all(t.language_instruction for t in discover_tasks())


@_needs_rerun
@pytest.mark.parametrize("phase", ["eval", "collection"])
def test_for_phase_builds_a_blueprint(phase):
    import rerun.blueprint as rrb

    from dual_flexiv_control.dashboard import blueprints

    bp = blueprints.for_phase(phase, task_name="handover")
    assert isinstance(bp, rrb.Blueprint)


@_needs_rerun
def test_for_phase_rejects_unknown_phase():
    from dual_flexiv_control.dashboard import blueprints

    with pytest.raises(ValueError):
        blueprints.for_phase("nonsense")


@_needs_rerun
def test_welcome_blueprint_builds():
    import rerun.blueprint as rrb

    from dual_flexiv_control.dashboard import blueprints

    assert isinstance(blueprints.welcome_blueprint(), rrb.Blueprint)


@_needs_rerun
def test_eval_probe_blueprint_builds():
    # The eval no-motion probe uses this focused TCP-position + dq layout.
    import rerun.blueprint as rrb

    from dual_flexiv_control.dashboard import blueprints

    assert isinstance(blueprints.eval_probe_blueprint("handover"), rrb.Blueprint)


@_needs_rerun
def test_metric_panels_split_per_arm():
    # Each arm gets its own column: a left column references only .../left entities and
    # a right column only .../right — no more both-arms-overlaid panels.
    from dual_flexiv_control.dashboard import blueprints

    left = [str(v.origin) for v in blueprints._arm_column("left", ("eef_pos", "dq")).contents]
    right = [str(v.origin) for v in blueprints._arm_column("right", ("eef_pos", "dq")).contents]
    assert left == ["/proprio/eef_pos/left", "/proprio/dq/left"]
    assert right == ["/proprio/eef_pos/right", "/proprio/dq/right"]


def _view_origins(blueprint) -> list[str]:
    """All view origins in a blueprint (containers have ``contents``, views ``origin``)."""
    origins: list[str] = []

    def walk(node) -> None:
        if hasattr(node, "origin"):
            origins.append(str(node.origin))
        for child in getattr(node, "contents", None) or []:
            walk(child)

    walk(blueprint.root_container)
    return origins


@_needs_rerun
@pytest.mark.parametrize("kind", ["collection", "eval", "welcome"])
def test_layouts_embed_robot_scene_not_eef_trace(kind):
    # The robot 3D scene replaced the old end-effector trace: every layout's 3D panel is
    # now the "/robot" scene (logged into the same metrics recording), and nothing
    # points at the removed "/eef" entities.
    from dual_flexiv_control.dashboard import blueprints

    if kind == "welcome":
        bp = blueprints.welcome_blueprint()
    elif kind == "eval":
        bp = blueprints.eval_probe_blueprint("handover")
    else:
        bp = blueprints.for_phase(kind, "handover")

    origins = _view_origins(bp)
    assert "/robot" in origins
    assert not any(o == "/eef" or o.startswith("/eef/") for o in origins)


def test_eef_position_is_a_time_series_metric():
    # "metrics for eef pos": TCP position (eef[:3]) is plotted as an x/y/z time
    # series (the 3D panel is the robot scene), so it must be a first-class series signal.
    from dual_flexiv_control.dashboard import blueprints

    assert "eef_pos" in blueprints.PROPRIO_SERIES
    assert blueprints.PROPRIO_DIMS["eef_pos"] == 3
    assert blueprints.PROPRIO_TITLES["eef_pos"] == "TCP position (m)"
    assert blueprints.proprio_group("eef_pos") == "proprio/eef_pos"
    assert blueprints.proprio_path("eef_pos", "left") == "proprio/eef_pos/left"
    # x/y/z labels/colours are paired for the static SeriesLines styling.
    assert len(blueprints.EEF_POS_COMPONENTS) == len(blueprints.EEF_POS_COLORS) == 3


def test_open_in_vscode_reports_missing_file(tmp_path):
    # Returns before shelling out to `code`, so it never touches a real editor.
    from dual_flexiv_control.dashboard.editor import OpenResult
    from dual_flexiv_control.dashboard.editor import open_in_vscode

    result = open_in_vscode(tmp_path / "does-not-exist.yaml")
    assert isinstance(result, OpenResult)
    assert not result.ok
    assert "not found" in result.message.lower()


def test_editor_discovery_helpers_return_str_or_none():
    from dual_flexiv_control.dashboard import editor

    code = editor._find_code_cli()
    assert code is None or isinstance(code, str)
    socket = editor._ipc_socket()
    assert socket is None or isinstance(socket, str)


def test_discover_camera_views_from_config():
    from dual_flexiv_control.dashboard.cameras import CameraView
    from dual_flexiv_control.dashboard.cameras import discover_camera_views

    views = discover_camera_views()
    keys = {v.key for v in views}
    assert {"cam/wrist_left/left", "cam/static/left", "cam/static/right"} <= keys
    assert all(isinstance(v, CameraView) for v in views)


def test_discover_arms_uses_config_names():
    from dual_flexiv_control.dashboard.arms import ArmInfo
    from dual_flexiv_control.dashboard.arms import discover_arms

    by_side = {a.side: a for a in discover_arms()}
    assert {"left", "right"} <= set(by_side)
    assert all(isinstance(a, ArmInfo) for a in by_side.values())
    assert by_side["left"].name == "Lauer"
    assert by_side["right"].name == "Rogers"


def test_discover_arms_carries_serials_for_the_probe():
    # The eval dq probe hands these serials straight to flexivrdk.
    from dual_flexiv_control.dashboard.arms import discover_arms

    by_side = {a.side: a for a in discover_arms()}
    assert by_side["left"].serial == "Rizon4s-062841"
    assert by_side["right"].serial == "Rizon4s-062837"
    assert all(a.dof == 7 for a in by_side.values())


def test_runtime_is_sim_returns_bool():
    from dual_flexiv_control.dashboard.arms import runtime_is_sim

    assert isinstance(runtime_is_sim(), bool)


def test_read_arm_status_placeholder_when_no_run(tmp_path):
    from dual_flexiv_control.dashboard.arms import ArmStatus
    from dual_flexiv_control.dashboard.arms import discover_arms
    from dual_flexiv_control.dashboard.arms import read_arm_status

    arm = discover_arms()[0]
    status = read_arm_status(arm, runtime_dir=str(tmp_path))
    assert isinstance(status, ArmStatus)
    assert status.source == "disconnected"
    assert status.mode == "disconnected"
    assert status.estop_pressed is None
    assert status.info is arm


def test_robot_urdf_chain_parses():
    # The 3D robot scene builds the arms from the vendor URDF: joint tree (FK) plus
    # the per-link visual meshes. Guard the serial chain + first/last bone offsets.
    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view.parse_chain()
    assert [l.name for l in chain] == [
        "base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "flange",
    ]
    assert chain[0].child_offset == (0.0, 0.0, 0.155)  # base -> link1 (URDF joint1 xyz)
    assert chain[-1].child_offset is None  # flange is the leaf (no bone)


def test_robot_urdf_visual_meshes_resolve():
    # Every <visual> mesh the URDF references must exist in assets/robot/meshes
    # (the ../meshes/... paths resolve relative to the urdf/ dir). link1..7 also
    # carry the scaled ring mesh; flange has no geometry.
    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view.parse_chain()
    by_name = {l.name: l for l in chain}
    assert robot_view._has_mesh_visuals(chain)
    for lk in chain:
        for vis in lk.visuals:
            assert vis.mesh.is_file(), f"missing mesh for {lk.name}: {vis.mesh}"
    assert [v.name for v in by_name["base_link"].visuals] == ["shell"]
    assert [v.name for v in by_name["link1"].visuals] == ["shell", "ring"]
    assert by_name["link1"].visuals[1].scale == (0.062, 0.062, 0.002)
    assert by_name["flange"].visuals == ()


def test_obj_loader_parts_colors_and_units():
    # The OBJ+MTL loader feeds Mesh3D directly: per-usemtl parts, Kd albedo,
    # metre-scale vertices (the exporter's "MilliMeters" comment is wrong),
    # per-corner normals, in-range indices.
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view.parse_chain()
    shell = next(v for v in chain[1].visuals if v.name == "shell")  # link1
    parts = robot_view._load_obj(shell.mesh)
    assert {p.material for p in parts} == {"cover", "robot"}
    for p in parts:
        assert np.isfinite(p.positions).all()
        assert p.indices.shape[1] == 3 and p.indices.max() < len(p.positions)
        assert p.normals is not None and p.normals.shape == p.positions.shape
        assert all(0.0 <= c <= 1.0 for c in p.albedo)
        assert float(np.abs(p.positions).max()) < 1.0  # metres, not millimetres
    cover = next(p for p in parts if p.material == "cover")
    assert cover.albedo == pytest.approx((0.969921, 0.969921, 0.969921))


@_needs_rerun
def test_arm_mesh_geometry_and_tint_smoke():
    # Real-mesh arm geometry + the stale/live albedo tint log to a sink-less
    # recording without error (the live recolor path in update_poses).
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view.parse_chain()
    rec = rr.RecordingStream("dfc-test-arm-meshes")
    robot_view._log_arm_geometry(rec, "left", chain, ghost=False)
    robot_view._tint_arm_meshes(rec, "left", chain, stale=True)
    robot_view._tint_arm_meshes(rec, "left", chain, stale=False)
    assert sum(1 for _ in robot_view._mesh_entities("left", chain)) > 0


@_needs_rerun
def test_fk_world_eef_matches_rig_geometry():
    # The horizon-target trace endpoints come from this numeric FK; sanity it against
    # the known rig: the arms sit on the 45°-outward mount plates, so home poses
    # mirror across the YZ plane, lean apart wider than the plate separation, sit
    # above the mount plane (z > 0), and actually move when a joint moves.
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    home = np.zeros(7)
    left = robot_view.fk_world_eef("left", home)
    right = robot_view.fk_world_eef("right", home)
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    assert left[2] > 0.5 and right[2] > 0.5  # home reaches well above the mount plane
    # equal-and-opposite tilts are both about world Y: the y components match exactly
    # (the x/z pair is NOT an exact mirror — the chain itself has a lateral offset)
    assert right[1] == pytest.approx(left[1])
    # the 45° outward lean spreads the EEFs far beyond the plate separation
    assert right[0] - left[0] > robot_view.MOUNT_SEP_M + 0.5
    bent = robot_view.fk_world_eef("left", np.array([0.0, 0.8, 0.0, 1.2, 0.0, 0.5, 0.0]))
    assert np.linalg.norm(bent - left) > 0.1


@_needs_rerun
def test_horizon_target_update_and_clear_smoke():
    # Purple horizon ghost + EEF trace log to a sink-less recording without error,
    # and clear resets the lazy-geometry tracking for the next eval run.
    import numpy as np
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view

    rec = rr.RecordingStream("dfc-test-horizon")
    robot_view.clear_horizon_targets(rec)  # idempotent when nothing is shown
    q = np.zeros(7)
    robot_view.update_horizon_targets(rec, {"left": q + 0.3}, {"left": q}, t=1.0)
    assert "left" in robot_view._shown_targets
    robot_view.update_horizon_targets(rec, {"left": q + 0.2}, {}, t=2.0)  # no measured q: no trace
    robot_view.clear_horizon_targets(rec)
    assert not robot_view._shown_targets


@_needs_rerun
def test_camera_frustum_logs_at_configured_pose():
    # The Robot-tab frustum: pose comes from conf/camera extrinsics, and logging
    # to a sink-less recording succeeds (no-op before the viewer starts).
    import numpy as np
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view
    from dual_flexiv_control.dashboard.cameras import camera_cfg

    cfg = camera_cfg("static")
    rot, t = robot_view.camera_world_pose(cfg)
    assert rot.shape == (3, 3) and np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    anchor = np.asarray(robot_view._MOUNTS["left"]["translation"])  # pose_frame: mount_left
    np.testing.assert_allclose(t, anchor + np.asarray(cfg.pose_xyz), atol=1e-12)

    robot_view.log_camera_frustum("static", cfg)  # viewer not started -> no-op
    prev = robot_view._REC
    robot_view._REC = rr.RecordingStream("dfc-test-frustum")
    try:
        robot_view.log_camera_frustum("static", cfg)
    finally:
        robot_view._REC = prev


def test_read_live_horizon_q_none_without_eval_run(tmp_path):
    from dual_flexiv_control.dashboard.arms import HORIZON_STREAM
    from dual_flexiv_control.dashboard.arms import read_live_horizon_q

    # Name must match what the eval node publishes (policy.loop.horizon_stream_name).
    from dual_flexiv_control.policy import horizon_stream_name

    assert HORIZON_STREAM.format(side="left") == horizon_stream_name("left")
    assert read_live_horizon_q("left", runtime_dir=str(tmp_path)) is None


def test_get_frame_falls_back_to_placeholder(tmp_path):
    # Empty runtime dir -> no live producer -> synthetic uint8 RGB frame.
    import numpy as np

    from dual_flexiv_control.dashboard.cameras import discover_camera_views
    from dual_flexiv_control.dashboard.cameras import get_frame

    rgb_view = next(v for v in discover_camera_views() if v.channels == 3)
    frame, source = get_frame(rgb_view, runtime_dir=str(tmp_path))
    assert source == "placeholder"
    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
