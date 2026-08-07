"""Dashboard logic: task discovery and per-phase blueprint construction.

These are pure, hardware-free checks. Anything that binds ports or starts the
Rerun servers is out of scope here (covered by the manual smoke run).
"""

from __future__ import annotations

import importlib.util

import pytest

from dual_flexiv_control.dashboard.tasks import TaskInfo
from dual_flexiv_control.dashboard.tasks import discover_policies
from dual_flexiv_control.dashboard.tasks import discover_rigs
from dual_flexiv_control.dashboard.tasks import discover_tasks

_HAVE_RERUN = importlib.util.find_spec("rerun") is not None
_needs_rerun = pytest.mark.skipif(not _HAVE_RERUN, reason="rerun-sdk not installed")


@_needs_rerun
def test_viewer_server_bounds_refresh_history(monkeypatch):
    from dual_flexiv_control.dashboard import viewer

    grpc_calls = []
    monkeypatch.setattr(viewer.rr, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        viewer.rr,
        "serve_grpc",
        lambda **kwargs: grpc_calls.append(kwargs) or "rerun+http://127.0.0.1:19876/proxy",
    )
    monkeypatch.setattr(viewer.rr, "serve_web_viewer", lambda **kwargs: None)
    monkeypatch.setattr(viewer, "_await_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(viewer, "_SERVERS", None)
    monkeypatch.setattr(viewer, "_WEB_VIEWER_PORT", None)

    viewer.start_servers(grpc_port=19876, web_port=19090)

    assert grpc_calls == [{
        "grpc_port": 19876,
        "server_memory_limit": viewer.DEFAULT_MEMORY_LIMIT,
        "cors_allow_origin": ["*"],
    }]


@_needs_rerun
def test_replay_replaces_server_store_for_each_episode(monkeypatch):
    from types import SimpleNamespace

    import rerun as rr

    from dual_flexiv_control.dashboard import replay

    events = []

    class FakeRecording:
        def __init__(self, app_id, recording_id):
            self.recording_id = recording_id
            events.append(("create", recording_id))

        def serve_grpc(self, **kwargs):
            events.append(("serve", self.recording_id, kwargs))
            return "rerun+http://127.0.0.1:19880/proxy"

        def disconnect(self):
            events.append(("disconnect", self.recording_id))

        def flush(self):
            events.append(("flush", self.recording_id))

    replay.reset()
    monkeypatch.setattr(rr, "RecordingStream", FakeRecording)
    monkeypatch.setattr(replay, "serve_grpc_checked", lambda serve, port, what: serve())
    monkeypatch.setattr(replay, "read_episode", lambda ds, index: ([{"index": index}], []))
    monkeypatch.setattr(
        replay,
        "_log_frames",
        lambda rec, frames, cams: events.append(("log", rec.recording_id, frames[0]["index"])),
    )

    try:
        replay.start_replay_viewer(web_port=19090, grpc_port=19880)
        replay.log_episode(SimpleNamespace(repo_id="dfc/test"), 1)
        replay.log_episode(SimpleNamespace(repo_id="dfc/test"), 2)
    finally:
        replay.reset()

    served_ids = [event[1] for event in events if event[0] == "serve"]
    assert served_ids[0] == "replay-host"
    assert served_ids[1].startswith("ep-dfc/test-1-")
    assert served_ids[2].startswith("ep-dfc/test-2-")
    assert ("disconnect", "replay-host") in events
    assert ("disconnect", served_ids[1]) in events
    for event in (e for e in events if e[0] == "serve"):
        assert event[2]["server_memory_limit"] == replay.DEFAULT_REPLAY_MEMORY_LIMIT


def test_dashboard_separates_control_modes_from_content_views():
    from dual_flexiv_control.dashboard import app

    assert app.CONTROL_MODES == ("Experiment", "Calibration")
    assert app.CONTENT_VIEWS == ("Viewer", "Cameras", "Storage", "Logs")
    assert not set(app.CONTROL_MODES) & set(app.CONTENT_VIEWS)


def test_calibration_has_no_viewer_or_rerun_feed():
    from dual_flexiv_control.dashboard import app
    from dual_flexiv_control.dashboard import calibration

    assert not hasattr(app, "_calibration_viewer")
    assert not hasattr(app, "_calibration_view_feed")
    assert not hasattr(calibration, "start_calib_viewer")
    assert not hasattr(calibration, "render")


def test_viewer_workspace_embeds_viser_scene_and_plotly_dash(monkeypatch):
    from types import SimpleNamespace

    from dual_flexiv_control.dashboard import app

    class Column:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            active_column.append(self.name)

        def __exit__(self, *_args):
            active_column.pop()

    active_column = []
    column_calls = []
    iframes = []

    def columns(spec, **kwargs):
        column_calls.append((spec, kwargs))
        return Column("scene"), Column("telemetry")

    monkeypatch.setattr(app.st, "columns", columns)
    monkeypatch.setattr(
        app.st,
        "iframe",
        lambda url, **kwargs: iframes.append((active_column[-1], url, kwargs)),
    )
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_render_skill_bar", lambda registry: None)
    monkeypatch.setattr(app._cameras, "depth_cameras", lambda: [])
    monkeypatch.setattr(app, "_robot_data_status", lambda: None)

    servers = SimpleNamespace(web_url="http://127.0.0.1:9090/?url=metrics")
    plots = SimpleNamespace(web_url="http://127.0.0.1:9094")
    app._render_viewer_workspace(servers, plots, object())

    assert column_calls == [
        ([2, 3], {"gap": "small", "vertical_alignment": "top"})
    ]
    assert iframes == [
        (
            "scene",
            f"{servers.web_url}&dfc_view={app.VISER_VIEW_REVISION}",
            {"height": app.LIVE_VIEWER_HEIGHT_PX},
        ),
        ("telemetry", plots.web_url, {"height": app.LIVE_VIEWER_HEIGHT_PX}),
    ]


def test_control_workspace_renders_only_selected_mode(monkeypatch):
    from types import SimpleNamespace

    from dual_flexiv_control.dashboard import app

    selected = ["Calibration", "Calibration", "Experiment"]
    calls = []
    monkeypatch.setattr(
        app.st,
        "segmented_control",
        lambda *_args, **_kwargs: selected.pop(0),
    )
    monkeypatch.setattr(app.st, "session_state", {})
    monkeypatch.setattr(
        app,
        "_render_calibration_controls",
        lambda: calls.append(("calibration",)),
    )
    monkeypatch.setattr(
        app,
        "_render_controls",
        lambda tasks, rig, registry: calls.append(("experiment",)),
    )
    monkeypatch.setattr(
        app._runner,
        "activate_metrics_view",
        lambda view: calls.append(("activate_metrics", view.state)),
    )
    monkeypatch.setattr(
        app._robot,
        "clear_calibration_targets",
        lambda: calls.append(("clear_calibration",)),
    )
    registry = SimpleNamespace(session_view=lambda: SimpleNamespace(state="viewing"))

    app._render_control_workspace([], None, registry)
    app._render_control_workspace([], None, registry)
    app._render_control_workspace([], None, registry)

    assert calls == [
        ("calibration",),
        ("calibration",),
        ("clear_calibration",),
        ("activate_metrics", "viewing"),
        ("experiment",),
    ]


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


def test_discover_policies_finds_shipped_types():
    # The eval launcher's policy-type dropdown: real entries only, no schema base.
    policies = discover_policies()
    assert {"acme", "openpi", "pi05_aloha"} <= set(policies)
    assert "base_policy" not in policies


def test_discover_rigs_finds_shipped_rigs_with_descriptions():
    rigs = {r.name: r for r in discover_rigs()}
    assert {"bimanual", "bench", "left_only", "right_only"} <= set(rigs)
    # Each rig file leads with a one-line hardware summary (after @package).
    assert all(r.description for r in rigs.values())
    assert "@package" not in rigs["bench"].description
    assert "dummy" in rigs["bench"].description.lower()


def test_policy_server_info_reports_protocol_and_checkpoint(tmp_path, monkeypatch):
    from dual_flexiv_control.dashboard import policy_servers

    (tmp_path / "custom.yaml").write_text(
        "adapter: acme\ntransport: http\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        policy_servers,
        "_probe_http",
        lambda endpoint, timeout: {
            "server": {"name": "ACME"},
            "conventions": {"checkpoint_dir": "/models/moose/checkpoint_12000"},
        },
    )

    info = policy_servers.inspect_policy_server(
        "custom", "gpu-box", 53805, policy_dir=tmp_path,
    )

    assert info.reachable
    assert info.adapter == "acme"
    assert info.transport == "http"
    assert info.endpoint == "http://gpu-box:53805"
    assert info.checkpoint == "/models/moose/checkpoint_12000"
    help_text = policy_servers.policy_server_help(info)
    assert "Policy server details" in help_text
    assert "checkpoint_12000" in help_text
    assert '"conventions"' in help_text


def test_policy_server_info_keeps_failed_probe_informational(tmp_path, monkeypatch):
    from dual_flexiv_control.dashboard import policy_servers

    (tmp_path / "openpiish.yaml").write_text(
        "adapter: openpi\n",
        encoding="utf-8",
    )

    def fail(*_args):
        raise TimeoutError("metadata handshake timed out")

    monkeypatch.setattr(policy_servers, "_probe_websocket", fail)
    info = policy_servers.inspect_policy_server(
        "openpiish", "gpu-box", 8000, policy_dir=tmp_path,
    )

    assert not info.reachable
    assert info.transport == "websocket"
    assert "timed out" in policy_servers.policy_server_help(info)


def test_eval_launchability_does_not_depend_on_factr_leaders():
    """Eval is policy-driven: unavailable FACTR leaders only gate Collection."""
    from dataclasses import replace

    from dual_flexiv_control.dashboard import app
    from dual_flexiv_control.dashboard.session import SessionView

    ready = SessionView(
        state="viewing",
        factr_servers={
            "state": "running",
            "down": ["teleop:left", "teleop:right"],
        },
    )
    assert app._eval_launchable(ready)

    # Eval still needs live policy observations. Connected followers with an
    # E-stop/fault are handled daemon-side as a dry run, but a down arm has no
    # state stream from which to construct an observation.
    assert not app._eval_launchable(replace(ready, cameras_down=("zed:static",)))
    assert not app._eval_launchable(replace(ready, arms_down=("left",)))
    assert not app._eval_launchable(replace(ready, state="collection"))


@_needs_rerun
@pytest.mark.parametrize("phase", ["eval", "collection", "skill"])
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


def test_read_live_policy_comm_returns_buffered_events(tmp_path):
    # The mirror's comm reader: all buffered [kind, seq, elapsed_s] events (with
    # ring timestamps), None when no eval run publishes the stream.
    import numpy as np

    from dual_flexiv_control.dashboard.arms import POLICY_COMM_STREAM
    from dual_flexiv_control.dashboard.arms import read_live_policy_comm
    from dual_flexiv_control.streams import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    assert read_live_policy_comm(runtime_dir=str(tmp_path)) is None

    registry = StreamRegistry(str(tmp_path), "runX")
    writer = StreamWriter.create(
        StreamSpec(name=POLICY_COMM_STREAM, dim=3, capacity=512,
                   dtype="float64", rate_hz=15.0),
        "runX",
        registry,
    )
    try:
        writer.write(np.array([0.0, 1.0, 0.0]))   # sent
        writer.write(np.array([1.0, 1.0, 0.25]))  # received, 250 ms round trip
        samples = read_live_policy_comm(runtime_dir=str(tmp_path))
        assert samples is not None and samples.n == 2
        np.testing.assert_allclose(samples.data[0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(samples.data[1], [1.0, 1.0, 0.25])
        assert samples.t_ns[1] >= samples.t_ns[0] > 0
        assert list(samples.seq) == [0, 1]
    finally:
        writer.close()
        writer.unlink()


@_needs_rerun
def test_eval_layout_has_policy_comm_row_collection_does_not():
    # Eval adds the policy-server comms panels (packet send/receive activity +
    # round-trip latency) below the robot metrics; collection has no server.
    from dual_flexiv_control.dashboard import blueprints

    eval_origins = _view_origins(blueprints.for_phase("eval", "handover"))
    assert f"/{blueprints.POLICY_COMM_ROOT}" in eval_origins
    assert f"/{blueprints.POLICY_LATENCY_PATH}" in eval_origins
    for phase in ("collection", "viewing"):
        origins = _view_origins(blueprints.for_phase(phase, "handover"))
        assert not any(o.startswith("/policy") for o in origins)


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


def test_external_joint_torque_is_a_dashboard_metric():
    from dual_flexiv_control.dashboard import blueprints
    from dual_flexiv_control.dashboard import runner

    assert "tau_ext" in blueprints.PROPRIO_SERIES
    assert blueprints.PROPRIO_DIMS["tau_ext"] == 7
    assert "External joint torque" in blueprints.PROPRIO_TITLES["tau_ext"]
    assert ("tau_ext", "tau_ext", None) in runner._VIEW_STREAMS


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
    from dual_flexiv_control.dashboard import cameras as cam_mod
    from dual_flexiv_control.dashboard.arms import set_active_rig
    from dual_flexiv_control.dashboard.cameras import CameraView
    from dual_flexiv_control.dashboard.cameras import discover_camera_views

    set_active_rig("bimanual")  # full camera set
    cam_mod.reset()
    try:
        views = discover_camera_views()
        keys = {v.key for v in views}
        assert {"cam/wrist_left/left", "cam/static/left", "cam/static/right"} <= keys
        assert all(isinstance(v, CameraView) for v in views)
    finally:
        set_active_rig(None)
        cam_mod.reset()


def test_discover_arms_uses_config_names():
    from dual_flexiv_control.dashboard.arms import ArmInfo
    from dual_flexiv_control.dashboard.arms import discover_arms
    from dual_flexiv_control.dashboard.arms import set_active_rig

    set_active_rig("bimanual")  # two-arm rig
    try:
        by_side = {a.side: a for a in discover_arms()}
        assert {"left", "right"} <= set(by_side)
        assert all(isinstance(a, ArmInfo) for a in by_side.values())
        assert by_side["left"].name == "Lauer"
        assert by_side["right"].name == "Rogers"
    finally:
        set_active_rig(None)


def test_discover_arms_carries_serials_for_the_probe():
    # The eval dq probe hands these serials straight to flexivrdk.
    from dual_flexiv_control.dashboard.arms import discover_arms
    from dual_flexiv_control.dashboard.arms import set_active_rig

    set_active_rig("bimanual")  # two-arm rig
    try:
        by_side = {a.side: a for a in discover_arms()}
        assert by_side["left"].serial == "Rizon4s-062841"
        assert by_side["right"].serial == "Rizon4s-062837"
        assert all(a.dof == 7 for a in by_side.values())
    finally:
        set_active_rig(None)


def test_runtime_is_sim_returns_bool():
    from dual_flexiv_control.dashboard.arms import runtime_is_sim

    assert isinstance(runtime_is_sim(), bool)


def test_calibration_configured_leader_sides_from_rig():
    from dual_flexiv_control.dashboard import calibration
    from dual_flexiv_control.dashboard.arms import set_active_rig

    set_active_rig("left_only")  # ships just the left FACTR leader (:5000)
    try:
        assert calibration.configured_leader_sides() == ["left"]
    finally:
        set_active_rig(None)


def test_calibration_capture_joint_solves_one_index(monkeypatch):
    # Drive the real per-joint capture path against the synthetic (sim) FACTR source
    # so it needs no hardware; each capture solves exactly its one joint's offset.
    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    calibration.reset()  # drop any cached client so the sim override takes effect
    try:
        offsets = calibration.initial_offsets("left")
        assert len(offsets) == 7  # gripper dropped
        for j in (0, 3, 6):
            cap = calibration.capture_joint("left", j, samples=1)
            assert cap.joint == j
            assert -180.0 <= cap.offset_deg <= 180.0
            # captured joint reads its straight angle; offset is that negated (+ wrap)
            assert cap.offset_deg == pytest.approx(_wrap180(-cap.captured_deg), abs=1e-6)
    finally:
        calibration.reset()
        _arms.set_active_rig(None)


def test_calibration_capture_joint_rejects_out_of_range(monkeypatch):
    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    calibration.reset()
    try:
        with pytest.raises(RuntimeError):
            calibration.capture_joint("left", 7, samples=1)  # 7-DoF: valid indices 0..6
    finally:
        calibration.reset()
        _arms.set_active_rig(None)


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def test_calibration_reference_poses_make_every_sign_observable():
    # J1 must never pitch behind the robot. Straight-up (0) and the established
    # forward elbow bend (-90) are its only safe anchors. Every joint still needs
    # at least two distinct references, one nonzero, to make its sign observable.
    from dual_flexiv_control.dashboard.calibration import REFERENCE_POSES

    assert len(REFERENCE_POSES) >= 3
    for pose in REFERENCE_POSES:
        assert len(pose.q_deg) == 7
        assert all(v % 90 == 0 for v in pose.q_deg)
        assert pose.q_deg[1] in (0.0, -90.0)
    for j in range(7):
        values = [p.q_deg[j] for p in REFERENCE_POSES]
        assert len(set(values)) >= 2, f"J{j} never moves in the pose set"
        assert any(v != 0 for v in values), f"J{j} has no sign-observable target"


def test_calibration_solve_recovers_full_convention():
    # Synthesize leader readings from a known convention (l = wrap(s*r - off)) at
    # every reference pose; the solve must recover offsets AND sign flips exactly,
    # and the recovered convention must map each sample back onto its reference.
    import numpy as np

    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control.convention import convert_factr_to_rizon
    from dual_flexiv_control.dashboard.calibration import REFERENCE_POSES
    from dual_flexiv_control.dashboard.calibration import solve_from_samples

    true_off = [180.0, -90.0, -90.0, 90.0, 90.0, 180.0, -90.0]
    true_flips = {1, 3, 6}
    pairs = []
    for pose in REFERENCE_POSES:
        leader = [
            _wrap180((-r if j in true_flips else r) - true_off[j])
            for j, r in enumerate(pose.q_deg)
        ]
        pairs.append((leader, list(pose.q_deg)))

    fit = solve_from_samples(pairs, fallback_flips=())
    assert set(fit.sign_flip_joints) == true_flips
    assert fit.ambiguous_joints == []
    assert fit.n_samples == len(REFERENCE_POSES)
    for j in range(7):
        assert _wrap180(fit.offsets_deg[j] - true_off[j]) == pytest.approx(0.0, abs=1e-9)
        assert fit.residuals_deg[j] == pytest.approx(0.0, abs=1e-9)

    conv = JointConventionCfg(
        offsets_deg=list(fit.offsets_deg), sign_flip_joints=sorted(fit.sign_flip_joints)
    )
    for leader, ref in pairs:
        q = np.radians(np.asarray(leader + [0.0]))  # + trailing gripper value
        converted = np.degrees(convert_factr_to_rizon(q, conv))
        # Calibration identifies angles modulo 360 and live conversion selects the
        # canonical representative before it reaches the follower.
        assert [_wrap180(v - r) for v, r in zip(converted, ref)] == pytest.approx(
            [0.0] * len(ref), abs=1e-9
        )


def test_calibration_solve_home_only_degrades_to_straight_pose():
    # With only the all-zero pose captured, both signs fit every joint perfectly, so
    # the sign is ambiguous: the configured flips are kept and the offset is exactly
    # the straight-pose solve, wrap(-leader).
    from dual_flexiv_control.dashboard.calibration import solve_from_samples

    leader = [10.0, -170.0, 45.0, 0.0, 90.0, -30.0, 175.0]
    fit = solve_from_samples([(leader, [0.0] * 7)], fallback_flips=[2, 4])
    assert set(fit.sign_flip_joints) == {2, 4}
    assert fit.ambiguous_joints == list(range(7))
    for j in range(7):
        assert _wrap180(fit.offsets_deg[j] - _wrap180(-leader[j])) == pytest.approx(0.0, abs=1e-9)


def test_calibration_solve_residual_flags_inconsistent_joint():
    # A mis-struck pose shows up as a nonzero RMS residual on the affected joint
    # while the others stay clean (and their offsets stay exact).
    from dual_flexiv_control.dashboard.calibration import REFERENCE_POSES
    from dual_flexiv_control.dashboard.calibration import solve_from_samples

    pairs = []
    for k, pose in enumerate(REFERENCE_POSES):
        leader = [float(r) for r in pose.q_deg]  # identity convention: off=0, no flips
        if k == 1:
            leader[2] += 8.0  # joint 2 badly matched in one pose
        pairs.append((leader, list(pose.q_deg)))

    fit = solve_from_samples(pairs, fallback_flips=())
    assert fit.residuals_deg[2] > 2.0
    for j in (0, 1, 3, 4, 5, 6):
        assert fit.residuals_deg[j] == pytest.approx(0.0, abs=1e-9)
        assert fit.offsets_deg[j] == pytest.approx(0.0, abs=1e-9)



def test_calibration_model_home_solves_dfc_pose_and_derives_audit_offset():
    import numpy as np

    from dual_flexiv_control.dashboard.calibration import solve_model_home

    fit = solve_model_home(
        raw_leader_deg=[10.0, -20.0, 30.0, 40.0],
        offsets_deg=[5.0, 2.0, -3.0, 4.0],
        sign_flip_joints=[1],
        model_signs=[1.0, -1.0, 1.0, -1.0],
        target_q_rad=[0.0, 0.0, 0.0, 1.57],
    )
    expected_home = np.radians([15.0, 18.0, 27.0, 44.0])
    assert fit.home_q_rad == pytest.approx(expected_home)
    assert fit.derived_offset_rad == pytest.approx(
        np.asarray(fit.target_q_rad) - np.asarray(fit.signs) * expected_home
    )


def test_calibration_model_home_target_is_read_from_factr(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dual_flexiv_control.dashboard import calibration

    config_dir = tmp_path / "src" / "factr_teleop" / "factr_teleop" / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "factr_rizon_left.yaml").write_text(
        "arm_teleop:\n"
        "  initialization:\n"
        "    model_home_q_rad: [0.1, 0.2, 1.3]\n"
    )
    monkeypatch.setattr(calibration, "follower_dof", lambda _side: 3)
    monkeypatch.setattr(
        calibration._arms,
        "discover_factr",
        lambda: SimpleNamespace(launch=SimpleNamespace(workdir=str(tmp_path))),
    )

    assert calibration.factr_model_home("left") == pytest.approx([0.1, 0.2, 1.3])

def test_calibration_solve_pose_samples_ignores_stale_entries():
    # The UI wrapper drops samples whose pose name vanished or whose length no longer
    # matches the follower DoF, instead of crashing the solve.
    from dual_flexiv_control.dashboard import calibration
    from dual_flexiv_control.dashboard.arms import set_active_rig

    set_active_rig("left_only")
    try:
        poses = calibration.reference_poses("left")
        assert all(len(p.q_deg) == 7 for p in poses)
        samples = {
            poses[0].name: [0.0] * 7,
            poses[1].name: [0.0, 0.0, 0.0],  # stale: wrong length
            "no-such-pose": [0.0] * 7,       # stale: renamed away
        }
        fit = calibration.solve_pose_samples("left", samples)
        assert fit.n_samples == 1
    finally:
        set_active_rig(None)


def test_calibration_format_yaml_and_overrides():
    # Calibration targets DFC's per-leader factr config.
    from dual_flexiv_control.dashboard.calibration import format_overrides
    from dual_flexiv_control.dashboard.calibration import format_yaml

    offsets = [180.0, -90.0, -90.0, 90.0, 90.0, 180.0, -90.0]
    y = format_yaml("right", offsets, [1, 2, 3])
    assert "leaders:" in y and "raw_to_dfc:" in y
    assert "offsets_deg: [180.00, -90.00" in y
    assert "sign_flip_joints: [1, 2, 3]" in y
    assert "factr.leaders.right.raw_to_dfc.offsets_deg" in format_overrides(
        "right", offsets, [1, 2, 3]
    )


def test_factr_path_follows_active_rig_group():
    from dual_flexiv_control.dashboard import calibration

    assert calibration.factr_path("left_only").name == "left.yaml"
    assert calibration.factr_path("right_only").name == "right.yaml"
    assert calibration.factr_path("bimanual").name == "bimanual.yaml"


def test_apply_to_rig_merges_existing_gripper(tmp_path, monkeypatch):
    # An existing convention with gripper endpoints must survive a sync; offsets/flips
    # are (re)written. Idempotent: applying twice leaves a single valid convention.
    import yaml

    from dual_flexiv_control.dashboard import calibration

    factr_file = tmp_path / "factr.yaml"
    factr_file.write_text(
        "leaders:\n"
        "  left:\n"
        "    raw_to_dfc: { gripper_open: 0.1, gripper_closed: 1.2 }\n"
        "    home_q_rad: [1, 2, 3]\n"
        "    dfc_to_factr: { signs: [1, 1, 1], offset_rad: [0, 0, 0] }\n"
    )
    monkeypatch.setattr(calibration, "factr_path", lambda *_a, **_k: factr_file)

    offsets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    calibration.apply_to_rig("left", offsets, [3, 1])
    calibration.apply_to_rig("left", offsets, [3, 1])  # idempotent
    data = yaml.safe_load(factr_file.read_text())
    conv = data["leaders"]["left"]["raw_to_dfc"]
    assert conv["offsets_deg"] == [round(o, 2) for o in offsets]
    assert conv["sign_flip_joints"] == [1, 3]           # sorted+deduped
    assert conv["gripper_open"] == 0.1 and conv["gripper_closed"] == 1.2  # preserved
    assert data["leaders"]["left"]["home_q_rad"] == [1, 2, 3]
    assert "offset_rad" not in data["leaders"]["left"]["dfc_to_factr"]



def test_apply_to_rig_saves_complete_measured_leader_calibration(tmp_path, monkeypatch):
    import yaml

    from dual_flexiv_control.dashboard import calibration

    factr_file = tmp_path / "factr.yaml"
    factr_file.write_text(
        "leaders:\n"
        "  left:\n"
        "    raw_to_dfc: {}\n"
        "    home_q_rad: [9, 9, 9]\n"
        "    dfc_to_factr: { signs: [1, 1, 1], offset_rad: [9, 9, 9] }\n"
    )
    monkeypatch.setattr(calibration, "factr_path", lambda *_a, **_k: factr_file)
    offsets = [1.0, 2.0, 3.0]
    model = calibration.solve_model_home(
        [10.0, -20.0, 30.0], offsets, [1], [1.0, -1.0, 1.0], [0.0, 0.0, 1.57]
    )

    calibration.apply_to_rig("left", offsets, [1], model_home=model)

    leader = yaml.safe_load(factr_file.read_text())["leaders"]["left"]
    assert leader["home_q_rad"] == pytest.approx(model.home_q_rad)
    assert leader["dfc_to_factr"]["signs"] == [1, -1, 1]
    assert "offset_rad" not in leader["dfc_to_factr"]
    preview = calibration.format_yaml("left", offsets, [1], model_home=model)
    assert "home_q_rad:" in preview and "dfc_to_factr:" in preview
    overrides = calibration.format_overrides("left", offsets, [1], model_home=model)
    assert "dfc_to_factr.offset_rad" not in overrides

def test_apply_to_rig_rejects_missing_side(tmp_path, monkeypatch):
    from dual_flexiv_control.dashboard import calibration

    factr_file = tmp_path / "r.yaml"
    factr_file.write_text("leaders:\n  left:\n    raw_to_dfc: {}\n")
    monkeypatch.setattr(calibration, "factr_path", lambda *_a, **_k: factr_file)
    with pytest.raises(RuntimeError):
        calibration.apply_to_rig("right", [0.0] * 7, [])


def test_gripper_read_and_preview(monkeypatch):
    import numpy as np

    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    calibration.reset()
    try:
        g = calibration.read_gripper("left", samples=1)
        assert np.isfinite(g)  # trailing element of the DoF+1 sim vector
        prev = calibration.gripper_preview(0.2, 1.2)
        labels = {lab: frac for lab, _raw, frac in prev}
        assert labels["open"] == pytest.approx(0.0)
        assert labels["mid"] == pytest.approx(0.5)
        assert labels["closed"] == pytest.approx(1.0)
        assert calibration.gripper_preview(0.5, 0.5) is None   # equal endpoints -> no map
        assert calibration.gripper_preview(None, 1.0) is None   # unset -> no map
    finally:
        calibration.reset()
        _arms.set_active_rig(None)


def test_format_and_apply_include_gripper(tmp_path, monkeypatch):
    import yaml

    from dual_flexiv_control.dashboard import calibration

    offs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    y = calibration.format_yaml("left", offs, [1, 2], gripper_open=0.1, gripper_closed=1.25)
    assert "gripper_open: 0.1000" in y and "gripper_closed: 1.2500" in y
    ov = calibration.format_overrides("left", offs, [1, 2], gripper_open=0.1, gripper_closed=1.25)
    assert "raw_to_dfc.gripper_open=0.1000" in ov and "raw_to_dfc.gripper_closed=1.2500" in ov
    # omitted -> no gripper keys emitted
    assert "gripper" not in calibration.format_yaml("left", offs, [1, 2])

    factr_file = tmp_path / "r.yaml"
    factr_file.write_text("leaders:\n  left:\n    raw_to_dfc: {}\n")
    monkeypatch.setattr(calibration, "factr_path", lambda *_a, **_k: factr_file)
    calibration.apply_to_rig("left", offs, [1], gripper_open=0.3, gripper_closed=1.4)
    conv = yaml.safe_load(factr_file.read_text())["leaders"]["left"]["raw_to_dfc"]
    assert conv["gripper_open"] == 0.3 and conv["gripper_closed"] == 1.4


def test_follower_dof_and_initial_offsets_length(monkeypatch):
    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    try:
        assert calibration.follower_dof("left") == 7  # server dof 8 - drop_trailing 1
        assert len(calibration.initial_offsets("left")) == 7
        # A mis-sized config offsets list is normalized to the follower DoF, not echoed.
        monkeypatch.setattr(calibration, "current_convention",
                            lambda side: _FakeConv([1.0, 2.0, 3.0]))  # too short
        assert len(calibration.initial_offsets("left")) == 7
        assert calibration.initial_offsets("left")[:3] == [1.0, 2.0, 3.0]
        assert calibration.initial_offsets("left")[3:] == [0.0, 0.0, 0.0, 0.0]  # zero-padded
    finally:
        _arms.set_active_rig(None)


class _FakeConv:
    def __init__(self, offsets):
        self.offsets_deg = offsets
        self.sign_flip_joints = [1, 2, 3]
        self.drop_trailing = 1


def test_calibration_reset_closes_client(monkeypatch):
    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    calibration.reset()
    try:
        client = calibration._get_client()
        assert client is not None
        calibration.reset()
        assert calibration._CLIENT is None  # reset drops (and closes) the cached client
    finally:
        calibration.reset()
        _arms.set_active_rig(None)


def test_read_arm_status_placeholder_when_no_run(tmp_path):
    from dual_flexiv_control.dashboard.arms import ArmStatus
    from dual_flexiv_control.dashboard.arms import discover_arms
    from dual_flexiv_control.dashboard.arms import read_arm_status

    arm = discover_arms()[0]
    status = read_arm_status(arm, runtime_dir=str(tmp_path))
    assert isinstance(status, ArmStatus)
    assert status.source == "disconnected"
    assert status.mode == "unknown"
    assert status.operational_status == "disconnected"
    assert status.estop_pressed is None
    assert status.servo_enabled is None
    assert status.info is arm


def test_read_arm_status_live_iff_stream_fresh(tmp_path):
    # The arm row must show the ACTUAL robot state: a fresh status sample reads
    # live (with the RDK mode decoded), while the frozen last sample of a dead
    # producer reads disconnected — never a stale green "Mode: Idle".
    import time as _time

    import numpy as np

    from dual_flexiv_control.dashboard.arms import ArmInfo
    from dual_flexiv_control.dashboard.arms import read_arm_status
    from dual_flexiv_control.streams import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    arm = ArmInfo(side="left", name="Lauer", serial="", dof=7)
    registry = StreamRegistry(str(tmp_path), "runX")
    writer = StreamWriter.create(
        StreamSpec(name="left/status", dim=5, capacity=64,
                   dtype="float64", rate_hz=10.0),
        "runX",
        registry,
    )
    # READY, E-stop clear, controlling, servo on, mode 6 = NRT_JOINT_POSITION.
    vec = np.array([1.0, 0.0, 1.0, 1.0, 6.0])
    try:
        # Stale sample (dead producer) -> disconnected, mode unknown.
        writer.write(vec, _time.monotonic_ns() - int(60e9))
        stale = read_arm_status(arm, runtime_dir=str(tmp_path))
        assert stale.source == "disconnected"
        assert stale.mode == "unknown"
        assert stale.estop_pressed is None

        # Fresh sample -> live, with the actual RDK mode decoded into the label.
        writer.write(vec, _time.monotonic_ns())
        fresh = read_arm_status(arm, runtime_dir=str(tmp_path))
        assert fresh.source == "live"
        assert fresh.control_active is True
        assert fresh.estop_pressed is False
        try:
            import flexivrdk  # noqa: F401

            assert fresh.mode.lower().replace(" ", "_") == "nrt_joint_position"
        except ImportError:
            assert fresh.mode == "mode 6"
    finally:
        writer.close()


def test_read_leader_status_reachable_iff_stream_fresh(tmp_path, monkeypatch):
    # The leader status probe reads the factr/<side> stream (the same samples
    # control consumes): fresh sample -> reachable (trailing gripper split off);
    # no stream (no producer running) -> disconnected; stale sample -> disconnected.
    import time as _time

    import numpy as np

    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard.arms import LeaderStatus
    from dual_flexiv_control.dashboard.arms import read_leader_status
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.streams import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    monkeypatch.setenv("DFC_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    _arms.reset()
    sides = _arms.configured_leader_sides()
    assert sides, "the default rig should configure at least one FACTR leader"
    side = sides[0]

    # No producer -> no stream -> disconnected (even in sim: the probe is a reader).
    status = read_leader_status(side)
    assert status.reachable is False

    registry = StreamRegistry(str(tmp_path), "runX")
    writer = StreamWriter.create(
        StreamSpec(name=factr_stream_name(side), dim=8, capacity=64,
                   dtype="float64", rate_hz=100.0),
        "runX",
        registry,
    )
    try:
        # Stale sample (older than factr.max_age_s) -> still disconnected.
        writer.write(np.linspace(0.1, 0.8, 8), _time.monotonic_ns() - int(60e9))
        assert read_leader_status(side).reachable is False

        # Fresh sample -> reachable, DoF+gripper split, sim flag carried through.
        writer.write(np.linspace(0.1, 0.8, 8), _time.monotonic_ns())
        status = read_leader_status(side)
        assert isinstance(status, LeaderStatus)
        assert status.reachable is True
        assert status.sim is True
        assert status.dof == 7
        assert status.gripper == pytest.approx(0.8)
        assert status.grav_comp_enabled is False
        assert status.grav_comp_gain == 0.0
        assert status.force_feedback_enabled is False
        assert status.force_feedback_gain == 0.0
    finally:
        writer.close()
        writer.unlink()
        _arms.reset()


def test_read_leader_status_disconnected_when_unconfigured(monkeypatch):
    # A side the rig does not serve reads as disconnected (no signal), never crashes.
    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard.arms import read_leader_status

    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    _arms.reset()
    status = read_leader_status("nonexistent-side")
    assert status.reachable is False
    assert status.gripper is None
    _arms.reset()


def test_read_leader_status_reports_actual_grav_comp(tmp_path, monkeypatch):
    import time as _time
    import numpy as np

    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.streams import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    monkeypatch.setenv("DFC_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: False)
    monkeypatch.setattr(
        _arms, "read_leader_grav_comp_status",
        lambda side: {
            "grav_comp_enabled": side == "left",
            "grav_comp_gain": 1.0,
            "grav_comp_gain_target": 1.0,
            "force_feedback_gain": 1.0 if side == "left" else 0.0,
            "force_feedback_gain_target": 1.0 if side == "left" else 0.0,
            "force_feedback_enabled": side == "left",
        },
    )
    _arms.reset()
    side = _arms.configured_leader_sides()[0]
    registry = StreamRegistry(str(tmp_path), "runY")
    writer = StreamWriter.create(
        StreamSpec(name=factr_stream_name(side), dim=8, capacity=64,
                   dtype="float64", rate_hz=100.0),
        "runY", registry,
    )
    try:
        writer.write(np.zeros(8), _time.monotonic_ns())
        status = _arms.read_leader_status(side)
        assert status.grav_comp_enabled is (side == "left")
        assert status.grav_comp_gain == 1.0
        assert status.force_feedback_enabled is (side == "left")
        assert status.force_feedback_gain == (1.0 if side == "left" else 0.0)
    finally:
        writer.close()
        writer.unlink()
        _arms.reset()


def test_running_factr_service_does_not_mask_unreachable_leader_as_booting(monkeypatch):
    """A started relay with no usable stream is disconnected, not forever booting."""
    from dual_flexiv_control.dashboard import app
    from dual_flexiv_control.dashboard.arms import LeaderStatus

    rendered = []
    monkeypatch.setattr(app.st, "markdown", rendered.append)
    status = LeaderStatus(
        "left", "Left", reachable=False, dof=0, gripper=None, sim=False,
    )

    app._render_leader_row(status, {"state": "unreachable"}, "running")

    assert rendered and "no signal" in rendered[-1]
    assert "Booting" not in rendered[-1]


def test_grav_comp_display_uses_live_gain_not_process_health():
    from dual_flexiv_control.dashboard.arms import grav_comp_display

    assert "enabled" in grav_comp_display({
        "grav_comp_enabled": True,
        "grav_comp_gain": 1.0,
        "grav_comp_gain_target": 1.0,
    })[1]
    assert "enabling" in grav_comp_display({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.4,
        "grav_comp_gain_target": 1.0,
    })[1]
    assert "disabling" in grav_comp_display({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.4,
        "grav_comp_gain_target": 0.0,
    })[1]
    assert grav_comp_display({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.0,
        "grav_comp_gain_target": 0.0,
    })[1] == "gain `0.00`"
    assert "unknown" in grav_comp_display(None)[1]


def test_grav_comp_state_drives_g_hotkey():
    from dual_flexiv_control.dashboard.arms import grav_comp_state

    assert grav_comp_state({
        "grav_comp_enabled": True,
        "grav_comp_gain": 1.0,
        "grav_comp_gain_target": 1.0,
    }) == "enabled"
    assert grav_comp_state({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.4,
        "grav_comp_gain_target": 1.0,
    }) == "enabling"
    assert grav_comp_state({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.4,
        "grav_comp_gain_target": 0.0,
    }) == "disabling"
    assert grav_comp_state({
        "grav_comp_enabled": False,
        "grav_comp_gain": 0.0,
        "grav_comp_gain_target": 0.0,
    }) == "disabled"
    assert grav_comp_state(None) == "unknown"
    assert grav_comp_state({"grav_comp_gain": "bogus"}) == "unknown"


def test_factr_single_toggle_chooses_action_from_aggregate_state():
    from dual_flexiv_control.dashboard.app import _factr_toggle_action

    assert _factr_toggle_action(["disabled", "disabled"]) == "enable"
    assert _factr_toggle_action(["disabling", "disabled"]) == "enable"
    assert _factr_toggle_action(["enabled", "disabled"]) == "disable"
    assert _factr_toggle_action(["enabling", "unknown"]) == "disable"
    assert _factr_toggle_action(["unknown", "unknown"]) is None
    assert _factr_toggle_action([]) is None


def test_force_feedback_display_handles_live_and_legacy_status():
    from dual_flexiv_control.dashboard.arms import force_feedback_display
    from dual_flexiv_control.dashboard.arms import force_feedback_state

    enabled = {
        "force_feedback_enabled": True,
        "force_feedback_gain": 1.0,
        "force_feedback_gain_target": 1.0,
    }
    disabled = {
        "force_feedback_enabled": False,
        "force_feedback_gain": 0.0,
        "force_feedback_gain_target": 0.0,
    }
    assert "enabled" in force_feedback_display(enabled)
    assert "disabled" in force_feedback_display(disabled)
    assert "unknown" in force_feedback_display(None)
    assert force_feedback_state(enabled) == "enabled"
    assert force_feedback_state({
        **disabled, "force_feedback_gain_target": 1.0,
    }) == "enabling"


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


def test_factr_urdf_path_comes_from_server_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dual_flexiv_control.dashboard import arms
    from dual_flexiv_control.dashboard import robot_view

    package = tmp_path / "src" / "factr_teleop" / "factr_teleop"
    (package / "configs").mkdir(parents=True)
    (package / "urdf").mkdir()
    selected = package / "urdf" / "actual-left.urdf"
    selected.write_text("<robot name='actual'/>")
    (package / "configs" / "factr_rizon_left.yaml").write_text(
        "arm_teleop:\n  leader_urdf: actual-left.urdf\n"
    )
    monkeypatch.setattr(
        arms,
        "discover_factr",
        lambda: SimpleNamespace(launch=SimpleNamespace(workdir=str(tmp_path))),
    )

    assert robot_view.factr_urdf_path("left") == selected.resolve()


def test_factr_current_model_uses_calibrated_factr_pose(monkeypatch):
    import numpy as np

    from dual_flexiv_control.dashboard import arms
    from dual_flexiv_control.dashboard import runner

    model_q = np.linspace(0.1, 0.7, 7)
    monkeypatch.setattr(
        arms,
        "_read_live_stream_sample",
        lambda _name, _runtime_dir, max_age_s=None: (model_q, 123),
    )
    monkeypatch.setattr(arms, "factr_max_age_s", lambda: 0.25)

    current = runner._factr_configs({"left": np.zeros(8)})
    np.testing.assert_allclose(current["left"], model_q)


def test_factr_current_model_has_no_dfc_fallback(monkeypatch):
    import numpy as np

    from dual_flexiv_control.dashboard import arms
    from dual_flexiv_control.dashboard import runner

    monkeypatch.setattr(
        arms,
        "_read_live_stream_sample",
        lambda _name, _runtime_dir, max_age_s=None: None,
    )
    monkeypatch.setattr(arms, "factr_max_age_s", lambda: 0.25)

    assert runner._factr_configs({"left": np.arange(8.0)}) == {}


def test_live_factr_model_hides_without_model_q_and_recovers(monkeypatch):
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    visibility = []
    poses = []

    class FakeRecording:
        def set_time(self, *_args, **_kwargs):
            pass

    rec = FakeRecording()
    monkeypatch.setattr(robot_view, "_chain", lambda: [])
    monkeypatch.setattr(robot_view, "_factr_chain", lambda _side: [])
    monkeypatch.setattr(robot_view, "_has_mesh_visuals", lambda _chain: False)
    monkeypatch.setattr(robot_view, "_log_real_skeleton", lambda *_a, **_k: None)
    monkeypatch.setattr(
        robot_view,
        "_log_arm_pose",
        lambda _rec, side, _chain, q, **kwargs: poses.append((side, q, kwargs.get("root"))),
    )
    monkeypatch.setattr(
        robot_view,
        "_set_factr_visible",
        lambda _rec, side, visible: visibility.append((side, visible)),
    )
    robot_view._arm_has_live.clear()
    robot_view._factr_has_live.clear()

    robot_view.update_poses(rec, {}, {}, 0.0, factr_q={})
    assert visibility == [("left", False), ("right", False)]
    assert poses == []

    model_q = np.arange(7.0)
    robot_view.update_poses(rec, {}, {}, 0.1, factr_q={"left": model_q})
    assert visibility[-1] == ("left", True)
    assert poses[-1][0] == "left"
    np.testing.assert_array_equal(poses[-1][1], model_q)
    assert poses[-1][2] == robot_view._factr_root("left")


def test_factr_mount_flips_base_x_and_y_only():
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    for side in ("left", "right"):
        vendor = robot_view._MOUNTS[side]
        factr = robot_view._factr_mount(side)
        np.testing.assert_allclose(factr["translation"], vendor["translation"])
        np.testing.assert_allclose(
            factr["rot"],
            np.asarray(vendor["rot"]) @ np.diag([-1.0, -1.0, 1.0]),
            atol=1e-12,
        )


def test_factr_base_yaw_aligns_zero_pose_axes_with_rizon():
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    def base_axes(chain):
        rotation = np.eye(3)
        axes = []
        for index, link in enumerate(chain):
            rotation = rotation @ robot_view._rot_from_rpy(*link.rpy)
            if index:
                axes.append(rotation @ np.asarray(link.axis, dtype=float))
        return np.asarray(axes)

    factr_axes = base_axes(robot_view._factr_chain("left"))
    vendor_axes = base_axes(robot_view.parse_chain())[: len(factr_axes)]
    correction = robot_view._rot_from_axis_angle(
        (0.0, 0.0, 1.0), robot_view.FACTR_BASE_YAW_RAD
    )
    corrected = (correction @ factr_axes.T).T
    # The vendor URDF contains milliradian factory corrections, so compare the
    # physical axis directions rather than demanding byte-identical cardinal axes.
    dots = np.sum(vendor_axes * corrected, axis=1)
    assert dots == pytest.approx(np.ones(7), abs=1e-4)


def test_legacy_ghost_keeps_dfc_convention():
    import numpy as np

    from dual_flexiv_control.dashboard import runner

    converted = np.linspace(-0.3, 0.4, 8)
    ghost = runner._ghost_configs({"left": converted})
    np.testing.assert_allclose(ghost["left"], converted[:7])


def test_factr_left_binary_stl_meshes_load():
    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view._factr_chain("left")
    meshes = [visual.mesh for link in chain for visual in link.visuals]
    assert meshes
    assert all(path.suffix.lower() == ".stl" for path in meshes)
    assert all(robot_view._load_mesh(path) for path in meshes)


def test_factr_terminal_mesh_uses_side_asset_and_millimetre_scale():
    from dual_flexiv_control.dashboard import robot_view

    expected = {
        "left": ("GripperLeft.stl", (-0.047, -0.08005, -0.11480)),
        "right": ("GripperRight.stl", (-0.047, 0.00145, -0.11480)),
    }
    for side, (mesh_name, origin) in expected.items():
        terminal = robot_view._factr_chain(side)[-1]
        assert terminal.name == "rail_carriage_link"
        assert terminal.visuals
        visual = terminal.visuals[0]
        assert visual.mesh.name == mesh_name
        assert visual.mesh.is_file()
        assert visual.xyz == pytest.approx(origin)
        assert visual.scale == pytest.approx((0.001, 0.001, 0.001))


def test_factr_live_display_scale_matches_follower_link_lengths():
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    follower = robot_view._chain()
    leader = robot_view._factr_chain("left")
    # Exclude the follower-only base riser and flange. The six comparable serial
    # segments are manufactured at an exact 2:1 follower-to-leader scale.
    follower_lengths = np.asarray(
        [np.linalg.norm(link.xyz) for link in follower[2:8]]
    )
    leader_lengths = np.asarray(
        [np.linalg.norm(link.xyz) for link in leader[2:8]]
    )
    np.testing.assert_allclose(
        follower_lengths / leader_lengths,
        robot_view.FACTR_LIVE_DISPLAY_SCALE,
        rtol=1e-5,
    )


def test_live_scene_overlays_are_opt_in(tmp_path, monkeypatch):
    from dual_flexiv_control.dashboard import robot_view

    scales = []
    grippers = []

    class FakeRecording:
        def log(self, *_args, **_kwargs):
            pass

        def set_time(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(robot_view, "PEDESTAL_GLB", tmp_path / "missing.glb")
    monkeypatch.setattr(robot_view, "_chain", lambda: [])
    monkeypatch.setattr(robot_view, "_factr_chain", lambda _side: [])
    monkeypatch.setattr(robot_view, "_log_arm_geometry", lambda *_a, **_k: None)
    monkeypatch.setattr(robot_view, "_log_arm_pose", lambda *_a, **_k: None)
    monkeypatch.setattr(
        robot_view,
        "_log_follower_gripper",
        lambda _rec, side, _chain, *, ghost: grippers.append((side, ghost)),
    )
    monkeypatch.setattr(
        robot_view,
        "_log_factr_geometry",
        lambda _rec, _side, _chain, *, display_scale=1.0: scales.append(display_scale),
    )

    robot_view.log_scene(FakeRecording())
    assert scales == [1.0, 1.0]  # replay/default
    assert grippers == []
    scales.clear()
    robot_view.log_scene(
        FakeRecording(), scale_factr_leaders=True, show_follower_grippers=True
    )
    assert scales == [2.0, 2.0]  # live dashboard
    assert grippers == [
        ("left", False),
        ("left", True),
        ("right", False),
        ("right", True),
    ]


def test_grav_follower_gripper_mesh_and_flange_transform():
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    visual = robot_view._GRAV_GRIPPER_VISUAL
    assert visual.mesh.is_file()
    assert visual.scale == (0.001, 0.001, 0.001)
    assert visual.rpy == pytest.approx((np.pi / 2, 0.0, np.pi / 2))

    positions = robot_view._load_stl(visual.mesh)[0].positions
    rotation = robot_view._rot_from_rpy(*visual.rpy)
    mounted = (rotation @ (positions * np.asarray(visual.scale)).T).T + np.asarray(
        visual.xyz
    )
    # The circular mating face is flush to flange Z=0 and fingers extend +190 mm.
    assert mounted[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert mounted[:, 2].max() == pytest.approx(0.1902876, abs=1e-6)


def test_grav_follower_gripper_ghost_uses_ghost_tree_and_tint(monkeypatch):
    from dual_flexiv_control.dashboard import robot_view

    logged = []

    class FakeMesh3D:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRecording:
        def log(self, path, value, **kwargs):
            logged.append((path, value, kwargs))

    monkeypatch.setattr(robot_view.rr, "Mesh3D", FakeMesh3D)
    robot_view._log_follower_gripper(
        FakeRecording(), "left", robot_view._chain(), ghost=True
    )

    mesh_logs = [item for item in logged if isinstance(item[1], FakeMesh3D)]
    assert mesh_logs
    assert all(path.startswith("robot/left_ghost/") for path, _, _ in logged)
    expected = tuple(c / 255.0 for c in robot_view._GHOST_COLOR["left"])
    assert mesh_logs[0][1].kwargs["albedo_factor"] == pytest.approx(expected)


def test_factr_base_frame_stls_are_localized_for_fk():
    from dual_flexiv_control.dashboard import robot_view

    chain = robot_view._factr_chain("left")
    assert chain[0].visuals[0].xyz == pytest.approx((0.0, 0.0, 0.0))
    assert chain[2].visuals[0].xyz != pytest.approx((0.0, 0.0, 0.0))
    assert all(
        visual.xyz == pytest.approx((0.0, 0.0, 0.0))
        for link in robot_view._localize_factr_base_frame_meshes(
            [
                robot_view._Link(
                    name="base_link",
                    xyz=(0.0, 0.0, 0.0),
                    rpy=(0.0, 0.0, 0.0),
                    axis=(0.0, 0.0, 1.0),
                    child_offset=None,
                    visuals=(
                        robot_view._Visual(
                            name="local",
                            mesh=robot_view.Path("/tmp/meshes_ros_zup_local/link.stl"),
                            xyz=(0.0, 0.0, 0.0),
                            rpy=(0.0, 0.0, 0.0),
                            scale=(1.0, 1.0, 1.0),
                        ),
                    ),
                )
            ]
        )
        for visual in link.visuals
    )


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
    assert "left" in robot_view._shown_traces
    robot_view.update_horizon_targets(rec, {"left": q + 0.2}, {}, t=2.0)  # no measured q: no trace
    robot_view.clear_horizon_targets(rec)
    assert not robot_view._shown_targets
    assert not robot_view._shown_traces


@_needs_rerun
def test_horizon_target_renders_fk_at_every_joint_waypoint(monkeypatch):
    import numpy as np
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view

    captured = {}

    def capture_trace(_rec, side, p_now, p_end, waypoints=None):
        captured[side] = (p_now, p_end, np.asarray(waypoints))

    monkeypatch.setattr(robot_view, "_log_trace", capture_trace)
    rec = rr.RecordingStream("dfc-test-horizon-path")
    q0 = np.zeros(7)
    q_path = np.vstack([q0, q0 + 0.1, q0 + 0.2, q0 + 0.3])
    robot_view.update_horizon_targets(
        rec,
        {"left": q_path[-1]},
        {"left": q0},
        t=1.0,
        target_q_paths={"left": q_path},
    )

    p_now, p_end, waypoints = captured["left"]
    np.testing.assert_allclose(p_now, robot_view.fk_world_eef("left", q0))
    np.testing.assert_allclose(p_end, robot_view.fk_world_eef("left", q_path[-1]))
    assert waypoints.shape == (4, 3)
    np.testing.assert_allclose(
        waypoints,
        np.asarray([robot_view.fk_world_eef("left", q) for q in q_path]),
    )
    robot_view.clear_horizon_targets(rec)


@_needs_rerun
def test_calibration_target_update_and_clear_smoke(monkeypatch):
    # Calibration uses a distinct purple target subtree in the existing viewer;
    # switching reference poses re-poses it and leaving the tab removes it.
    import numpy as np
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view

    rec = rr.RecordingStream("dfc-test-calibration-target")
    monkeypatch.setattr(robot_view, "_REC", rec)
    robot_view.clear_calibration_targets()
    robot_view.show_calibration_target("left", np.zeros(7))
    assert robot_view._shown_calibration_targets == {"left"}
    assert robot_view._calibration_target_q["left"] == (0.0,) * 7

    robot_view.show_calibration_target("right", np.full(7, np.pi / 2))
    assert robot_view._shown_calibration_targets == {"right"}
    assert robot_view._calibration_target_q["right"] == (np.pi / 2,) * 7

    robot_view.clear_calibration_targets()
    assert not robot_view._shown_calibration_targets
    assert not robot_view._calibration_target_q

    robot_view.show_calibration_target("left", np.zeros(7))
    assert robot_view._shown_calibration_targets == {"left"}
    robot_view.clear_calibration_targets()


@_needs_rerun
def test_horizon_eef_target_traces_without_ghost():
    # Cartesian-kind predictions carry only a TCP position: a trace + tip are drawn
    # (from the measured eef when present, else FK of measured q) but no ghost.
    import numpy as np
    import rerun as rr

    from dual_flexiv_control.dashboard import robot_view

    rec = rr.RecordingStream("dfc-test-horizon-eef")
    p = np.array([0.4, 0.0, 0.3])
    robot_view.update_horizon_targets(
        rec, {}, {}, t=1.0,
        target_eef={"right": p}, real_eef={"right": np.array([0.4, 0.1, 0.3])},
    )
    assert "right" in robot_view._shown_traces
    assert "right" not in robot_view._shown_targets  # no joint target -> no ghost
    # no measured eef: falls back to FK of measured q; with neither, tip only
    robot_view.update_horizon_targets(rec, {}, {"right": np.zeros(7)}, t=2.0, target_eef={"right": p})
    robot_view.update_horizon_targets(rec, {}, {}, t=3.0, target_eef={"right": p})
    robot_view.clear_horizon_targets(rec)
    assert not robot_view._shown_traces


@_needs_rerun
def test_mount_world_point_matches_fk_base():
    # The mount transform applied to the base-frame origin lands on the arm's mount
    # (= the FK world position of the chain root), so measured/predicted eef points
    # render in the same frame as the FK trace endpoints.
    import numpy as np

    from dual_flexiv_control.dashboard import robot_view

    for side in ("left", "right"):
        origin = robot_view.mount_world_point(side, np.zeros(3))
        np.testing.assert_allclose(
            origin, np.asarray(robot_view._MOUNTS[side]["translation"]), atol=1e-12
        )
    # a point along base +Z leans outward with the 45° plate tilt (x moves, z rises)
    up = robot_view.mount_world_point("right", np.array([0.0, 0.0, 1.0]))
    anchor = np.asarray(robot_view._MOUNTS["right"]["translation"])
    assert up[0] > anchor[0] + 0.5 and up[2] > 0.5


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


def test_read_live_horizon_q_trajectory_returns_latest_timestamp_group(tmp_path):
    import time

    import numpy as np

    from dual_flexiv_control.dashboard.arms import HORIZON_STREAM
    from dual_flexiv_control.dashboard.arms import read_live_horizon_q
    from dual_flexiv_control.dashboard.arms import read_live_horizon_q_trajectory
    from dual_flexiv_control.streams import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    registry = StreamRegistry(str(tmp_path), "runX")
    writer = StreamWriter.create(
        StreamSpec(
            name=HORIZON_STREAM.format(side="left"),
            dim=7,
            capacity=64,
            dtype="float64",
            rate_hz=15.0,
        ),
        "runX",
        registry,
    )
    try:
        old_stamp = time.monotonic_ns()
        writer.write(np.full(7, -1.0), t_ns=old_stamp)
        new_stamp = old_stamp + 1
        expected = np.vstack([np.arange(7.0) + i for i in range(4)])
        for row in expected:
            writer.write(row, t_ns=new_stamp)

        path = read_live_horizon_q_trajectory("left", runtime_dir=str(tmp_path))
        np.testing.assert_allclose(path, expected)
        np.testing.assert_allclose(
            read_live_horizon_q("left", runtime_dir=str(tmp_path)), expected[-1]
        )
    finally:
        writer.close()
        writer.unlink()


def test_read_live_horizon_eef_none_without_eval_run(tmp_path):
    from dual_flexiv_control.dashboard.arms import EEF_HORIZON_STREAM
    from dual_flexiv_control.dashboard.arms import read_live_horizon_eef

    # Name must match what the eval node publishes (policy.loop.eef_horizon_stream_name).
    from dual_flexiv_control.policy import eef_horizon_stream_name

    assert EEF_HORIZON_STREAM.format(side="left") == eef_horizon_stream_name("left")
    assert read_live_horizon_eef("left", runtime_dir=str(tmp_path)) is None


def test_get_frame_missing_when_no_live_producer(tmp_path):
    # Empty runtime dir -> no live producer -> the frame reads as MISSING (no
    # fabricated placeholder), so the dashboard can surface it as an error.
    from dual_flexiv_control.dashboard.cameras import discover_camera_views
    from dual_flexiv_control.dashboard.cameras import get_frame

    rgb_view = next(v for v in discover_camera_views() if v.channels == 3)
    frame, source = get_frame(rgb_view, runtime_dir=str(tmp_path))
    assert source == "missing"
    assert frame is None


def test_discover_logs_orders_newest_first(tmp_path):
    # Each Hydra run dir holds a system.log; discovery lists them newest-first so the
    # Logs tab defaults to the most recent run.
    import os

    from dual_flexiv_control.dashboard import logs

    for name, mtime in (("2026-01-01_00-00-00", 1_000), ("2026-01-02_00-00-00", 2_000)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        p = run_dir / "system.log"
        p.write_text("hello\n")
        os.utime(p, (mtime, mtime))

    found = logs.discover_logs(tmp_path)
    assert [f.name for f in found] == ["2026-01-02_00-00-00", "2026-01-01_00-00-00"]
    assert all(f.size_bytes > 0 for f in found)


def test_discover_logs_empty_when_no_outputs(tmp_path):
    from dual_flexiv_control.dashboard import logs

    assert logs.discover_logs(tmp_path) == []


def test_read_tail_truncates_large_logs(tmp_path):
    from dual_flexiv_control.dashboard import logs

    p = tmp_path / "system.log"
    p.write_text("".join(f"line {i}\n" for i in range(100_000)))
    tail = logs.read_tail(p, max_bytes=1024)
    # Bounded read + a truncation marker; the final line is always intact, and
    # every line in the window is kept (no line cap — the box scrolls).
    assert len(tail) < 2048
    assert tail.startswith("… (showing last")
    assert tail.rstrip().endswith("line 99999")


def test_read_tail_returns_whole_small_log(tmp_path):
    from dual_flexiv_control.dashboard import logs

    p = tmp_path / "system.log"
    p.write_text("only line\n")
    assert logs.read_tail(p) == "only line\n"


def test_live_daemon_log_wraps_existing_path(tmp_path):
    # Dashboard-launched runs never write an outputs/ system.log (the persistent
    # session daemon composes each run in-process, skipping the Hydra job
    # machinery that creates it) — their output all lands in the daemon's own
    # log file instead, which the Logs tab must surface via this wrapper.
    from dual_flexiv_control.dashboard import logs

    p = tmp_path / "dfc-session-abc123.log"
    p.write_text("hello\n")
    found = logs.live_daemon_log(str(p))
    assert found is not None
    assert found.path == p
    assert found.size_bytes > 0


def test_live_daemon_log_none_when_no_path():
    from dual_flexiv_control.dashboard import logs

    assert logs.live_daemon_log(None) is None


def test_live_daemon_log_none_when_path_missing(tmp_path):
    from dual_flexiv_control.dashboard import logs

    assert logs.live_daemon_log(str(tmp_path / "gone.log")) is None


def test_factr_gain_display_keeps_separate_arm_values():
    from dual_flexiv_control.dashboard.app import _format_factr_gain

    assert _format_factr_gain(
        ["left"], [{"grav_comp_gain": 0.375}], "grav_comp_gain"
    ) == "0.38"
    assert _format_factr_gain(
        ["left", "right"],
        [{"force_feedback_gain": 0.25}, {"force_feedback_gain": 0.875}],
        "force_feedback_gain",
    ) == "L 0.25 · R 0.88"
    assert _format_factr_gain(
        ["left", "right"], [{"force_feedback_gain": 0.25}, None],
        "force_feedback_gain",
    ) == "L 0.25 · R —"


# ---------------------------------------------------------------------------
# Launcher rig option (dfc-dashboard --rig <name>)
# ---------------------------------------------------------------------------


def test_pop_rig_arg_extracts_both_forms():
    from dual_flexiv_control.dashboard.launch import _pop_rig_arg

    assert _pop_rig_arg(["--rig", "bench"]) == ("bench", [])
    assert _pop_rig_arg(["--rig=left_only"]) == ("left_only", [])
    # Streamlit args pass through untouched, wherever --rig sits among them.
    rig, rest = _pop_rig_arg(["--server.port", "8502", "--rig", "bimanual", "-v"])
    assert rig == "bimanual"
    assert rest == ["--server.port", "8502", "-v"]
    assert _pop_rig_arg([]) == (None, [])


def test_pop_rig_arg_rejects_missing_value():
    from dual_flexiv_control.dashboard.launch import _pop_rig_arg

    with pytest.raises(SystemExit):
        _pop_rig_arg(["--rig"])


def test_validate_rig_accepts_shipped_and_rejects_unknown():
    from dual_flexiv_control.dashboard.launch import _validate_rig

    _validate_rig("bimanual")  # shipped rig: no error
    with pytest.raises(SystemExit, match="unknown rig"):
        _validate_rig("no_such_rig")
