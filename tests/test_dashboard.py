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
    monkeypatch.setattr(viewer, "_SERVERS", None)
    monkeypatch.setattr(viewer, "_WEB_VIEWER_PORT", None)

    viewer.start_servers(grpc_port=19876, web_port=19090)

    assert grpc_calls == [{
        "grpc_port": 19876,
        "server_memory_limit": viewer.DEFAULT_MEMORY_LIMIT,
        "cors_allow_origin": ["*"],
    }]


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
    assert {"acme", "openpi"} <= set(policies)
    assert "base_policy" not in policies


def test_discover_rigs_finds_shipped_rigs_with_descriptions():
    rigs = {r.name: r for r in discover_rigs()}
    assert {"bimanual", "bench", "left_only"} <= set(rigs)
    # Each rig file leads with a one-line hardware summary (after @package).
    assert all(r.description for r in rigs.values())
    assert "@package" not in rigs["bench"].description
    assert "dummy" in rigs["bench"].description.lower()


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

    set_active_rig("bimanual")  # full camera set (default rig is now left_only)
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

    set_active_rig("bimanual")  # two-arm rig (shipped default is now left_only)
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

    set_active_rig("bimanual")  # two-arm rig (shipped default is now left_only)
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


def test_calibration_format_yaml_and_overrides():
    from dual_flexiv_control.dashboard.calibration import format_overrides
    from dual_flexiv_control.dashboard.calibration import format_yaml

    offsets = [180.0, -90.0, -90.0, 90.0, 90.0, 180.0, -90.0]
    y = format_yaml("right", offsets, [1, 2, 3])
    assert "right:" in y and "offsets_deg: [180.00, -90.00" in y and "sign_flip_joints: [1, 2, 3]" in y
    assert format_overrides("right", offsets, [1, 2, 3]) == (
        "arms.right.convention.offsets_deg='[180.00,-90.00,-90.00,90.00,90.00,180.00,-90.00]' "
        "arms.right.convention.sign_flip_joints='[1,2,3]'"
    )


def test_splice_convention_inserts_and_preserves_comments():
    # On a real rig file with no active convention, splicing inserts one inline line,
    # the result parses, the convention is set, and existing comments survive.
    # The shipped rig may already carry a live calibration (💾 Sync writes into it),
    # so strip any active convention line first — the test targets insertion.
    import yaml

    from dual_flexiv_control.dashboard import calibration

    path = calibration.rig_path("left_only")
    text = "\n".join(
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith("convention:")
    ) + "\n"
    assert "convention:" not in text.replace("# convention:", "")  # none active
    out = calibration._splice_convention(
        text, "left", {"offsets_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "sign_flip_joints": [1, 2]}
    )
    data = yaml.safe_load(out)
    assert data["arms"]["left"]["convention"]["offsets_deg"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert data["arms"]["left"]["convention"]["sign_flip_joints"] == [1, 2]
    assert data["arms"]["left"]["serial"] == "Rizon4s-062841"  # untouched
    assert "# NOTE: a camera exists only" in out                # comments preserved


def test_apply_to_rig_merges_existing_gripper(tmp_path, monkeypatch):
    # An existing convention with gripper endpoints must survive a sync; offsets/flips
    # are (re)written. Idempotent: applying twice leaves a single valid convention.
    import yaml

    from dual_flexiv_control.dashboard import calibration

    rig_file = tmp_path / "myrig.yaml"
    rig_file.write_text(
        "# @package _global_\n"
        "arms:\n"
        "  left:\n"
        "    name: \"Lauer\"\n"
        "    serial: Rizon4s-000000\n"
        "    convention: { gripper_open: 0.1, gripper_closed: 1.2 }\n"
    )
    monkeypatch.setattr(calibration, "rig_path", lambda *_a, **_k: rig_file)

    offsets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    calibration.apply_to_rig("left", offsets, [3, 1])
    calibration.apply_to_rig("left", offsets, [3, 1])  # idempotent
    data = yaml.safe_load(rig_file.read_text())
    conv = data["arms"]["left"]["convention"]
    assert conv["offsets_deg"] == [round(o, 2) for o in offsets]
    assert conv["sign_flip_joints"] == [1, 3]           # sorted+deduped
    assert conv["gripper_open"] == 0.1 and conv["gripper_closed"] == 1.2  # preserved
    assert data["arms"]["left"]["serial"] == "Rizon4s-000000"


def test_apply_to_rig_rejects_missing_side(tmp_path, monkeypatch):
    from dual_flexiv_control.dashboard import calibration

    rig_file = tmp_path / "r.yaml"
    rig_file.write_text("arms:\n  left:\n    serial: X\n")
    monkeypatch.setattr(calibration, "rig_path", lambda *_a, **_k: rig_file)
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
    assert "convention.gripper_open=0.1000" in ov and "convention.gripper_closed=1.2500" in ov
    # omitted -> no gripper keys emitted
    assert "gripper" not in calibration.format_yaml("left", offs, [1, 2])

    rig_file = tmp_path / "r.yaml"
    rig_file.write_text("arms:\n  left:\n    serial: X\n")
    monkeypatch.setattr(calibration, "rig_path", lambda *_a, **_k: rig_file)
    calibration.apply_to_rig("left", offs, [1], gripper_open=0.3, gripper_closed=1.4)
    conv = yaml.safe_load(rig_file.read_text())["arms"]["left"]["convention"]
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


def test_calibration_render_smoke(monkeypatch):
    # The live-view path (render -> commanded_follower_q -> robot_view.update_poses)
    # must run end-to-end without throwing; drive it against a buffered recording (no
    # gRPC server) and the synthetic FACTR source.
    import rerun as rr

    from dual_flexiv_control.dashboard import arms as _arms
    from dual_flexiv_control.dashboard import calibration

    _arms.set_active_rig("left_only")
    monkeypatch.setattr(_arms, "runtime_is_sim", lambda: True)
    calibration.reset()
    rec = rr.RecordingStream("dfc-test-calib", recording_id="t")
    from dual_flexiv_control.dashboard import robot_view
    robot_view.log_scene(rec)
    monkeypatch.setattr(calibration, "_REC", rec)
    try:
        q = calibration.render("left", calibration.initial_offsets("left"), [1, 2, 3])
        assert q.shape == (7,)
    finally:
        monkeypatch.setattr(calibration, "_REC", None)
        calibration.reset()
        _arms.set_active_rig(None)


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
        assert status.force_gain == 0.0
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
        lambda side: {"grav_comp_enabled": side == "left", "force_gain": 1.0},
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
        assert status.force_gain == 1.0
    finally:
        writer.close()
        writer.unlink()
        _arms.reset()


def test_grav_comp_display_uses_live_gain_not_process_health():
    from dual_flexiv_control.dashboard.arms import grav_comp_display

    assert "enabled" in grav_comp_display({
        "grav_comp_enabled": True, "force_gain": 1.0, "force_gain_target": 1.0,
    })[1]
    assert "enabling" in grav_comp_display({
        "grav_comp_enabled": False, "force_gain": 0.4, "force_gain_target": 1.0,
    })[1]
    assert "disabling" in grav_comp_display({
        "grav_comp_enabled": False, "force_gain": 0.4, "force_gain_target": 0.0,
    })[1]
    assert "disabled (limp)" in grav_comp_display({
        "grav_comp_enabled": False, "force_gain": 0.0, "force_gain_target": 0.0,
    })[1]
    assert "unknown" in grav_comp_display(None)[1]


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
    assert "left" in robot_view._shown_traces
    robot_view.update_horizon_targets(rec, {"left": q + 0.2}, {}, t=2.0)  # no measured q: no trace
    robot_view.clear_horizon_targets(rec)
    assert not robot_view._shown_targets
    assert not robot_view._shown_traces


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
