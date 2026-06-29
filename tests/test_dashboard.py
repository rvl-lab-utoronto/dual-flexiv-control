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
def test_qvel_test_blueprint_builds():
    # The eval no-motion probe uses this focused dq-only layout.
    import rerun.blueprint as rrb

    from dual_flexiv_control.dashboard import blueprints

    assert isinstance(blueprints.qvel_test_blueprint("handover"), rrb.Blueprint)


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
    assert by_side["left"].serial == "Rizon-4s-062841"
    assert by_side["right"].serial == "Rizon-4s-062837"
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
