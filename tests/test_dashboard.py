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
