"""Dashboard run registry: async collection stop, outcome alerts, error surfacing.

The registry drives real subprocesses; here the collection system is stood in by
small ``bash`` scripts (crash with output / save-on-SIGINT) so the monitor, alert,
and stop paths are exercised end-to-end without hardware. The Rerun calls inside
the registry are no-ops without an active recording, so no viewer is needed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None
    or importlib.util.find_spec("streamlit") is None,
    reason="dashboard extras not installed",
)


@pytest.fixture(autouse=True)
def _rerun_recording():
    """A sink-less Rerun recording: the registry's viewer calls become no-ops.

    ``rr.send_blueprint`` raises without an active recording (production always
    has one, created by the dashboard's viewer startup).
    """
    import rerun as rr

    rr.init("dfc-test", spawn=False)
    yield


def _task():
    from dual_flexiv_control.dashboard.tasks import TaskInfo

    return TaskInfo(
        name="fake",
        language_instruction="do the thing",
        num_episodes=1,
        num_timesteps=None,
        path=Path("conf/task/fake.yaml"),
    )


def _registry_with_fake_proc(monkeypatch, tmp_path, script: str):
    """A RunRegistry whose collection subprocess is ``bash -c script``."""
    from dual_flexiv_control.dashboard import runner

    log_path = str(tmp_path / "collection.log")

    def fake_launch(task, run_id):
        logf = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                ["bash", "-c", script],
                start_new_session=True,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        finally:
            logf.close()
        return proc, log_path

    monkeypatch.setattr(runner, "_launch_collection", fake_launch)
    # The live view reads FACTR/shm/robot scene — irrelevant here.
    monkeypatch.setattr(runner, "_emit_collection_view", lambda stop, task: None)
    return runner.RunRegistry()


def _wait_alert(registry, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        alert = registry.take_alert()
        if alert is not None:
            return alert
        time.sleep(0.05)
    raise AssertionError("no alert within timeout")


def test_crash_surfaces_error_alert_with_log_tail(monkeypatch, tmp_path):
    """A collection subprocess that dies on its own raises an error alert + log tail."""
    registry = _registry_with_fake_proc(
        monkeypatch, tmp_path,
        "echo 'RuntimeError: camera static has auto_serial=true but no ZED'; exit 3",
    )
    registry.launch(_task(), "collection")
    alert = _wait_alert(registry)
    assert alert["kind"] == "error"
    assert "exit code 3" in alert["detail"]
    assert "no ZED" in (alert["tail"] or "")
    assert registry.active() is None            # monitor cleared the active slot
    (run,) = registry.history()
    assert run.outcome == "crashed"
    assert registry.take_alert() is None        # one-shot: taken exactly once


def test_stop_is_async_and_reports_saved(monkeypatch, tmp_path):
    """Stop returns immediately ("saving"), SIGINT saves, then an info alert lands."""
    registry = _registry_with_fake_proc(
        monkeypatch, tmp_path,
        # Stand-in for the orchestrator: on SIGINT, "save" briefly then exit 0.
        'trap "sleep 0.4; exit 0" INT; sleep 30 & wait $!',
    )
    registry.launch(_task(), "collection")
    time.sleep(0.3)  # let bash install its trap
    t0 = time.monotonic()
    registry.stop_active()
    assert time.monotonic() - t0 < 1.0, "stop must not block on the save"
    active = registry.active()
    assert active is not None and active.status == "stopping"
    cs = registry.collection_status()
    assert cs is not None and cs.state == "saving"

    alert = _wait_alert(registry)
    assert alert["kind"] == "info"
    assert "saved" in alert["detail"]
    assert registry.active() is None
    (run,) = registry.history()
    assert run.outcome == "saved"


def test_launch_refused_while_collection_active(monkeypatch, tmp_path):
    """Overlapping collection launches are refused (cameras/dataset conflict)."""
    registry = _registry_with_fake_proc(
        monkeypatch, tmp_path, 'trap "exit 0" INT; sleep 30 & wait $!'
    )
    registry.launch(_task(), "collection")
    with pytest.raises(RuntimeError, match="still active"):
        registry.launch(_task(), "collection")
    registry.stop_active()
    _wait_alert(registry)  # let it unwind before the tmpdir vanishes


def test_running_status_surfaces_collection_heartbeat(monkeypatch, tmp_path):
    """While recording, the status panel shows the loop's own heartbeat line."""
    registry = _registry_with_fake_proc(
        monkeypatch, tmp_path,
        "echo '2026-01-01 INFO collection dual_flexiv_control.collection.loop: "
        "collection: recording (42 frames this episode, 42 total)'; "
        'trap "exit 0" INT; sleep 30 & wait $!',
    )
    registry.launch(_task(), "collection")
    time.sleep(0.4)  # give bash time to write the heartbeat
    cs = registry.collection_status()
    assert cs is not None and cs.state == "running"
    assert "42 frames" in cs.detail
    registry.stop_active()
    _wait_alert(registry)


def test_log_tail_and_heartbeat_helpers(tmp_path):
    from dual_flexiv_control.dashboard.runner import _last_heartbeat
    from dual_flexiv_control.dashboard.runner import _log_tail

    p = tmp_path / "log.txt"
    p.write_text(
        "\n".join(
            [f"line {i}" for i in range(40)]
            + ["... loop: collection: NO frames recorded in the last ~2s — waiting on ['cam/x']"]
        )
    )
    tail = _log_tail(str(p), n=5)
    assert tail is not None and "line 39" in tail
    beat = _last_heartbeat(str(p))
    assert beat is not None and beat.startswith("NO frames recorded")
    assert _log_tail(str(tmp_path / "missing.txt")) is None
    assert _last_heartbeat(None) is None
