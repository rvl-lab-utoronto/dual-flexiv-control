"""Dashboard run registry: session-backed launches, outcome alerts, status mapping.

The registry is now a thin client over the session daemon (see
:mod:`dual_flexiv_control.session`); here the daemon is stood in by a scripted
:class:`FakeManager` so the launch gating, one-shot alert, status panel, and
history paths are exercised without processes. The Rerun calls inside the
registry are no-ops without an active recording, so no viewer is needed. The
real daemon protocol is covered end-to-end in ``test_session.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None
    or importlib.util.find_spec("streamlit") is None,
    reason="dashboard extras not installed",
)


@pytest.fixture(autouse=True)
def _rerun_recording():
    """A sink-less Rerun recording: the registry's viewer calls become no-ops.

    ``rr.log`` raises without an active recording (production always has one,
    created by the dashboard's viewer startup).
    """
    import rerun as rr

    rr.init("dfc-test", spawn=False)
    yield


def _task():
    from pathlib import Path

    from dual_flexiv_control.dashboard.tasks import TaskInfo

    return TaskInfo(
        name="fake",
        language_instruction="do the thing",
        num_episodes=1,
        num_timesteps=None,
        path=Path("conf/task/fake.yaml"),
    )


class FakeManager:
    """Scripted stand-in for :class:`~dual_flexiv_control.dashboard.session.SessionManager`."""

    def __init__(self, **view_kwargs):
        from dual_flexiv_control.dashboard.session import SessionView

        self._view = SessionView(state="viewing", rig="bimanual", sim=True,
                                 run_id="abc123", **view_kwargs)
        self.commands: list = []
        self.tail = "boom traceback"

    def set_view(self, **changes):
        from dataclasses import replace

        self._view = replace(self._view, **changes)

    def view(self):
        return self._view

    def ensure(self, rig, sim):
        return False

    def start_run(self, phase, task, policy=None, host=None, port=None, skill=None):
        self.commands.append(("start", phase, task, policy, host, port, skill))
        return True

    def stop_run(self):
        self.commands.append(("stop",))
        return True

    def log_tail(self, n=25):
        return self.tail


def _registry(**view_kwargs):
    from dual_flexiv_control.dashboard import runner

    mgr = FakeManager(**view_kwargs)
    return runner.RunRegistry(manager=mgr), mgr


def test_launch_sends_start_command_from_viewing():
    registry, mgr = _registry()
    registry.launch(_task(), "collection", rig="bimanual")
    assert mgr.commands == [("start", "collection", "fake", None, None, None, None)]


def test_launch_forwards_eval_policy_overrides():
    registry, mgr = _registry()
    registry.launch(_task(), "eval", rig="bimanual",
                    policy="acme", host="100.92.86.90", port=53866)
    assert mgr.commands == [
        ("start", "eval", "fake", "acme", "100.92.86.90", 53866, None)
    ]


def test_launch_skill_sends_start_command():
    registry, mgr = _registry()
    registry.launch_skill("pick-ep0", rig="bimanual")
    assert mgr.commands == [("start", "skill", "default", None, None, None, "pick-ep0")]


def test_launch_skill_shares_the_launch_gates():
    registry, mgr = _registry()
    mgr.set_view(state="skill", task="pick-ep0", phase="skill")
    with pytest.raises(RuntimeError, match="still active"):
        registry.launch_skill("pick-ep0", rig="bimanual")
    mgr.set_view(state="down")
    with pytest.raises(RuntimeError, match="not running"):
        registry.launch_skill("pick-ep0", rig="bimanual")
    mgr.set_view(state="viewing")
    with pytest.raises(RuntimeError, match="rig"):
        registry.launch_skill("pick-ep0", rig="bench")
    assert mgr.commands == []


def test_launch_refused_while_run_active():
    registry, mgr = _registry()
    mgr.set_view(state="collection", task="fake", phase="collection")
    with pytest.raises(RuntimeError, match="still active"):
        registry.launch(_task(), "collection")
    mgr.set_view(state="saving")  # saving counts as active too
    with pytest.raises(RuntimeError, match="still active"):
        registry.launch(_task(), "eval")
    assert mgr.commands == []


def test_launch_refused_when_session_down_or_starting():
    registry, mgr = _registry()
    mgr.set_view(state="down")
    with pytest.raises(RuntimeError, match="not running"):
        registry.launch(_task(), "collection")
    mgr.set_view(state="starting")
    with pytest.raises(RuntimeError, match="starting"):
        registry.launch(_task(), "collection")
    assert mgr.commands == []


def test_launch_refused_on_rig_mismatch():
    registry, mgr = _registry()
    with pytest.raises(RuntimeError, match="rig"):
        registry.launch(_task(), "collection", rig="bench")
    assert mgr.commands == []


def test_launch_rejects_unknown_phase():
    registry, _ = _registry()
    with pytest.raises(ValueError, match="unknown phase"):
        registry.launch(_task(), "bogus")


def test_active_and_status_follow_the_session_state():
    registry, mgr = _registry()
    assert registry.active() is None
    assert registry.collection_status() is None

    mgr.set_view(state="collection", task="fake", phase="collection",
                 run_seq=1, run_started_ts=1e9)
    active = registry.active()
    assert active is not None
    assert (active.phase, active.task, active.status) == ("collection", "fake", "running")
    cs = registry.collection_status()
    assert cs is not None and cs.state == "running"

    mgr.set_view(state="saving")
    assert registry.active().status == "stopping"
    assert registry.collection_status().state == "saving"

    mgr.set_view(state="eval", phase="eval")
    cs = registry.collection_status()
    assert cs is not None and cs.state == "running" and "rollout" in cs.detail

    mgr.set_view(state="skill", phase="skill")
    cs = registry.collection_status()
    assert cs is not None and cs.state == "running" and "skill" in cs.detail
    assert registry.active() is not None  # a skill run counts as active


def test_stop_active_sends_stop():
    registry, mgr = _registry()
    mgr.set_view(state="collection", task="fake", phase="collection")
    registry.stop_active()
    assert ("stop",) in mgr.commands


def test_outcome_surfaces_one_alert_and_lands_in_history():
    registry, mgr = _registry()
    mgr.set_view(last_outcome={
        "run_seq": 1, "phase": "collection", "task": "fake",
        "outcome": "saved", "exitcode": 0,
        "detail": "collection stopped — episode saved to the dataset.",
    })
    alert = registry.take_alert()
    assert alert is not None and alert["kind"] == "info"
    assert "saved" in alert["detail"]
    assert alert["tail"] is None
    assert registry.take_alert() is None        # one-shot: taken exactly once
    (run,) = registry.history()
    assert run.status == "saved" and run.task == "fake"


def test_crash_outcome_alerts_with_log_tail():
    registry, mgr = _registry()
    mgr.set_view(last_outcome={
        "run_seq": 1, "phase": "collection", "task": "fake",
        "outcome": "crashed", "exitcode": 3,
        "detail": "run stopped on its own (exit code 3) — a source likely failed.",
    })
    alert = registry.take_alert()
    assert alert is not None and alert["kind"] == "error"
    assert "exit code 3" in alert["detail"]
    assert alert["tail"] == mgr.tail
    (run,) = registry.history()
    assert run.status == "crashed"


def test_new_outcome_seq_raises_a_new_alert():
    registry, mgr = _registry()
    mgr.set_view(last_outcome={"run_seq": 1, "phase": "collection", "task": "fake",
                               "outcome": "saved", "exitcode": 0, "detail": "d1"})
    assert registry.take_alert() is not None
    mgr.set_view(last_outcome={"run_seq": 2, "phase": "eval", "task": "fake",
                               "outcome": "finished", "exitcode": 0, "detail": "d2"})
    alert = registry.take_alert()
    assert alert is not None and alert["detail"] == "d2"
    assert registry.take_alert() is None
    assert len(registry.history()) == 2


def test_reset_stops_run_and_clears_history():
    registry, mgr = _registry()
    mgr.set_view(last_outcome={"run_seq": 1, "phase": "collection", "task": "fake",
                               "outcome": "saved", "exitcode": 0, "detail": "d"})
    registry.take_alert()
    assert registry.history()
    mgr.set_view(state="collection", task="fake", phase="collection")
    registry.reset()
    assert ("stop",) in mgr.commands
    assert registry.history() == []


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


def test_running_status_surfaces_collection_heartbeat(tmp_path):
    """While recording, the status panel shows the loop's own heartbeat line
    (taken from the session daemon's log, where the consumer's output lands)."""
    registry, mgr = _registry()
    log = tmp_path / "daemon.log"
    log.write_text(
        "2026-01-01 INFO collection dual_flexiv_control.collection.loop: "
        "collection: recording (42 frames this episode, 42 total)\n"
    )
    mgr.set_view(state="collection", task="fake", phase="collection",
                 log_path=str(log))
    cs = registry.collection_status()
    assert cs is not None and cs.state == "running"
    assert "42 frames" in cs.detail


def test_log_policy_comm_counts_events_and_resets_between_runs(monkeypatch):
    """The mirror's comm logger: counters step per event at true event times,
    only not-yet-surfaced ring samples are logged, and the stream disappearing
    (run over) resets the counters for the next eval."""
    import numpy as np

    from dual_flexiv_control.dashboard import arms
    from dual_flexiv_control.dashboard import runner
    from dual_flexiv_control.streams.ring import Samples

    logged = []
    monkeypatch.setattr(runner.rr, "log", lambda path, val, **kw: logged.append((path, val)))
    monkeypatch.setattr(runner.rr, "set_time", lambda name, duration=None: None)
    monkeypatch.setattr(runner.rr, "Scalars", lambda values: list(values))

    feed = {"value": None}
    monkeypatch.setattr(
        arms, "read_live_policy_comm", lambda runtime_dir=None: feed["value"]
    )

    state = {"next_seq": 0, "sent": 0, "received": 0, "errors": 0}

    # sent -> received (200 ms round trip)
    feed["value"] = Samples(
        data=np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.2]]),
        t_ns=np.array([100, 300], np.int64),
        seq=np.array([0, 1], np.int64),
    )
    runner._log_policy_comm(state, t0_ns=0, now_t=5.0)
    assert state == {"next_seq": 2, "sent": 1, "received": 1, "errors": 0}
    assert ("policy/comm/sent", [1]) in logged
    assert ("policy/comm/received", [1]) in logged
    assert ("policy/comm/in_flight", [1.0]) in logged
    assert ("policy/comm/in_flight", [0.0]) in logged
    assert (runner.blueprints.POLICY_LATENCY_PATH, [pytest.approx(200.0)]) in logged

    # Same ring again: everything already surfaced -> nothing new logged.
    logged.clear()
    runner._log_policy_comm(state, t0_ns=0, now_t=6.0)
    assert logged == []

    # Run over: stream gone -> counters reset for the next eval.
    feed["value"] = None
    runner._log_policy_comm(state, t0_ns=0, now_t=7.0)
    assert state == {"next_seq": 0, "sent": 0, "received": 0, "errors": 0}

    # Next eval: a fresh ring starting at seq 0, whose request errors out.
    feed["value"] = Samples(
        data=np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 1.5]]),
        t_ns=np.array([900, 2400], np.int64),
        seq=np.array([0, 1], np.int64),
    )
    runner._log_policy_comm(state, t0_ns=0, now_t=8.0)
    assert state == {"next_seq": 2, "sent": 1, "received": 0, "errors": 1}
    assert ("policy/comm/errors", [1]) in logged
