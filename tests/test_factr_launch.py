"""The FACTR-Server process supervisor + its session-daemon integration.

The supervisor tests drive :class:`FactrServerSupervisor` against a fake
``subprocess.Popen`` (no bash, no ROS): countdown → spawn, teleop-vs-API death
policy, SIGINT-first stop with SIGTERM/SIGKILL escalation. The daemon tests
exercise the ``start_factr``/``stop_factr`` command gates and the state-file
sync in-process, same style as ``test_session.py``.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time

import pytest

from dual_flexiv_control.configs import FactrLaunchCfg
from dual_flexiv_control.configs import FactrLeaderCfg
from dual_flexiv_control.configs import FactrTransformCfg
from dual_flexiv_control.configs import JointConventionCfg
from dual_flexiv_control.interfaces.factr import launch as fl


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProc:
    """A Popen stand-in: records signals; dies on SIGINT unless ``stubborn``."""

    def __init__(self, cmd, stubborn=0, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.signals: list[int] = []
        self.stubborn = stubborn  # how many INT/TERM signals it shrugs off
        self._rc = None

    def poll(self):
        return self._rc

    def send_signal(self, sig):
        self.signals.append(sig)
        if self.stubborn > 0:
            self.stubborn -= 1
        else:
            self._rc = -sig

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.signals.append(signal.SIGKILL)
        self._rc = -signal.SIGKILL

    def die(self, rc=1):
        self._rc = rc


class _SpawnLog(list):
    """The FakeProcs created so far; ``make_stubborn(n)`` shapes future spawns."""

    def __init__(self):
        super().__init__()
        self.stubborn = 0

    def make_stubborn(self, n: int) -> None:
        self.stubborn = n


@pytest.fixture()
def spawned(monkeypatch):
    """Patch Popen in the launch module; returns the log of FakeProcs created."""
    procs = _SpawnLog()

    def _popen(cmd, **kwargs):
        proc = FakeProc(cmd, stubborn=procs.stubborn, **kwargs)
        procs.append(proc)
        return proc

    monkeypatch.setattr(fl.subprocess, "Popen", _popen)
    # No fuser in the sandbox: port freeing must silently skip.
    monkeypatch.setattr(fl.shutil, "which", lambda _: None)
    return procs


def _cfg(tmp_path, **kw) -> FactrLaunchCfg:
    defaults = dict(
        enabled=True,
        workdir=str(tmp_path / "FACTR_Teleop"),
        calib_delay_s=0.0,
        stop_grace_s=0.1,
        teleop_modules={
            "left": "src.factr_teleop.factr_teleop.factr_rizon_teleop",
            "right": "src.factr_teleop.factr_teleop.factr_rizon_dual_board",
        },
    )
    defaults.update(kw)
    return FactrLaunchCfg(**defaults)


def _running_supervisor(tmp_path, spawned, **kw):
    sup = fl.FactrServerSupervisor(_cfg(tmp_path, **kw), sides=["left", "right"])
    sup.request_start()
    sup.tend()  # calib_delay_s=0 -> countdown fires immediately
    assert sup.state == fl.RUNNING
    return sup


# ---------------------------------------------------------------------------
# Supervisor: launch
# ---------------------------------------------------------------------------


def test_countdown_gates_the_spawn(tmp_path, spawned):
    sup = fl.FactrServerSupervisor(_cfg(tmp_path, calib_delay_s=0.15), ["left", "right"])
    ok, detail = sup.request_start()
    assert ok and "delayed launch" in detail.lower()
    assert sup.state == fl.COUNTDOWN and not spawned
    ends = sup.status()["countdown_ends_ts"]
    assert ends == pytest.approx(time.time() + 0.15, abs=0.1)

    sup.tend()  # too early: still counting down, nothing spawned
    assert sup.state == fl.COUNTDOWN and not spawned

    time.sleep(0.2)
    sup.tend()
    assert sup.state == fl.RUNNING and len(spawned) == 3  # 2 teleops + api

    # A second start is refused while running.
    ok, detail = sup.request_start()
    assert not ok and "stop them first" in detail


def test_spawn_commands_and_logs(tmp_path, spawned):
    sup = _running_supervisor(tmp_path, spawned)
    by_name = {u.name: u for u in sup.units}
    left = by_name["teleop:left"].proc
    api = by_name["api"].proc

    # bash -c 'source … && source … && exec <python> -m <module>', cwd=workdir.
    assert left.cmd[:2] == ["bash", "-c"]
    script = left.cmd[2]
    assert "source /opt/ros/humble/setup.bash" in script
    assert "source install/setup.bash" in script
    assert script.endswith("exec /usr/bin/python3 -m src.factr_teleop.factr_teleop.factr_rizon_teleop")
    assert left.kwargs["cwd"] == str(tmp_path / "FACTR_Teleop")
    assert left.kwargs["start_new_session"] is True

    # Teleop stdout (the 500 Hz screen-clear) is discarded; the API's is kept.
    assert left.kwargs["stdout"] is subprocess.DEVNULL
    assert api.kwargs["stdout"] is not subprocess.DEVNULL

    # Health logs land where the FACTR-Server tail tasks expect them.
    logs = tmp_path / "FACTR_Teleop" / "logs"
    assert (logs / "factr_health_left.log").exists()
    assert (logs / "factr_health_right.log").exists()
    assert (logs / "factr_api.log").exists()

    # The status payload is what session.json carries: plain JSON.
    status = sup.status()
    json.dumps(status)
    assert status["state"] == "running" and status["down"] == []
    assert set(status["units"]) == {"teleop:left", "teleop:right", "api"}


def test_managed_teleop_receives_dfc_leader_contract(tmp_path, spawned):
    leader = FactrLeaderCfg(
        raw_to_dfc=JointConventionCfg(
            offsets_deg=[1.0] * 7,
            sign_flip_joints=[2],
            gripper_open=0.1,
            gripper_closed=0.9,
        ),
        home_q_rad=[0.0] * 7,
        dfc_to_factr=FactrTransformCfg(
            signs=[1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            offset_rad=[0.0] * 7,
        ),
    )
    sup = fl.FactrServerSupervisor(
        _cfg(tmp_path), ["left"], {"left": leader}
    )
    sup.start_now()
    teleop = next(u for u in sup.units if u.name == "teleop:left").proc
    payload = json.loads(teleop.kwargs["env"]["DFC_LEADER_CONFIG"])
    assert payload["side"] == "left"
    assert payload["raw_to_dfc"]["offsets_deg"] == [1.0] * 7
    assert payload["dfc_to_factr"]["signs"][2] == -1.0
    api = next(u for u in sup.units if u.name == "api").proc
    assert "DFC_LEADER_CONFIG" not in api.kwargs["env"]


def test_cancel_during_countdown_spawns_nothing(tmp_path, spawned):
    sup = fl.FactrServerSupervisor(_cfg(tmp_path, calib_delay_s=60.0), ["left"])
    sup.request_start()
    ok, detail = sup.request_stop()
    assert ok and "cancelled" in detail
    assert sup.state == fl.OFF and not spawned


# ---------------------------------------------------------------------------
# Supervisor: supervision policy
# ---------------------------------------------------------------------------


def test_dead_teleop_is_reported_never_respawned(tmp_path, spawned):
    sup = _running_supervisor(tmp_path, spawned)
    left = next(u for u in sup.units if u.name == "teleop:left")
    left.proc.die(rc=1)
    n = len(spawned)
    sup.tend()
    sup.tend()
    assert sup.status()["down"] == ["teleop:left"]
    assert len(spawned) == n  # re-energizing a leader is never automatic
    assert sup.status()["units"]["teleop:left"] == "down"
    assert sup.status()["units"]["api"] == "running"


def test_dead_api_is_respawned_paced(tmp_path, spawned, monkeypatch):
    sup = _running_supervisor(tmp_path, spawned)
    api = next(u for u in sup.units if u.name == "api")
    api.proc.die(rc=1)

    monkeypatch.setattr(fl, "_API_RESPAWN_S", 3600.0)
    n = len(spawned)
    sup.tend()
    assert sup.status()["down"] == ["api"] and len(spawned) == n  # paced: not yet

    monkeypatch.setattr(fl, "_API_RESPAWN_S", 0.0)
    sup.tend()
    assert len(spawned) == n + 1  # relay respawned: it never touches the servos
    assert sup.status()["down"] == []


# ---------------------------------------------------------------------------
# Supervisor: stop semantics (SIGINT de-energizes; escalation is last resort)
# ---------------------------------------------------------------------------


def test_stop_sigints_then_reaches_off(tmp_path, spawned):
    sup = _running_supervisor(tmp_path, spawned)
    ok, detail = sup.request_stop()
    assert ok and "de-energizing" in detail
    assert sup.state == fl.STOPPING
    assert all(p.signals == [signal.SIGINT] for p in spawned)
    sup.tend()  # all died on SIGINT
    assert sup.state == fl.OFF
    assert all(u.proc is None for u in sup.units)


def test_stop_escalates_term_then_kill(tmp_path, spawned, monkeypatch):
    spawned.make_stubborn(2)  # every proc shrugs off SIGINT and SIGTERM
    sup = _running_supervisor(tmp_path, spawned, stop_grace_s=0.05)
    monkeypatch.setattr(fl, "_TERM_GRACE_S", 0.05)
    sup.request_stop()
    sup.tend()
    assert sup.state == fl.STOPPING  # grace not over, still SIGINT-only

    time.sleep(0.08)
    sup.tend()  # grace expired -> SIGTERM
    assert sup.state == fl.STOPPING
    assert all(p.signals == [signal.SIGINT, signal.SIGTERM] for p in spawned)

    time.sleep(0.08)
    sup.tend()  # TERM grace expired -> SIGKILL, and the group reads off
    assert sup.state == fl.OFF
    assert all(p.signals[-1] == signal.SIGKILL for p in spawned)


def test_blocking_shutdown_runs_to_off(tmp_path, spawned):
    sup = _running_supervisor(tmp_path, spawned, stop_grace_s=0.2)
    sup.shutdown()
    assert sup.state == fl.OFF
    assert all(p.signals[0] == signal.SIGINT for p in spawned)


# ---------------------------------------------------------------------------
# Daemon integration: command gates + state sync
# ---------------------------------------------------------------------------


def _daemon(tmp_path, monkeypatch=None, managed=False, spawned=None):
    from test_session import _compose_config  # tests/ is not a package

    from dual_flexiv_control import session as sess

    config, overrides = _compose_config(tmp_path)
    daemon = sess.SessionDaemon(config, overrides)
    if managed:
        daemon.factr_servers = fl.FactrServerSupervisor(_cfg(tmp_path), ["left"])
        daemon.state.factr_servers = daemon.factr_servers.status()
    return daemon


def test_daemon_refuses_unmanaged_factr_commands(tmp_path):
    daemon = _daemon(tmp_path)  # sim compose -> supervisor is None
    assert daemon.factr_servers is None
    daemon._handle_start_factr()
    assert "not managed" in daemon.state.message
    daemon._handle_stop_factr()
    assert "not managed" in daemon.state.message


def test_daemon_gates_launch_on_idle_and_stop_on_collection(tmp_path, spawned):
    import threading

    from dual_flexiv_control import session as sess

    daemon = _daemon(tmp_path, managed=True)

    # Launch refused while a run is active (the leaders must be posable).
    daemon.run = sess._ActiveRun(
        phase="collection", task="t", proc=None,
        stop_event=threading.Event(), command_sides=["left"],
    )
    daemon._handle_start_factr()
    assert "stop the run first" in daemon.state.message
    assert daemon.factr_servers.state == fl.OFF

    # Stop refused mid-collection (teleop is fed by these leaders)…
    daemon.factr_servers.request_start()
    daemon.factr_servers.tend()
    daemon._handle_stop_factr()
    assert "stop the run first" in daemon.state.message
    assert daemon.factr_servers.state == fl.RUNNING

    # …but allowed once the run is gone, and mirrored into the state file dict.
    daemon.run = None
    daemon._handle_stop_factr()
    assert daemon.factr_servers.state == fl.STOPPING
    assert daemon.state.factr_servers["state"] == "stopping"

    # Idle: launch arms the countdown and syncs state.
    daemon.factr_servers.tend()  # fakes die on SIGINT -> off
    daemon._handle_start_factr()
    assert daemon.factr_servers.state == fl.COUNTDOWN
    assert daemon.state.factr_servers["state"] == "countdown"


def test_daemon_tend_syncs_state_and_flags_teleop_death_mid_collection(tmp_path, spawned):
    import threading

    from dual_flexiv_control import session as sess

    daemon = _daemon(tmp_path, managed=True)
    daemon._handle_start_factr()
    assert daemon._tend_factr_servers() is True  # countdown fired -> running
    assert daemon.state.factr_servers["state"] == "running"
    assert daemon._tend_factr_servers() is False  # steady state: no rewrite

    daemon.run = sess._ActiveRun(
        phase="collection", task="t", proc=None,
        stop_event=threading.Event(), command_sides=["left"],
    )
    next(u for u in daemon.factr_servers.units if u.kind == "teleop").proc.die(rc=1)
    assert daemon._tend_factr_servers() is True
    assert daemon.state.factr_servers["down"] == ["teleop:left"]
    assert "died mid-collection" in daemon.state.message
