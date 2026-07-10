"""The session daemon: protocol units + the sim end-to-end state machine.

The end-to-end test spawns the REAL daemon (``python -m dual_flexiv_control.session``,
all sim) and drives it exactly like the dashboard does — JSON lines on stdin,
state from ``session.json`` — through VIEWING → COLLECTION → stop/save → VIEWING →
EVAL (serverless ``policy.kind=hold``) → VIEWING → shutdown, checking the arms'
``control_active`` flag, the recorded dataset, and that no shared memory leaks.
"""

from __future__ import annotations

import json
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dual_flexiv_control.session import classify_outcome
from dual_flexiv_control.session import parse_command
from dual_flexiv_control.session import read_state
from dual_flexiv_control.session import run_overrides
from dual_flexiv_control.session import state_path


# ---------------------------------------------------------------------------
# Protocol units (fast)
# ---------------------------------------------------------------------------


def test_parse_command_accepts_json_dicts_only():
    assert parse_command('{"cmd": "stop"}') == {"cmd": "stop"}
    assert parse_command('  {"cmd": "start", "phase": "eval", "task": "t"} \n') == {
        "cmd": "start", "phase": "eval", "task": "t"
    }
    assert parse_command("") is None
    assert parse_command("not json") is None
    assert parse_command('["cmd"]') is None
    assert parse_command('{"nocmd": 1}') is None


def test_classify_outcome_matrix():
    assert classify_outcome("collection", 0, stopping=True)[0] == "saved"
    assert classify_outcome("collection", 0, stopping=False)[0] == "finished"
    assert classify_outcome("eval", 0, stopping=True)[0] == "stopped"
    assert classify_outcome("eval", 0, stopping=False)[0] == "finished"
    assert classify_outcome("skill", 0, stopping=True)[0] == "stopped"
    assert classify_outcome("skill", 0, stopping=False)[0] == "finished"
    assert classify_outcome("collection", 3, stopping=False)[0] == "crashed"
    outcome, detail = classify_outcome("skill", 1, stopping=False)
    assert outcome == "crashed" and "start pose" in detail
    outcome, detail = classify_outcome("collection", 2, stopping=True)
    assert outcome == "stopped-error" and "code 2" in detail


def test_run_overrides_repin_task_and_phase():
    got = run_overrides(
        ["rig=bench", "runtime.sim=true", "task=old", "runtime.phase=eval"],
        task="handover",
        phase="collection",
    )
    assert got == ["rig=bench", "runtime.sim=true", "task=handover", "runtime.phase=collection"]
    # Value overrides under task.* are session-wide and survive the re-pin.
    got = run_overrides(["task.eval.num_timesteps=10"], task="t", phase="eval")
    assert got == ["task.eval.num_timesteps=10", "task=t", "runtime.phase=eval"]


def test_run_overrides_extra_repins_matching_session_override():
    got = run_overrides(
        ["rig=bench", "policy.port=1111"],
        task="t", phase="eval", extra=["policy.port=8000"],
    )
    assert got == ["rig=bench", "task=t", "runtime.phase=eval", "policy.port=8000"]


def test_state_file_roundtrip(tmp_path):
    from dual_flexiv_control.session import SessionState
    from dual_flexiv_control.session import StateFile

    sf = StateFile(state_path(tmp_path))
    state = SessionState(pid=123, run_id="r1", rig="bench", sim=True)
    sf.write(state)
    raw = read_state(tmp_path)
    assert raw is not None
    assert (raw["pid"], raw["run_id"], raw["rig"], raw["sim"]) == (123, "r1", "bench", True)
    assert raw["state"] == "viewing"
    assert raw["heartbeat_ts"] > 0
    sf.remove()
    assert read_state(tmp_path) is None
    sf.remove()  # idempotent


# ---------------------------------------------------------------------------
# End-to-end (sim): the daemon's whole state machine
# ---------------------------------------------------------------------------


def _spawn_daemon(tmp_path) -> tuple[subprocess.Popen, Path, Path]:
    """The real daemon in sim on a bimanual rig, left arm control-enabled."""
    runtime_dir = tmp_path / "runtime"
    datasets = tmp_path / "datasets"
    overrides = [
        "rig=bimanual",
        "runtime.sim=true",
        f"runtime.runtime_dir={runtime_dir}",
        f"recording.root={datasets}",
        f"skill.root={tmp_path / 'skills'}",
        "skill.start_timeout_s=15",
        "runtime.save_grace_s=60",
        "arms.left.control_enabled=true",
        "arms.left.control_rate_hz=100",
        "arms.left.rate_hz=200",
        "arms.right.rate_hz=200",
        "arms.left.control_attach_timeout_s=5",
        "arms.right.control_attach_timeout_s=5",
        # Light sim cameras so spawn + video encode stay fast.
        "cameras.wrist_left.width=64", "cameras.wrist_left.height=48",
        "cameras.wrist_right.width=64", "cameras.wrist_right.height=48",
        "cameras.static.width=64", "cameras.static.height=48",
        # Serverless eval: repeat the measured joint positions, tiny horizon.
        "policy.kind=hold",
        "task.eval.num_timesteps=10",
    ]
    log_path = tmp_path / "daemon.log"
    logf = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "dual_flexiv_control.session", *overrides],
            stdin=subprocess.PIPE,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        logf.close()
    return proc, runtime_dir, log_path


def _send(proc: subprocess.Popen, cmd: dict) -> None:
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()


def _wait_state(runtime_dir, want: str, timeout_s: float, log_path=None) -> dict:
    deadline = time.monotonic() + timeout_s
    raw = None
    while time.monotonic() < deadline:
        raw = read_state(runtime_dir)
        if raw is not None and raw.get("state") == want:
            return raw
        time.sleep(0.1)
    tail = ""
    if log_path is not None and os.path.isfile(log_path):
        tail = "".join(open(log_path, errors="replace").readlines()[-25:])
    raise AssertionError(
        f"daemon never reached state {want!r} (last: {raw!r})\n--- daemon log ---\n{tail}"
    )


def _left_control_active(runtime_dir) -> bool | None:
    from dual_flexiv_control.dashboard.arms import ArmInfo
    from dual_flexiv_control.dashboard.arms import read_arm_status

    info = ArmInfo(side="left", name="Lauer", serial="", dof=7)
    status = read_arm_status(info, runtime_dir=str(runtime_dir))
    return status.control_active if status.source == "live" else None


@pytest.mark.slow
def test_session_daemon_full_lifecycle_sim(tmp_path):
    """VIEWING → collection (arms controlling) → stop/save → VIEWING →
    eval (hold policy, self-finishing) → VIEWING → bad task refused → EOF shutdown."""
    proc, runtime_dir, log_path = _spawn_daemon(tmp_path)
    try:
        # -- boots into VIEWING with the arms live (read-only) -----------------
        raw = _wait_state(runtime_dir, "viewing", 60.0, log_path)
        run_id = raw["run_id"]  # scope the shm-leak check to THIS daemon's segments
        # (other sessions on this host may create/destroy their own dfc_* concurrently)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and _left_control_active(runtime_dir) is None:
            time.sleep(0.2)
        assert _left_control_active(runtime_dir) is False, (
            "left arm must be live and IDLE in viewing"
        )
        # Launches are gated until every camera streams; wait out the boot window.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and (read_state(runtime_dir) or {}).get("cameras_down"):
            time.sleep(0.2)
        assert not (read_state(runtime_dir) or {}).get("cameras_down"), (
            "sim cameras never started producing"
        )

        # -- collection: arms enter control, frames record, stop saves ---------
        _send(proc, {"cmd": "start", "phase": "collection", "task": "default"})
        _wait_state(runtime_dir, "collection", 30.0, log_path)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and _left_control_active(runtime_dir) is not True:
            time.sleep(0.2)
        assert _left_control_active(runtime_dir) is True, (
            "left arm never entered its control session during collection"
        )
        time.sleep(2.0)  # record some frames
        _send(proc, {"cmd": "stop"})
        raw = _wait_state(runtime_dir, "viewing", 90.0, log_path)
        assert raw["last_outcome"]["outcome"] == "saved", raw["last_outcome"]
        assert raw["last_outcome"]["phase"] == "collection"
        # The dataset landed under the session's recording root (task default →
        # repo_id dfc/default) and the arm is back IDLE with telemetry flowing.
        assert (tmp_path / "datasets" / "dfc" / "default").is_dir()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and _left_control_active(runtime_dir) is not False:
            time.sleep(0.2)
        assert _left_control_active(runtime_dir) is False

        # -- eval: hold policy, finishes on its own ------------------------------
        _send(proc, {"cmd": "start", "phase": "eval", "task": "default"})
        _wait_state(runtime_dir, "eval", 30.0, log_path)
        raw = _wait_state(runtime_dir, "viewing", 60.0, log_path)
        assert raw["last_outcome"]["outcome"] == "finished", raw["last_outcome"]
        assert raw["last_outcome"]["phase"] == "eval"

        # -- skill: teach-and-repeat replay, finishes on its own ------------------
        # Teach a short trajectory (as the dashboard's 🎓 Teach would save it);
        # the sim arm tracks commanded q, so convergence + replay run through.
        import numpy as np

        from dual_flexiv_control import skills as skills_mod

        traj = np.linspace(0.0, 0.2, 20)[:, None] * np.ones(7)
        skills_mod.save_skill(
            skills_mod.Skill(name="taught", fps=15.0, q={"left": traj}),
            tmp_path / "skills",
        )
        _send(proc, {"cmd": "start", "phase": "skill", "skill": "taught"})
        _wait_state(runtime_dir, "skill", 30.0, log_path)
        assert read_state(runtime_dir)["task"] == "taught"  # labelled by skill
        raw = _wait_state(runtime_dir, "viewing", 60.0, log_path)
        assert raw["last_outcome"]["outcome"] == "finished", raw["last_outcome"]
        assert raw["last_outcome"]["phase"] == "skill"

        # -- an unknown skill is refused with a message; the session survives ----
        _send(proc, {"cmd": "start", "phase": "skill", "skill": "no_such_skill"})
        deadline = time.monotonic() + 15.0
        msg = None
        while time.monotonic() < deadline:
            raw = read_state(runtime_dir)
            msg = (raw or {}).get("message")
            if msg:
                break
            time.sleep(0.2)
        assert msg and "start failed" in msg and "no_such_skill" in msg
        assert read_state(runtime_dir)["state"] == "viewing"

        # -- a bad task is refused with a message; the session survives ---------
        _send(proc, {"cmd": "start", "phase": "collection", "task": "no_such_task"})
        deadline = time.monotonic() + 15.0
        msg = None
        while time.monotonic() < deadline:
            raw = read_state(runtime_dir)
            msg = (raw or {}).get("message")
            if msg:
                break
            time.sleep(0.2)
        assert msg and "start failed" in msg
        assert read_state(runtime_dir)["state"] == "viewing"

        # -- stdin EOF == shutdown: clean exit, state file + shm swept ----------
        proc.stdin.close()
        proc.wait(timeout=60.0)
        assert proc.returncode == 0, f"daemon exited {proc.returncode}"
        assert read_state(runtime_dir) is None, "session.json must be removed on shutdown"
        from dual_flexiv_control.streams.registry import sanitize

        leaked = glob.glob(f"/dev/shm/dfc_{sanitize(run_id)}_*")
        assert not leaked, f"leaked shm segments: {leaked}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10.0)


@pytest.mark.slow
def test_camera_death_is_not_fatal_and_respawns(tmp_path, monkeypatch):
    """A dead camera must not kill the session: it reads as down (gating runs,
    with the reason in ``message``), the arms stay up, and the camera respawns
    and recovers. Exercises the daemon's internals in-process (no subprocess)."""
    from hydra import compose
    from hydra import initialize_config_module
    from omegaconf import OmegaConf

    from dual_flexiv_control import session as sess
    from dual_flexiv_control.configs import register_configs

    register_configs()
    overrides = [
        "rig=bimanual",
        "runtime.sim=true",
        f"runtime.runtime_dir={tmp_path}/runtime",
        "arms.left.rate_hz=200", "arms.right.rate_hz=200",
        "cameras.wrist_left.width=64", "cameras.wrist_left.height=48",
        "cameras.wrist_right.width=64", "cameras.wrist_right.height=48",
        "cameras.static.width=64", "cameras.static.height=48",
    ]
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    config = OmegaConf.to_object(cfg)
    config.runtime.runtime_dir = str(tmp_path / "runtime")

    monkeypatch.setattr(sess, "_CAMERA_RESPAWN_S", 0.5)
    monkeypatch.setattr(sess, "_CAMERA_BOOT_GRACE_S", 10.0)
    monkeypatch.setattr(sess, "_CAMERA_FRESH_S", 1.0)
    monkeypatch.setattr(sess, "_CAMERA_KILL_GRACE_S", 0.2)
    daemon = sess.SessionDaemon(config, overrides)
    try:
        daemon.start_hardware()

        def _tend_until(cond, timeout_s, msg):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                daemon._tend_cameras()
                if cond():
                    return
                time.sleep(0.2)
            raise AssertionError(f"{msg} (cameras_down={daemon.state.cameras_down})")

        # All three sim cameras come up producing.
        _tend_until(lambda: daemon.state.cameras_down == [], 30.0,
                    "cameras never all came up")

        # Kill one camera hard (as a wedged ZED would): session must survive.
        victim = daemon.cameras[0]
        victim.proc.kill()
        victim.proc.join(timeout=10.0)
        _tend_until(lambda: victim.node.name in daemon.state.cameras_down, 15.0,
                    "dead camera never reported down")
        assert daemon._arms_alive(), "arms must be unaffected by a camera death"

        # Runs are refused while a camera is down, with the reason.
        daemon._handle_start({"cmd": "start", "phase": "collection", "task": "default"})
        assert daemon.run is None
        assert "camera(s) down" in (daemon.state.message or "")

        # The camera respawns (paced) and recovers on its own.
        _tend_until(lambda: daemon.state.cameras_down == [], 30.0,
                    "killed camera never respawned/recovered")

        # Wedge a camera ALIVE (SIGSTOP freezes it mid-loop, like a ZED whose
        # device dropped: process up, frames stale). The supervisor must report
        # it down, then kill + respawn it — a wedged handle never heals in
        # place, so waiting on the old process would show "waiting on camera"
        # forever even after a replug.
        import signal as _signal
        victim = daemon.cameras[1]
        wedged_pid = victim.proc.pid
        os.kill(wedged_pid, _signal.SIGSTOP)
        try:
            _tend_until(lambda: victim.node.name in daemon.state.cameras_down, 20.0,
                        "stalled camera never reported down")
            _tend_until(lambda: daemon.state.cameras_down == [], 30.0,
                        "stalled camera never killed/respawned/recovered")
        finally:
            try:  # unfreeze on failure so teardown/escalation can reap it
                os.kill(wedged_pid, _signal.SIGCONT)
            except ProcessLookupError:
                pass
        assert victim.proc.pid != wedged_pid, "stalled camera was not replaced"
    finally:
        daemon._teardown()


def test_session_manager_backoff_and_give_up_on_boot_crashes(tmp_path, monkeypatch):
    """A daemon that dies at boot must NOT be hot-respawned on every rerun: each
    attempt seizes cameras/arms. ensure() backs off, gives up after 3 early
    deaths (surfacing why), and re-arms on shutdown() (Reset services)."""
    pytest.importorskip("streamlit")
    import time as _time

    from dual_flexiv_control.dashboard import session as dsession

    monkeypatch.chdir(tmp_path)
    # The "daemon" exits 3 immediately, like a rig whose camera cannot open.
    monkeypatch.setattr(dsession, "_daemon_cmd", lambda rig, sim: ["bash", "-c", "exit 3"])
    manager = dsession.SessionManager()

    assert manager.ensure("bimanual", sim=False) is True    # spawn #1
    manager._proc.wait(timeout=10.0)                        # let it die

    # Within the backoff window nothing respawns.
    assert manager.ensure("bimanual", sim=False) is False
    view = manager.view()
    assert view.state == "down" and "retrying" in (view.message or "")

    # Age past the backoff to allow attempts #2 and #3.
    for expected_deaths in (2, 3):
        manager._spawned_at = _time.monotonic() - dsession._RESPAWN_BACKOFF_S - 1
        assert manager.ensure("bimanual", sim=False) is True
        manager._proc.wait(timeout=10.0)
        manager._spawned_at = _time.monotonic() - dsession._RESPAWN_BACKOFF_S - 1
        if expected_deaths < 3:
            continue
        # Third early death: give up — no more respawns, message says so.
        assert manager.ensure("bimanual", sim=False) is False
        assert manager._early_deaths == 3
    view = manager.view()
    assert view.state == "down"
    assert "auto-restart paused" in (view.message or "")

    # A different target re-arms (deliberate change)…
    assert manager.ensure("bimanual", sim=True) is True
    manager._proc.wait(timeout=10.0)
    # …and so does shutdown() (the Reset services path).
    manager._early_deaths = 3
    manager._gave_up = True
    manager.shutdown()
    assert manager._gave_up is False and manager._early_deaths == 0
    manager._spawned_at = _time.monotonic() - dsession._RESPAWN_BACKOFF_S - 1
    assert manager.ensure("bimanual", sim=True) is True
    manager._proc.wait(timeout=10.0)


@pytest.mark.slow
def test_session_manager_spawns_views_and_shuts_down(tmp_path, monkeypatch):
    """The dashboard's SessionManager drives a real (sim) daemon: spawn on
    ensure(), read state through view(), refuse a redundant respawn, and shut
    down cleanly (stdin EOF) leaving no state file behind."""
    pytest.importorskip("streamlit")
    from dual_flexiv_control.dashboard.session import SessionManager

    # The daemon resolves runtime_dir against its cwd; the manager reads
    # session.json the same way — chdir to the tmp dir so both land there.
    monkeypatch.chdir(tmp_path)
    manager = SessionManager()
    try:
        assert manager.view().state == "down"
        assert manager.ensure("bimanual", sim=True) is True   # spawned
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and manager.view().state != "viewing":
            time.sleep(0.2)
        view = manager.view()
        assert view.state == "viewing", f"never reached viewing: {view}"
        assert view.rig == "bimanual" and view.sim is True
        assert manager.ensure("bimanual", sim=True) is False  # already matching
        assert manager.send({"cmd": "stop"}) is True          # reachable (no-op stop)
    finally:
        manager.shutdown()
    assert manager.view().state == "down"
    assert read_state(tmp_path / "runtime") is None, "session.json must be swept"


@pytest.mark.slow
def test_second_daemon_on_same_runtime_dir_is_refused(tmp_path):
    """One session per runtime dir: a second daemon must refuse to start."""
    proc, runtime_dir, log_path = _spawn_daemon(tmp_path)
    try:
        _wait_state(runtime_dir, "viewing", 60.0, log_path)
        second = subprocess.run(
            [
                sys.executable, "-m", "dual_flexiv_control.session",
                "runtime.sim=true", f"runtime.runtime_dir={runtime_dir}",
            ],
            capture_output=True, text=True, timeout=60.0,
        )
        assert second.returncode != 0
        assert "already owns" in (second.stderr + second.stdout)
    finally:
        if proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
