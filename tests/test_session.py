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
from types import SimpleNamespace

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


def test_force_feedback_command_fans_out_to_every_factr_leader(monkeypatch):
    """The dashboard command uses each side's bodyless FACTR control route."""
    from dual_flexiv_control import session as sess

    requests = []

    class Response:
        status = 200

        def read(self):
            return b"{}"

    class Connection:
        def __init__(self, host, port, timeout):
            self.address = host, port, timeout

        def request(self, method, route):
            requests.append((*self.address, method, route))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPConnection", Connection)
    daemon = sess.SessionDaemon.__new__(sess.SessionDaemon)
    daemon.config = SimpleNamespace(
        runtime=SimpleNamespace(sim=False),
        factr=SimpleNamespace(servers={
            "left": SimpleNamespace(host="leader-a", port=5000, request_timeout_s=0.5),
            "right": SimpleNamespace(host="leader-b", port=5001, request_timeout_s=0.7),
        }),
    )
    daemon.state = SimpleNamespace(message=None)

    daemon._handle_force_feedback(enable=True)
    daemon._handle_force_feedback(enable=False)

    assert requests == [
        ("leader-a", 5000, 0.5, "POST", "/enable_force_feedback_left"),
        ("leader-b", 5001, 0.7, "POST", "/enable_force_feedback_right"),
        ("leader-a", 5000, 0.5, "POST", "/disable_force_feedback_left"),
        ("leader-b", 5001, 0.7, "POST", "/disable_force_feedback_right"),
    ]
    assert daemon.state.message == "force feedback disabled — leader(s) left, right"


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


def _compose_config(tmp_path, extra=(), sim=True):
    """A validated sim config for in-process daemon tests (no @hydra.main)."""
    from hydra import compose
    from hydra import initialize_config_module
    from omegaconf import OmegaConf

    from dual_flexiv_control.configs import register_configs

    register_configs()
    overrides = [
        "rig=bimanual",
        f"runtime.sim={'true' if sim else 'false'}",
        f"runtime.runtime_dir={tmp_path}/runtime",
        "arms.left.rate_hz=200", "arms.right.rate_hz=200",
        "cameras.wrist_left.width=64", "cameras.wrist_left.height=48",
        "cameras.wrist_right.width=64", "cameras.wrist_right.height=48",
        "cameras.static.width=64", "cameras.static.height=48",
        *extra,
    ]
    with initialize_config_module(config_module="dual_flexiv_control.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    config = OmegaConf.to_object(cfg)
    config.runtime.runtime_dir = str(tmp_path / "runtime")
    return config, overrides


def _free_closed_port() -> int:
    """A localhost port with nothing listening (connect gets refused)."""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Arm supervision (WS1) + recovery commands (WS4), in-process
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_arm_death_is_not_fatal_and_respawns(tmp_path, monkeypatch):
    """A dead arm must not kill the session: its streams are swept, its session
    queue drained (a stale EnterControl must never fire on respawn), starts that
    need it are refused, and it respawns to IDLE telemetry. Mid-run, its death
    stops the run instead. reconnect_arm / respawn_camera replace nodes on
    demand and are refused while a run needs them."""
    from dual_flexiv_control import session as sess
    from dual_flexiv_control.interfaces.flexiv.interface import EnterControl

    config, overrides = _compose_config(
        tmp_path, extra=["arms.left.control_enabled=true", "arms.left.control_rate_hz=100"]
    )
    monkeypatch.setattr(sess, "_ARM_RESPAWN_S", 0.5)
    monkeypatch.setattr(sess, "_ARM_FRESH_S", 5.0)
    monkeypatch.setattr(sess, "_CAMERA_BOOT_GRACE_S", 10.0)
    daemon = sess.SessionDaemon(config, overrides)
    try:
        daemon.start_hardware()

        def _tend_until(cond, timeout_s, msg):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                daemon._tend_arms()
                daemon._tend_cameras()
                if cond():
                    return
                time.sleep(0.2)
            raise AssertionError(
                f"{msg} (arms_down={daemon.state.arms_down}, "
                f"cameras_down={daemon.state.cameras_down})"
            )

        _tend_until(
            lambda: daemon.state.arms_down == [] and daemon.state.cameras_down == [],
            60.0, "hardware never came up",
        )

        # -- idle death: swept, drained, refused, respawned ---------------------
        victim = next(u for u in daemon.arms if u.side == "left")
        old_pid = victim.proc.pid
        daemon.session_qs["left"].put(EnterControl())  # stale message to drain
        time.sleep(0.3)  # let the queue's feeder thread deliver it
        victim.proc.kill()
        victim.proc.join(timeout=10.0)
        daemon._tend_arms()
        assert victim.node.name in daemon.state.arms_down
        assert daemon._registry.get("left/status") is None, "dead arm's streams not swept"
        try:
            stale = daemon.session_qs["left"].get_nowait()
        except Exception:  # noqa: BLE001 - queue.Empty (mp re-exports it)
            stale = None
        assert stale is None, "stale EnterControl survived the drain"

        daemon._handle_start({"cmd": "start", "phase": "collection", "task": "default"})
        assert daemon.run is None
        assert "arm(s)" in (daemon.state.message or "")

        _tend_until(lambda: daemon.state.arms_down == [], 60.0,
                    "dead arm never respawned/recovered")
        assert victim.proc.pid != old_pid, "arm proc was not replaced"
        assert daemon._arms_alive()

        # -- mid-run death stops the run; no respawn while the run is active ----
        stop_event = daemon.ctx.Event()
        consumer = daemon.ctx.Process(target=time.sleep, args=(60,), name="collection")
        consumer.start()
        daemon.run = sess._ActiveRun(
            phase="collection", task="t", proc=consumer,
            stop_event=stop_event, command_sides=["left"],
        )
        daemon.state.state = sess.COLLECTION

        # Recovery commands are refused while the run needs the hardware.
        daemon._handle_reconnect_arm({"cmd": "reconnect_arm", "side": "left"})
        assert "during a run" in (daemon.state.message or "")
        daemon._handle_respawn_camera(
            {"cmd": "respawn_camera", "name": daemon.cameras[0].node.name}
        )
        assert "during a" in (daemon.state.message or "")

        dead_pid = victim.proc.pid
        victim.proc.kill()
        victim.proc.join(timeout=10.0)
        daemon._tend_arms()
        assert daemon.run.stopping, "arm death mid-run must stop the run"
        assert "died mid-run" in (daemon.state.message or "")
        daemon._tend_arms()
        assert not victim.proc.is_alive() and victim.proc.pid == dead_pid, (
            "arm must not respawn while a run is active"
        )
        consumer.terminate()
        consumer.join(timeout=10.0)
        assert daemon._reap_consumer()
        assert daemon.run is None
        _tend_until(lambda: daemon.state.arms_down == [], 60.0,
                    "arm never respawned after the run was reaped")

        # -- reconnect_arm replaces a LIVE node immediately ----------------------
        live_pid = victim.proc.pid
        daemon._handle_reconnect_arm({"cmd": "reconnect_arm", "side": "left"})
        assert victim.proc.pid != live_pid, "reconnect_arm did not replace the node"
        assert "reconnecting" in (daemon.state.message or "")
        daemon._handle_reconnect_arm({"cmd": "reconnect_arm", "side": "nope"})
        assert "unknown arm" in (daemon.state.message or "")

        # -- respawn_camera replaces a camera immediately (no pacing) ------------
        cam = daemon.cameras[0]
        cam_pid = cam.proc.pid
        daemon._handle_respawn_camera({"cmd": "respawn_camera", "name": cam.node.name})
        assert cam.proc.pid != cam_pid, "respawn_camera did not replace the node"
        daemon._handle_respawn_camera({"cmd": "respawn_camera", "name": "zed:nope"})
        assert "unknown camera" in (daemon.state.message or "")

        _tend_until(
            lambda: daemon.state.arms_down == [] and daemon.state.cameras_down == [],
            60.0, "hardware never recovered after recovery commands",
        )
    finally:
        daemon._teardown()


# ---------------------------------------------------------------------------
# Preflight (WS2)
# ---------------------------------------------------------------------------


def test_preflight_run_gates(tmp_path):
    """preflight_run: hold-eval/skill/collection pass here (collection's leader
    liveness is judged on the factr/<side> streams by the daemon's start gate,
    not an HTTP probe); a remote eval policy with nothing listening is refused."""
    from dual_flexiv_control.session import preflight_run

    port = _free_closed_port()
    cfg_hold, _ = _compose_config(tmp_path, extra=["policy.kind=hold", "runtime.phase=eval"])
    assert preflight_run(cfg_hold, "eval") is None
    assert preflight_run(cfg_hold, "skill") is None

    cfg_remote, _ = _compose_config(
        tmp_path,
        extra=["runtime.phase=eval", "policy.kind=remote",
               "policy.host=127.0.0.1", f"policy.port={port}"],
    )
    msg = preflight_run(cfg_remote, "eval")
    assert msg is not None and "not reachable" in msg and str(port) in msg

    cfg_col, _ = _compose_config(tmp_path, extra=["runtime.phase=collection"])
    assert preflight_run(cfg_col, "collection") is None  # stream gate lives elsewhere


def test_factr_stale_sides_judges_the_leader_streams(tmp_path):
    """The collection start gate reads the factr/<side> streams: absent -> stale,
    stale sample -> stale, fresh sample -> ok. No factr node (rig without
    leaders) -> nothing to gate."""
    import time as _time

    import numpy as np

    from dual_flexiv_control import session as sess
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    config, overrides = _compose_config(tmp_path)
    daemon = sess.SessionDaemon(config, overrides)
    assert daemon._factr_stale_sides() == []  # no factr unit -> gate open

    # Fabricate the unit (no live process needed: the gate only reads streams).
    daemon.factr = sess._FactrUnit(
        node=None, proc=None, stop_event=None, spawned_at=0.0
    )
    sides = list(config.factr.servers)
    assert set(daemon._factr_stale_sides()) == set(sides)  # no streams yet

    writers = []
    try:
        for i, side in enumerate(sides):
            w = StreamWriter.create(
                StreamSpec(name=factr_stream_name(side), dim=8, capacity=64,
                           dtype="float64", rate_hz=100.0),
                daemon.run_id, daemon._registry,
            )
            writers.append(w)
            # left gets a FRESH sample; every other side a stale one.
            age_ns = 0 if i == 0 else int(60e9)
            w.write(np.zeros(8), _time.monotonic_ns() - age_ns)
        stale = daemon._factr_stale_sides()
        assert sides[0] not in stale
        assert set(stale) == set(sides[1:])
    finally:
        for w in writers:
            w.close()
            w.unlink()


def test_preflight_refusal_blocks_start_before_entercontrol(tmp_path, monkeypatch):
    """A preflight refusal must leave the session untouched: no consumer, no
    EnterControl on any arm's session queue, the reason in state.message."""
    from dual_flexiv_control import session as sess

    config, overrides = _compose_config(
        tmp_path,
        extra=["arms.left.control_enabled=true",
               f"recording.root={tmp_path}/datasets"],
    )
    daemon = sess.SessionDaemon(config, overrides)
    monkeypatch.setattr(sess, "preflight_run", lambda cfg, phase: "start refused — test gate")
    daemon._handle_start({"cmd": "start", "phase": "collection", "task": "default"})
    assert daemon.run is None
    assert daemon.state.message == "start refused — test gate"
    assert daemon.session_qs["left"].empty(), "EnterControl must not fire on a refused start"


def test_follower_control_blocker_is_fail_safe():
    """READY/disabled followers may enter control; safety/error states may not."""
    import numpy as np

    from dual_flexiv_control.session import follower_control_blocker

    assert follower_control_blocker(np.array([1.0, 0.0])) is None   # READY
    assert follower_control_blocker(np.array([4.0, 0.0])) is None   # NOT_ENABLED
    assert follower_control_blocker(np.array([1.0, 1.0])) == "E-stop pressed"
    assert "critical fault" in follower_control_blocker(np.array([7.0, 0.0]))
    assert follower_control_blocker(None) == "status unavailable"


def test_eval_follower_error_forces_dry_run(tmp_path, monkeypatch):
    """An unsafe follower keeps policy/viz alive but removes every write path."""
    import numpy as np

    from dual_flexiv_control import session as sess

    # Compose directly instead of using the bimanual E2E fixture: this unit only
    # needs one control-capable follower and no hardware processes are started.
    overrides = [
        "rig=left_only",
        "runtime.sim=true",
        f"runtime.runtime_dir={tmp_path}/runtime",
        "arms.left.control_enabled=true",
    ]
    config = sess.compose_config(overrides)
    daemon = sess.SessionDaemon(config, overrides)
    built = []

    class _Consumer:
        name = "eval"
        drive_sides = None

    def _build(run_config, _run_id):
        built.append(run_config)
        return _Consumer()

    monkeypatch.setattr(sess, "build_consumer", _build)
    monkeypatch.setattr(sess, "preflight_run", lambda _cfg, _phase: "test stop")
    monkeypatch.setattr(
        daemon, "_arm_status",
        lambda _side: np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    )

    daemon._handle_start({"cmd": "start", "phase": "eval", "task": "default"})

    assert built and all(not arm.control_enabled for arm in built[0].arms.values())
    assert daemon.run is None
    assert daemon.session_qs["left"].empty(), "dry eval must not send EnterControl"


# ---------------------------------------------------------------------------
# Switch + pending queue (WS3), in-process units
# ---------------------------------------------------------------------------


def test_validate_start_matrix():
    from dual_flexiv_control.session import SessionDaemon

    v = SessionDaemon._validate_start
    assert v({"cmd": "start", "phase": "collection", "task": "t"}) is None
    assert v({"cmd": "switch", "phase": "eval", "task": "t", "port": 8000}) is None
    assert v({"cmd": "start", "phase": "skill", "skill": "s1"}) is None  # task defaults
    assert "bad start" in v({"cmd": "start", "phase": "bogus", "task": "t"})
    assert "bad start" in v({"cmd": "start", "phase": "collection"})  # no task
    assert "skill" in v({"cmd": "start", "phase": "skill"})  # no skill name
    assert "port" in v({"cmd": "start", "phase": "eval", "task": "t", "port": 99999})
    assert "host" in v({"cmd": "start", "phase": "eval", "task": "t", "host": "bad host"})
    assert "policy" in v({"cmd": "start", "phase": "eval", "task": "t", "policy": "../x"})


def test_switch_and_pending_lifecycle(tmp_path, monkeypatch):
    """switch: starts immediately when idle; queues + stops the run when busy;
    the queued start fires once idle, is abandoned past the wind-down deadline,
    and is cancelled by stop."""
    import threading

    from dual_flexiv_control import session as sess

    config, overrides = _compose_config(tmp_path)
    daemon = sess.SessionDaemon(config, overrides)
    started: list = []
    monkeypatch.setattr(
        daemon, "_handle_start", lambda cmd: started.append(cmd), raising=False
    )

    # Malformed switch: refused outright, nothing queued.
    daemon._handle_switch({"cmd": "switch", "phase": "bogus", "task": "t"})
    assert daemon.pending is None and "bad start" in daemon.state.message

    # Idle: a switch is just a start.
    daemon._handle_switch({"cmd": "switch", "phase": "eval", "task": "t"})
    assert started and started[-1]["phase"] == "eval" and daemon.pending is None

    # Active run: switch stops it and queues the start (deadline arms post-reap).
    run = sess._ActiveRun(
        phase="collection", task="t", proc=None,
        stop_event=threading.Event(), command_sides=["left"],
    )
    daemon.run = run
    daemon.state.state = sess.COLLECTION
    daemon._handle_switch({"cmd": "switch", "phase": "eval", "task": "t"})
    assert run.stopping and run.stop_event.is_set()
    assert daemon.pending is not None and daemon.pending.deadline is None
    assert daemon.state.pending == {"phase": "eval", "task": "t", "skill": None}
    assert "switching to eval" in daemon.state.message

    # Still running: nothing fires.
    n_started = len(started)
    assert daemon._tend_pending() is False
    assert len(started) == n_started

    # Run reaped, arms still winding down: deadline arms; expiry abandons it.
    daemon.run = None
    monkeypatch.setattr(daemon, "_arms_still_in_control", lambda: ["left"], raising=False)
    monkeypatch.setattr(sess, "_PENDING_WINDDOWN_S", 0.1)
    assert daemon._tend_pending() is False  # arms busy, deadline just armed
    assert daemon.pending.deadline is not None
    time.sleep(0.15)
    assert daemon._tend_pending() is True
    assert daemon.pending is None and daemon.state.pending is None
    assert "abandoned" in daemon.state.message
    assert len(started) == n_started

    # Queued start fires the moment the arms go idle.
    monkeypatch.setattr(daemon, "_arms_still_in_control", lambda: [], raising=False)
    daemon._handle_switch({"cmd": "switch", "phase": "collection", "task": "t2"})
    assert started[-1]["task"] == "t2"  # idle again: fired immediately

    # stop cancels a queued switch.
    daemon._set_pending({"cmd": "switch", "phase": "eval", "task": "t"}, deadline=None)
    daemon._handle_stop()
    assert daemon.pending is None and daemon.state.pending is None
    assert "cancelled" in daemon.state.message


@pytest.mark.slow
def test_switch_collection_to_eval_sim(tmp_path):
    """End-to-end atomic switch: one command takes a live collection run through
    saving (episode committed) into an eval run, with zero manual retries."""
    proc, runtime_dir, log_path = _spawn_daemon(tmp_path)

    def _wait(pred, timeout_s, what):
        deadline = time.monotonic() + timeout_s
        raw = None
        while time.monotonic() < deadline:
            raw = read_state(runtime_dir)
            if raw is not None and pred(raw):
                return raw
            time.sleep(0.1)
        tail = ""
        if os.path.isfile(log_path):
            tail = "".join(open(log_path, errors="replace").readlines()[-30:])
        raise AssertionError(f"never observed: {what} (last: {raw!r})\n{tail}")

    try:
        _wait_state(runtime_dir, "viewing", 60.0, log_path)
        _wait(lambda r: not r.get("cameras_down") and not r.get("arms_down"),
              30.0, "hardware up")

        _send(proc, {"cmd": "start", "phase": "collection", "task": "default"})
        _wait_state(runtime_dir, "collection", 30.0, log_path)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and _left_control_active(runtime_dir) is not True:
            time.sleep(0.2)
        assert _left_control_active(runtime_dir) is True
        time.sleep(1.0)  # record some frames

        _send(proc, {"cmd": "switch", "phase": "eval", "task": "default"})
        # The collection episode commits…
        raw = _wait(
            lambda r: (r.get("last_outcome") or {}).get("phase") == "collection",
            120.0, "collection outcome recorded",
        )
        assert raw["last_outcome"]["outcome"] == "saved", raw["last_outcome"]
        # …and the eval run starts on its own and finishes (hold policy).
        raw = _wait(
            lambda r: (r.get("last_outcome") or {}).get("phase") == "eval",
            120.0, "eval ran after the switch",
        )
        assert raw["last_outcome"]["outcome"] == "finished", raw["last_outcome"]
        assert (tmp_path / "datasets" / "dfc" / "default").is_dir()

        proc.stdin.close()
        proc.wait(timeout=60.0)
        assert proc.returncode == 0
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
