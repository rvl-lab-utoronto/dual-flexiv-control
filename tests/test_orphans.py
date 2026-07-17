"""Orphan detection/reaping: classification, live decoys, and the data sweeps.

The detectors only ever act on a **provably dead supervisor** — these tests pin
both directions: dead-tag / dead-creator debris is found and reclaimed, while
anything a live pid protects survives untouched. The shm/run-dir sweeps run
against tmp dirs so the host's real ``/dev/shm`` is never mutated here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from dual_flexiv_control.orphans import ENV_TAG
from dual_flexiv_control.orphans import ProcRecord
from dual_flexiv_control.orphans import classify_orphan
from dual_flexiv_control.orphans import find_orphans
from dual_flexiv_control.orphans import kill_orphans
from dual_flexiv_control.orphans import pid_alive
from dual_flexiv_control.orphans import sweep_stale_run_dirs
from dual_flexiv_control.orphans import sweep_stale_shm
from dual_flexiv_control.orphans import tag_supervisor


def _dead_pid() -> int:
    """A pid that is definitely dead (a just-reaped child of ours)."""
    proc = subprocess.Popen(["true"])
    proc.wait(timeout=10.0)
    return proc.pid


def test_pid_alive_basics():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(_dead_pid()) is False
    assert pid_alive(None) is False
    assert pid_alive("garbage") is False
    assert pid_alive(-1) is False


def test_tag_supervisor_stamps_the_environment(monkeypatch):
    monkeypatch.delenv(ENV_TAG, raising=False)
    tag_supervisor()
    assert os.environ[ENV_TAG] == str(os.getpid())


def test_classify_orphan_matrix():
    """Pure classification over plain records (no /proc needed)."""
    alive = {100}.__contains__  # pid 100 lives; everything else is dead

    def rec(pid=5, tag=None, cmdline=(), creators=()):
        return ProcRecord(pid=pid, tag=tag, cmdline=tuple(cmdline),
                          mapped_creators=tuple(creators))

    spawn = ("python", "-c", "from multiprocessing.spawn import spawn_main; spawn_main(...)")

    # Tagged with a dead supervisor: orphan. Live supervisor: protected —
    # even when the legacy evidence would otherwise match.
    assert classify_orphan(rec(tag="999"), self_pid=1, alive=alive) == "tagged"
    assert classify_orphan(rec(tag="100"), self_pid=1, alive=alive) is None
    assert classify_orphan(
        rec(tag="100", cmdline=spawn, creators=(999,)), self_pid=1, alive=alive
    ) is None
    # A malformed tag is unknown provenance: never touched.
    assert classify_orphan(rec(tag="junk"), self_pid=1, alive=alive) is None
    # Never self.
    assert classify_orphan(rec(pid=1, tag="999"), self_pid=1, alive=alive) is None

    # Legacy: an mp spawn child mapping only dead-creator dfc shm.
    assert classify_orphan(
        rec(cmdline=spawn, creators=(999,)), self_pid=1, alive=alive
    ) == "legacy"
    # One live creator protects it (better to leak than kill a live run's node).
    assert classify_orphan(
        rec(cmdline=spawn, creators=(999, 100)), self_pid=1, alive=alive
    ) is None
    # No dfc shm mapped, or not a spawn child: not our evidence.
    assert classify_orphan(rec(cmdline=spawn), self_pid=1, alive=alive) is None
    assert classify_orphan(
        rec(cmdline=("sleep", "60"), creators=(999,)), self_pid=1, alive=alive
    ) is None


def test_find_and_kill_tagged_orphans_with_live_decoys():
    """A real /proc scan: a sleeper tagged with a dead supervisor pid is found
    and killed; one tagged with a live pid survives."""
    dead = _dead_pid()
    env_dead = {**os.environ, ENV_TAG: str(dead)}
    env_live = {**os.environ, ENV_TAG: str(os.getpid())}
    orphan = subprocess.Popen(["sleep", "60"], env=env_dead)
    protected = subprocess.Popen(["sleep", "60"], env=env_live)
    try:
        found = {rec.pid: reason for rec, reason in find_orphans()}
        assert found.get(orphan.pid) == "tagged"
        assert protected.pid not in found

        killed = kill_orphans([orphan.pid], term_grace_s=5.0)
        assert orphan.pid in killed
        orphan.wait(timeout=10.0)
        assert orphan.poll() is not None
        assert protected.poll() is None, "the live-tagged decoy must survive"
    finally:
        for proc in (orphan, protected):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10.0)


def test_sweep_stale_shm_only_removes_dead_creators(tmp_path):
    dead = _dead_pid()
    live = os.getpid()
    (tmp_path / f"dfc_{dead}_deadbeef_left_q").write_bytes(b"x")
    (tmp_path / f"dfc_{dead}_0badf00d_left_q").write_bytes(b"x")
    (tmp_path / f"dfc_{live}_cafe0001_left_q").write_bytes(b"x")
    (tmp_path / "unrelated_file").write_bytes(b"x")
    (tmp_path / "dfc_notarunid").write_bytes(b"x")

    removed = sweep_stale_shm(
        keep_run_ids=(f"{dead}_0badf00d",), shm_dir=str(tmp_path)
    )
    assert removed == 1
    assert not (tmp_path / f"dfc_{dead}_deadbeef_left_q").exists()
    assert (tmp_path / f"dfc_{dead}_0badf00d_left_q").exists()  # kept explicitly
    assert (tmp_path / f"dfc_{live}_cafe0001_left_q").exists()  # live creator
    assert (tmp_path / "unrelated_file").exists()
    assert (tmp_path / "dfc_notarunid").exists()


def test_sweep_stale_run_dirs_cleans_dead_manifest_trees(tmp_path):
    dead = _dead_pid()
    live = os.getpid()
    dead_dir = tmp_path / f"{dead}_deadbeef" / "streams"
    dead_dir.mkdir(parents=True)
    (dead_dir / "left_q.json").write_text("{not json")
    live_dir = tmp_path / f"{live}_cafe0001" / "streams"
    live_dir.mkdir(parents=True)
    (tmp_path / "not_a_run_dir").mkdir()

    cleaned = sweep_stale_run_dirs(tmp_path)
    assert cleaned == 1
    assert not (tmp_path / f"{dead}_deadbeef").exists()
    assert (tmp_path / f"{live}_cafe0001").exists()
    assert (tmp_path / "not_a_run_dir").exists()


def test_spawn_children_inherit_the_tag():
    """The tag survives into a real child's exec-time /proc environ — the
    property the tag-based detector rests on."""
    dead = _dead_pid()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env={**os.environ, ENV_TAG: str(dead)},
    )
    try:
        deadline = time.monotonic() + 10.0
        found = {}
        while time.monotonic() < deadline:
            found = {rec.pid: reason for rec, reason in find_orphans()}
            if child.pid in found:
                break
            time.sleep(0.2)
        assert found.get(child.pid) == "tagged"
    finally:
        child.kill()
        child.wait(timeout=10.0)
