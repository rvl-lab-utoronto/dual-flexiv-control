"""The Grav comp panel's health-log parsing (dashboard/factr_servers.py).

Sample lines are verbatim from this machine's FACTR_Teleop/logs — the rclpy
stderr format the grav-comp teleops actually emit.
"""

from __future__ import annotations

import time

import pytest

from dual_flexiv_control.dashboard import factr_servers as fs

_HEALTH = (
    "[INFO] [1784237385.093860358] [factr_teleop_0]: [health] "
    "id1:T=41C I=-241/910  id2:T=37C I=118/1193  id3:T=39C I=-154/910  "
    "id4:T=35C I=-33/1193  id5:T=35C I=-66/910  id6:T=34C I=43/910  "
    "id7:T=33C I=6/910  id8:T=37C I=0/910"
)
_OFFSETS = (
    "[INFO] [1784237300.000000000] [factr_teleop_0]: FACTR TELEOP factr_teleop_0: "
    "best offsets: [1.571, -0.003, 0.008, 1.566, 0.001, -0.002, 0.004, 2.113]"
)
_MATCH_NAG = (
    "[INFO] [1784237310.000000000] [factr_teleop_0]: FACTR TELEOP factr_teleop_0: "
    "Please match starting joint pos. Current error: 8.312 | current joint pos: "
    "[0.001, -0.700, 0.012, 1.571, 0.030, 0.700, 0.002]"
)
_MATCHED = (
    "[INFO] [1784237320.000000000] [factr_teleop_0]: FACTR TELEOP factr_teleop_0: "
    "Initial joint position matched."
)
_HW_ERROR = (
    "[WARN] [1784237386.000000000] [factr_teleop_0]: [health] servo id1 "
    "HARDWARE ERROR 0x20 (overload) — its gravity comp is dead until power-cycled"
)
_SELF_DISABLE = (
    "[WARN] [1784237387.000000000] [factr_teleop_0]: [health] servo id4 disabled "
    "its own torque while the node still expects it on"
)


def _write(tmp_path, *lines):
    path = tmp_path / "factr_health_left.log"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_parse_servos_reads_every_reading():
    servos = fs.parse_servos(_HEALTH)
    assert len(servos) == 8
    assert servos[0] == fs.ServoHealth(sid=1, temp_c=41, current=-241, current_limit=910)
    assert servos[3].current_limit == 1193
    assert max(servos, key=lambda s: s.temp_c).sid == 1


def test_tail_lines_bounded_read(tmp_path):
    path = _write(tmp_path, *(f"line {i}" for i in range(500)))
    tail = fs.tail_lines(path, max_bytes=200)
    assert tail and tail[-1] == "line 499"
    assert len(tail) < 40  # bounded, and the cut first line was dropped
    assert fs.tail_lines(str(tmp_path / "missing.log")) == []


def test_phase_progression(tmp_path):
    # Fresh boot: no marker yet -> the calibration read is in progress.
    h = fs.read_teleop_health(_write(tmp_path, "[INFO] […] booting"))
    assert h.phase == fs.CALIBRATING and h.servos == () and h.alerts == ()

    # Offsets logged -> the match loop is next.
    h = fs.read_teleop_health(_write(tmp_path, _OFFSETS))
    assert h.phase == fs.MATCHING and h.match_error is None

    # Nagging for the start pose, with the operator's distance-to-go.
    h = fs.read_teleop_health(_write(tmp_path, _OFFSETS, _MATCH_NAG))
    assert h.phase == fs.MATCHING and h.match_error == 8.312

    # Matched -> grav comp begins (no health summary yet).
    h = fs.read_teleop_health(_write(tmp_path, _OFFSETS, _MATCH_NAG, _MATCHED))
    assert h.phase == fs.RUNNING and h.servos == ()

    # Health summaries flowing: servos + a wall-clock age.
    h = fs.read_teleop_health(_write(tmp_path, _OFFSETS, _MATCH_NAG, _MATCHED, _HEALTH))
    assert h.phase == fs.RUNNING and len(h.servos) == 8
    assert h.ts == 1784237385.093860358
    assert h.age_s == pytest.approx(time.time() - h.ts, abs=5.0) and h.age_s >= 0

    assert fs.read_teleop_health(str(tmp_path / "missing.log")) is None


def test_fault_lines_alert_without_masking_the_phase(tmp_path):
    path = _write(tmp_path, _MATCHED, _HEALTH, _HW_ERROR, _SELF_DISABLE)
    h = fs.read_teleop_health(path)
    assert h.phase == fs.RUNNING  # fault lines are skipped by the phase scan
    assert len(h.servos) == 8
    assert len(h.alerts) == 2
    assert "HARDWARE ERROR" in h.alerts[0]
    assert "disabled its own torque" in h.alerts[1]


def test_old_faults_age_out_of_the_alert_window(tmp_path):
    filler = [_HEALTH] * (fs._ALERT_WINDOW_LINES + 5)
    h = fs.read_teleop_health(_write(tmp_path, _HW_ERROR, *filler))
    assert h.alerts == ()  # ancient fault: history, not an active alert


def test_supervisor_status_carries_log_paths(tmp_path, monkeypatch):
    """The panel's log tailing keys off status()['logs'] — pin the contract."""
    from dual_flexiv_control.configs import FactrLaunchCfg
    from dual_flexiv_control.interfaces.factr import launch as fl

    cfg = FactrLaunchCfg(
        enabled=True, workdir=str(tmp_path),
        teleop_modules={"left": "m.left"}, calib_delay_s=7.5,
    )
    sup = fl.FactrServerSupervisor(cfg, ["left"])
    status = sup.status()
    assert status["countdown_total_s"] == 7.5
    assert status["logs"]["teleop:left"].endswith("logs/factr_health_left.log")
    assert status["logs"]["api"].endswith("logs/factr_api.log")
