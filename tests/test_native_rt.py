from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dual_flexiv_control.interfaces.flexiv.native_rt import controller_args
from dual_flexiv_control.interfaces.flexiv.native_rt import find_controller
from dual_flexiv_control.streams.ring import SharedRingBuffer


def _arm():
    channel = SimpleNamespace(deadman_ms=100.0, deadman_hard_ms=500.0)
    control = SimpleNamespace(channel=channel)
    gripper = SimpleNamespace(
        enabled=True,
        name="Flexiv-GN01",
        velocity=0.1,
        force=40.0,
        move_rate_hz=15.0,
        deadband=0.02,
        init_on_start=True,
        open_width=None,
        closed_width=0.0,
    )
    return SimpleNamespace(
        serial="Rizon4s-test",
        dof=7,
        control=control,
        wrench_frame="local",
        control_safety_check=False,
        control_tolerance=0.5,
        verbose_rdk=False,
        rt_priority=0,
        gripper=gripper,
    )


def _coeffs():
    return SimpleNamespace(max_joint_vel=1.5, max_joint_acc=2.0)


def test_controller_argv_contains_all_ipc_and_safety_inputs():
    telemetry = {
        "q": "q",
        "dq": "dq",
        "tau": "tau",
        "tau_ext": "tau_ext",
        "wrench": "wrench",
        "eef": "eef",
        "eef_vel": "eef_vel",
        "status": "status",
    }
    args = controller_args(
        Path("/controller"),
        side="right",
        arm=_arm(),
        control=_arm().control,
        coeffs=_coeffs(),
        setpoint_shm="setpoint",
        command_shm="command",
        telemetry_shm=telemetry,
        gripper_shm="gripper",
    )
    pairs = dict(zip(args[1::2], args[2::2], strict=True))

    assert pairs["--serial"] == "Rizon4s-test"
    assert pairs["--setpoint-shm"] == "setpoint"
    assert pairs["--q-shm"] == "q"
    assert pairs["--status-shm"] == "status"
    assert pairs["--safety-check"] == "0"
    assert pairs["--deadman-hard-ms"] == "500.0"
    assert pairs["--max-joint-vel"] == "1.5"
    assert pairs["--max-joint-acc"] == "2.0"
    assert pairs["--gripper-name"] == "Flexiv-GN01"
    assert pairs["--gripper-closed-width"] == "0.0"
    assert "--gripper-open-width" not in pairs


def test_native_binary_reads_python_ring_abi():
    try:
        executable = find_controller()
    except FileNotFoundError:
        pytest.skip("native controller has not been built")

    name = f"dfc_native_rt_test_{time.monotonic_ns()}"
    ring = SharedRingBuffer.create(name, capacity=4, dim=3, dtype="float64")
    try:
        stamp = time.monotonic_ns()
        ring.append(np.array([1.0, 2.0, 3.0]), stamp)
        result = subprocess.run(
            [
                str(executable),
                "--validate-ring",
                ring.name,
                "--expected-dim",
                "3",
                "--roundtrip",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        assert result.returncode == 0, result.stderr
        assert "dim=3" in result.stdout
        assert f"timestamp={stamp}" in result.stdout
        returned = ring.latest()
        assert returned.n == 1
        assert returned.newest_t_ns == stamp + 1
        np.testing.assert_allclose(returned.newest, [1.0, 2.0, 3.0])
    finally:
        ring.close()
        ring.unlink()


def test_native_trajectory_generator_respects_limits():
    try:
        executable = find_controller()
    except FileNotFoundError:
        pytest.skip("native controller has not been built")
    result = subprocess.run(
        [str(executable), "--validate-trajectory"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert result.returncode == 0, result.stderr
    assert "ok trajectory" in result.stdout
