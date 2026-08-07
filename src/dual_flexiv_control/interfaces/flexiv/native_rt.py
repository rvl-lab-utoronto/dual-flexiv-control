"""Launch the RDK 1.8 C++ real-time joint-position controller.

The Python arm process owns the shared-memory segments for its whole lifetime.
During a hardware qpos control session it releases its Python ``Robot`` handle
and this process temporarily becomes the sole RDK owner, consuming the existing
brain channels and publishing into the existing telemetry rings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


EXECUTABLE = "dfc-flexiv-rt-controller"


def find_controller() -> Path:
    """Resolve the native controller, with a useful source-tree default."""
    override = os.environ.get("DFC_FLEXIV_RT_CONTROLLER")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    installed = shutil.which(EXECUTABLE)
    if installed:
        candidates.append(Path(installed))
    # native_rt.py -> flexiv -> interfaces -> dual_flexiv_control -> src -> repo
    repo = Path(__file__).resolve().parents[4]
    candidates.append(repo / "build" / "native-flexiv-rt" / EXECUTABLE)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "native Flexiv RT controller is not built; run "
        "native/flexiv_rt_controller/build.sh or set DFC_FLEXIV_RT_CONTROLLER"
    )


def _add(args: list[str], key: str, value) -> None:
    args.extend((key, str(value)))


def controller_args(
    executable: Path,
    *,
    side: str,
    arm,
    control,
    coeffs,
    setpoint_shm: str,
    command_shm: str,
    telemetry_shm: dict[str, str],
    gripper_shm: str | None,
) -> list[str]:
    """Build the sidecar argv without a shell (values remain literal)."""
    args = [str(executable)]
    _add(args, "--serial", arm.serial)
    _add(args, "--dof", arm.dof)
    _add(args, "--setpoint-shm", setpoint_shm)
    _add(args, "--command-shm", command_shm)
    for signal, flag in (
        ("q", "--q-shm"),
        ("dq", "--dq-shm"),
        ("tau", "--tau-shm"),
        ("tau_ext", "--tau-ext-shm"),
        ("wrench", "--wrench-shm"),
        ("eef", "--eef-shm"),
        ("eef_vel", "--eef-vel-shm"),
        ("status", "--status-shm"),
    ):
        _add(args, flag, telemetry_shm[signal])
    _add(args, "--wrench-frame", arm.wrench_frame)
    _add(args, "--deadman-ms", control.channel.deadman_ms)
    _add(args, "--deadman-hard-ms", control.channel.deadman_hard_ms)
    _add(args, "--safety-check", int(arm.control_safety_check))
    _add(args, "--tolerance", arm.control_tolerance)
    _add(args, "--verbose", int(arm.verbose_rdk))
    _add(args, "--rt-priority", arm.rt_priority)
    _add(args, "--max-joint-vel", coeffs.max_joint_vel)
    _add(args, "--max-joint-acc", coeffs.max_joint_acc)
    if gripper_shm is not None and arm.gripper.enabled and arm.gripper.name:
        _add(args, "--gripper-shm", gripper_shm)
        _add(args, "--gripper-name", arm.gripper.name)
        _add(args, "--gripper-velocity", arm.gripper.velocity)
        _add(args, "--gripper-force", arm.gripper.force)
        _add(args, "--gripper-rate", arm.gripper.move_rate_hz)
        _add(args, "--gripper-deadband", arm.gripper.deadband)
        _add(args, "--gripper-init", int(arm.gripper.init_on_start))
        if arm.gripper.open_width is not None:
            _add(args, "--gripper-open-width", arm.gripper.open_width)
        if arm.gripper.closed_width is not None:
            _add(args, "--gripper-closed-width", arm.gripper.closed_width)
    return args


def run_controller(args: list[str], stop_event) -> None:
    """Run until STOP/deadman or process shutdown, then require a clean exit."""
    process = subprocess.Popen(args)
    try:
        while process.poll() is None:
            if stop_event.is_set():
                process.terminate()
                break
            time.sleep(0.01)
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=2.0)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
    if returncode != 0:
        raise RuntimeError(f"native RT controller exited with status {returncode}")
