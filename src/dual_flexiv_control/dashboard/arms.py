"""Per-arm identity + read-only operation status for the dashboard.

Display **names** come from the composed Hydra config (`arms.<side>.name` in
`conf/config.yaml`, customizable per arm; falls back to the side).

**Operation mode** (Auto / Auto-Remote / Manual / …) and the **E-stop** state are
*read-only*: `flexivrdk` can read them (`Robot.operational_status()`,
`Robot.estop_released`) but cannot switch operation mode — that's a physical
slide switch + Flexiv Elements. They are read live from a per-arm status stream
``<side>/status`` ( ``[operational_status_code, estop_pressed]`` ) when the
running system publishes it; otherwise the arm reads as **disconnected**.

This is a **monitoring view, not a safety interlock.**
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")

#: Per-arm status stream: a 2-vector ``[operational_status_code, estop_pressed]``.
#: ``operational_status_code`` is ``flexivrdk.OperationalStatus(...).value``.
STATUS_STREAM = "{side}/status"

#: Friendly labels for flexivrdk OperationalStatus names (RDK 1.8.0).
_MODE_LABELS = {
    "READY": "Auto (Remote)",
    "IN_AUTO_MODE": "Auto (local)",
    "IN_MANUAL_MODE": "Manual",
    "IN_REDUCED_STATE": "Reduced",
    "NOT_ENABLED": "Not enabled",
    "ESTOP_NOT_RELEASED": "E-stop",
    "BOOTING": "Booting",
    "RELEASING_BRAKE": "Releasing brake",
    "IN_RECOVERY_STATE": "Recovery",
    "MINOR_FAULT": "Minor fault",
    "CRITICAL_FAULT": "Critical fault",
    "UNKNOWN": "Unknown",
}


@dataclass(frozen=True)
class ArmInfo:
    side: str  # "left" | "right"
    name: str  # display name from config
    serial: str = ""  # robot serial (used by the read-only eval dq probe)
    dof: int = 7


@dataclass(frozen=True)
class ArmStatus:
    info: ArmInfo
    mode: str  # friendly operation-mode label
    estop_pressed: bool | None  # None = unknown
    source: str  # "live" | "placeholder"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
#: Cached single compose of the bits the dashboard needs: the arms + the sim flag.
_SNAPSHOT: tuple[tuple[ArmInfo, ...], bool] | None = None


def discover_arms() -> list[ArmInfo]:
    """The configured arms with display names + serials (computed once, then cached)."""
    return list(_compose()[0])


def runtime_is_sim() -> bool:
    """Whether ``runtime.sim`` selects simulated sources (no hardware)."""
    return _compose()[1]


def _compose() -> tuple[tuple[ArmInfo, ...], bool]:
    """Compose the config once (lock-guarded) and cache the arms + sim flag.

    Sharing one compose behind one lock keeps the (global, non-reentrant) Hydra
    init off the concurrent path that two callers — the status fragment and the
    eval probe thread — would otherwise take.
    """
    global _SNAPSHOT
    with _LOCK:
        if _SNAPSHOT is None:
            _SNAPSHOT = _compose_uncached()
        return _SNAPSHOT


def _compose_uncached() -> tuple[tuple[ArmInfo, ...], bool]:
    from hydra import compose
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config")

    arms: list[ArmInfo] = []
    for side, arm in cfg.arms.items():
        name = (str(getattr(arm, "name", "") or "")).strip() or str(side).capitalize()
        arms.append(
            ArmInfo(
                side=str(side),
                name=name,
                serial=str(getattr(arm, "serial", "") or ""),
                dof=int(getattr(arm, "dof", 7)),
            )
        )
    sim = bool(getattr(cfg.runtime, "sim", False))
    return tuple(arms), sim


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def read_arm_status(arm: ArmInfo, runtime_dir: str | None = None) -> ArmStatus:
    """Live operation mode + E-stop for ``arm``, or **disconnected** if none is published.

    "Connected" means a running ``FlexivInterface`` is publishing this arm's
    ``<side>/status`` stream; until then (or if its run ends) the arm reads as
    disconnected with unknown E-stop.
    """
    live = _read_live_status(arm, runtime_dir)
    if live is not None:
        code, estop = live
        return ArmStatus(arm, _label_for_code(code), bool(estop), "live")
    return ArmStatus(arm, mode="disconnected", estop_pressed=None, source="disconnected")


def _runtime_root(runtime_dir: str | None) -> Path:
    rd = runtime_dir or os.environ.get("DFC_RUNTIME_DIR", "runtime")
    return Path(rd if os.path.isabs(rd) else os.path.join(os.getcwd(), rd))


def _read_live_status(arm: ArmInfo, runtime_dir: str | None) -> tuple[float, float] | None:
    root = _runtime_root(runtime_dir)
    if not root.is_dir():
        return None
    try:
        from dual_flexiv_control.streams import StreamReader
        from dual_flexiv_control.streams import StreamRegistry
    except Exception:  # noqa: BLE001 - streams stack unavailable -> placeholder
        return None

    stream = STATUS_STREAM.format(side=arm.side)
    run_dirs = sorted(
        (p for p in root.iterdir() if (p / "streams").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        try:
            entry = StreamRegistry(str(root), run_dir.name).get(stream)
            if entry is None:
                continue
            reader = StreamReader.attach(entry)
            try:
                samples = reader.latest()
                if samples.n > 0:
                    vec = np.asarray(samples.newest)
                    return float(vec[0]), float(vec[1])
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - dead run / lapped buffer -> try next
            continue
    return None


def _label_for_code(code: float) -> str:
    """Map an OperationalStatus code to a friendly label (via the SDK enum)."""
    try:
        import flexivrdk

        name = flexivrdk.OperationalStatus(int(code)).name
    except Exception:  # noqa: BLE001 - flexivrdk absent / unknown code
        return f"status {int(code)}"
    return _MODE_LABELS.get(name, name.replace("_", " ").title())
