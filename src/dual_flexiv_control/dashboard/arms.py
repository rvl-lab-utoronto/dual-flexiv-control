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

#: Per-arm status stream: ``[operational_status_code, estop_pressed, control_active]``.
#: ``operational_status_code`` is ``flexivrdk.OperationalStatus(...).value``;
#: ``control_active`` is 1.0 while the arm is inside a control session (a
#: collection/eval run) and 0.0 while idle (viewing). Older 2-wide streams (no
#: ``control_active``) are tolerated.
STATUS_STREAM = "{side}/status"

#: Per-arm measured joint-position stream (published by ``FlexivInterface`` as
#: ``<side>/q``, link-side joint positions in rad). Read by the 3D robot scene to
#: pose the solid arms at the real configuration.
JOINT_POS_STREAM = "{side}/q"

#: Per-arm policy horizon target (published by the eval node once per inference:
#: the joint target at the END of the returned action chunk — see
#: ``policy.loop.horizon_stream_name``). Read by the 3D robot scene to pose the
#: purple horizon-target ghost + the current→target EEF trace during eval runs.
HORIZON_STREAM = "eval/{side}/q_horizon"

#: Cartesian sibling of :data:`HORIZON_STREAM` (control kinds with no joint target
#: — ``end_effector``/``eef_vel``): the estimated base-frame TCP position
#: ``[x y z]`` at the END of the chunk — see ``policy.loop.eef_horizon_stream_name``.
#: Read by the 3D robot scene to draw the current→predicted EEF trace (no ghost).
EEF_HORIZON_STREAM = "eval/{side}/eef_horizon"

#: Policy-server comm events published by a running eval (one ``[kind, seq,
#: elapsed_s]`` sample per packet sent / received / failed — see
#: ``policy.loop.comm_stream_name``). Read by the mirror to plot send/receive
#: activity and round-trip latency below the robot metrics.
POLICY_COMM_STREAM = "eval/policy_comm"

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
    source: str  # "live" | "disconnected"
    control_active: bool | None = None  # in a control session? None = unknown/old stream


@dataclass(frozen=True)
class LeaderStatus:
    """Reachability of one FACTR teleop leader (the arm the operator moves)."""

    side: str  # "left" | "right"
    name: str  # display name (side, capitalized)
    reachable: bool  # did the leader's server answer?
    dof: int  # arm joints (trailing gripper dropped), when reachable
    gripper: float | None  # raw trailing gripper reading, when reachable
    sim: bool  # runtime.sim -> the reading is synthetic, not real hardware


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
#: Cached single compose of the bits the dashboard needs: arms, sim flag, FACTR cfg,
#: per-side leader→Rizon joint conventions.
_SNAPSHOT: tuple[tuple[ArmInfo, ...], bool, object, dict] | None = None
#: The rig the dashboard composes against (``rig=<name>`` override); None = the
#: config default. Owned here so every dashboard compose (arms, cameras, storage)
#: follows the same selection.
_ACTIVE_RIG: str | None = None

_LEADER_LOCK = threading.Lock()
#: Cached FACTR client for the read-only leader-status probe. Its own client (not
#: the Calibrate tab's) so the always-on status poll and the on-demand calibration
#: render never share one keep-alive HTTP connection across threads. Dropped by
#: :func:`reset` so a rig / ``runtime.sim`` change rebuilds it.
_LEADER_CLIENT = None


def set_active_rig(rig: str | None) -> None:
    """Select the rig every dashboard compose uses; drops the cached snapshot.

    The dashboard pins this once at startup from its launch option
    (``dfc-dashboard --rig <name>``); ``cameras.reset()`` / ``storage.reset()``
    must be called alongside so their composes follow too.
    """
    global _ACTIVE_RIG, _SNAPSHOT
    with _LOCK:
        _ACTIVE_RIG = rig
        _SNAPSHOT = None


def active_rig() -> str | None:
    """The rig name dashboard composes are pinned to, or None for the default.

    Lock-free read (atomic in CPython) — called from inside :func:`_compose`,
    which already holds the non-reentrant ``_LOCK``.
    """
    return _ACTIVE_RIG


def compose_overrides() -> list[str]:
    """The Hydra overrides every dashboard compose should apply (the rig pin)."""
    rig = active_rig()
    return [f"rig={rig}"] if rig else []


def discover_arms() -> list[ArmInfo]:
    """The configured arms with display names + serials (computed once, then cached)."""
    return list(_compose()[0])


def runtime_is_sim() -> bool:
    """Whether ``runtime.sim`` selects simulated sources (no hardware)."""
    return _compose()[1]


def discover_factr() -> object:
    """Composed FACTR config (``cfg.factr``): one server entry per leader side.

    Returned as-is (an OmegaConf node) for :meth:`FactrClient.from_config`; the
    dashboard emitter uses it to stream live leader joint positions.
    """
    return _compose()[2]


def discover_conventions() -> dict:
    """Per-side FACTR-leader → Rizon joint conventions (``{side: JointConventionCfg}``).

    Plain dataclass instances (resolved via ``OmegaConf.to_object``) so
    :func:`~dual_flexiv_control.control.convention.convert_factr_to_rizon` can read
    ``offsets_deg`` / ``sign_flip_joints`` / ``wrap_deg`` / ``drop_trailing`` directly.
    Used to render each leader's *commanded* Rizon config (the teleop ghost).
    """
    return dict(_compose()[3])


def reset() -> None:
    """Drop the cached config snapshot so the next call re-composes.

    Called by the dashboard's *Reset services* action: forces a fresh Hydra
    compose on the next :func:`discover_arms` / :func:`runtime_is_sim` / … call,
    so edits to ``conf`` (or a changed ``runtime.sim``) take effect without a
    process restart.
    """
    global _SNAPSHOT
    with _LOCK:
        _SNAPSHOT = None
    global _LEADER_CLIENT
    with _LEADER_LOCK:
        if _LEADER_CLIENT is not None:
            try:
                _LEADER_CLIENT.close()
            except Exception:  # noqa: BLE001 - already broken; just drop it
                pass
        _LEADER_CLIENT = None


def _compose() -> tuple[tuple[ArmInfo, ...], bool, object, dict]:
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


def _compose_uncached() -> tuple[tuple[ArmInfo, ...], bool, object, dict]:
    from hydra import compose
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config", overrides=compose_overrides())

    arms: list[ArmInfo] = []
    conventions: dict = {}
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
        conv = getattr(arm, "convention", None)
        if conv is not None:
            conventions[str(side)] = OmegaConf.to_object(conv)
    sim = bool(getattr(cfg.runtime, "sim", False))
    return tuple(arms), sim, cfg.factr, conventions


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
        code, estop, control = live
        return ArmStatus(
            arm, _label_for_code(code), bool(estop), "live",
            control_active=None if control is None else bool(control),
        )
    return ArmStatus(arm, mode="disconnected", estop_pressed=None, source="disconnected")


def configured_leader_sides() -> list[str]:
    """The teleop-leader sides the active rig serves (keys of ``cfg.factr.servers``)."""
    servers = getattr(discover_factr(), "servers", None)
    return list(servers) if servers else []


def _leader_client():
    """The cached read-only leader client (built lazily), or None if none composes.

    Honours ``runtime.sim``; construction only builds the per-side HTTP clients (no
    connection), so this returns a client whenever the rig configures any FACTR
    server. :func:`reset` drops it so a rig / sim change rebuilds it.
    """
    from ..interfaces.factr.client import FactrClient

    global _LEADER_CLIENT
    with _LEADER_LOCK:
        if _LEADER_CLIENT is None:
            try:
                client = FactrClient.from_config(discover_factr(), sim=runtime_is_sim())
            except Exception:  # noqa: BLE001 - no/invalid FACTR config -> no leaders
                return None
            _LEADER_CLIENT = client if client.sides else None
        return _LEADER_CLIENT


def read_leader_status(side: str) -> LeaderStatus:
    """Single read-only reachability probe for one FACTR teleop leader.

    One HTTP GET (never a robot connection): **reachable** if the leader's server
    answers with a valid ``DoF+1`` sample, else **disconnected**. In ``runtime.sim``
    the reading is synthetic (``sim=True``) — the leader always reads reachable, so
    the row flags it rather than implying live hardware.
    """
    name = side.capitalize()
    sim = runtime_is_sim()
    client = _leader_client()
    if client is None or side not in getattr(client, "sides", []):
        return LeaderStatus(side, name, reachable=False, dof=0, gripper=None, sim=sim)
    with _LEADER_LOCK:  # serialize the shared keep-alive connection across fragments
        try:
            jp = np.asarray(client.get_joint_positions_for(side), dtype=float).ravel()
        except Exception:  # noqa: BLE001 - leader down / bad response -> disconnected
            return LeaderStatus(side, name, reachable=False, dof=0, gripper=None, sim=sim)
    grip = float(jp[-1]) if jp.size else None
    return LeaderStatus(side, name, reachable=True, dof=max(0, jp.size - 1), gripper=grip, sim=sim)


def _runtime_root(runtime_dir: str | None) -> Path:
    rd = runtime_dir or os.environ.get("DFC_RUNTIME_DIR", "runtime")
    return Path(rd if os.path.isabs(rd) else os.path.join(os.getcwd(), rd))


def _read_live_stream_newest(stream: str, runtime_dir: str | None) -> np.ndarray | None:
    """Newest sample vector of a shared-memory ``stream`` across the latest runs, or None.

    Scans the runtime run dirs newest-first, attaches the stream if present, and
    returns its most recent vector. None means the streams stack is unavailable, no
    run publishes the stream, or its buffer is empty — i.e. nothing is currently
    producing it. Read-only shared-memory access: never opens a robot connection, so
    it cannot conflict with the system that owns the arm.
    """
    root = _runtime_root(runtime_dir)
    if not root.is_dir():
        return None
    try:
        from dual_flexiv_control.streams import StreamReader
        from dual_flexiv_control.streams import StreamRegistry
    except Exception:  # noqa: BLE001 - streams stack unavailable -> no live data
        return None

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
                    return np.asarray(samples.newest)
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - dead run / lapped buffer -> try next
            continue
    return None


def _read_live_status(
    arm: ArmInfo, runtime_dir: str | None
) -> tuple[float, float, float | None] | None:
    vec = _read_live_stream_newest(STATUS_STREAM.format(side=arm.side), runtime_dir)
    if vec is None or len(vec) < 2:
        return None
    control = float(vec[2]) if len(vec) >= 3 else None  # 2-wide: pre-session stream
    return float(vec[0]), float(vec[1]), control


def read_live_joint_positions(side: str, runtime_dir: str | None = None) -> np.ndarray | None:
    """Latest measured joint positions ``<side>/q`` from the running system, or None.

    The real ``FlexivInterface`` publishes ``<side>/q`` (link-side joint positions,
    rad). Returns the newest vector when a system is running and producing it, else
    None — nothing connected (e.g. control boxes off) → the caller shows no motion
    for that arm rather than fabricating it.
    """
    return _read_live_stream_newest(JOINT_POS_STREAM.format(side=side), runtime_dir)


def read_live_stream(name: str, runtime_dir: str | None = None) -> np.ndarray | None:
    """Newest sample of any live shared-memory stream by full name, or None.

    Generic read-only sibling of :func:`read_live_joint_positions` for the other
    proprio streams a running system publishes (``<side>/dq``, ``<side>/tau``,
    ``<side>/wrench``, ``<side>/eef``, ``<side>/eef_vel``, …). Never opens a robot
    connection; None simply means nothing is producing that stream right now.
    """
    return _read_live_stream_newest(name, runtime_dir)


def read_live_horizon_q(side: str, runtime_dir: str | None = None) -> np.ndarray | None:
    """Latest policy horizon-end joint target ``eval/<side>/q_horizon``, or None.

    Only a running **eval** system publishes this (once per inference); None
    outside eval runs → the caller hides the horizon-target ghost.
    """
    return _read_live_stream_newest(HORIZON_STREAM.format(side=side), runtime_dir)


def read_live_horizon_eef(side: str, runtime_dir: str | None = None) -> np.ndarray | None:
    """Latest estimated horizon-end TCP position ``eval/<side>/eef_horizon``, or None.

    Base-frame ``[x y z]``, published (once per inference) by an eval run whose
    arm is under a cartesian control kind; None otherwise → the caller hides the
    predicted-EEF trace.
    """
    return _read_live_stream_newest(EEF_HORIZON_STREAM.format(side=side), runtime_dir)


def read_live_policy_comm(runtime_dir: str | None = None):
    """All buffered policy-server comm events, or None outside eval runs.

    Returns the stream's :class:`~dual_flexiv_control.streams.ring.Samples`
    (``[kind, seq, elapsed_s]`` rows with per-sample monotonic timestamps and
    gap-free ring sequence numbers) so the caller can pick out the events it
    has not surfaced yet. None means no running eval publishes the stream.
    """
    root = _runtime_root(runtime_dir)
    if not root.is_dir():
        return None
    try:
        from dual_flexiv_control.streams import StreamReader
        from dual_flexiv_control.streams import StreamRegistry
    except Exception:  # noqa: BLE001 - streams stack unavailable -> no live data
        return None
    run_dirs = sorted(
        (p for p in root.iterdir() if (p / "streams").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        try:
            entry = StreamRegistry(str(root), run_dir.name).get(POLICY_COMM_STREAM)
            if entry is None:
                continue
            reader = StreamReader.attach(entry)
            try:
                samples = reader.last(reader.capacity)
                if samples.n > 0:
                    return samples
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
