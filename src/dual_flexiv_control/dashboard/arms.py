"""Per-arm identity + read-only operation status for the dashboard.

Display **names** come from the composed Hydra config (`arms.<side>.name` in
`conf/config.yaml`, customizable per arm; falls back to the side).

The operational state (Auto / Manual / fault), actual RDK control mode, servo
state, and E-stop are read-only telemetry from ``Robot``. They are published on
the per-arm ``<side>/status`` stream; otherwise the arm reads as disconnected.

This is a **monitoring view, not a safety interlock.**
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDES = ("left", "right")

#: Per-arm status stream: ``[operational_status_code, estop_pressed, control_active,
#: servo_enabled, mode_code]``.
#: ``operational_status_code`` is ``flexivrdk.OperationalStatus(...).value``;
#: ``control_active`` is 1.0 while the arm is inside a control session (a
#: collection/eval run) and 0.0 while idle (viewing). Older 2-wide streams (no
#: ``control_active``) are tolerated.
STATUS_STREAM = "{side}/status"

#: Freshness gate for :data:`STATUS_STREAM`: the newest sample must be at most this
#: old to count as live. The arm node writes status every telemetry tick (at least
#: ``arm.rate_hz``), so anything older means the producer is gone — a dead run's
#: frozen last sample must read as disconnected, not as a live mode/E-stop state.
STATUS_MAX_AGE_S = 2.0

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
    servo_enabled: bool | None = None
    operational_status: str = "unknown"


@dataclass(frozen=True)
class LeaderStatus:
    """Reachability of one FACTR teleop leader (the arm the operator moves)."""

    side: str  # "left" | "right"
    name: str  # display name (side, capitalized)
    reachable: bool  # did the leader's server answer?
    dof: int  # arm joints (trailing gripper dropped), when reachable
    gripper: float | None  # raw trailing gripper reading, when reachable
    sim: bool  # runtime.sim -> the reading is synthetic, not real hardware
    grav_comp_enabled: bool | None = None  # None = status endpoint unavailable
    force_gain: float | None = None  # live master output multiplier, 0..1
    force_gain_target: float | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
#: Cached single compose of the bits the dashboard needs: arms, sim flag, FACTR cfg,
#: per-side leader→Rizon joint conventions.
_SNAPSHOT: tuple[tuple[ArmInfo, ...], bool, object, dict] | None = None
_LEADER_STATUS_LOCK = threading.Lock()
_LEADER_STATUS_CLIENT = None
_LEADER_CONVENTION_LOCK = threading.Lock()
_LEADER_CONVENTIONS: dict = {}
_LEADER_CONVENTION_STATUS: dict[str, dict] = {}
#: The rig the dashboard composes against (``rig=<name>`` override); None = the
#: config default. Owned here so every dashboard compose (arms, cameras, storage)
#: follows the same selection.
_ACTIVE_RIG: str | None = None



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
    """Poll leader conventions without making FACTR a dashboard dependency.

    A failed/unavailable endpoint returns the last valid per-side value (or omits the
    side before the first success). Dashboard reruns call this again and recover
    automatically when FACTR appears.
    """
    from ..configs import JointConventionCfg
    from ..interfaces.factr.client import FactrClient
    from ..interfaces.factr.interface import FactrInterface

    _arms_info, sim, factr, _legacy = _compose()
    if sim:
        return {
            side: JointConventionCfg(
                offsets_deg=[0.0] * (int(server.dof) - 1), sign_flip_joints=[],
                drop_trailing=1, wrap_deg=True, gripper_open=0.0, gripper_closed=1.0,
            )
            for side, server in factr.servers.items()
        }
    discovered = {}
    attempts = {}
    client = FactrClient.from_config(factr, sim=False)
    try:
        for side, server in factr.servers.items():
            attempted_at = time.time()
            try:
                data = client.get_diagnostics_for(side)
                if data.get("available") is True:
                    discovered[side] = FactrInterface._convention_from_diagnostics(
                        side, int(server.dof), data
                    )
                    attempts[side] = {
                        "state": "live", "attempted_at": attempted_at,
                        "succeeded_at": attempted_at, "message": "calibration received",
                    }
                else:
                    attempts[side] = {
                        "state": "waiting", "attempted_at": attempted_at,
                        "message": "FACTR is starting; diagnostics not available yet",
                    }
            except Exception as exc:  # FACTR is optional; poll again next fragment rerun.
                attempts[side] = {
                    "state": "unreachable", "attempted_at": attempted_at,
                    "message": str(exc),
                }
    finally:
        client.close()
    with _LEADER_CONVENTION_LOCK:
        _LEADER_CONVENTIONS.update(discovered)
        for side, attempt in attempts.items():
            previous = _LEADER_CONVENTION_STATUS.get(side, {})
            if side not in discovered and previous.get("succeeded_at") is not None:
                attempt["succeeded_at"] = previous["succeeded_at"]
                attempt["state"] = "stale"
                attempt["message"] += "; retaining last valid calibration"
            _LEADER_CONVENTION_STATUS[side] = attempt
        return dict(_LEADER_CONVENTIONS)


def convention_poll_status(side: str) -> dict:
    """Latest non-blocking dashboard diagnostics-poll state for one leader."""
    with _LEADER_CONVENTION_LOCK:
        return dict(_LEADER_CONVENTION_STATUS.get(side, {
            "state": "waiting", "message": "waiting for first diagnostics poll",
        }))


def reset() -> None:
    """Drop the cached config snapshot so the next call re-composes.

    Called by the dashboard's *Reset services* action: forces a fresh Hydra
    compose on the next :func:`discover_arms` / :func:`runtime_is_sim` / … call,
    so edits to ``conf`` (or a changed ``runtime.sim``) take effect without a
    process restart.
    """
    global _SNAPSHOT, _LEADER_STATUS_CLIENT
    with _LOCK:
        _SNAPSHOT = None
    with _LEADER_STATUS_LOCK:
        if _LEADER_STATUS_CLIENT is not None:
            _LEADER_STATUS_CLIENT.close()
            _LEADER_STATUS_CLIENT = None
    with _LEADER_CONVENTION_LOCK:
        _LEADER_CONVENTIONS.clear()
        _LEADER_CONVENTION_STATUS.clear()
        now = time.time()
        for side in configured_leader_sides():
            _LEADER_CONVENTION_STATUS[side] = {
                "state": "reset", "attempted_at": now,
                "message": "calibration cache reset; waiting for next 2 s poll",
            }


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
    return tuple(arms), sim, cfg.factr, {}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def read_arm_status(arm: ArmInfo, runtime_dir: str | None = None) -> ArmStatus:
    """Live operation mode + E-stop for ``arm``, or **disconnected** if none is published.

    "Connected" means a running ``FlexivInterface`` is publishing this arm's
    ``<side>/status`` stream with a FRESH sample (:data:`STATUS_MAX_AGE_S`); until
    then (or if its run ends / its producer dies) the arm reads as disconnected
    with unknown E-stop — never as the frozen last state of a dead run.
    """
    live = _read_live_status(arm, runtime_dir)
    if live is not None:
        code, estop, control, servo, actual_mode = live
        return ArmStatus(
            arm, _mode_label(actual_mode), bool(estop), "live",
            control_active=None if control is None else bool(control),
            servo_enabled=None if servo is None else bool(servo),
            operational_status=_label_for_code(code),
        )
    return ArmStatus(
        arm, mode="unknown", estop_pressed=None, source="disconnected",
        operational_status="disconnected",
    )


def configured_leader_sides() -> list[str]:
    """The teleop-leader sides the active rig serves (keys of ``cfg.factr.servers``)."""
    servers = getattr(discover_factr(), "servers", None)
    return list(servers) if servers else []


def factr_max_age_s() -> float:
    """The composed freshness gate for leader streams (``factr.max_age_s``)."""
    return float(getattr(discover_factr(), "max_age_s", 0.5))


def read_live_leader(side: str, runtime_dir: str | None = None) -> np.ndarray | None:
    """Newest FRESH sample of one leader's ``factr/<side>`` stream, or None.

    The single source of truth for leader data: the same stream the control loop
    converts into setpoints (published by the ``FactrInterface`` producer). None
    when no system publishes the stream, or its newest sample is older than
    ``factr.max_age_s`` — a stale leader must read as disconnected, not as its
    frozen last pose.
    """
    from ..interfaces.factr import factr_stream_name

    sample = _read_live_stream_sample(factr_stream_name(side), runtime_dir)
    if sample is None:
        return None
    vec, t_ns = sample
    import time as _time

    if (_time.monotonic_ns() - t_ns) > factr_max_age_s() * 1e9:
        return None
    return vec


def read_leader_status(side: str) -> LeaderStatus:
    """Read-only status of one FACTR teleop leader, from its ``factr/<side>`` stream.

    **Reachable** iff the stream carries a fresh ``DoF+1`` sample — exactly the
    condition under which teleop is being fed, judged on the same samples control
    reads (no separate HTTP probe that could disagree). Disconnected covers: no
    running session, the producer down, or the leader server stale/unreachable.
    In ``runtime.sim`` the producer fabricates the readings (``sim=True``) — the
    leader always reads reachable, so the row flags it rather than implying live
    hardware.
    """
    name = side.capitalize()
    sim = runtime_is_sim()
    if side not in configured_leader_sides():
        return LeaderStatus(side, name, reachable=False, dof=0, gripper=None, sim=sim)
    jp = read_live_leader(side)
    if jp is None or jp.size == 0:
        return LeaderStatus(side, name, reachable=False, dof=0, gripper=None, sim=sim)
    jp = np.asarray(jp, dtype=float).ravel()
    grip = float(jp[-1]) if jp.size else None
    enabled = None
    gain = None
    gain_target = None
    if sim:
        enabled = False
        gain = 0.0
        gain_target = 0.0
    else:
        payload = read_leader_grav_comp_status(side)
        if payload is not None:
            try:
                enabled = bool(payload["grav_comp_enabled"])
                gain = float(payload["force_gain"])
                gain_target = float(payload.get("force_gain_target", gain))
            except (KeyError, TypeError, ValueError):
                enabled = None
                gain = None
                gain_target = None
    return LeaderStatus(
        side, name, reachable=True, dof=max(0, jp.size - 1), gripper=grip, sim=sim,
        grav_comp_enabled=enabled, force_gain=gain, force_gain_target=gain_target,
    )


def read_leader_grav_comp_status(side: str) -> dict | None:
    """Read the authoritative per-leader gain state without changing it."""
    from ..interfaces.factr.client import FactrClient

    global _LEADER_STATUS_CLIENT
    with _LEADER_STATUS_LOCK:
        try:
            if _LEADER_STATUS_CLIENT is None:
                _LEADER_STATUS_CLIENT = FactrClient.from_config(discover_factr(), sim=False)
            if side not in _LEADER_STATUS_CLIENT.sides:
                return None
            return _LEADER_STATUS_CLIENT.get_status_for(side)
        except Exception:  # noqa: BLE001 - status is optional; stream still proves reachability
            return None


def grav_comp_display(status: dict | None) -> tuple[str, str]:
    """Return the dashboard icon and label for a leader's actual output gain."""
    if status is None:
        return "🟡", ":gray[grav comp unknown]"
    try:
        gain = float(status["force_gain"])
        target = float(status["force_gain_target"])
        enabled = status["grav_comp_enabled"] is True
    except (KeyError, TypeError, ValueError):
        return "🟡", ":gray[grav comp unknown]"
    if enabled:
        return "🟢", f":green[grav comp enabled] · gain `{gain:.2f}`"
    if target >= 0.99 and gain < 0.99:
        return "🟡", f":orange[enabling grav comp] · gain `{gain:.2f}`"
    if target <= 0.01 and gain > 0.01:
        return "🟡", f":orange[disabling grav comp] · gain `{gain:.2f}`"
    return "⚫", f":gray[grav comp disabled (limp)] · gain `{gain:.2f}`"


def _runtime_root(runtime_dir: str | None) -> Path:
    rd = runtime_dir or os.environ.get("DFC_RUNTIME_DIR", "runtime")
    return Path(rd if os.path.isabs(rd) else os.path.join(os.getcwd(), rd))


def _read_live_stream_sample(
    stream: str, runtime_dir: str | None, max_age_s: float | None = None
) -> tuple[np.ndarray, int] | None:
    """Newest ``(vector, t_ns)`` of a shared-memory ``stream`` across the latest runs.

    Scans the runtime run dirs newest-first, attaches the stream if present, and
    returns its most recent vector with its producer timestamp (monotonic ns —
    comparable across processes, for freshness gating). With ``max_age_s`` set, a
    candidate whose newest sample is older is skipped like an absent stream (the
    scan continues into older run dirs), so a dead run's frozen leftover cannot
    shadow a live producer. None means the streams stack is unavailable, no run
    publishes the stream (fresh enough), or its buffer is empty — i.e. nothing is
    currently producing it. Read-only shared-memory access: never opens a robot
    connection, so it cannot conflict with the system that owns the arm.
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
                    t_ns = int(samples.newest_t_ns)
                    if (
                        max_age_s is not None
                        and (time.monotonic_ns() - t_ns) > max_age_s * 1e9
                    ):
                        continue  # frozen leftover of a dead producer
                    return np.asarray(samples.newest), t_ns
            finally:
                reader.close()
        except Exception:  # noqa: BLE001 - dead run / lapped buffer -> try next
            continue
    return None


def _read_live_stream_newest(stream: str, runtime_dir: str | None) -> np.ndarray | None:
    """Newest sample vector of a live ``stream``, or None (see `_read_live_stream_sample`)."""
    sample = _read_live_stream_sample(stream, runtime_dir)
    return None if sample is None else sample[0]


def _read_live_status(
    arm: ArmInfo, runtime_dir: str | None
) -> tuple[float, float, float | None, float | None, float | None] | None:
    sample = _read_live_stream_sample(
        STATUS_STREAM.format(side=arm.side), runtime_dir, max_age_s=STATUS_MAX_AGE_S
    )
    vec = None if sample is None else sample[0]
    if vec is None or len(vec) < 2:
        return None
    control = float(vec[2]) if len(vec) >= 3 else None  # 2-wide: pre-session stream
    servo = float(vec[3]) if len(vec) >= 4 else None
    actual_mode = float(vec[4]) if len(vec) >= 5 else None
    return float(vec[0]), float(vec[1]), control, servo, actual_mode


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


def _mode_label(code: float | None) -> str:
    if code is None:
        return "unknown"
    try:
        import flexivrdk

        name = flexivrdk.Mode(int(code)).name
    except Exception:  # noqa: BLE001 - SDK absent / unknown future mode
        return f"mode {int(code)}"
    return name.replace("NRT_", "NRT ").replace("_", " ").title()
