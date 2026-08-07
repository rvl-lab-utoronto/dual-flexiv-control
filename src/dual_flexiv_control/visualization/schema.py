"""Typed stream contract shared by live visualization backends.

Visualization is deliberately downstream of the shared-memory registry.  This
module is the small, backend-free mapping from producer-owned stream names to
viewer semantics; neither Viser nor the deprecated Rerun adapter gets to invent
another data path.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..interfaces.factr.interface import TELEMETRY_SCALAR_FIELDS
from ..interfaces.factr.interface import TELEMETRY_VECTOR_FIELDS
from ..proprio import PROPRIO_SIGNALS

SIDES = ("left", "right")

PROPRIO_TITLES = {
    "q": "Joint position q (rad)",
    "dq": "Joint velocity dq (rad/s)",
    "tau": "Measured joint torque τ (Nm)",
    "tau_ext": "External joint torque τ_ext (Nm)",
    "wrench": "TCP wrench (N, Nm)",
    "eef": "TCP position (m)",
    "eef_vel": "TCP twist (m/s, rad/s)",
}
FACTR_TITLES = {
    "q": "Leader joint q — DFC/Rizon convention (rad)",
    "grip": "Leader gripper",
    "raw": "Raw leader position",
    **{field: field.replace("_", " ") for field in TELEMETRY_VECTOR_FIELDS},
    **{field: field.replace("_", " ") for field in TELEMETRY_SCALAR_FIELDS},
}
EEF_COLORS = {"left": (80, 160, 255), "right": (255, 140, 80)}


@dataclass(frozen=True, slots=True)
class StreamRoute:
    """A validated input stream route understood by live visualizers."""

    kind: str
    side: str | None
    signal: str


def parse_stream_name(name: str) -> StreamRoute:
    """Classify a supported stream name, rejecting untyped flattened data."""
    parts = name.split("/")
    if len(parts) == 2 and parts[0] in SIDES and parts[1] in PROPRIO_SIGNALS:
        return StreamRoute("proprio", parts[0], parts[1])
    if len(parts) == 2 and parts[0] == "factr" and parts[1] in SIDES:
        return StreamRoute("factr", parts[1], "leader")
    if len(parts) == 3 and parts[:2] == ["factr", "raw"] and parts[2] in SIDES:
        return StreamRoute("factr", parts[2], "raw")
    if (
        len(parts) == 4
        and parts[:2] == ["factr", "telemetry"]
        and parts[2] in SIDES
        and parts[3] in TELEMETRY_VECTOR_FIELDS + TELEMETRY_SCALAR_FIELDS
    ):
        return StreamRoute("factr", parts[2], parts[3])
    if len(parts) == 3 and parts[0] == "eval" and parts[1] in SIDES:
        if parts[2] in ("q_horizon", "eef_horizon"):
            return StreamRoute("horizon", parts[1], parts[2])
    if name == "eval/policy_comm":
        return StreamRoute("policy", None, "comm")
    if len(parts) == 3 and parts[0] == "cam":
        return StreamRoute("camera", parts[1], parts[2])
    raise ValueError(f"no visualization adapter for stream {name!r}")


def entry_matches(route: StreamRoute, entry) -> bool:
    """Whether registry metadata satisfies the routed visualization contract."""
    if route.kind == "camera":
        return entry.dim > 0
    if entry.dtype not in ("float32", "float64") or entry.dim <= 0:
        return False
    if route.kind == "proprio" and route.signal == "eef":
        return entry.dim >= 3
    if route.kind == "policy":
        return entry.dim == 3
    return True


def default_stream_names(
    sides: tuple[str, ...] = SIDES,
    camera_names: tuple[str, ...] = (),
) -> list[str]:
    """Every typed stream used by the live viewer.

    This includes all follower proprioception and all FACTR telemetry, not only
    the fields currently used to pose a model.  Camera RGB/depth names are added
    from composed rig metadata by the service.
    """
    names = [f"{side}/{signal}" for side in sides for signal in PROPRIO_SIGNALS]
    for side in sides:
        names.extend((f"factr/{side}", f"factr/raw/{side}"))
        names.extend(
            f"factr/telemetry/{side}/{field}"
            for field in TELEMETRY_VECTOR_FIELDS + TELEMETRY_SCALAR_FIELDS
        )
        names.extend((f"eval/{side}/q_horizon", f"eval/{side}/eef_horizon"))
    names.append("eval/policy_comm")
    for camera in camera_names:
        names.extend((f"cam/{camera}/left", f"cam/{camera}/depth"))
    return names


def plot_title(route: StreamRoute) -> str:
    if route.kind == "proprio":
        return PROPRIO_TITLES[route.signal]
    if route.kind == "factr":
        return FACTR_TITLES.get(route.signal, route.signal.replace("_", " "))
    return route.signal.replace("_", " ")
