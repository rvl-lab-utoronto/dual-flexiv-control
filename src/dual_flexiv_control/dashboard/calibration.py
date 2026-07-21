"""Leader-arm joint-offset calibration for the dashboard's Calibrate tab.

The FACTR leader→Rizon follower mapping (:func:`convert_factr_to_rizon`) adds a
per-joint ``offsets_deg`` before the sign flip and wrap. Those offsets are the one
part of the convention that drifts per physical leader and must be measured (the
right arm's especially — do NOT assume symmetry with the left). This module drives
the measurement **one joint at a time**: straighten a single link, capture it, and
solve for that joint's offset (repeat per joint). Capturing per joint — rather than
the whole pose at once — lets the operator align and verify each link on its own.

Solving one joint: at the target the follower joint reads 0, and the sign flip
multiplies zero, so ``offset[j] = wrap(-degrees(q_leader_arm[j]))`` — independent of
the other joints (see :func:`~dual_flexiv_control.control.offsets_from_straight_pose`,
from which we take the one index). Un-captured joints keep their existing config
offset, so a partial calibration is well-defined.

A **live Rerun view** poses a Flexiv arm at the follower config the current leader
maps to *with the offsets captured so far* (solid), against the straight/home target
(translucent ghost): when a joint is correctly calibrated and physically straight,
its link snaps onto the ghost, so the operator can see the result align. The view is
its own gRPC recording embedded in the shared web viewer (same approach as replay).

This module *reads* the FACTR HTTP endpoint (honouring ``runtime.sim``) and renders —
it never opens a robot connection. The only file it touches is the rig YAML, and only
via the explicit **Sync to file** action (:func:`apply_to_rig`), which merges the
measured convention in place (preserving comments + gripper endpoints). Mirrors
``scripts/factr_gripper_calibrate.py`` (the gripper-endpoint sibling), for the arm
joints and surfaced as a dashboard tab.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from dataclasses import replace
from urllib.parse import quote

import numpy as np

from . import arms as _arms
from .viewer import serve_grpc_checked

CALIB_APP_ID = "dual-flexiv-calibration"
DEFAULT_CALIB_GRPC_PORT = 9881  # replay uses 9880; metrics 9876

_LOCK = threading.Lock()
#: Cached leader client (keep-alive HTTP), rebuilt on demand — avoids reconnect
#: churn under the tab's live-render fragment. Dropped by :func:`reset`.
_CLIENT = None


# ---------------------------------------------------------------------------
# Config + leader reads
# ---------------------------------------------------------------------------


def configured_leader_sides() -> list[str]:
    """The leader sides the active rig serves (keys of ``cfg.factr.servers``)."""
    factr = _arms.discover_factr()
    servers = getattr(factr, "servers", None)
    return list(servers) if servers else []


def current_convention(side: str):
    """Latest leader convention, or an uncalibrated UI placeholder while unavailable."""
    from ..configs import JointConventionCfg

    conventions = _arms.discover_conventions()
    return conventions.get(side) or JointConventionCfg()


def follower_dof(side: str) -> int:
    """Follower joint count = FACTR server DoF minus the dropped trailing gripper(s).

    This — not the length of the configured ``offsets_deg`` — is the number of joints
    the per-joint UI must expose and the length ``convert_factr_to_rizon`` consumes, so
    a mis-sized ``offsets_deg`` in config cannot hide joints or trigger a broadcast error.
    """
    conv = current_convention(side)
    servers = getattr(_arms.discover_factr(), "servers", None) or {}
    server = servers.get(side)
    dof = int(getattr(server, "dof", 8)) if server is not None else 8
    return max(1, dof - int(conv.drop_trailing))


def initial_offsets(side: str) -> list[float]:
    """The side's starting offsets, length-normalized to the follower DoF.

    Seeded from config ``offsets_deg`` but padded with zeros / truncated to the follower
    joint count (:func:`follower_dof`) so the per-joint UI always matches the arm even if
    the configured list is a different length.
    """
    dof = follower_dof(side)
    cfg = [float(o) for o in current_convention(side).offsets_deg]
    return (cfg + [0.0] * dof)[:dof]


def _get_client():
    """The cached FACTR client (built lazily), or ``None`` if unavailable.

    Honours ``runtime.sim``. Cached so the live-render fragment reuses one keep-alive
    connection instead of reconnecting each tick; :func:`reset` drops it.
    """
    from ..interfaces.factr.client import FactrClient

    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            try:
                client = FactrClient.from_config(_arms.discover_factr(), sim=_arms.runtime_is_sim())
            except Exception:  # noqa: BLE001 - no/invalid FACTR config -> no source
                return None
            _CLIENT = client if client.sides else None
        return _CLIENT


def read_leader(side: str, samples: int = 5) -> np.ndarray:
    """A robust leader sample (rad, ``DoF+1``): the per-joint median of a few reads.

    Raises :class:`~dual_flexiv_control.interfaces.factr.client.FactrError` (or
    ``RuntimeError`` if no client composes / the side is not served) so the caller
    can surface why the leader could not be read.
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("no FACTR leader is configured for the active rig")
    if side not in client.sides:
        raise RuntimeError(f"leader {side!r} is not configured (have {client.sides})")
    reads = [
        np.asarray(client.get_joint_positions_for(side), dtype=np.float64).ravel()
        for _ in range(max(1, samples))
    ]
    return np.median(np.vstack(reads), axis=0)


def _arm_joints_deg(q_leader: np.ndarray, conv) -> np.ndarray:
    """Leader arm joints in degrees (trailing gripper dropped per the convention)."""
    q = np.asarray(q_leader, dtype=np.float64).ravel()
    if conv.drop_trailing:
        q = q[: len(q) - conv.drop_trailing]
    return np.degrees(q)


def read_gripper(side: str, samples: int = 5) -> float:
    """The current raw FACTR gripper value — the trailing element of the DoF+1 vector.

    Median of a few reads (robust to jitter). This is the un-normalized servo angle in
    radians that ``gripper_open``/``gripper_closed`` calibrate; record it at the fully
    open and fully closed trigger (see
    :func:`~dual_flexiv_control.control.normalize_gripper`).
    """
    return float(read_leader(side, samples)[-1])


def gripper_preview(open_v, closed_v):
    """Raw→normalized ``[(label, raw, frac), …]`` at open/mid/closed, or None if unset.

    Uses the real :func:`~dual_flexiv_control.control.normalize_gripper` so the tab's
    sanity check matches what recording will produce (open→0, mid→0.5, closed→1).
    """
    if open_v is None or closed_v is None or float(open_v) == float(closed_v):
        return None
    from ..configs import JointConventionCfg
    from ..control.convention import normalize_gripper

    conv = JointConventionCfg(gripper_open=float(open_v), gripper_closed=float(closed_v))
    pts = [("open", float(open_v)), ("mid", (float(open_v) + float(closed_v)) / 2.0), ("closed", float(closed_v))]
    return [(lab, raw, normalize_gripper(raw, conv)) for lab, raw in pts]


# ---------------------------------------------------------------------------
# Per-joint capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointCapture:
    joint: int
    captured_deg: float   # the leader arm joint (deg) at capture — should be "straight"
    offset_deg: float     # solved offset that maps this joint to 0


def capture_joint(side: str, joint: int, samples: int = 5) -> JointCapture:
    """Read the leader and solve ``offsets_deg[joint]`` from the current (straight) link.

    Only this one joint is measured; the operator has straightened just this link.
    """
    from ..control.convention import offsets_from_straight_pose

    conv = current_convention(side)
    q_leader = read_leader(side, samples)
    offsets = offsets_from_straight_pose(q_leader, conv)
    if not 0 <= joint < len(offsets):
        raise RuntimeError(f"joint {joint} out of range for a {len(offsets)}-DoF leader")
    return JointCapture(
        joint=joint,
        captured_deg=float(_arm_joints_deg(q_leader, conv)[joint]),
        offset_deg=offsets[joint],
    )


def commanded_follower_q(
    side: str, offsets_deg: list[float], sign_flip_joints: list[int]
) -> np.ndarray:
    """The follower config the *current* leader maps to under the in-progress convention.

    A single fast read (no medianing) for the live view, converted with the working
    ``offsets_deg`` **and** ``sign_flip_joints`` — so toggling a joint's sign visibly
    mirrors that link in the view, letting the operator confirm the flip direction.
    """
    from ..control.convention import convert_factr_to_rizon

    conv = replace(
        current_convention(side),
        offsets_deg=list(offsets_deg),
        sign_flip_joints=list(sign_flip_joints),
    )
    return convert_factr_to_rizon(read_leader(side, samples=1), conv)


# ---------------------------------------------------------------------------
# Config formatting (paste into conf/rig)
# ---------------------------------------------------------------------------


def _fmt_list(values, decimals: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def format_yaml(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
) -> str:
    """A FACTR leader-YAML initialization snippet for the measured convention."""
    lines = [
        "arm_teleop:",
        "  initialization:",
        f"    dfc_raw_offsets_deg: {_fmt_list(offsets_deg)}",
        f"    dfc_sign_flip_joints: {list(sign_flip_joints)}",
        "    dfc_wrap_deg: true",
        "    dfc_drop_trailing: 1",
    ]
    if gripper_open is not None:
        lines.append(f"    dfc_gripper_open: {float(gripper_open):.4f}")
    if gripper_closed is not None:
        lines.append(f"    dfc_gripper_closed: {float(gripper_closed):.4f}")
    return "\n".join(lines)


def format_overrides(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
) -> str:
    """Removed: leader conversion cannot be overridden through follower Hydra config."""
    raise RuntimeError("FACTR conversion is leader-owned; edit the FACTR arm YAML")
    offsets = "[" + ",".join(f"{v:.2f}" for v in offsets_deg) + "]"
    flips = "[" + ",".join(str(int(j)) for j in sign_flip_joints) + "]"
    parts = [
        f"arms.{side}.convention.offsets_deg='{offsets}'",
        f"arms.{side}.convention.sign_flip_joints='{flips}'",
    ]
    if gripper_open is not None:
        parts.append(f"arms.{side}.convention.gripper_open={float(gripper_open):.4f}")
    if gripper_closed is not None:
        parts.append(f"arms.{side}.convention.gripper_closed={float(gripper_closed):.4f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Write-back: sync the measured convention into the rig YAML in place
# ---------------------------------------------------------------------------


def _default_rig(conf_dir) -> str:
    """The rig ``conf/config.yaml`` composes by default (its ``defaults: - rig: …``)."""
    import yaml
    from pathlib import Path

    data = yaml.safe_load(Path(conf_dir, "config.yaml").read_text()) or {}
    for item in data.get("defaults", []):
        if isinstance(item, dict) and "rig" in item:
            return str(item["rig"])
    raise RuntimeError("could not determine the default rig from conf/config.yaml")


def rig_path(rig: str | None = None):
    """Filesystem path of the rig YAML the dashboard composes against.

    ``rig`` defaults to the dashboard's active rig pin, else the config default.
    """
    import dual_flexiv_control
    from pathlib import Path

    conf = Path(dual_flexiv_control.__file__).resolve().parent / "conf"
    name = rig or _arms.active_rig() or _default_rig(conf)
    return conf / "rig" / f"{name}.yaml"


def _fmt_conv_inline(conv: dict) -> str:
    """One YAML flow-map for a convention dict, matching the rig files' inline style."""
    parts = []
    for k, v in conv.items():
        if isinstance(v, list):
            if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
                s = "[" + ", ".join(str(int(x)) for x in v) + "]"
            else:
                s = "[" + ", ".join(f"{float(x):.2f}" for x in v) + "]"
        elif isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, (int, float)):
            s = f"{v}"
        else:
            s = str(v)
        parts.append(f"{k}: {s}")
    return "{ " + ", ".join(parts) + " }"


def _splice_convention(text: str, side: str, conv: dict) -> str:
    """Return ``text`` with ``arms.<side>.convention`` set to ``conv`` (inline), comments kept.

    Text surgery (not a full YAML re-dump) so the rig file's extensive comments and
    layout survive. Handles: no existing convention (insert), an existing inline
    ``convention: {…}`` (replace the line), and an existing block ``convention:`` with
    indented children (replace the whole block). Raises if the ``arms.<side>`` block is
    absent or written as an inline flow map (which this simple splicer won't edit).
    """
    lines = text.splitlines()

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    # Locate `arms:` (top-level), then the `<side>:` child header.
    arms_i = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith("arms:") and indent_of(ln) == 0),
        None,
    )
    if arms_i is None:
        raise RuntimeError("no top-level `arms:` block in the rig file")
    side_i = None
    for i in range(arms_i + 1, len(lines)):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if indent_of(ln) == 0:
            break  # left the arms block
        if indent_of(ln) > 0 and ln.strip().startswith(f"{side}:"):
            rest = ln.split(":", 1)[1].strip()
            if rest and rest != "":  # inline flow map like `left: { ... }`
                raise RuntimeError(
                    f"arms.{side} is written inline (`{side}: {{…}}`) — expand it to a "
                    "block before syncing, or paste the snippet manually"
                )
            side_i = i
            break
    if side_i is None:
        raise RuntimeError(f"no `arms.{side}:` block in the rig file")

    side_indent = indent_of(lines[side_i])
    child_indent = side_indent + 2

    # Body of the side block: lines until the next line indented <= side_indent (a
    # sibling/top-level key), skipping blanks/comments that belong to the block.
    body_end = len(lines)
    for i in range(side_i + 1, len(lines)):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if indent_of(ln) <= side_indent:
            body_end = i
            break

    # An existing ACTIVE convention line within the body?
    conv_i = None
    for i in range(side_i + 1, body_end):
        ln = lines[i]
        if ln.lstrip().startswith("convention:") and indent_of(ln) >= child_indent:
            conv_i = i
            break

    new_line = f"{' ' * child_indent}convention: {_fmt_conv_inline(conv)}"

    if conv_i is None:
        lines.insert(side_i + 1, new_line)
    else:
        # Remove the convention line + any deeper-indented children (block form).
        conv_indent = indent_of(lines[conv_i])
        end = conv_i + 1
        while end < body_end and (
            not lines[end].strip()
            or lines[end].lstrip().startswith("#")
            or indent_of(lines[end]) > conv_indent
        ):
            # stop if a blank/comment is actually followed by a shallower sibling
            if lines[end].strip() and not lines[end].lstrip().startswith("#"):
                if indent_of(lines[end]) <= conv_indent:
                    break
            end += 1
        lines[conv_i:end] = [new_line]

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def apply_to_rig(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    *,
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
    rig: str | None = None,
):
    """Removed: conversion calibration belongs to the FACTR leader YAML.

    Sets offsets + sign flips, plus ``gripper_open``/``gripper_closed`` when provided
    (recorded endpoints win; unprovided endpoints fall back to any existing value).
    Merges into any existing convention, edits the file text in place (keeping
    comments), and returns the written :class:`~pathlib.Path`. Validated by re-parsing
    before the write, so a splice that would corrupt the file raises instead of leaving
    it broken. *Reset services* re-composes so the change takes effect.
    """
    raise RuntimeError("refusing follower config write: edit the FACTR arm YAML")
    import yaml

    path = rig_path(rig)
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    existing = (((data.get("arms") or {}).get(side) or {}).get("convention") or {})
    if not isinstance(existing, dict):
        existing = {}

    conv: dict = {
        "offsets_deg": [round(float(o), 2) for o in offsets_deg],
        "sign_flip_joints": [int(j) for j in sorted(set(int(x) for x in sign_flip_joints))],
    }
    if gripper_open is not None:
        conv["gripper_open"] = round(float(gripper_open), 4)
    if gripper_closed is not None:
        conv["gripper_closed"] = round(float(gripper_closed), 4)
    for k, v in existing.items():  # keep gripper endpoints / any other tuned keys not re-measured
        if k not in conv:
            conv[k] = v

    new_text = _splice_convention(text, side, conv)

    # Validate before writing: the spliced convention must parse back to `conv`.
    reparsed = yaml.safe_load(new_text) or {}
    got = (((reparsed.get("arms") or {}).get(side) or {}).get("convention") or {})
    if got.get("offsets_deg") != conv["offsets_deg"] or got.get("sign_flip_joints") != conv["sign_flip_joints"]:
        raise RuntimeError(f"write-back validation failed for arms.{side}.convention in {path.name}")

    path.write_text(new_text)
    return path


# ---------------------------------------------------------------------------
# Live Rerun view (dedicated gRPC recording, embedded in the shared web viewer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibViewer:
    #: The SHARED metrics web-viewer port; calibration has no web host of its own,
    #: it embeds that viewer pointed at ``grpc_uri`` (same approach as replay).
    web_port: int
    grpc_uri: str

    @property
    def web_url(self) -> str:
        base = f"http://127.0.0.1:{self.web_port}"
        return f"{base}/?url={quote(self.grpc_uri, safe='')}&persist=0&renderer=webgl"


_VIEWER: "CalibViewer | None" = None
_REC = None  # the served, live-updated recording (kept alive module-scoped)
_T = 0.0     # monotonic-ish frame time (incremented per render; Date/time avoided)


def grpc_port_from_env() -> int:
    """Calibration gRPC-server port, honouring ``DFC_CALIB_GRPC_PORT``."""
    return int(os.environ.get("DFC_CALIB_GRPC_PORT", DEFAULT_CALIB_GRPC_PORT))


def _blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/robot",
            name="Calibration — follower at commanded pose (solid) vs straight/home (ghost)",
        ),
        collapse_panels=True,
    )


def start_calib_viewer(web_port: int, grpc_port: int | None = None) -> CalibViewer:
    """Serve the calibration gRPC recording, embedded in the SHARED web viewer.

    One persistent recording (not fresh-per-frame like replay): the live view is a
    single arm updated in place, like the metrics scene. The static robot geometry is
    logged once here; :func:`render` updates the poses. Idempotent while up; after
    :func:`reset` (paired with :func:`~.viewer.teardown`) the next call re-serves.
    """
    import rerun as rr

    from . import robot_view

    global _VIEWER, _REC
    with _LOCK:
        if _VIEWER is not None:
            return _VIEWER
        gp = grpc_port or grpc_port_from_env()
        rec = rr.RecordingStream(CALIB_APP_ID, recording_id="calibration")
        uri = serve_grpc_checked(
            lambda: rec.serve_grpc(
                grpc_port=gp, default_blueprint=_blueprint(), cors_allow_origin=["*"]
            ),
            gp,
            what="calibration",
        )
        try:
            robot_view.log_scene(rec)  # static pedestal + arm geometry (home)
        except Exception:  # noqa: BLE001 - missing URDF must not break the tab
            pass
        rec.send_blueprint(_blueprint())
        _REC = rec
        _VIEWER = CalibViewer(web_port=web_port, grpc_uri=uri)
        return _VIEWER


def render(side: str, offsets_deg: list[float], sign_flip_joints: list[int]) -> np.ndarray:
    """Pose the arm at the current leader's commanded config; return that config (rad).

    Solid = follower under the in-progress ``offsets_deg`` + ``sign_flip_joints``;
    translucent ghost = the straight/home target (zeros). When a joint is correctly
    calibrated and physically straight, its solid link overlaps the ghost. Raises if
    the viewer is not started or the leader cannot be read.
    """
    from . import robot_view

    global _T
    # Capture the recording + timeline tick under the lock: keeps _T atomic across
    # sessions and avoids a use-after-reset if Reset services nulls _REC mid-render.
    with _LOCK:
        rec = _REC
        if rec is None:
            raise RuntimeError("calibration viewer not started")
        _T += 1.0
        t = _T
    q = commanded_follower_q(side, offsets_deg, sign_flip_joints)  # HTTP read: outside the lock
    home = np.zeros_like(q)
    # Pose BOTH arms so the non-calibrated arm is NOT tinted stale-red ("no live data"):
    # the calibrated side tracks the leader, the other rests solid at home for context.
    real = {s: (q if s == side else home) for s in ("left", "right")}
    robot_view.update_poses(rec, real, {side: home}, t)
    rec.flush()
    return q


def reset() -> None:
    """Drop the viewer + client singletons so the next use re-serves/reconnects.

    Paired with :func:`~.viewer.teardown` — that global ``rerun_shutdown`` releases
    this gRPC port too, so after *Reset services* the next render rebinds a fresh
    calibration server (reusing the shared web viewer).
    """
    global _VIEWER, _REC, _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:  # noqa: BLE001
                pass
        _VIEWER = None
        _REC = None
        _CLIENT = None
