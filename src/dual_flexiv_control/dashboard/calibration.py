"""Leader-arm joint-offset calibration for the dashboard's Calibration controls.

The FACTR leader→Rizon follower mapping (:func:`convert_factr_to_rizon`) adds a
per-joint ``offsets_deg`` before the sign flip and wrap. Those offsets (and the
``sign_flip_joints``) are the part of the convention that drifts per physical leader
and must be measured (the right arm's especially — do NOT assume symmetry with the
left). Two measurement flows share this module:

**Pose-set calibration (primary).** A fixed set of reference poses
(:data:`REFERENCE_POSES`, follower configs in degrees). The operator moves the
leader into each pose, captures a sample (:func:`read_leader_arm_deg`), and once a
handful of poses are captured :func:`solve_pose_samples` solves the whole
convention at once: per joint, a circular least-squares fit of
``ref = wrap(sign * (leader + offset))`` over the samples, trying both signs and
keeping the better fit. Because the poses drive every joint to distinct nonzero
targets, the fit recovers the *sign flips* too (a single straight pose cannot — at
zero both signs agree), and the per-joint RMS residual exposes a badly-matched pose.

**Per-joint capture (fine-tune).** Straighten a single link and capture it; at the
straight target the follower joint reads 0, and the sign flip multiplies zero, so
``offset[j] = wrap(-degrees(q_leader_arm[j]))`` — independent of the other joints
(see :func:`~dual_flexiv_control.control.offsets_from_straight_pose`, from which we
take the one index). Un-captured joints keep their existing config offset, so a
partial calibration is well-defined.

The same session also captures the physical leader at FACTR's dynamics-model home.
Once the raw→DFC convention is known, that sample determines the side-specific
canonical DFC home. FACTR owns the model-home target and derives the affine zero
offset at process launch; the derived value is never persisted by DFC.

This module reads the FACTR WebSocket cache (honouring ``runtime.sim``); it never
opens a robot connection or creates a Rerun recording. Its explicit save action
writes all measured leader fields together in the active DFC ``conf/factr/*.yaml``:
``raw_to_dfc``, ``home_q_rad``, and ``dfc_to_factr``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from dataclasses import replace
import numpy as np

from . import arms as _arms

_LOCK = threading.Lock()
#: Cached leader client (persistent WebSocket), rebuilt on demand — avoids reconnect
#: churn under the calibration live-render fragment. Dropped by :func:`reset`.
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


def current_model_signs(side: str) -> list[float]:
    """Configured DFC→FACTR model-axis signs for one physical leader."""
    leaders = getattr(_arms.discover_factr(), "leaders", None) or {}
    leader = leaders.get(side)
    transform = getattr(leader, "dfc_to_factr", None) if leader is not None else None
    signs = list(getattr(transform, "signs", []) or [])
    dof = follower_dof(side)
    if len(signs) != dof or any(float(v) not in (-1.0, 1.0) for v in signs):
        raise RuntimeError(
            f"leaders.{side}.dfc_to_factr.signs must contain {dof} values of ±1"
        )
    return [float(v) for v in signs]


def factr_model_home(side: str) -> list[float]:
    """FACTR-owned dynamics-model home from the side's mechanism configuration."""
    import yaml
    from pathlib import Path

    dof = follower_dof(side)
    factr = _arms.discover_factr()
    workdir = Path(str(factr.launch.workdir)).expanduser().resolve()
    config_path = (
        workdir / "src" / "factr_teleop" / "factr_teleop" / "configs"
        / f"factr_rizon_{side}.yaml"
    )
    data = yaml.safe_load(config_path.read_text()) or {}
    try:
        values = data["arm_teleop"]["initialization"]["model_home_q_rad"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"FACTR {side} config has no arm_teleop.initialization.model_home_q_rad"
        ) from exc
    target = np.asarray(values, dtype=np.float64)
    if target.shape != (dof,) or not np.all(np.isfinite(target)):
        raise RuntimeError(
            f"FACTR {side} model_home_q_rad must contain {dof} finite values"
        )
    return [float(v) for v in target]


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
# Pose-set calibration: fixed reference poses -> samples -> full-convention solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefPose:
    """One fixed calibration pose: a follower joint config the leader is moved to match."""

    name: str
    hint: str  # how to physically pose the leader (the 3D view shows it exactly)
    q_deg: tuple[float, ...]  # follower arm-joint targets, degrees


#: The calibration pose set. J1 is the shoulder pitch that decides whether the arm
#: stays in front of the robot: every target therefore uses either straight-up
#: ``J1 = 0°`` or the known-reachable elbow-bend ``J1 = -90°``. The remaining
#: poses start from that elbow bend and vary only one roll joint at a time. Comparing
#: each nonzero target with Straight up makes every joint's sign observable without
#: asking the operator to reproduce a compound pose that reaches behind the base.
#: Angles remain multiples of 90° and inside the Rizon 4s limits.
REFERENCE_POSES: tuple[RefPose, ...] = (
    RefPose(
        "Straight up",
        "Every link straight: the whole arm points straight up from its mount.",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    RefPose(
        "Elbow bend",
        "Set J1 to the known-reachable -90° shoulder target, then bend the elbow "
        "and wrist +90° to make a simple square profile in front of the robot.",
        (0.0, -90.0, 0.0, 90.0, 0.0, 90.0, 0.0),
    ),
    RefPose(
        "Base roll",
        "Start from Elbow bend, then turn only J0 +90°. Keep J1 at the same "
        "forward elbow-bend angle.",
        (90.0, -90.0, 0.0, 90.0, 0.0, 90.0, 0.0),
    ),
    RefPose(
        "Upper-arm roll",
        "Start from Elbow bend, then turn only J2 +90°. Keep the shoulder and "
        "the rest of the square profile fixed.",
        (0.0, -90.0, 90.0, 90.0, 0.0, 90.0, 0.0),
    ),
    RefPose(
        "Forearm roll",
        "Start from Elbow bend, then turn only J4 +90°. Keep J1 at -90° so the "
        "arm remains in front of the robot.",
        (0.0, -90.0, 0.0, 90.0, 90.0, 90.0, 0.0),
    ),
    RefPose(
        "Flange roll",
        "Start from Elbow bend, then turn only J6 +90°. All other joints remain "
        "at the reachable elbow-bend target.",
        (0.0, -90.0, 0.0, 90.0, 0.0, 90.0, 90.0),
    ),
)

#: Residual (deg RMS) above which the UI flags a joint's fit as suspect — a badly
#: matched pose or a mis-seated leader shows up here rather than silently skewing
#: the offsets.
RESIDUAL_WARN_DEG = 5.0

#: Sign fits whose residuals differ by less than this are treated as *ambiguous*
#: (the sample set cannot tell the signs apart — e.g. only the straight pose was
#: captured) and the joint keeps its configured sign flip instead.
_SIGN_TIE_EPS_DEG = 1e-6


def reference_poses(side: str) -> list[RefPose]:
    """:data:`REFERENCE_POSES` length-normalized to the side's follower DoF."""
    dof = follower_dof(side)
    return [
        replace(pose, q_deg=tuple((list(pose.q_deg) + [0.0] * dof)[:dof]))
        for pose in REFERENCE_POSES
    ]


def read_leader_arm_deg(side: str, samples: int = 5) -> list[float]:
    """One pose-set sample: the leader's arm joints in degrees (gripper dropped)."""
    conv = current_convention(side)
    return [float(v) for v in _arm_joints_deg(read_leader(side, samples), conv)]


def _wrap_deg(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _circular_mean_deg(vals) -> float:
    """Mean of angles in degrees via the resultant vector (offsets near ±180 safe)."""
    r = np.radians(np.asarray(vals, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


@dataclass(frozen=True)
class PoseFit:
    """A full-convention solve from pose-set samples (per-joint, indexed 0..DoF-1)."""

    offsets_deg: list[float]
    sign_flip_joints: list[int]
    residuals_deg: list[float]   # per-joint RMS of the kept sign's fit
    ambiguous_joints: list[int]  # sign not decidable from the samples (config kept)
    n_samples: int


def solve_from_samples(
    pairs: list[tuple[list[float], list[float]]],
    fallback_flips=(),
) -> PoseFit:
    """Solve the whole convention from ``(leader_deg, ref_deg)`` sample pairs.

    Per joint ``j`` the model is ``ref = wrap(sign * (leader + offset))`` (exactly
    :func:`convert_factr_to_rizon` in degrees), i.e. ``offset = wrap(sign*ref −
    leader)`` for every sample. Both signs are tried: each candidate's offset is the
    *circular mean* of its per-sample estimates and its residual the RMS of the
    wrapped deviations; the lower-residual sign wins. When the two signs fit equally
    well the samples cannot decide (every reference for that joint sat at 0/±180 —
    or only one pose was captured, which any offset fits perfectly under either
    sign): the joint is reported in ``ambiguous_joints`` and keeps its
    ``fallback_flips`` membership, so a home-pose-only capture degrades exactly to
    the straight-pose solve.
    """
    if not pairs:
        raise RuntimeError("no captured pose samples to solve from")
    leader = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ref = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if leader.ndim != 2 or leader.shape != ref.shape:
        raise RuntimeError(
            f"sample shape mismatch: leader {leader.shape} vs reference {ref.shape}"
        )
    fallback = {int(j) for j in fallback_flips}
    offsets: list[float] = []
    flips: list[int] = []
    residuals: list[float] = []
    ambiguous: list[int] = []
    for j in range(leader.shape[1]):
        fits: dict[float, tuple[float, float]] = {}
        for sign in (1.0, -1.0):
            est = _wrap_deg(sign * ref[:, j] - leader[:, j])
            off = _circular_mean_deg(est)
            rms = float(np.sqrt(np.mean(_wrap_deg(est - off) ** 2)))
            fits[sign] = (off, rms)
        if abs(fits[1.0][1] - fits[-1.0][1]) <= _SIGN_TIE_EPS_DEG:
            sign = -1.0 if j in fallback else 1.0
            ambiguous.append(j)
        else:
            sign = min(fits, key=lambda s: fits[s][1])
        off, rms = fits[sign]
        offsets.append(off)
        if sign < 0:
            flips.append(j)
        residuals.append(rms)
    return PoseFit(
        offsets_deg=offsets,
        sign_flip_joints=flips,
        residuals_deg=residuals,
        ambiguous_joints=ambiguous,
        n_samples=len(pairs),
    )


def solve_pose_samples(side: str, samples: dict[str, list[float]]) -> PoseFit:
    """UI wrapper: pair captured samples with their reference poses and solve.

    ``samples`` maps pose *name* → captured leader arm degrees
    (:func:`read_leader_arm_deg`). Names not in the current pose set — or samples
    whose length no longer matches the follower DoF — are ignored (stale keys after
    a pose-set or rig edit). The configured ``sign_flip_joints`` seed the
    ambiguous-joint fallback.
    """
    poses = {p.name: p for p in reference_poses(side)}
    pairs = [
        (list(q), list(poses[name].q_deg))
        for name, q in samples.items()
        if name in poses and len(q) == len(poses[name].q_deg)
    ]
    if not pairs:
        raise RuntimeError("no captured pose samples to solve from")
    return solve_from_samples(
        pairs, fallback_flips=current_convention(side).sign_flip_joints
    )


@dataclass(frozen=True)
class ModelHomeFit:
    """Measured DFC pose and axis convention at FACTR model home."""

    home_q_rad: list[float]
    signs: list[float]
    target_q_rad: list[float]

    @property
    def derived_offset_rad(self) -> list[float]:
        """Non-persisted affine offset, useful only for display/audit."""
        target = np.asarray(self.target_q_rad, dtype=np.float64)
        signs = np.asarray(self.signs, dtype=np.float64)
        home = np.asarray(self.home_q_rad, dtype=np.float64)
        return [float(v) for v in target - signs * home]


def solve_model_home(
    raw_leader_deg: list[float],
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    model_signs: list[float],
    target_q_rad: list[float],
) -> ModelHomeFit:
    """Convert a raw sample at FACTR's authoritative model home into DFC coordinates.

    The sample is first converted through the *in-progress* raw→DFC convention,
    so one calibration session owns both coordinate mappings. Model-axis signs are
    mechanism/URDF facts and remain configured; this capture measures their zero
    offsets.
    """
    raw = np.asarray(raw_leader_deg, dtype=np.float64)
    offsets = np.asarray(offsets_deg, dtype=np.float64)
    signs = np.asarray(model_signs, dtype=np.float64)
    target = np.asarray(target_q_rad, dtype=np.float64)
    if raw.ndim != 1 or not (raw.shape == offsets.shape == signs.shape == target.shape):
        raise RuntimeError(
            "model-home shape mismatch: "
            f"raw {raw.shape}, offsets {offsets.shape}, signs {signs.shape}, "
            f"target {target.shape}"
        )
    if not np.all(np.isfinite(np.concatenate((raw, offsets, signs, target)))):
        raise RuntimeError("model-home calibration contains non-finite values")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise RuntimeError("DFC-to-FACTR model signs must be -1 or +1")
    flips = {int(j) for j in sign_flip_joints}
    if any(j < 0 or j >= len(raw) for j in flips):
        raise RuntimeError(f"invalid sign-flip joints: {sorted(flips)}")

    q_dfc_deg = raw + offsets
    for j in flips:
        q_dfc_deg[j] = -q_dfc_deg[j]
    q_dfc_deg = _wrap_deg(q_dfc_deg)
    home_q = np.radians(q_dfc_deg)
    return ModelHomeFit(
        home_q_rad=[float(v) for v in home_q],
        signs=[float(v) for v in signs],
        target_q_rad=[float(v) for v in target],
    )


def capture_model_home(
    side: str, offsets_deg: list[float], sign_flip_joints: list[int], samples: int = 5
) -> tuple[list[float], ModelHomeFit]:
    """Capture raw arm joints at FACTR model home in canonical DFC coordinates."""
    raw = read_leader_arm_deg(side, samples)
    fit = solve_model_home(
        raw, offsets_deg, sign_flip_joints, current_model_signs(side), factr_model_home(side)
    )
    return raw, fit


# ---------------------------------------------------------------------------
# Config formatting (paste into conf/factr)
# ---------------------------------------------------------------------------


def _fmt_list(values, decimals: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def format_yaml(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
    model_home: ModelHomeFit | None = None,
) -> str:
    """A DFC factr-group leader snippet for the measured convention."""
    lines = [
        "leaders:",
        f"  {side}:",
        "    raw_to_dfc:",
        f"      offsets_deg: {_fmt_list(offsets_deg)}",
        f"      sign_flip_joints: {list(sign_flip_joints)}",
        "      wrap_deg: true",
        "      drop_trailing: 1",
    ]
    if gripper_open is not None:
        lines.append(f"      gripper_open: {float(gripper_open):.4f}")
    if gripper_closed is not None:
        lines.append(f"      gripper_closed: {float(gripper_closed):.4f}")
    if model_home is not None:
        lines.extend([
            f"    home_q_rad: {_fmt_list(model_home.home_q_rad, 10)}",
            "    dfc_to_factr:",
            f"      signs: {[int(v) for v in model_home.signs]}",
        ])
    return "\n".join(lines)


def format_overrides(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
    model_home: ModelHomeFit | None = None,
) -> str:
    """Hydra overrides for DFC's selected leader calibration."""
    offsets = "[" + ",".join(f"{v:.2f}" for v in offsets_deg) + "]"
    flips = "[" + ",".join(str(int(j)) for j in sign_flip_joints) + "]"
    parts = [
        f"factr.leaders.{side}.raw_to_dfc.offsets_deg='{offsets}'",
        f"factr.leaders.{side}.raw_to_dfc.sign_flip_joints='{flips}'",
    ]
    if gripper_open is not None:
        parts.append(f"factr.leaders.{side}.raw_to_dfc.gripper_open={float(gripper_open):.4f}")
    if gripper_closed is not None:
        parts.append(f"factr.leaders.{side}.raw_to_dfc.gripper_closed={float(gripper_closed):.4f}")
    if model_home is not None:
        homes = "[" + ",".join(f"{v:.10f}" for v in model_home.home_q_rad) + "]"
        signs = "[" + ",".join(str(int(v)) for v in model_home.signs) + "]"
        parts.extend([
            f"factr.leaders.{side}.home_q_rad='{homes}'",
            f"factr.leaders.{side}.dfc_to_factr.signs='{signs}'",
        ])
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


def factr_path(rig: str | None = None):
    """Path of the DFC factr-group YAML selected by the active rig."""
    import dual_flexiv_control
    import yaml
    from pathlib import Path

    rig_file = rig_path(rig)
    data = yaml.safe_load(rig_file.read_text()) or {}
    group = None
    for item in data.get("defaults", []):
        if isinstance(item, dict):
            for key in ("/factr", "factr"):
                if key in item:
                    group = str(item[key])
                    break
        if group:
            break
    if not group:
        raise RuntimeError(f"no /factr group selected by {rig_file.name}")
    conf = Path(dual_flexiv_control.__file__).resolve().parent / "conf"
    return conf / "factr" / f"{group}.yaml"


def apply_to_leader(
    side: str,
    offsets_deg: list[float],
    sign_flip_joints: list[int],
    *,
    gripper_open: float | None = None,
    gripper_closed: float | None = None,
    model_home: ModelHomeFit | None = None,
    rig: str | None = None,
):
    """Load, update, validate, and atomically save the active DFC factr YAML."""
    import os
    import tempfile
    import yaml

    path = factr_path(rig)
    data = yaml.safe_load(path.read_text()) or {}
    leaders = data.get("leaders")
    if not isinstance(leaders, dict) or side not in leaders:
        raise RuntimeError(f"no `leaders.{side}` object in {path.name}")
    leader = leaders[side]
    if not isinstance(leader, dict):
        raise RuntimeError(f"`leaders.{side}` is not a YAML object in {path.name}")
    conv = leader.get("raw_to_dfc")
    if not isinstance(conv, dict):
        raise RuntimeError(
            f"`leaders.{side}.raw_to_dfc` is not a YAML object in {path.name}"
        )
    conv["offsets_deg"] = [round(float(o), 2) for o in offsets_deg]
    conv["sign_flip_joints"] = [
        int(j) for j in sorted(set(int(x) for x in sign_flip_joints))
    ]
    conv.setdefault("wrap_deg", True)
    conv.setdefault("drop_trailing", 1)
    if gripper_open is not None:
        conv["gripper_open"] = round(float(gripper_open), 4)
    if gripper_closed is not None:
        conv["gripper_closed"] = round(float(gripper_closed), 4)
    if model_home is not None:
        dof = len(conv["offsets_deg"])
        if not (
            len(model_home.home_q_rad)
            == len(model_home.signs)
            == dof
        ):
            raise RuntimeError(f"model-home calibration must contain {dof} arm joints")
        leader["home_q_rad"] = [round(float(v), 10) for v in model_home.home_q_rad]
        transform = leader.setdefault("dfc_to_factr", {})
        if not isinstance(transform, dict):
            raise RuntimeError(
                f"leaders.{side}.dfc_to_factr is not a YAML object in {path.name}"
            )
        transform["signs"] = [int(v) for v in model_home.signs]
    transform = leader.get("dfc_to_factr")
    if isinstance(transform, dict):
        transform.pop("offset_rad", None)
    rendered = yaml.safe_dump(data, sort_keys=False, width=120)
    reparsed = yaml.safe_load(rendered) or {}
    got_leader = (reparsed.get("leaders") or {}).get(side) or {}
    got = got_leader.get("raw_to_dfc") or {}
    if got != conv:
        raise RuntimeError(
            f"write-back validation failed for leaders.{side}.raw_to_dfc in {path.name}"
        )
    if model_home is not None:
        if (
            got_leader.get("home_q_rad") != leader["home_q_rad"]
            or got_leader.get("dfc_to_factr") != leader["dfc_to_factr"]
        ):
            raise RuntimeError(
                f"write-back validation failed for leaders.{side} model transform "
                f"in {path.name}"
            )
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            temp_path = tmp.name
            tmp.write(rendered)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
    return path


def apply_to_rig(*args, **kwargs):
    """Compatibility alias; calibration now saves to DFC's leader config."""
    return apply_to_leader(*args, **kwargs)


def reset() -> None:
    """Close and forget the cached leader client."""
    global _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:  # noqa: BLE001
                pass
        _CLIENT = None
