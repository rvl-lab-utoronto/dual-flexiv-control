"""Structured (typed) configuration schema, composed by Hydra.

These dataclasses ARE the schema: they are registered in Hydra's ConfigStore and
the YAML under ``conf/`` is validated against them at compose time. The runtime
reads back fully-typed objects via ``OmegaConf.to_object`` (see ``system.py``),
so every node receives plain, picklable dataclasses across the spawn boundary.

The hierarchy mirrors the stream paths discussed for the system:

    arms.left.streams.{q,dq,tau,wrench,eef,eef_vel}   -> streams "left/<sig>"
    arms.right.streams.{...}                           -> streams "right/<sig>"
    factr.{host,port,endpoint,...}                     -> on-request FactrClient
    arms.{left,right}.control                          -> per-arm ControlCfg
                                                          (qpos|qvel|end_effector|force)
    task.{language_instruction, collection, eval}      -> active task + per-phase templates

Control configs lay out the *command schema* (command field -> dim) plus the
controller setters, aligned with the flexivrdk 1.8 flat send API (verified by API
introspection). They describe the SHAPE of control the arm controller fills and
sends each tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from omegaconf import MISSING

# ---------------------------------------------------------------------------
# Stream schema
# ---------------------------------------------------------------------------


@dataclass
class StreamCfg:
    """Schema for a single shared-memory stream (one signal)."""

    dim: int = MISSING
    """Scalars per sample (e.g. 7 joint torques, 6 wrench components)."""

    dtype: str = "float64"
    """``float32`` or ``float64``."""

    capacity: int = 4096
    """Ring-buffer length (number of recent samples retained)."""

    rate_hz: float = MISSING
    """Nominal production rate, informational."""


# ---------------------------------------------------------------------------
# Control schemas — aligned with the flexivrdk 1.8 flat send API
# ---------------------------------------------------------------------------


@dataclass
class JointImpedanceCfg:
    """``SetJointImpedance(K_q, Z_q)`` — used by NRT_JOINT_IMPEDANCE (1.8 is NRT-only)."""

    K_q: List[float] = field(default_factory=list)   # [DoF] stiffness [Nm/rad], <= K_q_nom
    Z_q: List[float] = field(default_factory=list)   # [DoF] damping ratio, [0.3, 0.8]


@dataclass
class CartesianImpedanceCfg:
    """``SetCartesianImpedance(K_x, Z_x)`` — used by Cartesian motion-force modes."""

    # [6] linear (N/m) + angular (Nm/rad) stiffness: [kx,ky,kz,kRx,kRy,kRz]
    K_x: List[float] = field(default_factory=lambda: [2000.0, 2000.0, 2000.0, 200.0, 200.0, 200.0])
    # [6] damping ratio, valid [0.3, 0.8]
    Z_x: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7])


@dataclass
class ForceControlFrameCfg:
    """``SetForceControlFrame(root_coord, T_in_root)``."""

    root_coord: str = "WORLD"   # WORLD | TCP  (flexivrdk.CoordType)
    # [7] transform root->force frame: [x,y,z,qw,qx,qy,qz]; identity by default
    T_in_root: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


@dataclass
class ControlChannelCfg:
    """Shared-memory control-channel sizing + deadman thresholds (per arm).

    The setpoint channel is a latest-wins mailbox (drop-stale, high rate); the
    command channel is a small reliable queue for discrete events. ``deadman_ms``
    is the soft-hold threshold — when no fresh setpoint has arrived within it the
    arm stops issuing new commands and the NRT motion generator parks at the last
    target; ``deadman_hard_ms`` is the hard threshold that calls ``robot.Stop()``
    and aborts the loop, bounding a crashed or hung brain.
    """

    setpoint_capacity: int = 8       # ring depth for the latest-wins setpoint mailbox
    command_capacity: int = 64       # ring depth for the reliable command queue
    command_dim: int = 8             # width of an encoded command vector [kind, *args]
    dtype: str = "float64"
    rate_hz: float = 50.0            # nominal brain post rate (informational)
    deadman_ms: float = 100.0        # soft-hold staleness threshold
    deadman_hard_ms: float = 500.0   # hard Stop() staleness threshold

    def __post_init__(self) -> None:
        # The control loop checks hard-stop before soft-hold, so an inverted or
        # non-positive pair would make the soft hold unreachable (every stale tick
        # would hard-stop) or break actuation. Fail fast at compose time.
        if not self.deadman_ms > 0:
            raise ValueError(f"deadman_ms must be > 0, got {self.deadman_ms}")
        if not self.deadman_ms < self.deadman_hard_ms:
            raise ValueError(
                f"deadman_hard_ms ({self.deadman_hard_ms}) must exceed "
                f"deadman_ms ({self.deadman_ms})"
            )


@dataclass
class ControlCoeffsCfg:
    """Per-phase controller coefficients (impedances + motion limits).

    A separate importable config group (``control_coeffs``) so each task can give
    different coefficients to collection (training) vs eval. The arm controller
    applies, AFTER ``SwitchMode``, only the subset its mode accepts (verified
    against flexivrdk 1.8.0):

    * NRT_JOINT_POSITION (qpos/qvel): uses ``max_joint_vel``/``max_joint_acc`` as
      the ``max_vel``/``max_acc`` args of ``SendJointPosition``. ``joint_impedance``
      is NOT settable in this mode (only the JOINT_IMPEDANCE modes) and is ignored.
    * NRT_CARTESIAN_MOTION_FORCE (end_effector/eef_vel/force): SetCartesianImpedance,
      SetMaxContactWrench, SetNullSpacePosture apply; ``max_{linear,angular}_*``
      feed the scalar limit args of ``SendCartesianMotionForce``.
    """

    joint_impedance: Optional[JointImpedanceCfg] = None         # SetJointImpedance(K_q,Z_q) — impedance modes only
    cartesian_impedance: Optional[CartesianImpedanceCfg] = None # SetCartesianImpedance(K_x,Z_x) — cartesian modes
    max_contact_wrench: Optional[List[float]] = None            # [6] SetMaxContactWrench [N,Nm]
    null_space_posture: Optional[List[float]] = None            # [DoF] SetNullSpacePosture [rad]
    # NRT joint motion limits -> SendJointPosition(..., max_vel, max_acc):
    max_joint_vel: float = 2.0       # [rad/s]
    max_joint_acc: float = 3.0       # [rad/s^2]
    # NRT cartesian motion limits -> SendCartesianMotionForce scalar caps:
    max_linear_vel: float = 0.5      # [m/s]
    max_angular_vel: float = 1.0     # [rad/s]
    max_linear_acc: float = 2.0      # [m/s^2]
    max_angular_acc: float = 5.0     # [rad/s^2]


@dataclass
class ControlCfg:
    """A control type: which RDK mode/method to use and the command schema.

    ``command`` maps each command field to its dim; ``streamed`` lists which of
    those fields the brain posts per tick on the setpoint channel (the rest are
    static limits drawn from the per-phase :class:`ControlCoeffsCfg`). The
    coefficients live in the ``control_coeffs`` group, imported per task phase —
    NOT here. Verified against flexivrdk 1.8.0 (all NRT, flat send API — the
    command fields are the positional args of ``send_fn``, brain-driven over IPC)::

      qpos          NRT_JOINT_POSITION         SendJointPosition         (q_d primary)
      qvel          NRT_JOINT_POSITION         SendJointPosition         (dq_d primary; q_d integrated)
      end_effector  NRT_CARTESIAN_MOTION_FORCE SendCartesianMotionForce  (pose_d primary)
      eef_vel       NRT_CARTESIAN_MOTION_FORCE SendCartesianMotionForce  (twist_d primary; pose_d integrated)
      force         NRT_CARTESIAN_MOTION_FORCE SendCartesianMotionForce  (wrench_d primary)
    """

    kind: str = MISSING          # qpos | qvel | end_effector | eef_vel | force
    mode: str = MISSING          # flexivrdk.Mode name, e.g. NRT_JOINT_POSITION
    send_fn: str = MISSING       # Robot send method, e.g. SendJointPosition
    command: Dict[str, int] = MISSING   # command field -> dim (full command schema)
    streamed: List[str] = field(default_factory=list)   # fields posted per tick on the setpoint channel

    channel: ControlChannelCfg = field(default_factory=ControlChannelCfg)

    # Structural force-control config (cartesian `force` kind only; NOT a tunable coeff):
    force_control_axes: Optional[List[bool]] = None        # [6] [X,Y,Z,Rx,Ry,Rz] force-controlled axes
    force_control_frame: Optional[ForceControlFrameCfg] = None
    force_axis_max_linear_vel: Optional[List[float]] = None  # [3] SetForceControlAxis vel cap [m/s]


# ---------------------------------------------------------------------------
# Task schema — a manipulation task with per-phase templates
# ---------------------------------------------------------------------------


@dataclass
class CollectionCfg:
    """Collection-phase template: teleoperated demonstration gathering.

    Demos are variable-length (the operator ends each one), so collection is
    bounded by a count of episodes, not a timestep horizon.
    """

    num_episodes: int = MISSING
    """How many demonstration episodes to teleoperate and record."""

    coeffs: ControlCoeffsCfg = field(default_factory=ControlCoeffsCfg)
    """Controller coefficients used during collection (imported from `control_coeffs`)."""


@dataclass
class EvalCfg:
    """Evaluation-phase template: online policy rollouts.

    A rollout has no operator to end it, so it is bounded by a fixed horizon.
    """

    num_timesteps: int = MISSING
    """Rollout horizon — max timesteps before an eval episode is cut off."""

    coeffs: ControlCoeffsCfg = field(default_factory=ControlCoeffsCfg)
    """Controller coefficients used during eval (imported from `control_coeffs`)."""


@dataclass
class TaskCfg:
    """A manipulation task: one shared spec plus per-phase templates.

    ``language_instruction`` is shared across both phases (the natural-language
    goal handed to the teleoperator during collection and to the policy during
    eval). The ``collection`` and ``eval`` sub-configs hold only what differs
    between the two phases; shared fields live directly on this node.
    """

    language_instruction: str = MISSING                            # shared by both phases
    collection: CollectionCfg = field(default_factory=CollectionCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass
class JointConventionCfg:
    """FACTR leader → Rizon follower joint mapping (used by the brain, pure math).

    Captured from the hardware-validated teleop test. ``offsets_deg`` is added
    per-joint after converting the leader's radians to degrees; ``sign_flip_joints``
    negates those joint indices; the result is wrapped to ``[-180,180]`` and
    converted back to radians. ``drop_trailing`` discards FACTR's trailing gripper
    value(s) (its payload is ``DoF+1``). Only the LEFT arm's values are known from
    the test — the right arm's must be measured (do NOT assume symmetry).
    """

    offsets_deg: List[float] = field(default_factory=lambda: [180.0, -90.0, 90.0, 90.0, 90.0, 180.0, -90.0])
    sign_flip_joints: List[int] = field(default_factory=lambda: [3])
    wrap_deg: bool = True
    drop_trailing: int = 1


@dataclass
class ArmCfg:
    """One Flexiv arm: connection, read settings, stream schemas, default controller."""

    serial: str = MISSING
    name: str = ""                       # display name (e.g. dashboard); "" => the side
    dof: int = 7
    wrench_frame: str = "local"          # local (TCP) | world
    require_operational: bool = False
    verbose_rdk: bool = False
    rate_hz: float = 1000.0              # read-only telemetry loop rate (control off)
    streams: Dict[str, StreamCfg] = MISSING   # signal -> StreamCfg
    control: ControlCfg = MISSING        # this arm's control schema (composed from the `control` group)

    # -- control loop (opt-in; off => the legacy read-only telemetry node) -----
    control_enabled: bool = False        # run the merged telemetry+control loop and actuate
    control_rate_hz: float = 200.0       # merged loop rate when control_enabled
    control_safety_check: bool = True    # L-inf joint-error gate vs measured q (joint kinds)
    control_tolerance: float = 0.5       # [rad] L-inf gate threshold (~28 deg)
    control_attach_timeout_s: float = 10.0   # wait for the brain's channels to appear
    convention: JointConventionCfg = field(default_factory=JointConventionCfg)


@dataclass
class CameraCfg:
    """One ZED camera publishing image streams (one process per camera).

    A camera publishes one stream per entry in ``views`` — canonical view names
    (see :mod:`dual_flexiv_control.cameras`): ``left``/``right`` RGB (uint8,
    HxWx3) and optional ``depth`` (float32, HxW, metres). Per-stream ``dim`` is
    *derived* from ``width``/``height`` (× channels), so image sizes are never
    hand-computed. Streams are named ``"cam/<name>/<view>"`` where ``<name>`` is
    the camera's key in :attr:`Config.cameras` (e.g. ``cam/wrist_left/left``).

    ``resolution`` is the ZED SDK ``sl.RESOLUTION`` enum name handed to the real
    camera; ``width``/``height`` must be what that resolution yields (the real
    source validates this at open and fails fast on mismatch). Valid enums are
    model-specific — ZED 2: HD2K/HD1080/HD720/VGA; ZED X (Nano): HD1200/HD1080/
    SVGA — confirm against your installed SDK.
    """

    model: str = MISSING          # "zed2" | "zedx_nano" (informational; SDK auto-detects)
    serial: str = ""              # ZED serial number (numeric); "" => first available
    placement: str = MISSING      # "wrist_left" | "wrist_right" | "static"
    resolution: str = "HD720"     # sl.RESOLUTION enum name handed to the real camera
    width: int = MISSING          # frame width  [px]; must match `resolution`
    height: int = MISSING         # frame height [px]; must match `resolution`
    fps: float = 30.0             # capture rate -> this producer's rate_hz
    depth_mode: str = "NONE"      # sl.DEPTH_MODE name; must be != NONE if "depth" in views
    views: List[str] = MISSING    # canonical views to publish, e.g. ["left"] or ["left","right"]
    capacity: int = 16            # ring depth (frames retained) for each of this camera's streams


@dataclass
class FactrServerCfg:
    """One FACTR leader-arm teleop server (FastAPI), queried on request.

    Each leader runs its own server on its own ``port``. A single
    ``GET http://{host}:{port}/{endpoint}`` returns *that* leader's joint
    positions (``dof`` joints).
    """

    host: str = "localhost"
    port: int = MISSING
    endpoint: str = "get_joint_positions"
    request_timeout_s: float = 0.5
    dof: int = 7


@dataclass
class FactrCfg:
    """FACTR teleop: one HTTP server per leader arm, each on its own port.

    Not polled and not streamed — the brain holds a ``FactrClient`` that queries
    every configured server on demand; ``get_joint_positions()`` returns
    ``{side: joint_positions}``.
    """

    servers: Dict[str, FactrServerCfg] = field(default_factory=dict)


@dataclass
class BrainCfg:
    """The main processing pipeline (consumer)."""

    rate_hz: float = 100.0
    attach_timeout_s: float = 10.0
    # Streams to subscribe to; empty => default (both arms' proprio + FACTR).
    subscribe: List[str] = field(default_factory=list)


@dataclass
class RuntimeCfg:
    """Process/runtime knobs."""

    runtime_dir: str = "runtime"   # resolved to absolute against the launch cwd
    sim: bool = False              # use simulated sources (no hardware)
    duration_s: Optional[float] = None   # auto-stop after N seconds (None = run until Ctrl-C)
    phase: str = "collection"      # collection | eval — selects which per-task coeffs the arms apply


@dataclass
class Config:
    """Top-level composed configuration."""

    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    brain: BrainCfg = field(default_factory=BrainCfg)
    factr: FactrCfg = MISSING
    task: TaskCfg = MISSING       # the active task (composed from the `task` group)
    # Populated by package-directed group defaults (arm@arms.left, control@arms.left.control, ...).
    arms: Dict[str, ArmCfg] = field(default_factory=dict)
    # Likewise camera@cameras.<name> — two ZED X Nano wrist cams + one static ZED 2.
    cameras: Dict[str, CameraCfg] = field(default_factory=dict)


def register_configs() -> None:
    """Register schemas in Hydra's ConfigStore so YAML is type-validated.

    Group schemas are referenced as the first entry in each group file's
    ``defaults`` list (e.g. ``arm/flexiv.yaml`` -> ``- base_arm``), which makes
    the composed node at any package path validate against the dataclass.
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="runtime", name="base_runtime", node=RuntimeCfg)
    cs.store(group="brain", name="base_brain", node=BrainCfg)
    cs.store(group="factr", name="base_factr", node=FactrCfg)
    cs.store(group="task", name="base_task", node=TaskCfg)
    cs.store(group="arm", name="base_arm", node=ArmCfg)
    cs.store(group="camera", name="base_camera", node=CameraCfg)
    cs.store(group="control", name="base_control", node=ControlCfg)
    cs.store(group="control_coeffs", name="base_control_coeffs", node=ControlCoeffsCfg)
