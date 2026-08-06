"""Structured (typed) configuration schema, composed by Hydra.

These dataclasses ARE the schema: they are registered in Hydra's ConfigStore and
the YAML under ``conf/`` is validated against them at compose time. The runtime
reads back fully-typed objects via ``OmegaConf.to_object`` (see ``system.py``),
so every node receives plain, picklable dataclasses across the spawn boundary.

The hierarchy mirrors the stream paths discussed for the system:

    arms.left.streams.{q,dq,tau,tau_ext,wrench,eef,eef_vel} -> "left/<sig>"
    arms.right.streams.{...}                           -> streams "right/<sig>"
    factr.servers.{left,right}                         -> streams "factr/<side>"
    policy.control                                     -> policy action/controller semantics
                                                          (qpos|qvel|end_effector|force)
    arms.{left,right}.control                          -> interpolation of policy.control
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

    dummy: bool = False
    """Publish fabricated ("dummy") data for this stream instead of reading hardware.

    For a Flexiv arm the proprio streams come from one robot connection, so a real
    robot cannot selectively fabricate a subset — mark **all** of an arm's streams
    ``dummy`` (see ``arm/flexiv_dummy.yaml``) and its interface uses the simulated
    source (no ``flexivrdk`` connection). Handy for a test task that runs a real
    camera against a robot that isn't connected (e.g. bring-up on a bench)."""


# ---------------------------------------------------------------------------
# Control schemas — aligned with the flexivrdk 1.8 flat send API
# ---------------------------------------------------------------------------


@dataclass
class JointImpedanceCfg:
    """``SetJointImpedance(K_q, Z_q)`` — used by NRT_JOINT_IMPEDANCE (1.8 is NRT-only)."""

    K_q: List[float] = field(default_factory=list)   # [DoF] stiffness [Nm/rad], <= K_q_nom
    Z_q: List[float] = field(default_factory=list)   # [DoF] damping ratio, [0.3, 0.8]

    K_q_fraction: Optional[float] = None
    """Stiffness as a fraction of the connected robot's own ``K_q_nom`` (uniform
    across joints), resolved live at apply-time instead of a hand-picked absolute
    ``K_q``. Takes precedence over ``K_q`` when set — the safer way to ask for
    "a small fraction of however stiff this robot's joints nominally are" without
    guessing per-model Nm/rad numbers."""


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
    """Per-phase motion, contact, stiffness-scale, and null-space limits.

    A separate importable config group (``control_coeffs``) so each task can give
    different coefficients to collection (training) vs eval. The arm controller
    applies, AFTER ``SwitchMode``, only the subset its mode accepts (verified
    against flexivrdk 1.8.0):

    Joint limits feed ``SendJointPosition``; Cartesian limits feed
    ``SendCartesianMotionForce``. Base impedance comes from the selected
    :class:`ControlCfg`; ``joint_stiffness_scale`` scales its joint stiffness for
    this phase after ``SwitchMode``.
    """

    max_contact_wrench: Optional[List[float]] = None            # [6] SetMaxContactWrench [N,Nm]
    null_space_posture: Optional[List[float]] = None            # [DoF] SetNullSpacePosture [rad]
    joint_stiffness_scale: float = 1.0
    """Multiplier on the selected joint-impedance controller's base ``K_q``.
    Collection uses one third; eval and non-impedance modes retain their baseline."""
    # NRT joint motion limits -> SendJointPosition(..., max_vel, max_acc):
    max_joint_vel: float = 2.0       # [rad/s]
    max_joint_acc: float = 3.0       # [rad/s^2]
    # NRT cartesian motion limits -> SendCartesianMotionForce scalar caps:
    max_linear_vel: float = 0.5      # [m/s]
    max_angular_vel: float = 1.0     # [rad/s]
    max_linear_acc: float = 2.0      # [m/s^2]
    max_angular_acc: float = 5.0     # [rad/s^2]


# -- named per-phase presets. Base impedance lives in conf/control/*.yaml; the
#    per-phase stiffness multiplier lives here.


def compliant_coeffs() -> ControlCoeffsCfg:
    """Low stiffness + gentle motion limits — safe teleoperated collection.

    Gentle motion/contact limits for ``task.collection.coeffs``.
    """
    return ControlCoeffsCfg(
        max_contact_wrench=[40.0, 40.0, 40.0, 12.0, 12.0, 12.0],
        joint_stiffness_scale=1.0 / 3.0,
        max_joint_vel=1.5,
        max_joint_acc=2.0,
        max_linear_vel=0.3,
        max_angular_vel=0.8,
        max_linear_acc=1.5,
        max_angular_acc=4.0,
    )


def stiff_coeffs() -> ControlCoeffsCfg:
    """High stiffness + faster motion limits — precise policy eval / tracking.

    The schema default for ``task.eval.coeffs``.
    """
    return ControlCoeffsCfg(
        max_contact_wrench=[60.0, 60.0, 60.0, 18.0, 18.0, 18.0],
        max_joint_vel=2.5,
        max_joint_acc=3.5,
        max_linear_vel=0.6,
        max_angular_vel=1.2,
        max_linear_acc=2.5,
        max_angular_acc=6.0,
    )


def default_coeffs() -> ControlCoeffsCfg:
    """Moderate middle ground between :func:`compliant_coeffs` and :func:`stiff_coeffs`."""
    return ControlCoeffsCfg(
        max_contact_wrench=[50.0, 50.0, 50.0, 15.0, 15.0, 15.0],
        max_joint_vel=2.0,
        max_joint_acc=3.0,
        max_linear_vel=0.5,
        max_angular_vel=1.0,
        max_linear_acc=2.0,
        max_angular_acc=5.0,
    )


def very_compliant_coeffs() -> ControlCoeffsCfg:
    """Legacy cautious limit preset; impedance is selected independently by control."""
    return ControlCoeffsCfg(
        max_joint_vel=1.0,
        max_joint_acc=1.5,
    )


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

    ``qpos``'s command/streamed schema also backs ``control/qpos_overdamped`` —
    the same table row with NRT_JOINT_IMPEDANCE and explicit damping.
    """

    kind: str = MISSING          # qpos | qvel | end_effector | eef_vel | force
    mode: str = MISSING          # flexivrdk.Mode name, e.g. NRT_JOINT_POSITION
    send_fn: str = MISSING       # Robot send method, e.g. SendJointPosition
    command: Dict[str, int] = MISSING   # command field -> dim (full command schema)
    streamed: List[str] = field(default_factory=list)   # fields posted per tick on the setpoint channel

    channel: ControlChannelCfg = field(default_factory=ControlChannelCfg)

    # Impedance belongs to the control choice (conf/control/*.yaml), not a task phase.
    joint_impedance: Optional[JointImpedanceCfg] = None
    cartesian_impedance: Optional[CartesianImpedanceCfg] = None

    # Structural force-control config (cartesian `force` kind only; NOT a tunable coeff):
    force_control_axes: Optional[List[bool]] = None        # [6] [X,Y,Z,Rx,Ry,Rz] force-controlled axes
    force_control_frame: Optional[ForceControlFrameCfg] = None
    force_axis_max_linear_vel: Optional[List[float]] = None  # [3] SetForceControlAxis vel cap [m/s]


# ---------------------------------------------------------------------------
# Task schema — a manipulation task with per-phase templates
# ---------------------------------------------------------------------------


@dataclass
class RecordingCfg:
    """Dataset-export engineering: how LeRobot episodes are written to disk.

    Deliberately separate from the task: these knobs describe the recording
    *machinery* (destination root, encoders, writer threading), not what is being
    demonstrated. One shared node for the whole run — a rig may override
    ``root`` to quarantine its output (e.g. the bench rig records under
    ``datasets/bench``). Composed from the ``recording`` group.
    """

    root: str = "datasets"
    """Directory holding the LeRobot dataset(s); resolved absolute against the launch cwd."""

    video: bool = True
    """Store camera views as encoded MP4 (LeRobot ``use_videos``); False => PNG frames."""

    streaming_encoding: bool = True
    """Encode video in real time as frames arrive (no PNG staging; no inter-episode stall)."""

    image_writer_processes: int = 0
    """Async image-writer subprocesses (0 => threads only). Keeps disk I/O off the loop."""

    image_writer_threads: int = 4
    """Async image-writer threads (per camera) so recording never blocks the loop."""

    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub after each episode."""

    resume: bool = True
    """Append to an existing dataset at ``root/repo_id`` instead of failing if present."""

    keyboard: bool = True
    """Use tty keys to end/discard/stop episodes; auto-disabled when stdin is not a tty."""


@dataclass
class CollectionCfg:
    """Collection-phase template: teleoperated demonstration gathering.

    Demos are variable-length (the operator ends each one), so collection is
    bounded by a count of episodes, not a timestep horizon. The collection loop
    (:mod:`dual_flexiv_control.collection`) runs at :attr:`frequency_hz`: each
    tick it reads the FACTR leaders + the follower proprio streams, posts the
    converted joint setpoints (commanding the arms), samples every camera's
    latest frame (software-synchronised), and records one LeRobot frame.
    The dataset-export machinery lives in :class:`RecordingCfg`.
    """

    num_episodes: int = MISSING
    """How many demonstration episodes to teleoperate and record."""

    repo_id: str = MISSING
    """LeRobot dataset id (``<namespace>/<name>``); the on-disk folder under the
    recording root. Required per task so tasks never silently share a dataset."""

    coeffs: ControlCoeffsCfg = field(default_factory=compliant_coeffs)
    """Motion/contact limits and stiffness scale during collection. Base impedance
    lives on the selected policy controller."""

    frequency_hz: float = 15.0
    """Collection loop rate: command + record cadence (also the dataset ``fps``)."""


@dataclass
class EvalCfg:
    """Evaluation-phase template: online policy rollouts.

    A rollout has no operator to end it, so it is bounded by a fixed horizon.
    The eval loop's observation schema mirrors collection by construction (same
    ``state_signals`` + cameras + instruction — see
    :mod:`dual_flexiv_control.policy.observation`), so only rollout-specific
    knobs live here.
    """

    num_timesteps: int = MISSING
    """Rollout horizon — max timesteps before an eval episode is cut off."""

    coeffs: ControlCoeffsCfg = field(default_factory=stiff_coeffs)
    """Controller coefficients during eval; baseline stiffness and faster limits.
    Swap with ``control_coeffs@task.eval.coeffs=<preset>``."""

    frequency_hz: float = 15.0
    """Eval loop rate: one policy action step per tick. Must match the fps the
    policy was trained at (normally ``collection.frequency_hz``)."""


@dataclass
class TaskCfg:
    """A manipulation task: one shared spec plus per-phase templates.

    ``language_instruction`` and ``state_signals`` are shared across both phases
    — together they define the task's observation schema (the instruction handed
    to the teleoperator / policy, and the proprio composition of
    ``observation.state``, which eval must mirror exactly for the policy to see
    what it was trained on). The ``collection`` and ``eval`` sub-configs hold
    only what differs between the two phases.
    """

    language_instruction: str = MISSING                            # shared by both phases
    state_signals: List[str] = field(default_factory=lambda: ["q"])
    """Proprio signals concatenated (per arm, side order) into ``observation.state``
    — shared by collection (recorded) and eval (observed) by construction."""

    collection: CollectionCfg = field(default_factory=CollectionCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass
class JointConventionCfg:
    """Raw FACTR/Dynamixel → canonical DFC/Rizon mapping.

    This belongs to a DFC ``FactrLeaderCfg``, never an ``ArmCfg``/follower
    setting. ``offsets_deg`` is added
    per-joint after converting the leader's radians to degrees; ``sign_flip_joints``
    negates those joint indices, wraps the result to ``[-180, 180)``, and converts
    it back to radians. The canonical branch prevents raw encoder turns from becoming
    literal multi-revolution follower commands. ``drop_trailing`` discards FACTR's
    trailing gripper value(s) (its payload is ``DoF+1``).

    ``gripper_open``/``gripper_closed`` calibrate that trailing gripper value. FACTR
    serves it as an un-normalized servo angle in radians. Both endpoints are required
    by the strict ingestion contract and map open→0 and closed→1 (clipped).
    """

    offsets_deg: List[float] = field(default_factory=list)
    sign_flip_joints: List[int] = field(default_factory=list)
    drop_trailing: int = 1
    gripper_open: Optional[float] = None       # raw FACTR gripper value [rad] mapped to normalized 0.0
    gripper_closed: Optional[float] = None      # raw FACTR gripper value [rad] mapped to normalized 1.0


@dataclass
class FactrTransformCfg:
    """Canonical DFC/Rizon → FACTR dynamics-model axis directions.

    The zero offset is deliberately not persisted: FACTR derives it at launch from
    the captured :attr:`FactrLeaderCfg.home_q_rad` and its authoritative model-home
    target. This leaves no independently editable derived calibration value.
    """

    signs: List[float] = field(default_factory=list)


@dataclass
class FactrLeaderCfg:
    """All calibration for one physical FACTR leader, expressed on the DFC side."""

    raw_to_dfc: JointConventionCfg = field(default_factory=JointConventionCfg)
    home_q_rad: List[float] = field(default_factory=list)
    """The leader's calibration/home pose in canonical DFC/Rizon coordinates."""

    dfc_to_factr: FactrTransformCfg = field(default_factory=FactrTransformCfg)
    """Conversion used only to derive FACTR's runtime inverse-dynamics model."""


@dataclass
class GripperCfg:
    """Follower end-effector gripper actuation (opt-in; teleoperated from FACTR).

    Off by default (``enabled=False``): a rig with no gripper, or an arm whose
    gripper should not move, is completely unaffected — no gripper channel is
    opened and the arm never touches ``flexivrdk.Gripper``. When enabled, the
    brain posts the leader's normalized trigger (0=open, 1=closed — see
    :func:`~dual_flexiv_control.control.normalize_gripper`, calibrated via the
    convention's ``gripper_open``/``gripper_closed``) on a dedicated latest-wins
    channel (``cmd/<side>/gripper``), and the arm maps it to a physical width and
    drives the gripper with ``Gripper.Move``.

    ``name`` is the gripper's device name as registered in Flexiv Elements Studio
    — the argument to ``Gripper.Enable(name)``; it is deployment-specific, so it
    MUST be set to enable actuation (an empty name disables it with a warning).
    ``velocity``/``force`` are the ``Move`` limits (clamped to the gripper's own
    reported ``params()`` range). The physical open/closed widths are taken live
    from ``params()`` (``max_width``→open, ``min_width``→closed) unless overridden.
    """

    enabled: bool = False
    """Master switch: actuate the follower gripper from the leader trigger."""

    name: str = ""
    """``Gripper.Enable(name)`` device name (from Flexiv Elements Studio). Required
    to actuate — an empty name leaves the gripper untouched (logged once)."""

    velocity: float = 0.1
    """``Move`` velocity [m/s], clamped to the gripper's reported ``[min,max]_vel``."""

    force: float = 20.0
    """``Move`` force [N], clamped to the gripper's reported ``[min,max]_force``."""

    move_rate_hz: float = 15.0
    """Max ``Move`` command rate. The control loop is far faster (``control_rate_hz``);
    grippers can't accept every-tick commands, so sends are throttled to this."""

    deadband: float = 0.02
    """Minimum change in the normalized target (0..1) that triggers a new ``Move``;
    smaller jitter is ignored so the gripper isn't dithered."""

    init_on_start: bool = True
    """Run ``Gripper.Init()`` (homing) once when a control session first actuates
    the gripper. Homing is a physical motion — disable if the gripper is already
    initialized and re-homing at session start is undesirable."""

    open_width: Optional[float] = None
    """Override the physical width [m] mapped from normalized 0 (fully open). None
    => the gripper's reported ``params().max_width``."""

    closed_width: Optional[float] = None
    """Override the physical width [m] mapped from normalized 1 (fully closed). None
    => the gripper's reported ``params().min_width``."""


@dataclass
class ArmCfg:
    """One Flexiv arm: hardware settings plus the active policy controller."""

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
    control_eager_enable: bool = True    # servo-on (Enable/brake release) BEFORE waiting
                                         # for the brain's channels, overlapping it with
                                         # the consumer's spawn — cuts seconds off every
                                         # run start. Off: enable only after the channels
                                         # attach (the arm is never enabled for a consumer
                                         # that dies before publishing them).
    gripper: GripperCfg = field(default_factory=GripperCfg)   # follower gripper (opt-in)


@dataclass
class CameraCfg:
    """One camera publishing image streams (one process per camera).

    A camera publishes one stream per entry in ``views`` — canonical view names
    (see :mod:`dual_flexiv_control.cameras`): ZED ``left``/``right`` or
    RealSense ``color`` RGB (uint8, HxWx3), and optional ``depth`` (float32,
    HxW, metres). Per-stream ``dim`` is
    *derived* from ``width``/``height`` (× channels), so image sizes are never
    hand-computed. Streams are named ``"cam/<name>/<view>"`` where ``<name>`` is
    the camera's key in :attr:`Config.cameras` (e.g. ``cam/wrist_left/left``).

    For ZED, ``resolution`` is the SDK ``sl.RESOLUTION`` enum name handed to the real
    camera; ``width``/``height`` must be what that resolution yields (the real
    source validates this at open and fails fast on mismatch). Valid enums are
    model-specific — ZED 2: HD2K/HD1080/HD720/VGA; ZED X (Nano): HD1200/HD1080/
    SVGA — confirm against your installed SDK.
    """

    backend: str = "zed"          # "zed" | "realsense"
    model: str = MISSING           # informational model name
    serial: str = ""              # device serial; "" => first available
    auto_serial: bool = False     # bind to a connected camera's serial at open
                                  # (overrides `serial`; use only when exactly one
                                  # ZED is attached, e.g. single-camera bringup)
    placement: str = MISSING      # "wrist_left" | "wrist_right" | "static"
    resolution: str = "HD720"     # ZED enum name; informational for RealSense
    width: int = MISSING          # frame width  [px]; must match `resolution`
    height: int = MISSING         # frame height [px]; must match `resolution`
    fps: float = 30.0             # capture rate -> this producer's rate_hz
    depth_mode: str = "NONE"      # sl.DEPTH_MODE name; must be != NONE if "depth" in views
    align_depth: bool = True       # RealSense: align depth pixels to the color frame
    views: List[str] = MISSING    # canonical views to publish, e.g. ["left"] or ["left","right"]
    capacity: int = 16            # ring depth (frames retained) for each of this camera's streams

    # -- extrinsics/intrinsics for the dashboard's RGB-D overlay (viz only) ----
    # Pose of the camera's *optical* frame (X right, Y down, Z forward) in the
    # robot scene, anchored to `pose_frame`: "world" (the pedestal-top origin of
    # the robot view) or an arm's URDF base — "mount_left" | "mount_right".
    # `pose_rpy` is URDF-style fixed-axis roll/pitch/yaw [rad]. `hfov_deg` is the
    # horizontal FOV used for the pinhole projection (from factory calibration:
    # hfov = 2*atan((W/2)/fx)); lens-dependent, so set it per camera.
    pose_frame: str = "world"
    pose_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose_rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hfov_deg: float = 101.0


@dataclass
class FactrServerCfg:
    """One FACTR leader-arm WebSocket stream.

    ``ws://{host}:{port}/{endpoint}`` pushes typed reading and telemetry frames
    for that leader. ``{side}`` in the endpoint is expanded by the client. A
    reading contains ``dof`` values: arm joints plus the trailing gripper.
    """

    host: str = "localhost"
    port: int = MISSING
    endpoint: str = "ws/{side}"
    request_timeout_s: float = 0.5
    dof: int = 7


@dataclass
class FactrLaunchCfg:
    """Launching the external FACTR-Server processes from the session daemon.

    Replaces the FACTR-Server repo's VS Code "Launch EVERYTHING" task: when
    ``enabled`` (and not ``runtime.sim``), the daemon owns one grav-comp teleop
    process per leader arm plus the WebSocket/control relay, supervised like the
    hardware
    nodes (see :class:`~dual_flexiv_control.interfaces.factr.FactrServerSupervisor`)
    and driven from the dashboard's Teleop leaders panel. The daemon injects
    each leader's DFC-owned calibration into its teleop process at launch;
    teleops never infer calibration from their startup pose.

    The processes run outside our environment on purpose: ``python_exe`` is the
    system interpreter (conda's python cannot load rclpy) and ``setup_scripts``
    are sourced first (ROS 2 + the FACTR workspace), from ``workdir`` (the
    ``FACTR_Teleop`` directory — module paths and relative setup scripts resolve
    against it).
    """

    enabled: bool = False
    """Master switch (rig-provided). Off: the daemon never touches FACTR-Server
    and the dashboard shows no launch controls (external launch still works)."""

    workdir: str = "~/FACTR-Server/FACTR_Teleop"
    """The FACTR_Teleop directory: cwd for every process; ``~`` expands."""

    python_exe: str = "/usr/bin/python3"
    """System python (3.10) — conda's 3.13 cannot load rclpy."""

    setup_scripts: List[str] = field(
        default_factory=lambda: ["/opt/ros/humble/setup.bash", "install/setup.bash"]
    )
    """Sourced (in order) before exec; relative paths resolve against ``workdir``."""

    teleop_modules: Dict[str, str] = field(default_factory=dict)
    """side -> python module of that leader's grav-comp teleop (torque ON), e.g.
    left: the single-board ``factr_rizon_teleop``, right: ``factr_rizon_dual_board``.
    Only sides present both here and in ``factr.servers`` get a process."""

    api_module: str = "src.factr_fastapi.factr_fastapi.factr_api"
    """ROS->WebSocket/control relay for every leader (left :5000, right :5001)."""

    api_ports: List[int] = field(default_factory=lambda: [5000, 5001])
    """Ports freed (``fuser -k``) before the relay spawns — a stale/orphaned
    relay holding them would kill the fresh one at bind."""

    calib_delay_s: float = 30.0
    """Legacy/manual launch countdown in seconds. It only delays process spawn;
    calibration is already supplied by DFC."""

    calib_pose: str = "[0, 0, 0, 1.57, 0, 0, 0] (J4 ≈ 90°)"
    """Deprecated dashboard hint retained for status-payload compatibility."""

    stop_grace_s: float = 10.0
    """Cooperative window after SIGINT (the only stop that de-energizes the
    servos) before SIGTERM/SIGKILL escalation."""


@dataclass
class FactrCfg:
    """FACTR teleop: one WebSocket per leader arm, consumed by ONE producer.

    The :class:`~dual_flexiv_control.interfaces.factr.FactrInterface` node is
    the single WebSocket consumer: it reads each configured server's latest
    cached frame at ``rate_hz`` and
    publishes each leader's raw payload as ``factr/raw/<side>``, its DFC/Rizon
    conversion as ``factr/<side>``, and live FACTR control telemetry as
    per-field ``factr/telemetry/<side>/<field>`` streams. Consumers attach
    read-only to those streams — never to the servers — so control and the
    viewer see identical samples by construction.
    """

    servers: Dict[str, FactrServerCfg] = field(default_factory=dict)
    leaders: Dict[str, FactrLeaderCfg] = field(default_factory=dict)
    """side → DFC-owned leader calibration. Its keys must match ``servers``."""

    rate_hz: float = 100.0
    """The producer's shared-memory publish rate. Far above the collection
    frequency (15 Hz) and below the FACTR WebSocket's 200 Hz broadcast rate."""

    max_age_s: float = 0.5
    """Freshness gate for readers: a ``factr/<side>`` sample older than this is
    treated as a leader dropout (consumers hold their last real target; the
    dashboard shows the leader as disconnected)."""

    launch: FactrLaunchCfg = field(default_factory=FactrLaunchCfg)
    """Daemon-managed launch of the FACTR-Server processes (off by default)."""


@dataclass
class IndexCopyCfg:
    """A human-readable vector copy: ``target[...] = source[...]``."""

    name: str = ""
    source: List[int] = field(default_factory=list)
    target: List[int] = field(default_factory=list)


@dataclass
class IndexConstantCfg:
    """A constant assigned to one or more target-vector indices."""

    name: str = ""
    target: List[int] = field(default_factory=list)
    value: float = 0.0


@dataclass
class OpenPIVectorCfg:
    """One OpenPI wire vector and its projection from the canonical DFC vector.

    ``dim=0`` means pass the complete canonical vector through unchanged.
    """

    key: str = "observation/state"
    dim: int = 0
    slices: List[IndexCopyCfg] = field(default_factory=list)
    constants: List[IndexConstantCfg] = field(default_factory=list)


@dataclass
class OpenPIImageCfg:
    """One canonical DFC camera view mapped to an OpenPI request image."""

    source: str = ""
    key: str = ""
    layout: str = "hwc"


@dataclass
class OpenPIRequestCfg:
    """General OpenPI request conventions plus YAML-owned vector/image mapping."""

    state: OpenPIVectorCfg = field(default_factory=OpenPIVectorCfg)
    images_key: str = ""
    images: List[OpenPIImageCfg] = field(default_factory=list)
    prompt_key: str = "prompt"
    image_key_template: str = "observation/images/{camera}"
    image_keys: Dict[str, str] = field(default_factory=dict)


@dataclass
class OpenPIActionsCfg:
    """OpenPI response actions projected into the canonical DFC action vector."""

    key: str = "actions"
    dim: int = 0
    slices: List[IndexCopyCfg] = field(default_factory=list)
    state_holds: List[IndexCopyCfg] = field(default_factory=list)
    constants: List[IndexConstantCfg] = field(default_factory=list)


@dataclass
class OpenPIResponseCfg:
    actions: OpenPIActionsCfg = field(default_factory=OpenPIActionsCfg)


@dataclass
class OpenPIEndpointCfg:
    """All OpenPI endpoint idiosyncrasies; embodiment copies remain in YAML."""

    request: OpenPIRequestCfg = field(default_factory=OpenPIRequestCfg)
    response: OpenPIResponseCfg = field(default_factory=OpenPIResponseCfg)


@dataclass
class PolicyCfg:
    """The eval policy: which server to query and how to shape requests/actions.

    The eval loop builds a *canonical* observation — the same LeRobot-keyed dict
    collection records (``observation.state``, ``observation.images.<cam>``,
    ``task``) — and a
    :class:`~dual_flexiv_control.policy.adapter.PolicyEndpointAdapter`
    (selected by ``adapter``) maps it onto one server's wire format and parses
    the response back into an action chunk. Supporting a new policy-server
    protocol = registering a new adapter (and, if the transport differs, a new
    client); this config stays the single switchboard.
    """

    control: ControlCfg = MISSING
    """Policy action semantics and matching physical controller. A qvel policy
    selects qvel here; an absolute-joint policy selects qpos. Collection recording
    remains controller-independent."""

    kind: str = "remote"
    """``remote`` — a live policy server; ``hold`` — no server: repeat the measured
    joint positions (stand still), an end-to-end smoke test of the eval path."""

    adapter: str = "openpi"
    """Registered request/response endpoint adapter (see ``policy/adapter.py``)."""

    transport: str = "websocket"
    """Wire transport to the server, chosen independently of ``adapter``:
    ``websocket`` — openpi msgpack-numpy frames; ``http`` — the ACME
    multipart/form-data ``POST /predict`` protocol (see ``policy/client.py``)."""

    host: str = "localhost"
    port: int = 8000
    api_key: Optional[str] = None
    connect_timeout_s: float = 60.0
    """Startup budget: how long to keep retrying the initial server connection."""

    infer_timeout_s: float = 10.0
    """Per-inference response timeout; a slow/hung server holds the arms instead."""

    max_consecutive_errors: int = 3
    """Abort an eval after this many consecutive inference failures.

    A small retry budget rides through a transient connection reset or policy
    server restart without hiding a persistent adapter/checkpoint error forever.
    Any successful inference resets the counter.
    """

    replan_steps: int = 0
    """Actions executed from each returned chunk before re-inferring (receding
    horizon); 0 executes the full chunk (open-loop within a chunk)."""

    # -- OpenPI endpoint protocol and declarative embodiment projection ----------
    openpi: OpenPIEndpointCfg = field(default_factory=OpenPIEndpointCfg)
    """OpenPI request/response rules. All index copies and held dimensions are
    declared in the selected policy YAML rather than hard-coded per embodiment."""

    # -- request mapping (ACME endpoint adapter) --------------------------------
    acme_image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "exterior_image_1_left": "static_left",
            "exterior_image_2_left": "static_right",
            "wrist_image_left": "wrist_left",
        }
    )
    """ACME multipart image slot -> canonical camera view (the key after
    ``observation.images.``). The ACME server validates strictly: every mapped
    slot must resolve to a present view. Map a slot to ``""`` to omit it (the
    server will 400 unless it also drops that key)."""

    qpos_slice: List[int] = field(default_factory=lambda: [0, 7])
    """``[start, stop]`` selecting the single arm's 7-dim ``qpos`` out of the
    (bimanual) ``observation.state``. Default is the left arm (first 7); use
    ``[7, 14]`` to drive from the right arm's joints."""

    obs_steps: int = 1
    """ACME ``obs_steps`` form field (the server consumes only the first
    timestep, so 1 is correct for this single-frame client)."""

    gripper_force: float = 0.0
    """Gripper input sent to the server, in ``[0, 1]`` (0=open, 1=closed). The
    ``pi05_base_qvel_acme`` checkpoint requires it even though ``/conventions``
    lists it as optional. The ``q``-only observation carries no gripper proprio,
    so this constant is sent every tick (default open); set to the current grip
    state, or wire real gripper proprio into ``state_signals``, for closed-loop
    grasping."""


@dataclass
class SkillCfg:
    """Teach-and-repeat skill replay (``runtime.phase=skill``).

    A skill is a taught joint trajectory saved under :attr:`root` (see
    :mod:`dual_flexiv_control.skills`); a skill run streams it back through the
    normal ``qpos`` control path. Task-independent — skills live beside the
    task/rig axes, so this is a top-level config, not a per-task template.
    """

    root: str = "skills"
    """Directory holding saved skills; resolved absolute against the launch cwd."""

    name: Optional[str] = None
    """The skill to repeat (its file stem). Required for a skill run: set per run
    with ``skill.name=<name>`` (the dashboard's Repeat button does this)."""

    coeffs: ControlCoeffsCfg = field(default_factory=default_coeffs)
    """Controller coefficients during a skill replay; the moderate preset by
    default (tracking a known-good demonstrated path). Swap with
    ``+control_coeffs@skill.coeffs=<preset>``."""

    frequency_hz: Optional[float] = None
    """Replay rate override; ``None`` replays at the fps the skill was taught at."""

    start_tolerance: float = 0.1
    """[rad] L-inf gate: the replay holds the first target until every driven
    arm's measured ``q`` is within this of it (the arm-side bootstrap MoveJs
    there first)."""

    start_timeout_s: float = 60.0
    """Abort the run if an arm has not reached the start pose within this window
    (an E-stopped/halted arm must fail loudly, not freeze silently)."""

    settle_s: float = 1.0
    """How long the final target is held after the last frame before the run ends."""


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
    phase: str = "collection"      # collection | eval | skill — selects the run's consumer + coeffs
    #: Shutdown window (s) granted to the recording node to finalize the in-progress
    #: episode's video encode before SIGKILL. Saving a long episode drains a
    #: multi-thousand-frame encode that far exceeds a hardware node's teardown, so
    #: this is generous; healthy short episodes finalize in seconds regardless.
    save_grace_s: float = 300.0


@dataclass
class Config:
    """Top-level composed configuration."""

    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    brain: BrainCfg = field(default_factory=BrainCfg)
    recording: RecordingCfg = field(default_factory=RecordingCfg)  # dataset-export machinery
    factr: FactrCfg = MISSING     # provided by the selected rig (composed from `factr`)
    task: TaskCfg = MISSING       # the active task (composed from the `task` group)
    policy: PolicyCfg = MISSING   # eval policy-server client (composed from the `policy` group)
    skill: SkillCfg = field(default_factory=SkillCfg)  # teach-and-repeat replay (phase=skill)
    # Populated by the selected `rig` (package-directed defaults: arm@arms.left, ...).
    arms: Dict[str, ArmCfg] = field(default_factory=dict)
    # Likewise camera@cameras.<name> — set per rig.
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
    cs.store(group="recording", name="base_recording", node=RecordingCfg)
    cs.store(group="factr", name="base_factr", node=FactrCfg)
    cs.store(group="task", name="base_task", node=TaskCfg)
    cs.store(group="policy", name="base_policy", node=PolicyCfg)
    cs.store(group="skill", name="base_skill", node=SkillCfg)
    cs.store(group="arm", name="base_arm", node=ArmCfg)
    cs.store(group="camera", name="base_camera", node=CameraCfg)
    cs.store(group="control", name="base_control", node=ControlCfg)
    cs.store(group="control_coeffs", name="base_control_coeffs", node=ControlCoeffsCfg)
    # Named presets (no YAML files — the values live in the factories above):
    cs.store(group="control_coeffs", name="compliant", node=compliant_coeffs())
    cs.store(group="control_coeffs", name="stiff", node=stiff_coeffs())
    cs.store(group="control_coeffs", name="default", node=default_coeffs())
    cs.store(group="control_coeffs", name="very_compliant", node=very_compliant_coeffs())
