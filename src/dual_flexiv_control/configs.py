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

Control configs lay out the *command schema* (RDK command-struct field -> dim)
plus the controller setters, aligned with flexivrdk 2.x (verified by API
introspection). They describe the SHAPE of control for the future control
implementation; no control is executed yet.
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
# Control schemas — aligned with flexivrdk 2.x command structs
# ---------------------------------------------------------------------------


@dataclass
class JointImpedanceCfg:
    """``SetJointImpedance(K_q, Z_q)`` — used by RT/NRT_JOINT_IMPEDANCE."""

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
class ControlCfg:
    """A control type: which RDK mode/stream to use and the command schema.

    ``command`` maps each command-struct field to its dimensionality — the schema
    of the control vector(s) the future controller will fill and stream. Optional
    sub-configs hold the controller setters relevant to that mode (impedance,
    force-control axes/frame, contact-wrench limits, null-space posture).
    """

    kind: str = MISSING          # qpos | qvel | end_effector | force
    mode: str = MISSING          # flexivrdk.Mode name, e.g. RT_JOINT_POSITION
    stream_fn: str = MISSING     # Robot streaming method, e.g. StreamJointPosition
    cmd_struct: str = MISSING    # RDK command struct, e.g. RtJointPositionCmd
    command: Dict[str, int] = MISSING   # struct field -> dim (the command schema)

    # Controller setters (present only where the mode uses them):
    joint_impedance: Optional[JointImpedanceCfg] = None
    cartesian_impedance: Optional[CartesianImpedanceCfg] = None
    force_control_axes: Optional[List[bool]] = None        # [6] [X,Y,Z,Rx,Ry,Rz]
    force_control_frame: Optional[ForceControlFrameCfg] = None
    max_linear_vel: Optional[List[float]] = None           # [3] force-axis vel cap [m/s]
    max_contact_wrench: Optional[List[float]] = None        # [6] [N,Nm]
    null_space_posture: Optional[List[float]] = None        # [DoF] [rad]


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass
class ArmCfg:
    """One Flexiv arm: connection, read settings, stream schemas, default controller."""

    serial: str = MISSING
    dof: int = 7
    wrench_frame: str = "local"          # local (TCP) | world
    require_operational: bool = False
    verbose_rdk: bool = False
    rate_hz: float = 1000.0
    streams: Dict[str, StreamCfg] = MISSING   # signal -> StreamCfg
    control: ControlCfg = MISSING        # this arm's control schema (composed from the `control` group)


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


@dataclass
class Config:
    """Top-level composed configuration."""

    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    brain: BrainCfg = field(default_factory=BrainCfg)
    factr: FactrCfg = MISSING
    # Populated by package-directed group defaults (arm@arms.left, control@arms.left.control, ...).
    arms: Dict[str, ArmCfg] = field(default_factory=dict)


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
    cs.store(group="arm", name="base_arm", node=ArmCfg)
    cs.store(group="control", name="base_control", node=ControlCfg)
