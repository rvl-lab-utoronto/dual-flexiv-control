"""Data sources for a single Flexiv arm.

:class:`FlexivSource` wraps a real ``flexivrdk.Robot`` connection (RDK 1.8).
:class:`FakeFlexivSource` synthesises plausible states so the entire
multiprocess + shared-memory pipeline can run with no hardware. Both expose the
same tiny interface: ``open()`` then repeated ``read()`` returning an object
with the RDK ``RobotStates`` fields we consume.
"""

from __future__ import annotations

import logging
import math
import time
from types import SimpleNamespace

import numpy as np

from ...proprio import CART_DOF
from ...proprio import POSE_SIZE

log = logging.getLogger(__name__)


def _quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``[qw, qx, qy, qz]`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _integrate_pose(pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a TCP pose ``[x,y,z,qw,qx,qy,qz]`` by a world-frame twist for ``dt``.

    Linear: ``p += v·dt``. Angular: first-order quaternion integration
    ``q += 0.5·(ω⊗q)·dt`` (ω a world-frame angular velocity), renormalised. Used to
    realise end-effector *velocity* control, for which flexivrdk 1.8 has no native
    mode — the arm integrates the commanded twist into the ``pose_d`` it sends.
    """
    pose = np.asarray(pose, dtype=np.float64)
    pos = pose[:3] + np.asarray(twist[:3], dtype=np.float64) * dt
    q = pose[3:7]
    w = np.asarray(twist[3:6], dtype=np.float64)
    qdot = 0.5 * _quat_mult(np.array([0.0, w[0], w[1], w[2]]), q)
    qn = q + qdot * dt
    n = float(np.linalg.norm(qn))
    qn = qn / n if n > 1e-9 else q
    return np.concatenate([pos, qn])


class SafetyHalt(RuntimeError):
    """Raised by the control send path when a safety gate trips; the caller halts."""


class FlexivSource:
    """Real ``flexivrdk`` (RDK 1.x) connection to one arm.

    One ``Robot`` connection per arm; ``read()`` returns that arm's flat
    ``RobotStates`` snapshot. Read-only until :meth:`enter_control` is called.
    """

    def __init__(
        self,
        serial: str,
        *,
        dof: int = 7,
        require_operational: bool = False,
        verbose: bool = False,
        connect_timeout_s: float = 30.0,
    ) -> None:
        self.serial = serial
        self.dof = dof
        self.require_operational = require_operational
        self.verbose = verbose
        self.connect_timeout_s = connect_timeout_s
        self._robot = None
        #: Integrated target for velocity kinds (q_d for qvel, pose_d for eef_vel).
        self._control_target = None

    def open(self) -> None:
        import flexivrdk  # imported in the child process only

        log.info("connecting to Flexiv %s", self.serial)
        robot = flexivrdk.Robot(self.serial, verbose=self.verbose)

        # Clear any standing fault so states stream cleanly.
        if robot.fault():
            log.warning("%s has a fault; attempting ClearFault()", self.serial)
            if not robot.ClearFault():
                raise RuntimeError(f"failed to clear fault on {self.serial}")

        deadline = time.monotonic() + self.connect_timeout_s
        while not robot.connected():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.serial} did not connect in {self.connect_timeout_s}s")
            time.sleep(0.01)

        if self.require_operational and not robot.operational():
            # Enabling (releasing brakes) is a control action; only do it when
            # explicitly requested. (Reading states does not need it.)
            log.info("%s: Enable to reach operational state", self.serial)
            robot.Enable()
            while not robot.operational():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.serial} not operational in time")
                time.sleep(0.01)

        self._robot = robot
        log.info("%s connected", self.serial)

    def read(self):
        """Return this arm's ``RobotStates`` snapshot.

        RDK 1.x ``states()`` returns a single flat ``RobotStates`` for the connected
        arm (one ``Robot`` connection per arm), not a dict keyed by joint group.
        """
        return self._robot.states()

    # -- control half (write path) --------------------------------------------
    #
    # All flexivrdk calls below are the verified 1.8.0 forms (flat API; one Robot
    # per arm): Enable() enables; states()/primitive_states() are flat; the joint
    # send is SendJointPosition(q_d, dq_d, dq_max, ddq_max) and the cartesian send is
    # SendCartesianMotionForce(pose, wrench, velocity, max_lin_vel, max_ang_vel,
    # max_lin_acc, max_ang_acc); setters take no joint-group arg. 1.8 is NRT-only
    # (no RT modes / no StreamJointPosition).

    def fault(self) -> bool:
        return bool(self._robot.fault())

    def stop(self) -> None:
        """Blocking stop → IDLE. Safe to call from teardown."""
        if self._robot is not None:
            self._robot.Stop()

    def enter_control(self) -> None:
        """Clear faults, enable the robot, and wait until operational (no mode yet)."""
        robot = self._robot
        if robot.fault():
            log.warning("%s: clearing fault before control", self.serial)
            if not robot.ClearFault():
                raise RuntimeError(f"failed to clear fault on {self.serial}")
        log.info("%s: Enable for control", self.serial)
        robot.Enable()
        deadline = time.monotonic() + self.connect_timeout_s
        while not robot.operational():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.serial} not operational in time")
            time.sleep(0.01)

    def start_control(self, ctrl_cfg, coeffs, first_fields: dict, rs, abort=None) -> bool:
        """Switch into the control mode, bootstrap to the first target, apply coeffs.

        ``first_fields`` is the first setpoint already split into named fields (see
        :func:`~dual_flexiv_control.control.slice_streamed`); ``rs`` is the current
        ``RobotStates`` (for integrator/pose initialisation). ``abort`` is an optional
        predicate polled during the (possibly multi-second) MoveJ bootstrap so a STOP
        or shutdown can interrupt it; returns ``False`` if the bootstrap was aborted.
        """
        import flexivrdk

        robot, mode = self._robot, flexivrdk.Mode
        kind = ctrl_cfg.kind

        if kind in ("qpos", "qvel"):
            if kind == "qpos":
                # Smooth MoveJ to the first commanded pose to avoid a startup jump.
                if not self._bootstrap_movej(first_fields["q_d"], abort):
                    return False
                robot.SwitchMode(mode.NRT_JOINT_POSITION)
            else:  # qvel: start integrating from the measured joint positions
                robot.SwitchMode(mode.NRT_JOINT_POSITION)
                self._control_target = np.asarray(rs.q, dtype=np.float64).copy()
        elif kind in ("end_effector", "eef_vel", "force"):
            robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
            # eef_vel integrates pose from the measured TCP pose; pose kinds track it.
            self._control_target = np.asarray(rs.tcp_pose, dtype=np.float64).copy()
        else:
            raise ValueError(f"unknown control kind {kind!r}")

        self._apply_coeffs(ctrl_cfg, coeffs)
        return True

    def _bootstrap_movej(self, q_d_rad: np.ndarray, abort=None) -> bool:
        """Drive smoothly to ``q_d`` (rad) via the MoveJ primitive (RDK 1.x flat form).

        Returns ``True`` on reaching the target, ``False`` if ``abort()`` fired mid-move
        (the caller then routes to a clean ``Stop()`` via teardown).
        """
        import flexivrdk

        robot, mode = self._robot, flexivrdk.Mode
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive(
            "MoveJ",
            {"target": flexivrdk.JPos(np.degrees(q_d_rad).tolist())},   # JPos q_m takes DEGREES
            block_until_started=True,
        )
        while True:
            if abort is not None and abort():
                log.info("%s: MoveJ bootstrap aborted", self.serial)
                return False
            # 1.x primitive_states() is a FLAT dict {str: value}; reachedTarget is int 1/0.
            if int(robot.primitive_states().get("reachedTarget", 0)) == 1:
                return True
            time.sleep(0.05)

    def _apply_coeffs(self, ctrl_cfg, coeffs) -> None:
        """Apply the coefficient setters valid for the active mode (after SwitchMode).

        RDK 1.x setters take no joint-group argument (one Robot per arm).
        """
        import flexivrdk

        robot = self._robot
        if ctrl_cfg.mode == "NRT_CARTESIAN_MOTION_FORCE":
            if coeffs.cartesian_impedance is not None:
                robot.SetCartesianImpedance(
                    coeffs.cartesian_impedance.K_x, coeffs.cartesian_impedance.Z_x
                )
            if coeffs.max_contact_wrench is not None:
                robot.SetMaxContactWrench(coeffs.max_contact_wrench)
            if coeffs.null_space_posture is not None:
                robot.SetNullSpacePosture(coeffs.null_space_posture)
            if ctrl_cfg.force_control_axes is not None:
                mlv = ctrl_cfg.force_axis_max_linear_vel or [1.0, 1.0, 1.0]
                robot.SetForceControlAxis(ctrl_cfg.force_control_axes, mlv)
            if ctrl_cfg.force_control_frame is not None:
                coord = getattr(flexivrdk.CoordType, ctrl_cfg.force_control_frame.root_coord)
                robot.SetForceControlFrame(coord, ctrl_cfg.force_control_frame.T_in_root)
        elif ctrl_cfg.mode == "NRT_JOINT_IMPEDANCE":
            if coeffs.joint_impedance is not None:
                robot.SetJointImpedance(coeffs.joint_impedance.K_q, coeffs.joint_impedance.Z_q)
        # NRT_JOINT_POSITION (qpos/qvel): no impedance setter applies; dq_max/ddq_max
        # ride on every SendJointPosition call instead.

    def send_control(
        self, ctrl_cfg, coeffs, fields: dict, rs, dt: float,
        *, safety_check: bool = False, tolerance: float = 0.5,
    ) -> None:
        """Build and send one command for the active kind (verified 1.8.0 flat forms).

        For joint kinds the L-inf safety gate is evaluated here, against the *effective*
        commanded ``q_d`` — including qvel's integrated target — and raises
        :class:`SafetyHalt` BEFORE committing the integrator or sending, so a tripped
        gate never advances qvel's open-loop integrator.
        """
        robot = self._robot
        kind = ctrl_cfg.kind

        if kind in ("qpos", "qvel"):
            dof = self.dof
            if kind == "qpos":
                q_d = np.asarray(fields["q_d"], dtype=np.float64)
                dq_d = np.asarray(fields.get("dq_d", np.zeros(dof)), dtype=np.float64)
            else:  # qvel: candidate integrated target (commit only if it passes the gate)
                dq_d = np.asarray(fields["dq_d"], dtype=np.float64)
                q_d = self._control_target + dq_d * dt
            if safety_check:
                err = float(np.max(np.abs(q_d - np.asarray(rs.q, dtype=np.float64))))
                if err > tolerance:
                    raise SafetyHalt(
                        f"L-inf joint error {err:.3f} rad > tol {tolerance:.3f} ({kind})"
                    )
            if kind == "qvel":
                self._control_target = q_d  # commit now that it passed
            dq_max = [float(coeffs.max_joint_vel)] * dof
            ddq_max = [float(coeffs.max_joint_acc)] * dof
            # 1.8 flat form: SendJointPosition(positions, velocities, max_vel, max_acc).
            robot.SendJointPosition(q_d.tolist(), dq_d.tolist(), dq_max, ddq_max)
        else:  # cartesian kinds
            if kind == "eef_vel":
                twist = np.asarray(fields["twist_d"], dtype=np.float64)
                self._control_target = _integrate_pose(self._control_target, twist, dt)
                pose_d = self._control_target
                wrench_d = np.zeros(6)
                twist_d = twist
            elif kind == "end_effector":
                pose_d = np.asarray(fields["pose_d"], dtype=np.float64)
                twist_d = np.asarray(fields.get("twist_d", np.zeros(6)), dtype=np.float64)
                wrench_d = np.zeros(6)
            else:  # force
                pose_d = np.asarray(fields.get("pose_d", self._control_target), dtype=np.float64)
                wrench_d = np.asarray(fields["wrench_d"], dtype=np.float64)
                twist_d = np.zeros(6)
            # 1.8 flat form: SendCartesianMotionForce(pose, wrench, velocity,
            # max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc).
            robot.SendCartesianMotionForce(
                pose_d.tolist(), wrench_d.tolist(), twist_d.tolist(),
                float(coeffs.max_linear_vel), float(coeffs.max_angular_vel),
                float(coeffs.max_linear_acc), float(coeffs.max_angular_acc),
            )

    def close(self) -> None:
        # We never took control, so there is nothing to stop. Drop the handle;
        # the RDK client shuts down its services when garbage collected.
        self._robot = None
        self._control_target = None


class FakeFlexivSource:
    """Synthetic states for hardware-free runs. Same shape as ``RobotStates``."""

    def __init__(self, serial: str, *, dof: int = 7, seed: int = 0) -> None:
        self.serial = serial
        self.dof = dof
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(seed)
        self._phase = self._rng.uniform(0, 2 * math.pi, size=dof)
        # Control sim state: when controlling, telemetry tracks the last command so
        # the L-inf safety gate passes and the full loop is exercised hardware-free.
        self._ctrl = False
        self._tracked_q = None
        self._tracked_pose = None
        self._control_target = None
        self.last_command = None   # introspection hook for tests

    def open(self) -> None:
        self._t0 = time.monotonic()

    def read(self):
        t = time.monotonic() - self._t0
        dof = self.dof
        if self._ctrl and self._tracked_q is not None:
            # Track the last commanded joint target (+ tiny noise) and pose.
            q = np.asarray(self._tracked_q, dtype=np.float64) + 1e-4 * np.sin(t + self._phase)
            dq = np.zeros(dof)
            tau = np.zeros(dof)
            tcp_pose = (
                np.asarray(self._tracked_pose, dtype=np.float64)
                if self._tracked_pose is not None
                else np.array([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
            )
            tcp_vel = np.zeros(6)
            wrench = np.zeros(6)
            return SimpleNamespace(
                q=q, dq=dq, tau=tau, tcp_pose=tcp_pose, tcp_vel=tcp_vel,
                ext_wrench_in_tcp=wrench, ext_wrench_in_world=wrench,
            )
        q = 0.5 * np.sin(t + self._phase)
        dq = 0.5 * np.cos(t + self._phase)
        tau = 2.0 * np.sin(0.5 * t + self._phase)
        # TCP pose: small circular motion + identity-ish quaternion.
        pos = np.array([0.5 + 0.05 * math.sin(t), 0.05 * math.cos(t), 0.4])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tcp_pose = np.concatenate([pos, quat])
        tcp_vel = np.array(
            [0.05 * math.cos(t), -0.05 * math.sin(t), 0.0, 0.0, 0.0, 0.0]
        )
        wrench = np.concatenate(
            [np.array([math.sin(t), math.cos(t), 0.5 * math.sin(2 * t)]), np.zeros(3)]
        )
        return SimpleNamespace(
            q=q,
            dq=dq,
            tau=tau,
            tcp_pose=tcp_pose,
            tcp_vel=tcp_vel,
            ext_wrench_in_tcp=wrench,
            ext_wrench_in_world=wrench,
        )

    # -- control half (no hardware; mirrors FlexivSource's contract) ----------

    def fault(self) -> bool:
        return False

    def stop(self) -> None:
        self._ctrl = False

    def enter_control(self) -> None:
        pass

    def start_control(self, ctrl_cfg, coeffs, first_fields: dict, rs, abort=None) -> bool:
        self._ctrl = True
        kind = ctrl_cfg.kind
        if kind == "qpos":
            self._tracked_q = np.asarray(first_fields["q_d"], dtype=np.float64).copy()
        elif kind == "qvel":
            self._control_target = np.asarray(rs.q, dtype=np.float64).copy()
            self._tracked_q = self._control_target.copy()
        else:  # cartesian kinds
            self._control_target = np.asarray(rs.tcp_pose, dtype=np.float64).copy()
            self._tracked_pose = self._control_target.copy()
            self._tracked_q = np.asarray(rs.q, dtype=np.float64).copy()
        return True

    def send_control(
        self, ctrl_cfg, coeffs, fields: dict, rs, dt: float,
        *, safety_check: bool = False, tolerance: float = 0.5,
    ) -> None:
        self.last_command = (ctrl_cfg.kind, {k: np.asarray(v) for k, v in fields.items()})
        kind = ctrl_cfg.kind
        if kind == "qpos":
            self._tracked_q = np.asarray(fields["q_d"], dtype=np.float64).copy()
        elif kind == "qvel":
            self._control_target = self._control_target + np.asarray(fields["dq_d"], dtype=np.float64) * dt
            self._tracked_q = self._control_target.copy()
        elif kind == "eef_vel":
            self._control_target = _integrate_pose(self._control_target, np.asarray(fields["twist_d"]), dt)
            self._tracked_pose = self._control_target.copy()
        elif kind == "end_effector":
            self._tracked_pose = np.asarray(fields["pose_d"], dtype=np.float64).copy()
        # force: pose tracks the motion target if present
        elif "pose_d" in fields:
            self._tracked_pose = np.asarray(fields["pose_d"], dtype=np.float64).copy()

    def close(self) -> None:
        pass
