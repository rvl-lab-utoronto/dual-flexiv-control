"""Data sources for a single Flexiv arm.

:class:`FlexivSource` wraps a real ``flexivrdk.Robot`` connection (RDK 2.x).
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


class FlexivSource:
    """Real ``flexivrdk`` connection to one arm; read-only state streaming.

    We only *read* states, so we do not enable the robot or release brakes by
    default. Construction connects to the robot; ``read()`` returns the
    single-arm joint group's ``RobotStates`` each call (a non-blocking snapshot).
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
        self._group = None

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
            # Releasing brakes / going operational is a control action; only do
            # it when explicitly requested. (Reading states does not need it.)
            log.info("%s: ServoOn to reach operational state", self.serial)
            robot.ServoOn()
            while not robot.operational():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.serial} not operational in time")
                time.sleep(0.01)

        self._robot = robot
        self._group = self._select_group(robot)
        log.info("%s connected; joint group = %s", self.serial, self._group)

    def _select_group(self, robot):
        """Pick the single-arm joint group whose states we publish."""
        import flexivrdk  # noqa: F401

        info = robot.info()
        states = robot.states()
        candidates = list(getattr(info, "single_arm_groups", []) or [])
        # Prefer a declared single-arm group that is actually present in states().
        for group in candidates:
            rs = states.get(group)
            if rs is not None and len(rs.q) == self.dof:
                return group
        # Fallback: any populated group with the expected DoF and a TCP pose.
        for group, rs in states.items():
            if len(rs.q) == self.dof and len(rs.tcp_pose) == POSE_SIZE:
                return group
        raise RuntimeError(
            f"{self.serial}: could not find a {self.dof}-DoF single-arm joint group "
            f"in states() keys {list(states.keys())}"
        )

    def read(self):
        """Return this arm's ``RobotStates`` snapshot."""
        return self._robot.states()[self._group]

    def close(self) -> None:
        # We never took control, so there is nothing to stop. Drop the handle;
        # the RDK client shuts down its services when garbage collected.
        self._robot = None
        self._group = None


class FakeFlexivSource:
    """Synthetic states for hardware-free runs. Same shape as ``RobotStates``."""

    def __init__(self, serial: str, *, dof: int = 7, seed: int = 0) -> None:
        self.serial = serial
        self.dof = dof
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(seed)
        self._phase = self._rng.uniform(0, 2 * math.pi, size=dof)

    def open(self) -> None:
        self._t0 = time.monotonic()

    def read(self):
        t = time.monotonic() - self._t0
        dof = self.dof
        q = 0.5 * np.sin(t + self._phase)
        dq = 0.5 * np.cos(t + self._phase)
        tau = 2.0 * np.sin(0.5 * t + self._phase)
        # TCP pose: small circular motion + identity-ish quaternion.
        pos = np.array([0.5 + 0.05 * math.sin(t), 0.05 * math.cos(t), 0.4])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tcp_pose = np.concatenate([pos, quat])
        tcp_twist = np.array(
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
            tcp_twist=tcp_twist,
            tcp_wrench=wrench,
            tcp_wrench_local=wrench,
        )

    def close(self) -> None:
        pass
