"""Map a Flexiv RDK 2.x ``RobotStates`` snapshot onto our proprio signal vectors.

The field names below were verified against the installed ``flexivrdk`` 2.1.0
(``RobotStates`` per-field docstrings), which differ from older RDK 1.x naming:

    our signal   RDK 2.x field        dim  meaning
    ----------   ------------------   ---  ----------------------------------
    q            q                    7    link-side joint positions   [rad]
    dq           dq                   7    link-side joint velocities  [rad/s]
    tau          tau                  7    measured joint torques      [Nm]
    wrench       tcp_wrench_local     6    ext. TCP wrench, TCP frame  [N,Nm]
                 (or tcp_wrench)            ext. TCP wrench, world frame
    eef          tcp_pose             7    TCP pose [x,y,z,qw,qx,qy,qz][m]
    eef_vel      tcp_twist            6    TCP twist [v(3), w(3)]  [m/s,rad/s]

``robot.states()`` returns ``dict[JointGroup, RobotStates]``; the caller selects
the single-arm joint group (via ``RobotInfo.single_arm_groups``) and passes that
group's ``RobotStates`` here.
"""

from __future__ import annotations

import numpy as np

#: Wrench frame -> RDK field. "local" == TCP frame (matches "TCP wrench").
_WRENCH_FIELD = {"local": "tcp_wrench_local", "world": "tcp_wrench"}


def map_states(rs, wrench_frame: str, dtype: str = "float64") -> dict[str, np.ndarray]:
    """Extract the six proprio vectors from one joint group's ``RobotStates``."""
    wrench_attr = _WRENCH_FIELD[wrench_frame]
    return {
        "q": np.asarray(rs.q, dtype=dtype),
        "dq": np.asarray(rs.dq, dtype=dtype),
        "tau": np.asarray(rs.tau, dtype=dtype),
        "wrench": np.asarray(getattr(rs, wrench_attr), dtype=dtype),
        "eef": np.asarray(rs.tcp_pose, dtype=dtype),
        "eef_vel": np.asarray(rs.tcp_twist, dtype=dtype),
    }
