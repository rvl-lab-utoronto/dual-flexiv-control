"""Map a Flexiv RDK 1.8 ``RobotStates`` snapshot onto our proprio signal vectors.

The field names below were verified against the installed ``flexivrdk`` 1.8.0
(``RobotStates`` members), which differ from the RDK 2.x naming the code briefly
targeted (2.x renamed ``tcp_vel``→``tcp_twist`` and split the external wrench into
``tcp_wrench_local``/``tcp_wrench``):

    our signal   RDK 1.8 field         dim  meaning
    ----------   -------------------   ---  ----------------------------------
    q            q                     7    link-side joint positions   [rad]
    dq           dq                    7    link-side joint velocities  [rad/s]
    tau          tau                   7    measured joint torques      [Nm]
    wrench       ext_wrench_in_tcp     6    ext. TCP wrench, TCP frame  [N,Nm]
                 (or ext_wrench_in_world)   ext. TCP wrench, world frame
    eef          tcp_pose             7    TCP pose [x,y,z,qw,qx,qy,qz][m]
    eef_vel      tcp_vel              6    TCP velocity [v(3), w(3)] [m/s,rad/s]

In RDK 1.8 ``robot.states()`` returns a single flat ``RobotStates`` for the
connected arm (one ``Robot`` per arm — no joint-group dict), so the caller passes
that ``RobotStates`` straight through.
"""

from __future__ import annotations

import numpy as np

#: Wrench frame -> RDK 1.8 field. "local" == TCP frame; "world" == world frame.
_WRENCH_FIELD = {"local": "ext_wrench_in_tcp", "world": "ext_wrench_in_world"}


def map_states(rs, wrench_frame: str, dtype: str = "float64") -> dict[str, np.ndarray]:
    """Extract the six proprio vectors from one arm's ``RobotStates``."""
    wrench_attr = _WRENCH_FIELD[wrench_frame]
    return {
        "q": np.asarray(rs.q, dtype=dtype),
        "dq": np.asarray(rs.dq, dtype=dtype),
        "tau": np.asarray(rs.tau, dtype=dtype),
        "wrench": np.asarray(getattr(rs, wrench_attr), dtype=dtype),
        "eef": np.asarray(rs.tcp_pose, dtype=dtype),
        "eef_vel": np.asarray(rs.tcp_vel, dtype=dtype),
    }
