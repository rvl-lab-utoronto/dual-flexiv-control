"""FACTR teleoperation: the leader-arm stream producer + its WebSocket clients.

:class:`FactrInterface` is the ONE process that consumes the FACTR WebSocket
server(s) and publishes each leader as converted ``factr/<side>`` and untouched
``factr/raw/<side>`` shared-memory streams. Every consumer — the
collection loop, the brain, the dashboard mirror and status probe — attaches
read-only to those streams; nothing else talks to the servers directly, so the
viewer and the control path see identical samples by construction.

:class:`FactrServerClient` / :class:`FactrClient` are the underlying WebSocket
clients the producer holds (also used by the interactive calibration tools).

:class:`FactrServerSupervisor` launches and supervises the external
FACTR-Server *processes* themselves (the grav-comp teleops + the API relay) on
the session daemon's behalf — see :mod:`.launch`.
"""

from .client import FactrClient
from .client import FactrError
from .client import FactrServerClient
from .interface import FactrInterface
from .interface import factr_stream_name
from .interface import fresh_leader_positions
from .interface import leader_stream_names
from .interface import raw_factr_stream_name
from .interface import wait_leaders_fresh
from .launch import FactrServerSupervisor

__all__ = [
    "FactrClient",
    "FactrServerClient",
    "FactrError",
    "FactrInterface",
    "FactrServerSupervisor",
    "factr_stream_name",
    "fresh_leader_positions",
    "leader_stream_names",
    "raw_factr_stream_name",
    "wait_leaders_fresh",
]
