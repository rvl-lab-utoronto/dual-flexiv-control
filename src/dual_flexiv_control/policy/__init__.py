"""Online policy eval: shape observations for a policy server, execute actions.

The eval path (``runtime.phase=eval``) mirrors collection in reverse: observe
proprio + cameras software-synchronised, hand the policy the same LeRobot-keyed
frame it was trained on, and post the returned joint targets on the control
channel.

* :class:`ObservationBuilder` — config -> canonical (training-frame) observation.
* :class:`PolicySchema` / :func:`register_schema` — pluggable wire schemas per
  policy-server family (:class:`OpenPISchema` ships first).
* :class:`WebsocketTransport` / :class:`RemotePolicy` — the openpi msgpack-numpy
  websocket protocol; :class:`HoldPolicy` — serverless stand-still smoke test.
* :class:`ActionLayout` — the collection-matching action vector, split per side.
* :class:`EvalLoop` — the reusable rollout core (testable in-process).
* :class:`EvalNode` — the spawned-process node wired into the live system.
"""

from .actions import ActionLayout
from .client import AcmeHttpTransport
from .client import HoldPolicy
from .client import Policy
from .client import PolicyError
from .client import PolicyUnavailable
from .client import RemotePolicy
from .client import WebsocketTransport
from .client import build_policy
from .loop import EvalLoop
from .loop import EvalNode
from .loop import horizon_stream_name
from .observation import ObservationBuilder
from .schema import AcmeSchema
from .schema import OpenPISchema
from .schema import PolicySchema
from .schema import build_schema
from .schema import register_schema

__all__ = [
    "AcmeHttpTransport",
    "AcmeSchema",
    "ActionLayout",
    "EvalLoop",
    "EvalNode",
    "HoldPolicy",
    "ObservationBuilder",
    "OpenPISchema",
    "Policy",
    "PolicyError",
    "PolicySchema",
    "PolicyUnavailable",
    "RemotePolicy",
    "WebsocketTransport",
    "build_policy",
    "build_schema",
    "horizon_stream_name",
    "register_schema",
]
