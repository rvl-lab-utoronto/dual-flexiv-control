"""Online policy eval: shape observations for a policy server, execute actions.

The eval path (``runtime.phase=eval``) mirrors collection in reverse: observe
proprio + cameras software-synchronised, hand the policy the same LeRobot-keyed
frame it was trained on, and post the returned joint targets on the control
channel.

* :class:`ObservationBuilder` — config -> canonical (training-frame) observation.
* :class:`PolicyEndpointAdapter` / :func:`register_adapter` — pluggable endpoint
  mappings per policy-server family (:class:`OpenPIEndpointAdapter` ships first).
* :class:`WebsocketTransport` / :class:`RemotePolicy` — the openpi msgpack-numpy
  websocket protocol; :class:`HoldPolicy` — serverless stand-still smoke test.
* :class:`DFCStateActionLayout` — the canonical state/action contract and slices.
* :class:`EvalLoop` — the reusable rollout core (testable in-process).
* :class:`EvalNode` — the spawned-process node wired into the live system.
"""

from ..layout import DFCStateActionLayout
from .adapter import AcmeEndpointAdapter
from .adapter import OpenPIEndpointAdapter
from .adapter import PolicyEndpointAdapter
from .adapter import build_adapter
from .adapter import register_adapter
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
from .loop import eef_horizon_stream_name
from .loop import horizon_stream_name
from .observation import ObservationBuilder

__all__ = [
    "AcmeHttpTransport",
    "AcmeEndpointAdapter",
    "DFCStateActionLayout",
    "EvalLoop",
    "EvalNode",
    "HoldPolicy",
    "ObservationBuilder",
    "OpenPIEndpointAdapter",
    "Policy",
    "PolicyError",
    "PolicyEndpointAdapter",
    "PolicyUnavailable",
    "RemotePolicy",
    "WebsocketTransport",
    "build_policy",
    "build_adapter",
    "eef_horizon_stream_name",
    "horizon_stream_name",
    "register_adapter",
]
