"""Deprecated policy-schema imports; use :mod:`.adapter`."""

from .adapter import AcmeEndpointAdapter
from .adapter import OpenPIEndpointAdapter
from .adapter import PolicyEndpointAdapter
from .adapter import build_adapter
from .adapter import register_adapter

AcmeSchema = AcmeEndpointAdapter
OpenPISchema = OpenPIEndpointAdapter
PolicySchema = PolicyEndpointAdapter
build_schema = build_adapter
register_schema = register_adapter

__all__ = [
    "AcmeSchema",
    "OpenPISchema",
    "PolicySchema",
    "build_schema",
    "register_schema",
]
