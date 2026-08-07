"""Backend-neutral contracts and robot geometry for live visualizers."""

from .schema import SIDES
from .schema import StreamRoute
from .schema import default_stream_names
from .schema import parse_stream_name

__all__ = ["SIDES", "StreamRoute", "default_stream_names", "parse_stream_name"]
