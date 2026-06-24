"""The brain: the main processing pipeline that observes data streams."""

from .brain import Brain
from .brain import BrainNode
from .brain import default_stream_names

__all__ = ["Brain", "BrainNode", "default_stream_names"]
