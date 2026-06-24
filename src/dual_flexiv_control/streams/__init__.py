"""Shared-memory data streams: the substrate every component reads and writes.

A *stream* is a fixed-dimension, timestamped, single-producer/multi-consumer
ring buffer in POSIX shared memory. Producers (the Flexiv and FACTR interfaces)
write samples; consumers (the brain) attach by name and pull the last ``k``.
"""

from .registry import AttachAborted
from .registry import RegistryEntry
from .registry import StreamRegistry
from .registry import cleanup_run
from .registry import shm_name_for
from .ring import Samples
from .ring import SharedRingBuffer
from .spec import StreamSpec
from .stream import StreamReader
from .stream import StreamWriter

__all__ = [
    "StreamSpec",
    "SharedRingBuffer",
    "Samples",
    "StreamWriter",
    "StreamReader",
    "StreamRegistry",
    "RegistryEntry",
    "AttachAborted",
    "shm_name_for",
    "cleanup_run",
]
