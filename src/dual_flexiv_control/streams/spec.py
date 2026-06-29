"""Declarative description of a single data stream.

A :class:`StreamSpec` is the *contract* for a stream: its logical name, the
fixed dimensionality of each sample, the numeric dtype, the ring-buffer
capacity (how many samples are retained), and an informational nominal rate.
Producers create shared memory from a spec; consumers attach to it. The spec is
plain data and is safe to pickle / send across processes (spawn start method).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: dtypes a stream may use. Floats carry proprioception (a fixed-dimension float
#: vector); ``uint8``/``uint16`` carry camera imagery (an HxWxC frame flattened to
#: a fixed-dimension vector — see :mod:`dual_flexiv_control.cameras`). The set is
#: deliberately small so the shared-memory header stays self-describing with a
#: tiny integer dtype code (see ``ring._DTYPE_TO_CODE``).
ALLOWED_DTYPES: tuple[str, ...] = ("float32", "float64", "uint8", "uint16")


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Immutable description of one fixed-dimension, timestamped stream."""

    name: str
    """Logical, hierarchical stream name, e.g. ``"right/tau"`` or ``"factr/left"``."""

    dim: int
    """Number of scalars per sample (e.g. 7 for joint torques on a 7-DoF arm)."""

    capacity: int
    """Ring-buffer length: the maximum number of recent samples retained."""

    dtype: str = "float64"
    """Sample element dtype; one of :data:`ALLOWED_DTYPES`."""

    rate_hz: float | None = None
    """Nominal production rate in Hz. Informational only (used for diagnostics)."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StreamSpec.name must be a non-empty string")
        if self.dim <= 0:
            raise ValueError(f"StreamSpec.dim must be positive, got {self.dim}")
        if self.capacity <= 0:
            raise ValueError(f"StreamSpec.capacity must be positive, got {self.capacity}")
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"StreamSpec.dtype must be one of {ALLOWED_DTYPES}, got {self.dtype!r}"
            )

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)
