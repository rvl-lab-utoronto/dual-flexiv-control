"""dual-flexiv-control: a bimanual Flexiv control brain.

Three precision-engineered components, decoupled by shared-memory data streams:

* :mod:`dual_flexiv_control.streams` — single-producer/multi-consumer ring buffers
  in POSIX shared memory; the substrate every component reads and writes.
* :mod:`dual_flexiv_control.interfaces.flexiv` — real ``flexivrdk`` proprioception,
  one process per arm (bimanual: ``left`` + ``right``).
* :mod:`dual_flexiv_control.interfaces.factr` — FACTR teleop, a black-box wrapper
  skeleton exposing "the last k elements".
* :mod:`dual_flexiv_control.brain` — the main processing pipeline: subscribes to
  streams, observes them, and gets the last ``k`` elements.

Configuration is composed by Hydra from ``conf/`` and validated against the
structured schema in :mod:`dual_flexiv_control.configs`. Each arm carries its own
control schema (qpos, qvel, end_effector, force), aligned with the flexivrdk
command structs for the future control implementation.

:func:`dual_flexiv_control.system.run_system` spawns and supervises the lot.
"""

from .configs import Config
from .proprio import PROPRIO_SIGNALS
from .proprio import Side

__version__ = "0.1.0"

__all__ = ["Config", "Side", "PROPRIO_SIGNALS", "__version__"]
