"""The Flexiv interface node: one process per arm, publishing proprio streams."""

from __future__ import annotations

import numpy as np

from ...configs import ArmCfg
from ...configs import RuntimeCfg
from ...process import StreamProducerNode
from ...proprio import streams_to_specs
from ...streams.spec import StreamSpec
from .source import FakeFlexivSource
from .source import FlexivSource
from .states import map_states


class FlexivInterface(StreamProducerNode):
    """Reads one Flexiv arm and publishes its proprio signals as streams.

    Stream names are ``"<side>/<signal>"`` (e.g. ``left/q``, ``right/tau``), with
    dims/dtype/capacity taken from ``arm.streams`` in the config. One
    :class:`FlexivInterface` runs per arm, so a bimanual setup spawns two.
    """

    def __init__(self, side: str, arm: ArmCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(
            name=f"flexiv:{side}",
            runtime_dir=runtime.runtime_dir,
            run_id=run_id,
            rate_hz=arm.rate_hz,
        )
        self.side = side
        self.arm = arm
        self.sim = runtime.sim
        self._source: FlexivSource | FakeFlexivSource | None = None

    def declare_streams(self) -> list[StreamSpec]:
        return streams_to_specs(self.side, self.arm.streams)

    def open_source(self) -> None:
        if self.sim:
            self._source = FakeFlexivSource(self.arm.serial, dof=self.arm.dof)
        else:
            self._source = FlexivSource(
                self.arm.serial,
                dof=self.arm.dof,
                require_operational=self.arm.require_operational,
                verbose=self.arm.verbose_rdk,
            )
        self._source.open()

    def poll(self) -> dict[str, np.ndarray] | None:
        rs = self._source.read()
        # map_states emits float64; each writer casts to its stream's dtype.
        signals = map_states(rs, self.arm.wrench_frame)
        return {f"{self.side}/{sig}": vec for sig, vec in signals.items()}

    def close_source(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None
