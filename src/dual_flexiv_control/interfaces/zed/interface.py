"""Compatibility wrapper for the generic camera interface."""

from __future__ import annotations

from ...configs import CameraCfg
from ...configs import RuntimeCfg
from ..camera import CameraInterface


class ZedInterface(CameraInterface):
    """ZED-named camera interface kept for existing imports/tests."""

    def __init__(self, name: str, cam: CameraCfg, runtime: RuntimeCfg, run_id: str) -> None:
        super().__init__(name, cam, runtime, run_id, node_prefix="zed")
