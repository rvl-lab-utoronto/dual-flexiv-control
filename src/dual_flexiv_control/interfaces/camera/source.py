"""Camera source factory and Isaac Sim camera source scaffold.

Camera sources expose the same tiny contract as the existing ZED source:
``open()`` then repeated ``read()`` returning ``{view: ndarray}``, with shaped
frames. The generic camera interface owns flattening and stream publication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from multiprocessing import shared_memory

from ...cameras import RGB_VIEWS
from ...cameras import VIEW_DEPTH
from ...cameras import VIEW_LEFT
from ...configs import CameraCfg
from ...streams.ring import _attach_untracked

CAMERA_SHM_MAGIC = 0x495341414343414D
CAMERA_SHM_VERSION = 1
CAMERA_SHM_HEADER_FIELDS = 8
CAMERA_SHM_HEADER_BYTES = CAMERA_SHM_HEADER_FIELDS * np.dtype(np.int64).itemsize
CAMERA_SHM_CHANNELS = 3
CAMERA_SHM_DTYPE = np.uint8
CAMERA_SHM_DTYPE_CODE = 2
_READ_RETRIES = 4


@dataclass
class CameraSharedMemoryData:
    name: str
    shm: shared_memory.SharedMemory
    header: np.ndarray
    rgb: np.ndarray


class CameraSource(Protocol):
    """Source-side contract for shaped camera frames."""

    def open(self) -> None:
        """Connect to the underlying camera producer."""

    def read(self) -> dict[str, np.ndarray] | None:
        """Return shaped frames keyed by view, or ``None`` when no frame is ready."""

    def close(self) -> None:
        """Release source resources."""


class IsaacSimCameraSource:
    """Read camera frames published by the Isaac bridge app.

    The bridge app owns Isaac's Python APIs and writes latest shaped frames into
    shared memory. This class will attach to those mailboxes and present the
    frames through the same source contract as :class:`ZedSource`.
    """

    def __init__(self, cam: CameraCfg) -> None:
        self.cam = cam
        self._cam: CameraSharedMemoryData | None = None
        self.latest_t_ns: int | None = None

    def open(self) -> None:
        cam = self.cam
        if VIEW_DEPTH in cam.views or cam.depth_mode.upper() != "NONE":
            raise ValueError(
                f"camera {cam.placement!r} requests Isaac Sim depth, but "
                "IsaacSimCameraSource currently supports RGB only"
            )

        unsupported = [view for view in cam.views if view not in RGB_VIEWS]
        if unsupported:
            raise ValueError(
                f"camera {cam.placement!r} has unsupported Isaac Sim views {unsupported}; "
                f"supported RGB views are {list(RGB_VIEWS)}"
            )
        if cam.views != [VIEW_LEFT]:
            raise ValueError(
                f"camera {cam.placement!r} has views {cam.views}; IsaacSimCameraSource "
                "currently expects exactly ['left'] from one RGB mailbox"
            )

        shm_name = _isaac_camera_shm_name(cam.model)
        shm = _attach_untracked(shm_name)
        expected_nbytes = (
            CAMERA_SHM_HEADER_BYTES
            + int(cam.height)
            * int(cam.width)
            * CAMERA_SHM_CHANNELS
            * np.dtype(CAMERA_SHM_DTYPE).itemsize
        )
        actual_nbytes = len(shm.buf)
        if actual_nbytes != expected_nbytes:
            shm.close()
            raise RuntimeError(
                f"Isaac Sim camera shared memory {shm_name!r} has {actual_nbytes} bytes, "
                f"but config expects {expected_nbytes} bytes for header + "
                f"{cam.width}x{cam.height} RGB"
            )

        header = np.ndarray(
            shape=(CAMERA_SHM_HEADER_FIELDS,),
            dtype=np.int64,
            buffer=shm.buf,
        )
        self._validate_header(header, shm_name)
        shm_rgb = np.ndarray(
            shape=(int(cam.height), int(cam.width), 3),
            dtype=np.uint8,
            buffer=shm.buf,
            offset=CAMERA_SHM_HEADER_BYTES,
        )
        self._cam = CameraSharedMemoryData(name=shm_name, shm=shm, header=header, rgb=shm_rgb)

    def read(self) -> dict[str, np.ndarray] | None:
        if self._cam is None:
            raise RuntimeError("IsaacSimCameraSource.open() must be called before read()")
        for _ in range(_READ_RETRIES):
            seq1 = int(self._cam.header[2])
            if seq1 <= 0 or seq1 % 2:
                continue
            rgb = self._cam.rgb.copy()
            t_ns = int(self._cam.header[3])
            seq2 = int(self._cam.header[2])
            if seq1 == seq2 and seq2 % 2 == 0:
                self.latest_t_ns = t_ns
                return {VIEW_LEFT: rgb}
        return None

    def close(self) -> None:
        if self._cam is not None:
            self._cam.header = None
            self._cam.rgb = None
            self._cam.shm.close()
            self._cam = None
        self.latest_t_ns = None

    def _validate_header(self, header: np.ndarray, shm_name: str) -> None:
        cam = self.cam
        if int(header[0]) != CAMERA_SHM_MAGIC:
            raise RuntimeError(f"Isaac Sim camera shared memory {shm_name!r} has bad magic")
        if int(header[1]) != CAMERA_SHM_VERSION:
            raise RuntimeError(
                f"Isaac Sim camera shared memory {shm_name!r} has version "
                f"{int(header[1])}, expected {CAMERA_SHM_VERSION}"
            )
        expected = {
            "height": int(cam.height),
            "width": int(cam.width),
            "channels": CAMERA_SHM_CHANNELS,
            "dtype_code": CAMERA_SHM_DTYPE_CODE,
        }
        actual = {
            "height": int(header[4]),
            "width": int(header[5]),
            "channels": int(header[6]),
            "dtype_code": int(header[7]),
        }
        if actual != expected:
            raise RuntimeError(
                f"Isaac Sim camera shared memory {shm_name!r} metadata {actual} "
                f"does not match config {expected}"
            )


def _isaac_camera_shm_name(camera_name: str) -> str:
    safe_camera_name = re.sub(r"[^0-9a-zA-Z_]", "_", camera_name)
    return f"isaac_sim_camera_{safe_camera_name}"


def make_camera_source(cam, *, runtime_sim: bool) -> CameraSource:
    """Create the concrete source selected by the camera config."""
    source = getattr(cam, "source", getattr(cam, "backend", "zed"))
    if source == "fake" or (runtime_sim and source == "zed"):
        from ..zed.source import FakeZedSource

        return FakeZedSource(cam)
    if source == "zed":
        from ..zed.source import ZedSource

        return ZedSource(cam)
    if source == "isaac_sim":
        return IsaacSimCameraSource(cam)
    raise ValueError(f"unknown camera source {source!r}")
