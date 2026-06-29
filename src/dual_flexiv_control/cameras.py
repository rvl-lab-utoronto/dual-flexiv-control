"""Canonical ZED camera view definitions and config -> StreamSpec builder.

Cameras are stream producers exactly like the Flexiv arms — this module is the
camera analogue of :mod:`dual_flexiv_control.proprio`. Each camera runs one
process and publishes one shared-memory stream per *view*. A frame is stored
flattened: the ring buffer holds fixed-dimension vectors, so an ``H×W×C`` frame
travels as an ``(H*W*C,)`` vector and the consumer reshapes it back with
:func:`reshape_frame`. (Self-describing per-sample shapes would be a ring/spec
schema change; flattening keeps the substrate unchanged, matching the existing
convention that consumers know a stream's shape out-of-band, from config.)

Three canonical views, mapped one-to-one onto the ZED SDK:

    view     SDK source                  dtype     frame shape  channels
    -------  --------------------------  --------  -----------  --------
    left     ``VIEW.LEFT``  (RGB)        uint8     ``(H,W,3)``  3
    right    ``VIEW.RIGHT`` (RGB)        uint8     ``(H,W,3)``  3
    depth    ``MEASURE.DEPTH`` (metres)  float32   ``(H,W)``    1

Stream names are ``"cam/<camera>/<view>"`` (e.g. ``"cam/wrist_left/left"``); the
``cam/`` namespace keeps them clear of the arm ``left``/``right`` and ``factr``
namespaces. The registry sanitises them to POSIX-shm tokens.
"""

from __future__ import annotations

import numpy as np

from .streams.spec import StreamSpec

#: Top-level namespace prefix for every camera stream.
CAMERA_NS = "cam"

VIEW_LEFT = "left"
VIEW_RIGHT = "right"
VIEW_DEPTH = "depth"

#: RGB stereo views (uint8, 3 channels).
RGB_VIEWS: tuple[str, ...] = (VIEW_LEFT, VIEW_RIGHT)
#: Stable ordering / canonical set of camera views.
CAMERA_VIEWS: tuple[str, ...] = (VIEW_LEFT, VIEW_RIGHT, VIEW_DEPTH)


def view_channels(view: str) -> int:
    """Channels for a canonical view: RGB -> 3, depth -> 1."""
    if view in RGB_VIEWS:
        return 3
    if view == VIEW_DEPTH:
        return 1
    raise ValueError(f"unknown camera view {view!r}; expected one of {CAMERA_VIEWS}")


def view_dtype(view: str) -> str:
    """Element dtype for a canonical view: RGB -> ``uint8``, depth -> ``float32``."""
    if view in RGB_VIEWS:
        return "uint8"
    if view == VIEW_DEPTH:
        return "float32"
    raise ValueError(f"unknown camera view {view!r}; expected one of {CAMERA_VIEWS}")


def view_shape(cam, view: str) -> tuple[int, ...]:
    """Unflattened frame shape for a view: ``(H, W, 3)`` RGB or ``(H, W)`` depth."""
    if view == VIEW_DEPTH:
        return (int(cam.height), int(cam.width))
    return (int(cam.height), int(cam.width), view_channels(view))


def camera_prefix(name: str) -> str:
    """Stream-name prefix for a camera, e.g. ``"cam/wrist_left"``."""
    return f"{CAMERA_NS}/{name}"


def camera_stream_name(name: str, view: str) -> str:
    """Full stream name for a camera's view, e.g. ``"cam/wrist_left/left"``."""
    return f"{camera_prefix(name)}/{view}"


def camera_streams_to_specs(name: str, cam) -> list[StreamSpec]:
    """Build :class:`StreamSpec` objects for one camera's declared views.

    ``name`` is the camera's key in the config (e.g. ``"wrist_left"``); ``cam`` is
    a ``CameraCfg``. Per-view ``dim`` is derived as ``H*W*channels`` so it is
    always consistent with ``width``/``height`` and never hand-computed; dtype
    comes from the view (RGB uint8 / depth float32) and ``rate_hz`` from ``fps``.
    Insertion order follows ``cam.views``.
    """
    specs: list[StreamSpec] = []
    for view in cam.views:
        h, w, c = int(cam.height), int(cam.width), view_channels(view)
        specs.append(
            StreamSpec(
                name=camera_stream_name(name, view),
                dim=h * w * c,
                capacity=int(cam.capacity),
                dtype=view_dtype(view),
                rate_hz=cam.fps,
            )
        )
    return specs


def camera_stream_names(cameras) -> list[str]:
    """Every camera stream name across a ``cameras`` mapping (name -> CameraCfg).

    Handy for a brain subscription, e.g. ``brain.subscribe`` (cameras are not in
    the proprio default subscription — see ``brain.default_stream_names``).
    """
    names: list[str] = []
    for name, cam in cameras.items():
        names.extend(camera_stream_name(name, v) for v in cam.views)
    return names


def reshape_frame(vec: np.ndarray, cam, view: str) -> np.ndarray:
    """Reshape a flattened stream sample back to its image shape.

    ``vec`` is a 1-D sample as returned by a reader (one row of ``Samples.data``,
    or ``Samples.newest``); the result is ``(H, W, 3)`` for an RGB view or
    ``(H, W)`` for depth.
    """
    return np.asarray(vec).reshape(view_shape(cam, view))
