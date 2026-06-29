"""Tests for camera view conventions, the spec builder, and the sim source."""

from __future__ import annotations

import numpy as np
import pytest

from dual_flexiv_control.cameras import camera_stream_name
from dual_flexiv_control.cameras import camera_stream_names
from dual_flexiv_control.cameras import camera_streams_to_specs
from dual_flexiv_control.cameras import reshape_frame
from dual_flexiv_control.cameras import view_channels
from dual_flexiv_control.cameras import view_dtype
from dual_flexiv_control.configs import CameraCfg
from dual_flexiv_control.interfaces.zed import FakeZedSource


def _cam(views, width=8, height=6, capacity=4, depth_mode="NONE", fps=30.0) -> CameraCfg:
    return CameraCfg(
        model="zed2",
        serial="",
        placement="static",
        resolution="VGA",
        width=width,
        height=height,
        fps=fps,
        depth_mode=depth_mode,
        views=list(views),
        capacity=capacity,
    )


def test_view_channels_and_dtype():
    assert view_channels("left") == 3
    assert view_channels("right") == 3
    assert view_channels("depth") == 1
    assert view_dtype("left") == "uint8"
    assert view_dtype("depth") == "float32"
    with pytest.raises(ValueError):
        view_channels("bogus")
    with pytest.raises(ValueError):
        view_dtype("bogus")


def test_stream_naming():
    assert camera_stream_name("wrist_left", "left") == "cam/wrist_left/left"
    cams = {"wrist_left": _cam(["left"]), "static": _cam(["left", "right"])}
    assert camera_stream_names(cams) == [
        "cam/wrist_left/left",
        "cam/static/left",
        "cam/static/right",
    ]


def test_streams_to_specs_derives_dim_dtype_rate():
    cam = _cam(["left", "right", "depth"], width=8, height=6, capacity=4, depth_mode="NEURAL", fps=15.0)
    specs = {s.name: s for s in camera_streams_to_specs("static", cam)}

    rgb = specs["cam/static/left"]
    assert rgb.dim == 8 * 6 * 3
    assert rgb.dtype == "uint8"
    assert rgb.capacity == 4
    assert rgb.rate_hz == 15.0

    depth = specs["cam/static/depth"]
    assert depth.dim == 8 * 6 * 1
    assert depth.dtype == "float32"
    # Order follows cam.views.
    assert [s.name for s in camera_streams_to_specs("static", cam)] == [
        "cam/static/left", "cam/static/right", "cam/static/depth",
    ]


def test_fake_source_frame_shapes_dtypes_and_stereo_differs():
    cam = _cam(["left", "right", "depth"], width=8, height=6, depth_mode="NEURAL")
    src = FakeZedSource(cam)
    src.open()
    frames = src.read()
    assert set(frames) == {"left", "right", "depth"}

    assert frames["left"].shape == (6, 8, 3)
    assert frames["left"].dtype == np.uint8
    assert frames["right"].shape == (6, 8, 3)
    assert frames["depth"].shape == (6, 8)
    assert frames["depth"].dtype == np.float32
    # Per-view tint => the two RGB eyes are not identical.
    assert not np.array_equal(frames["left"], frames["right"])
    src.close()


def test_reshape_frame_roundtrips_flattened_sample():
    cam = _cam(["left", "depth"], width=8, height=6, depth_mode="NEURAL")
    src = FakeZedSource(cam)
    src.open()
    frames = src.read()

    flat_rgb = np.ascontiguousarray(frames["left"]).reshape(-1)
    np.testing.assert_array_equal(reshape_frame(flat_rgb, cam, "left"), frames["left"])

    flat_depth = np.ascontiguousarray(frames["depth"]).reshape(-1)
    np.testing.assert_array_equal(reshape_frame(flat_depth, cam, "depth"), frames["depth"])
    src.close()
