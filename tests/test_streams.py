"""Tests for the writer/reader/registry stack (single process)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from dual_flexiv_control.streams.registry import StreamRegistry
from dual_flexiv_control.streams.registry import shm_name_for
from dual_flexiv_control.streams.spec import StreamSpec
from dual_flexiv_control.streams.stream import StreamReader
from dual_flexiv_control.streams.stream import StreamWriter


def _run_id() -> str:
    return "test_" + uuid.uuid4().hex[:8]


def test_spec_validation():
    with pytest.raises(ValueError):
        StreamSpec(name="", dim=3, capacity=8)
    with pytest.raises(ValueError):
        StreamSpec(name="x", dim=0, capacity=8)
    with pytest.raises(ValueError):
        StreamSpec(name="x", dim=3, capacity=8, dtype="int32")
    # Image dtypes are allowed (camera frames flatten to uint8 vectors).
    StreamSpec(name="x", dim=3, capacity=8, dtype="uint8")
    StreamSpec(name="x", dim=3, capacity=8, dtype="uint16")


def test_shm_name_sanitisation():
    assert shm_name_for("run1", "right/tau") == "dfc_run1_right_tau"


def test_writer_reader_roundtrip(tmp_path):
    run_id = _run_id()
    reg = StreamRegistry(tmp_path, run_id)
    spec = StreamSpec(name="right/tau", dim=7, capacity=64, dtype="float64", rate_hz=1000)
    writer = StreamWriter.create(spec, run_id, reg)
    try:
        for i in range(10):
            writer.write(np.full(7, float(i)))

        entry = reg.get("right/tau")
        assert entry is not None
        assert entry.dim == 7
        assert entry.shm_name == shm_name_for(run_id, "right/tau")

        reader = StreamReader.attach(entry)
        try:
            s = reader.last(3)
            assert s.n == 3
            np.testing.assert_allclose(s.data[:, 0], [7, 8, 9])
            assert reader.dim == 7
            assert reader.capacity == 64
            # timestamps auto-stamped, monotonic non-decreasing
            assert np.all(np.diff(s.t_ns) >= 0)
        finally:
            reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_uint8_image_stream_roundtrip(tmp_path):
    """A flattened uint8 frame survives the ring exactly and reshapes back."""
    run_id = _run_id()
    reg = StreamRegistry(tmp_path, run_id)
    h, w, c = 4, 5, 3
    spec = StreamSpec(name="cam/test/left", dim=h * w * c, capacity=8, dtype="uint8", rate_hz=30)
    writer = StreamWriter.create(spec, run_id, reg)
    try:
        frame = (np.arange(h * w * c) % 256).astype(np.uint8)
        writer.write(frame)

        reader = StreamReader.attach(reg.get("cam/test/left"))
        try:
            s = reader.latest()
            assert s.n == 1
            assert s.data.dtype == np.uint8
            assert reader.dim == h * w * c
            np.testing.assert_array_equal(s.newest, frame)
            np.testing.assert_array_equal(s.newest.reshape(h, w, c), frame.reshape(h, w, c))
        finally:
            reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_registry_discover_and_wait(tmp_path):
    run_id = _run_id()
    reg = StreamRegistry(tmp_path, run_id)
    spec_a = StreamSpec(name="a", dim=1, capacity=4)
    spec_b = StreamSpec(name="b", dim=1, capacity=4)
    wa = StreamWriter.create(spec_a, run_id, reg)
    wb = StreamWriter.create(spec_b, run_id, reg)
    try:
        found = reg.wait_for(["a", "b"], timeout_s=1.0)
        assert set(found) == {"a", "b"}
        with pytest.raises(TimeoutError):
            reg.wait_for(["a", "missing"], timeout_s=0.2)
    finally:
        for w in (wa, wb):
            w.close()
            w.unlink()
