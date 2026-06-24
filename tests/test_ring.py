"""Correctness tests for the shared-memory ring buffer (single process)."""

from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest

from dual_flexiv_control.streams.ring import SharedRingBuffer


def _make(cap: int = 8, dim: int = 3, dtype: str = "float64") -> SharedRingBuffer:
    name = "dfc_test_" + uuid.uuid4().hex[:12]
    return SharedRingBuffer.create(name, cap, dim, dtype)


def test_empty_buffer_returns_nothing():
    r = _make()
    try:
        assert r.last(4).n == 0
        assert r.latest().n == 0
        assert r.latest().newest is None
    finally:
        r.close()
        r.unlink()


def test_order_values_timestamps_and_seq():
    r = _make(cap=8, dim=2)
    try:
        for i in range(5):
            r.append(np.array([i, -i], dtype=float), t_ns=1000 + i)
        s = r.last(3)
        assert s.n == 3
        assert not s.overrun
        np.testing.assert_allclose(s.data, [[2, -2], [3, -3], [4, -4]])
        assert list(s.t_ns) == [1002, 1003, 1004]
        assert list(s.seq) == [2, 3, 4]
        np.testing.assert_allclose(r.latest().newest, [4, -4])
        assert r.write_count == 5
    finally:
        r.close()
        r.unlink()


def test_wraparound_keeps_newest():
    r = _make(cap=4, dim=1)
    try:
        for i in range(10):
            r.append(np.array([i], dtype=float), t_ns=i)
        np.testing.assert_allclose(r.last(4).data.ravel(), [6, 7, 8, 9])
        # Asking for more than capacity is clamped to the newest `capacity`.
        clamped = r.last(100)
        assert clamped.n == 4
        np.testing.assert_allclose(clamped.data.ravel(), [6, 7, 8, 9])
    finally:
        r.close()
        r.unlink()


def test_returned_arrays_are_private_copies():
    r = _make(cap=4, dim=1)
    try:
        r.append(np.array([1.0]), t_ns=0)
        s = r.last(1)
        s.data[0, 0] = 999.0  # mutate the copy
        assert r.latest().newest[0] == 1.0  # buffer unaffected
    finally:
        r.close()
        r.unlink()


def test_dim_mismatch_rejected():
    r = _make(cap=4, dim=3)
    try:
        with pytest.raises(ValueError):
            r.append(np.zeros(2), t_ns=0)
    finally:
        r.close()
        r.unlink()


def test_attach_sees_same_data():
    r = _make(cap=8, dim=2)
    try:
        for i in range(3):
            r.append(np.array([i, i], dtype=float), t_ns=i)
        reader = SharedRingBuffer.attach(r.name)
        try:
            np.testing.assert_allclose(reader.last(2).data, [[1, 1], [2, 2]])
        finally:
            reader.close()
    finally:
        r.close()
        r.unlink()


def test_float32_dtype():
    r = _make(cap=4, dim=2, dtype="float32")
    try:
        r.append(np.array([1.5, 2.5]), t_ns=0)
        s = r.last(1)
        assert s.data.dtype == np.float32
        np.testing.assert_allclose(s.data.ravel(), [1.5, 2.5])
    finally:
        r.close()
        r.unlink()


def test_threaded_seqlock_no_tearing():
    """One writer thread, one reader thread: every sample must be internally
    consistent (all elements equal its seq) and never torn."""
    r = _make(cap=256, dim=4)
    stop = threading.Event()
    n_writes = 20000
    errors: list[str] = []

    def writer():
        for i in range(n_writes):
            r.append(np.full(4, float(i)), t_ns=i)
        stop.set()

    def reader():
        while not stop.is_set():
            s = r.last(16)
            for row, sq in zip(s.data, s.seq):
                # The whole row was written atomically as [seq]*dim.
                if not np.all(row == row[0]):
                    errors.append(f"torn row {row} at seq {sq}")
                if row[0] != sq:
                    errors.append(f"value {row[0]} != seq {sq}")

    tw = threading.Thread(target=writer)
    tr = threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join()
    tr.join()
    try:
        assert not errors, errors[:5]
        np.testing.assert_allclose(r.latest().newest, np.full(4, float(n_writes - 1)))
    finally:
        r.close()
        r.unlink()
