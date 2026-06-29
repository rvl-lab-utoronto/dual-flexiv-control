"""Single-producer / multi-consumer shared-memory ring buffer.

This is the keystone of the whole system: every data stream (proprioception
from each Flexiv arm, ZED camera frames, FACTR teleop samples) is one of these
rings, living in a POSIX shared-memory segment so that the producing process and
any number of consuming processes (the brain) see the same bytes with no copy
and no IPC round-trip.

The payload dtype is parametric (float32/float64 for proprioception, uint8/
uint16 for flattened image frames); it is *orthogonal* to the seqlock, whose
coordination fields (write count, per-slot sequence stamps) are always int64, so
the concurrency argument below holds regardless of what the samples contain.

Concurrency model
-----------------
* **Exactly one producer** per ring (the owning interface process) calls
  :meth:`SharedRingBuffer.append`.
* **Any number of consumers** call :meth:`SharedRingBuffer.last` / ``latest``.
* No locks. Coordination is a *seqlock*: a monotonically increasing global
  write count published in the header, plus a per-slot sequence stamp that lets
  a reader detect a slot that was overwritten while it was being copied.

Correctness assumptions (documented on purpose — these are real constraints):

1. **Single producer.** The producer is the only writer of every shared field.
   Two producers on one ring is undefined.
2. **x86-TSO + aligned 8-byte stores.** All coordination fields (the global
   write count and the per-slot sequence stamps) are 8-byte-aligned ``int64``.
   On x86-64 such aligned stores/loads are atomic and stores are not reordered
   with one another (total store order), so a reader that observes a published
   ``write_count`` is guaranteed to also observe the payload that was written
   before it. The per-slot re-validation in :meth:`last` is a second line of
   defence that catches the only remaining race (the ring lapping a slow reader)
   regardless of memory model. This module is intended for x86-64 Linux.
3. **Capacity headroom.** Readers ask for the newest ``k`` samples; as long as
   the producer does not write ``capacity - k`` *more* samples during the few
   microseconds of a read, nothing the reader touches is recycled. With the
   default capacities (thousands) and millisecond-scale control loops this
   margin is enormous; the seqlock makes the rare violation safe, not corrupt.

Memory layout of the segment (all regions 8-byte aligned)::

    [ header : int64[HEADER_LEN] ]   magic, version, capacity, dim, dtype, write_count
    [ seq    : int64[capacity]   ]   per-slot published global index (-1 = empty/dirty)
    [ t_ns   : int64[capacity]   ]   per-slot timestamp, monotonic nanoseconds
    [ data   : dtype[capacity, dim] ] per-slot sample payload
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import numpy as np

# --- header layout -----------------------------------------------------------
_HEADER_LEN = 8
_H_MAGIC = 0
_H_VERSION = 1
_H_CAPACITY = 2
_H_DIM = 3
_H_DTYPE_CODE = 4
_H_WRITE_COUNT = 5  # monotonically increasing count of published samples
# indices 6,7 reserved for future use

_MAGIC = 0x44464252_52494E47 & 0x7FFFFFFFFFFFFFFF  # "DFBRRING" folded into int64 range
_VERSION = 1

# Payload dtype <-> stable integer code persisted in the header. Codes are
# append-only: never renumber an existing entry or old segments would misdecode.
# Floats serve proprioception; uint8/uint16 serve flattened camera frames.
_DTYPE_TO_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float64"): 1,
    np.dtype("uint8"): 2,
    np.dtype("uint16"): 3,
}
_CODE_TO_DTYPE = {code: dt for dt, code in _DTYPE_TO_CODE.items()}

#: Number of times :meth:`SharedRingBuffer.last` re-snapshots before giving up
#: and returning the validated newest-contiguous suffix with ``overrun=True``.
_READ_RETRIES = 4


@dataclass(frozen=True, slots=True)
class Samples:
    """A validated batch of recent samples, ordered oldest -> newest."""

    data: np.ndarray
    """Shape ``(n, dim)`` copy of the samples (private to the caller)."""

    t_ns: np.ndarray
    """Shape ``(n,)`` int64 monotonic-nanosecond timestamps, one per sample."""

    seq: np.ndarray
    """Shape ``(n,)`` int64 global sequence index of each sample (gap-free if not overrun)."""

    overrun: bool = False
    """True if the ring lapped the reader and only the newest valid suffix is returned."""

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def newest(self) -> np.ndarray | None:
        """The most recent sample as a ``(dim,)`` array, or ``None`` if empty."""
        return self.data[-1] if self.n else None

    @property
    def newest_t_ns(self) -> int | None:
        return int(self.t_ns[-1]) if self.n else None


def _shm_size(capacity: int, dim: int, itemsize: int) -> int:
    return (_HEADER_LEN + 2 * capacity) * 8 + capacity * dim * itemsize


class SharedRingBuffer:
    """A shared-memory ring buffer over a fixed-dimension stream.

    Samples are fixed-dimension vectors of one dtype (float32/float64 for
    proprioception, or uint8/uint16 for a flattened ``H*W*C`` image frame).

    Construct with :meth:`create` (producer; allocates the segment) or
    :meth:`attach` (consumer; maps an existing segment by name).
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        *,
        owner: bool,
    ) -> None:
        self._shm = shm
        self._owner = owner
        buf = shm.buf

        header = np.ndarray((_HEADER_LEN,), dtype=np.int64, buffer=buf, offset=0)
        if int(header[_H_MAGIC]) != _MAGIC:
            raise ValueError(f"shared memory {shm.name!r} has bad magic; not a ring buffer")
        capacity = int(header[_H_CAPACITY])
        dim = int(header[_H_DIM])
        dtype = _CODE_TO_DTYPE[int(header[_H_DTYPE_CODE])]

        seq_off = _HEADER_LEN * 8
        ts_off = seq_off + capacity * 8
        data_off = ts_off + capacity * 8

        self._header = header
        self._seq = np.ndarray((capacity,), dtype=np.int64, buffer=buf, offset=seq_off)
        self._t_ns = np.ndarray((capacity,), dtype=np.int64, buffer=buf, offset=ts_off)
        self._data = np.ndarray((capacity, dim), dtype=dtype, buffer=buf, offset=data_off)
        self._capacity = capacity
        self._dim = dim
        self._dtype = dtype

    # -- construction ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        capacity: int,
        dim: int,
        dtype: str | np.dtype = "float64",
    ) -> "SharedRingBuffer":
        """Allocate a new shared-memory segment and return a producer view."""
        np_dtype = np.dtype(dtype)
        if np_dtype not in _DTYPE_TO_CODE:
            raise ValueError(f"unsupported ring dtype {np_dtype!r}")
        size = _shm_size(capacity, dim, np_dtype.itemsize)
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        try:
            buf = shm.buf
            header = np.ndarray((_HEADER_LEN,), dtype=np.int64, buffer=buf, offset=0)
            header[:] = 0
            header[_H_CAPACITY] = capacity
            header[_H_DIM] = dim
            header[_H_DTYPE_CODE] = _DTYPE_TO_CODE[np_dtype]
            header[_H_VERSION] = _VERSION
            header[_H_WRITE_COUNT] = 0
            # Mark every slot empty. Done before the magic is published so that a
            # consumer never observes a valid magic over an uninitialised seq array.
            seq_off = _HEADER_LEN * 8
            seq = np.ndarray((capacity,), dtype=np.int64, buffer=buf, offset=seq_off)
            seq[:] = -1
            # Publish last: a reader keys off the magic to decide the segment is ready.
            header[_H_MAGIC] = _MAGIC
        except Exception:
            shm.close()
            shm.unlink()
            raise
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> "SharedRingBuffer":
        """Map an existing shared-memory segment read-only (consumer view).

        The consumer attaches WITHOUT registering the segment in its
        resource_tracker — see :func:`_attach_untracked` — so ownership and
        cleanup stay solely with the producer that called :meth:`create`.
        """
        shm = _attach_untracked(name)
        return cls(shm, owner=False)

    # -- producer side --------------------------------------------------------

    def append(self, vec: np.ndarray, t_ns: int) -> int:
        """Publish one sample. Returns its global sequence index.

        Single-producer only. The publish order (slot payload -> per-slot seq ->
        global write_count) is what makes a concurrent reader either see a fully
        written sample or not see it at all.
        """
        vec = np.ascontiguousarray(vec, dtype=self._dtype)
        if vec.shape != (self._dim,):
            raise ValueError(
                f"sample shape {vec.shape} != expected ({self._dim},)"
            )
        idx = int(self._header[_H_WRITE_COUNT])
        slot = idx % self._capacity

        # 1. Invalidate the slot so any reader copying it right now is forced to
        #    re-validate and discard (it will read -1 != idx).
        self._seq[slot] = -1
        # 2. Write the payload + timestamp.
        self._data[slot, :] = vec
        self._t_ns[slot] = t_ns
        # 3. Publish the slot with its global index (store-store ordered after 2).
        self._seq[slot] = idx
        # 4. Publish globally. A reader that observes write_count == idx+1 is
        #    guaranteed (x86-TSO) to also observe steps 2-3 for this slot.
        self._header[_H_WRITE_COUNT] = idx + 1
        return idx

    # -- consumer side --------------------------------------------------------

    @property
    def write_count(self) -> int:
        """Total number of samples ever published (monotonic)."""
        return int(self._header[_H_WRITE_COUNT])

    def last(self, k: int) -> Samples:
        """Return up to the newest ``k`` samples, oldest -> newest.

        The returned arrays are private copies, safe to keep and mutate. If the
        ring lapped the reader mid-copy (only possible when a consumer stalls for
        longer than it takes the producer to write ``capacity`` samples), only the
        validated newest-contiguous suffix is returned and ``overrun`` is True.
        """
        if k <= 0:
            return self._empty()
        cap = self._capacity
        last_ok: Samples | None = None
        for _ in range(_READ_RETRIES):
            wc = int(self._header[_H_WRITE_COUNT])
            if wc <= 0:
                return self._empty()
            n = min(k, cap, wc)
            gidx = np.arange(wc - n, wc, dtype=np.int64)
            slots = gidx % cap

            s1 = self._seq[slots].copy()        # snapshot stamps before copy
            data = self._data[slots].copy()     # fancy-index -> private copy
            t_ns = self._t_ns[slots].copy()
            s2 = self._seq[slots]               # re-read stamps after copy

            ok = (s1 == gidx) & (s2 == gidx)
            if bool(ok.all()):
                return Samples(data=data, t_ns=t_ns, seq=gidx, overrun=False)

            # Keep the newest contiguous run that validated, in case every retry
            # races (pathologically slow reader). Failures occur at the oldest
            # end first (those slots get recycled first).
            bad = np.nonzero(~ok)[0]
            start = int(bad[-1]) + 1 if bad.size else 0
            if start < n:
                last_ok = Samples(
                    data=data[start:],
                    t_ns=t_ns[start:],
                    seq=gidx[start:],
                    overrun=True,
                )
        return last_ok if last_ok is not None else self._empty()

    def latest(self) -> Samples:
        """Return the single newest sample (``n in {0, 1}``)."""
        return self.last(1)

    def _empty(self) -> Samples:
        return Samples(
            data=np.empty((0, self._dim), dtype=self._dtype),
            t_ns=np.empty((0,), dtype=np.int64),
            seq=np.empty((0,), dtype=np.int64),
            overrun=False,
        )

    # -- metadata / lifecycle -------------------------------------------------

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def close(self) -> None:
        """Unmap this process's view. Does not destroy the segment."""
        # Drop numpy views before releasing the buffer they alias, otherwise the
        # memoryview cannot be closed ("cannot close exported pointers exist").
        self._header = self._seq = self._t_ns = self._data = None  # type: ignore[assignment]
        try:
            self._shm.close()
        except BufferError:
            # A view somewhere still aliases the buffer; best-effort.
            pass

    def unlink(self) -> None:
        """Destroy the underlying segment. Only the owning producer should call this.

        ``SharedMemory.unlink`` already unregisters the segment from this
        process's resource_tracker, so we must NOT also call ``_untrack`` here:
        a second unregister would raise ``KeyError`` in the tracker process
        (its UNREGISTER handler does ``set.remove``).
        """
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


_ATTACH_LOCK = threading.Lock()


def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing segment WITHOUT registering it in the resource_tracker.

    Why not just attach normally, or attach-then-unregister?

    Under the spawn start method, every process in a run shares ONE
    resource_tracker process with a single global name set (no per-process
    refcount). CPython 3.12's ``SharedMemory.__init__`` unconditionally calls
    ``resource_tracker.register('/'+name)`` on attach. So:

    * If the consumer registers and then unregisters (the old ``_untrack``
      approach), it removes the *producer's* entry from the shared set — the
      producer's later ``unlink()`` then raises ``KeyError`` in the tracker.
    * If the consumer registers and leaves it, a consumer in a *separate*
      program (its own tracker) would unlink the producer's live segment when it
      exits.

    The robust fix on 3.12 (the ``track=`` kwarg only arrives in 3.13) is to stop
    the consumer from ever registering: temporarily no-op
    ``resource_tracker.register`` around construction. ``shared_memory`` looks the
    function up on the module at call time, so this is observed. The lock guards
    against a concurrent attach in another thread of the same process restoring
    the original mid-construction.
    """
    with _ATTACH_LOCK:
        original_register = resource_tracker.register
        resource_tracker.register = lambda *args, **kwargs: None
        try:
            return shared_memory.SharedMemory(name=name, create=False)
        finally:
            resource_tracker.register = original_register
