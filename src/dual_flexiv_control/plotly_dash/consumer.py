"""Isolated Plotly Dash server that consumes plot streams at a fixed rate."""

from __future__ import annotations

import atexit
import logging
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..process import ProcessNode
from ..process import run_node
from ..streams import RegistryEntry
from ..streams import StreamReader
from ..streams import StreamRegistry
from ..visualization import schema
from .view import PlotStore
from .view import SPECS_BY_STREAM
from .view import create_dash_app

log = logging.getLogger(__name__)

DEFAULT_PLOT_RATE_HZ = 3.0
DISCOVERY_PERIOD_S = 1.0
MIN_STALE_AFTER_S = 5.0
STALE_PERIODS = 10.0
STARTUP_TIMEOUT_S = 30.0


def _run_isolated(node: ProcessNode, stop_event) -> None:
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass
    run_node(node, stop_event)


class PlotlyDashConsumer(ProcessNode):
    """Sample typed rings and serve incremental Plotly graphs."""

    def __init__(
        self,
        runtime_dir: str,
        stream_names: list[str],
        host: str,
        port: int,
        ready_event,
        startup_queue,
        *,
        rate_hz: float = DEFAULT_PLOT_RATE_HZ,
    ) -> None:
        self.name = "plotly-dash-consumer"
        self.runtime_dir = os.path.abspath(runtime_dir)
        self.stream_names = list(stream_names)
        self.host, self.port = host, int(port)
        self.ready_event, self.startup_queue = ready_event, startup_queue
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError(f"plot rate must be finite and positive, got {rate_hz}")
        self.rate_hz = float(rate_hz)
        self.routes = {name: schema.parse_stream_name(name) for name in self.stream_names}

    def run(self, stop_event) -> None:
        http_server = None
        server_thread = None
        store = PlotStore()
        try:
            from werkzeug.serving import make_server

            # Browser polling is expected at 3 Hz per open dashboard. Access-log
            # every successful callback would drown out actual service errors.
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            app = create_dash_app(store, self.rate_hz)
            http_server = make_server(
                self.host, self.port, app.server, threaded=True
            )
            server_thread = threading.Thread(
                target=http_server.serve_forever,
                name="plotly-dash-http",
                daemon=True,
            )
            server_thread.start()
        except Exception as exc:
            self.startup_queue.put(repr(exc))
            self.ready_event.set()
            raise
        self.ready_event.set()

        readers: dict[str, StreamReader] = {}
        attached_run = None
        last_seq: dict[str, int] = {}
        next_discovery = 0.0
        policy_state = {
            "ring": None,
            "next_seq": 0,
            "sent": 0,
            "received": 0,
            "errors": 0,
            "in_flight": 0,
        }
        period = 1.0 / self.rate_hz

        try:
            while not stop_event.is_set():
                tick = time.monotonic()
                if not readers or tick >= next_discovery:
                    next_discovery = tick + DISCOVERY_PERIOD_S
                    candidate = self._newest_live_run()
                    if candidate is None:
                        self._close_readers(readers)
                        readers, attached_run = {}, None
                        last_seq.clear()
                        store.detach()
                    else:
                        run_id, entries = candidate
                        if run_id != attached_run:
                            self._close_readers(readers)
                            readers, attached_run = {}, run_id
                            last_seq.clear()
                            policy_state.update(
                                ring=None, next_seq=0, sent=0, received=0,
                                errors=0, in_flight=0,
                            )
                            store.attach(run_id)
                        self._sync_readers(readers, entries, last_seq)

                lost = []
                for name, reader in list(readers.items()):
                    route = self.routes[name]
                    try:
                        if route.kind == "policy":
                            self._consume_policy(reader, store, policy_state)
                            continue
                        samples = reader.latest()
                    except Exception as exc:  # noqa: BLE001 - producer may be exiting
                        log.debug("Plotly consumer lost %s: %s", name, exc)
                        lost.append(name)
                        continue
                    if samples.n == 0:
                        continue
                    seq = int(samples.seq[-1])
                    if last_seq.get(name) == seq:
                        continue
                    last_seq[name] = seq
                    self._append_stream(
                        name,
                        np.asarray(samples.newest).copy(),
                        int(samples.newest_t_ns),
                        store,
                    )

                for name in lost:
                    readers.pop(name).close()
                    last_seq.pop(name, None)

                elapsed = time.monotonic() - tick
                stop_event.wait(max(0.0, period - elapsed))
        finally:
            self._close_readers(readers)
            if http_server is not None:
                http_server.shutdown()
            if server_thread is not None:
                server_thread.join(timeout=3.0)

    @staticmethod
    def _append_stream(name: str, value: np.ndarray, t_ns: int, store: PlotStore) -> None:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        for spec in SPECS_BY_STREAM.get(name, ()):  # one leader ring feeds q + grip
            if spec.route.kind == "proprio" and spec.route.signal == "eef":
                selected = vector[:3]
            elif spec.route.kind == "factr" and spec.route.signal == "q":
                selected = vector[:7]
            elif spec.route.kind == "factr" and spec.route.signal == "grip":
                selected = vector[-1:]
            else:
                selected = vector[:spec.dim]
            store.append(spec.key, selected, t_ns)

    @staticmethod
    def _consume_policy(reader: StreamReader, store: PlotStore, state: dict) -> None:
        samples = reader.last(reader.capacity)
        if samples.n == 0:
            return
        ring = reader.entry.shm_name
        if state["ring"] != ring or int(samples.seq[-1]) + 1 < state["next_seq"]:
            state.update(
                ring=ring, next_seq=0, sent=0, received=0,
                errors=0, in_flight=0,
            )
        fresh = samples.seq >= state["next_seq"]
        if not fresh.any():
            return
        from ..policy.client import COMM_RECV
        from ..policy.client import COMM_SENT

        for row, t_ns in zip(samples.data[fresh], samples.t_ns[fresh]):
            kind, latency_s = float(row[0]), float(row[2])
            if kind == COMM_SENT:
                state["sent"] += 1
                state["in_flight"] = 1
            else:
                key = "received" if kind == COMM_RECV else "errors"
                state[key] += 1
                state["in_flight"] = 0
                if kind == COMM_RECV:
                    store.append("policy/latency_ms", [latency_s * 1000.0], int(t_ns))
            store.append(
                "policy/packets",
                [state["sent"], state["received"], state["errors"], state["in_flight"]],
                int(t_ns),
            )
        state["next_seq"] = int(samples.seq[-1]) + 1

    def _newest_live_run(self) -> tuple[str, dict[str, RegistryEntry]] | None:
        root = Path(self.runtime_dir)
        if not root.is_dir():
            return None
        try:
            run_dirs = sorted(
                (path for path in root.iterdir() if (path / "streams").is_dir()),
                key=lambda path: (path / "streams").stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None
        for run_dir in run_dirs:
            entries = StreamRegistry(root, run_dir.name).discover()
            live = {
                name: entries[name] for name in self.stream_names
                if name in entries
                and schema.entry_matches(self.routes[name], entries[name])
                and self._entry_is_live(entries[name])
            }
            if live:
                return run_dir.name, live
        return None

    @staticmethod
    def _entry_pid_is_live(entry: RegistryEntry) -> bool:
        try:
            os.kill(entry.pid, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    def _entry_is_live(entry: RegistryEntry) -> bool:
        if not PlotlyDashConsumer._entry_pid_is_live(entry):
            return False
        reader = None
        try:
            reader = StreamReader.attach(entry)
            sample = reader.latest()
        except Exception:  # noqa: BLE001
            return False
        finally:
            if reader is not None:
                reader.close()
        if sample.n == 0:
            return True
        rate = entry.rate_hz if entry.rate_hz and entry.rate_hz > 0 else None
        stale_after = max(
            MIN_STALE_AFTER_S,
            STALE_PERIODS / rate if rate else MIN_STALE_AFTER_S,
        )
        return time.monotonic_ns() - sample.newest_t_ns <= int(stale_after * 1e9)

    @staticmethod
    def _sync_readers(readers, entries, last_seq) -> None:
        for name in list(readers):
            entry = entries.get(name)
            if entry is None or readers[name].entry.shm_name != entry.shm_name:
                readers.pop(name).close()
                last_seq.pop(name, None)
        for name, entry in entries.items():
            if name not in readers:
                try:
                    readers[name] = StreamReader.attach(entry)
                except Exception:  # noqa: BLE001 - independent streams retry later
                    pass

    @staticmethod
    def _close_readers(readers) -> None:
        for reader in readers.values():
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass


@dataclass
class PlotlyDashConsumerHandle:
    process: object
    stop_event: object
    key: tuple

    def stop(self) -> None:
        self.stop_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)


_LOCK = threading.Lock()
_HANDLE: PlotlyDashConsumerHandle | None = None


def start_consumer(
    runtime_dir: str,
    stream_names: list[str],
    host: str,
    port: int,
    rate_hz: float = DEFAULT_PLOT_RATE_HZ,
) -> PlotlyDashConsumerHandle:
    """Start or reuse the singleton isolated Dash consumer/server process."""
    global _HANDLE
    with _LOCK:
        runtime_dir = os.path.abspath(runtime_dir)
        key = (runtime_dir, tuple(stream_names), host, int(port), float(rate_hz))
        if _HANDLE is not None and _HANDLE.process.is_alive():
            if _HANDLE.key == key:
                return _HANDLE
            _HANDLE.stop()
        ctx = mp.get_context("spawn")
        stop_event, ready_event = ctx.Event(), ctx.Event()
        startup_queue = ctx.Queue(maxsize=1)
        node = PlotlyDashConsumer(
            runtime_dir, stream_names, host, port, ready_event, startup_queue,
            rate_hz=rate_hz,
        )
        process = ctx.Process(target=_run_isolated, args=(node, stop_event), name=node.name)
        process.daemon = True
        process.start()
        handle = PlotlyDashConsumerHandle(process, stop_event, key)
        if not ready_event.wait(STARTUP_TIMEOUT_S):
            handle.stop()
            raise RuntimeError(f"Plotly Dash consumer did not bind {host}:{port}")
        try:
            error = startup_queue.get(timeout=0.1)
        except queue.Empty:
            error = None
        if error is not None:
            handle.stop()
            raise RuntimeError(f"Plotly Dash consumer failed to start: {error}")
        if not process.is_alive():
            raise RuntimeError("Plotly Dash consumer exited during startup")
        _HANDLE = handle
        return handle


def stop_consumer() -> None:
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.stop()
            _HANDLE = None


atexit.register(stop_consumer)
