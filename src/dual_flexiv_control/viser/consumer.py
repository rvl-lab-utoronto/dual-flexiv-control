"""Long-lived Viser server that directly consumes shared-memory streams."""

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
from ..session import read_state
from ..streams import RegistryEntry
from ..streams import StreamReader
from ..streams import StreamRegistry
from ..visualization import geometry
from ..visualization import schema
from .client import HIDDEN_PANEL_LABEL
from .client import create_live_server
from .view import ViserLiveView

log = logging.getLogger(__name__)

# A live viewer is deliberately lossy and non-authoritative. It observes the
# newest ring values slowly; recording/control retain their native rates.
DEFAULT_VIEWER_RATE_HZ = 3.0
DISCOVERY_PERIOD_S = 1.0
MIN_STALE_AFTER_S = 5.0
STALE_PERIODS = 10.0
STARTUP_TIMEOUT_S = 60.0
FACTR_MAX_AGE_S = 0.5


def _run_isolated(node: ProcessNode, stop_event) -> None:
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass
    run_node(node, stop_event)


class ViserConsumer(ProcessNode):
    """Serve Viser and sample typed producer streams at the viewer rate."""

    def __init__(
        self,
        runtime_dir: str,
        stream_names: list[str],
        host: str,
        port: int,
        command_queue,
        ready_event,
        startup_queue,
        *,
        rate_hz: float = DEFAULT_VIEWER_RATE_HZ,
        cameras: dict | None = None,
        factr_urdfs: dict | None = None,
        factr_max_age_s: float = FACTR_MAX_AGE_S,
    ) -> None:
        self.name = "viser-consumer"
        self.runtime_dir = os.path.abspath(runtime_dir)
        self.stream_names = list(stream_names)
        self.host, self.port = host, int(port)
        self.command_queue = command_queue
        self.ready_event, self.startup_queue = ready_event, startup_queue
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError(f"Viser viewer rate must be finite and positive, got {rate_hz}")
        self.rate_hz = float(rate_hz)
        self.cameras = cameras or {}
        self.factr_urdfs = factr_urdfs or {}
        self.factr_max_age_s = float(factr_max_age_s)
        self.routes = {name: schema.parse_stream_name(name) for name in self.stream_names}

    def run(self, stop_event) -> None:
        view = None
        try:
            import viser

            server = create_live_server(
                viser,
                host=self.host, port=self.port,
                label=HIDDEN_PANEL_LABEL, verbose=False,
            )
            view = ViserLiveView(
                server, self.rate_hz, factr_urdfs=self.factr_urdfs,
                cameras=self.cameras,
            )
        except Exception as exc:
            self.startup_queue.put(repr(exc))
            self.ready_event.set()
            raise
        self.ready_event.set()

        readers: dict[str, StreamReader] = {}
        attached_run = None
        last_seq: dict[str, int] = {}
        newest: dict[str, tuple[np.ndarray, int]] = {}
        horizon_paths: dict[str, np.ndarray] = {}
        next_discovery = 0.0
        period = 1.0 / self.rate_hz

        try:
            while not stop_event.is_set():
                tick = time.monotonic()
                self._drain_commands(view)
                state = read_state(self.runtime_dir)
                view.update_session(state)

                if not readers or tick >= next_discovery:
                    next_discovery = tick + DISCOVERY_PERIOD_S
                    candidate = self._newest_live_run()
                    if candidate is None:
                        self._close_readers(readers)
                        readers, attached_run = {}, None
                        newest.clear()
                        horizon_paths.clear()
                        last_seq.clear()
                        view.detach_run()
                    else:
                        run_id, entries = candidate
                        if run_id != attached_run:
                            self._close_readers(readers)
                            readers, attached_run = {}, run_id
                            newest.clear()
                            horizon_paths.clear()
                            last_seq.clear()
                            view.attach_run(run_id)
                        self._sync_readers(readers, entries, newest, last_seq)

                lost = []
                for name, reader in list(readers.items()):
                    route = self.routes[name]
                    try:
                        samples = reader.latest()
                    except Exception as exc:  # noqa: BLE001 - producer may be exiting
                        log.debug("Viser consumer lost %s: %s", name, exc)
                        lost.append(name)
                        continue
                    if samples.n == 0:
                        continue
                    seq = int(samples.seq[-1])
                    if last_seq.get(name) == seq:
                        continue
                    last_seq[name] = seq
                    value, t_ns = np.asarray(samples.newest).copy(), int(samples.newest_t_ns)
                    newest[name] = value, t_ns
                    if route.kind == "horizon" and route.signal == "q_horizon":
                        horizon_paths[route.side] = self._latest_timestamp_group(reader)

                for name in lost:
                    reader = readers.pop(name)
                    reader.close()
                    newest.pop(name, None)
                    last_seq.pop(name, None)

                self._render_scene(view, newest, horizon_paths, state)
                elapsed = time.monotonic() - tick
                stop_event.wait(max(0.0, period - elapsed))
        finally:
            self._close_readers(readers)
            if view is not None:
                view.close()

    def _render_scene(self, view, newest, horizon_paths, state) -> None:
        now_ns = time.monotonic_ns()
        follower_q, follower_eef, command_q, factr_q = {}, {}, {}, {}
        target_q, target_eef = {}, {}
        for side in schema.SIDES:
            q = newest.get(f"{side}/q")
            if q is not None:
                follower_q[side] = q[0][:7]
            eef = newest.get(f"{side}/eef")
            if eef is not None:
                follower_eef[side] = eef[0][:3]
            leader = newest.get(f"factr/{side}")
            if leader is not None and now_ns - leader[1] <= self.factr_max_age_s * 1e9:
                command_q[side] = leader[0][:7]
            model = newest.get(f"factr/telemetry/{side}/model_q_rad")
            if model is not None and now_ns - model[1] <= self.factr_max_age_s * 1e9:
                factr_q[side] = model[0][:7]
            horizon = newest.get(f"eval/{side}/q_horizon")
            if horizon is not None:
                target_q[side] = horizon[0][:7]
            eef_horizon = newest.get(f"eval/{side}/eef_horizon")
            if eef_horizon is not None:
                target_eef[side] = eef_horizon[0][:3]

        view.scene.update_followers(follower_q)
        view.scene.update_commands(command_q)
        view.scene.update_factr(factr_q)
        phase = (state or {}).get("phase") or (state or {}).get("state")
        if phase in ("eval", "skill"):
            view.scene.update_targets(
                target_q, follower_q, q_paths=horizon_paths,
                target_eef=target_eef, real_eef=follower_eef,
            )
        else:
            view.scene.clear_targets()
        self._render_depth(view, newest)

    def _render_depth(self, view, newest) -> None:
        for name, cam in self.cameras.items():
            rgb = newest.get(f"cam/{name}/left")
            depth = newest.get(f"cam/{name}/depth")
            if rgb is None or depth is None:
                continue
            try:
                image = np.asarray(rgb[0], dtype=np.uint8).reshape(
                    int(cam.height), int(cam.width), 3
                )
                z_image = np.asarray(depth[0], dtype=np.float32).reshape(
                    int(cam.height), int(cam.width)
                )
            except (TypeError, ValueError):
                continue
            stride = 4
            z = z_image[::stride, ::stride]
            colors = image[::stride, ::stride]
            vs, us = np.mgrid[0:int(cam.height):stride, 0:int(cam.width):stride]
            valid = np.isfinite(z) & (z > 0.2) & (z <= 5.0)
            if not valid.any():
                continue
            fx = (float(cam.width) / 2.0) / np.tan(np.deg2rad(float(cam.hfov_deg)) / 2.0)
            cx, cy = float(cam.width) / 2.0, float(cam.height) / 2.0
            zv = z[valid]
            points = np.stack(
                ((us[valid] - cx) * zv / fx, (vs[valid] - cy) * zv / fx, zv), axis=1
            ).astype(np.float32)
            rotation, translation = geometry.camera_world_pose(cam)
            view.scene.update_depth(points @ rotation.T + translation, colors[valid])
            return
        view.scene.update_depth(None, None)

    @staticmethod
    def _latest_timestamp_group(reader: StreamReader) -> np.ndarray:
        samples = reader.last(reader.capacity)
        if samples.n == 0:
            return np.empty((0, reader.dim))
        return np.asarray(samples.data[samples.t_ns == samples.t_ns[-1]]).copy()

    def _drain_commands(self, view) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            view.apply_command(command)

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
                and (
                    self._entry_pid_is_live(entries[name])
                    if self.routes[name].kind == "horizon"
                    else self._entry_is_live(entries[name])
                )
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
        if not ViserConsumer._entry_pid_is_live(entry):
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
        stale_after = max(MIN_STALE_AFTER_S, STALE_PERIODS / rate if rate else MIN_STALE_AFTER_S)
        return time.monotonic_ns() - sample.newest_t_ns <= int(stale_after * 1e9)

    @staticmethod
    def _sync_readers(readers, entries, newest, last_seq) -> None:
        for name in list(readers):
            entry = entries.get(name)
            if entry is None or readers[name].entry.shm_name != entry.shm_name:
                readers.pop(name).close()
                newest.pop(name, None)
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
class ViserConsumerHandle:
    process: object
    stop_event: object
    command_queue: object
    key: tuple

    def send(self, command: dict) -> None:
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            log.warning(
                "dropping viewer command because its queue is full: %s",
                command.get("kind"),
            )

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
_HANDLE: ViserConsumerHandle | None = None


def start_consumer(
    runtime_dir: str,
    stream_names: list[str],
    host: str,
    port: int,
    rate_hz: float = DEFAULT_VIEWER_RATE_HZ,
    *,
    cameras: dict | None = None,
    factr_urdfs: dict | None = None,
    factr_max_age_s: float = FACTR_MAX_AGE_S,
) -> ViserConsumerHandle:
    """Start or reuse the singleton isolated Viser consumer/server process."""
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
        command_queue, startup_queue = ctx.Queue(maxsize=128), ctx.Queue(maxsize=1)
        node = ViserConsumer(
            runtime_dir, stream_names, host, port, command_queue,
            ready_event, startup_queue, rate_hz=rate_hz,
            cameras=cameras, factr_urdfs=factr_urdfs,
            factr_max_age_s=factr_max_age_s,
        )
        process = ctx.Process(target=_run_isolated, args=(node, stop_event), name=node.name)
        process.daemon = True
        process.start()
        handle = ViserConsumerHandle(process, stop_event, command_queue, key)
        if not ready_event.wait(STARTUP_TIMEOUT_S):
            handle.stop()
            raise RuntimeError(f"Viser consumer did not bind {host}:{port}")
        try:
            error = startup_queue.get(timeout=0.1)
        except queue.Empty:
            error = None
        if error is not None:
            handle.stop()
            raise RuntimeError(f"Viser consumer failed to start: {error}")
        if not process.is_alive():
            raise RuntimeError("Viser consumer exited during startup")
        _HANDLE = handle
        return handle


def stop_consumer() -> None:
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.stop()
            _HANDLE = None


atexit.register(stop_consumer)
