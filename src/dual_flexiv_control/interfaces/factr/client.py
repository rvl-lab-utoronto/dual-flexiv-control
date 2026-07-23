"""Persistent WebSocket clients for the FACTR teleop servers.

FACTR (Force-Attending Curriculum Training, arXiv:2502.17432) runs one relay per
leader arm. There is **one server per leader, each on its own port** (e.g. left
on 5000, right on 5001). Each server pushes typed JSON frames on one WebSocket:

* ``{"type": "reading", "side": "left", "joint_pos": [...]}``
* ``{"type": "diagnostics", "side": "left", "available": true, ...}``

The leader-owned raw→DFC calibration contract is carried by the diagnostics
frame. The relay sends that frame first on every connection and again whenever
the teleop publishes a new snapshot.

Each :class:`FactrServerClient` owns a background receiver that continuously
drains the socket into latest-value caches. Public reads are therefore local and
never build a backlog when the server publishes faster than the DFC producer.
The receiver reconnects after an outage; a stale/missing reading raises
:class:`FactrError`, allowing the shared-memory leader stream to go stale.

* :class:`FactrClient` is the group the brain holds: one
  :class:`FactrServerClient` per side. :meth:`get_joint_positions` reads every
  client's latest cache and returns ``{side: joint_positions}``.

The socket is duplex: :meth:`send_force_feedback` pushes the follower's external
joint torques back up the same connection as ``force_feedback`` frames, which the
relay republishes to the leader's teleop (its force-feedback term). Joint-space
today (``"space": "joint"``, one torque per arm joint); a task-space variant
would swap the payload to a 6-D TCP wrench under ``"space": "tcp"`` without
changing the transport.

Gravity-comp status and enable/disable commands remain HTTP request/response
operations. ``sim=True`` returns synthetic positions with no network.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import threading
import time

import numpy as np
import websockets.exceptions
import websockets.sync.client

log = logging.getLogger(__name__)

#: Keys under which a flat joint-position list may be nested in a JSON object.
#: ``joint_pos`` is the reading key emitted by the real FACTR WebSocket server.
_POSITION_KEYS = ("joint_pos", "positions", "q", "joint_positions", "joints", "data")


class FactrError(RuntimeError):
    """Raised when a FACTR server cannot be reached or returns a bad response."""


def _extract_list(obj) -> list:
    """Pull a flat list of numbers out of a JSON value (bare list or wrapped dict)."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in _POSITION_KEYS:
            value = obj.get(key)
            if isinstance(value, list):
                return value
    raise FactrError(
        f"cannot extract a joint-position list from {type(obj).__name__}: {repr(obj)[:120]}"
    )


class FactrServerClient:
    """Receives one FACTR leader's readings and diagnostics over WebSocket."""

    def __init__(
        self,
        *,
        side: str,
        host: str,
        port: int,
        endpoint: str,
        dof: int,
        timeout_s: float,
        sim: bool,
    ) -> None:
        self.side = side
        self.host = host
        self.port = port
        self.path = "/" + endpoint.lstrip("/")
        self.dof = dof
        self.timeout_s = timeout_s
        self.sim = sim
        self._http_conn: http.client.HTTPConnection | None = None
        self._t0 = time.monotonic()

        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._receiver: threading.Thread | None = None
        self._ws = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_joint_pos_at = 0.0
        self._latest_diagnostics: dict | None = None
        self._last_stream_error = "stream has not connected"

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    # -- latest-value API -----------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Return this leader's joint positions ``(dof,)`` right now.

        Raises :class:`FactrError` if no fresh frame arrives within ``timeout_s``.
        The background receiver reconnects independently after transport failures.
        """
        if self.sim:
            return self._sim_positions()
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            self._ensure_receiver_locked()
            while True:
                now = time.monotonic()
                if (
                    self._latest_joint_pos is not None
                    and now - self._latest_joint_pos_at <= self.timeout_s
                ):
                    return self._latest_joint_pos.copy()
                remaining = deadline - now
                if remaining <= 0:
                    raise FactrError(
                        f"FACTR reading {self.side} from {self.url} unavailable: "
                        f"{self._last_stream_error or 'no fresh reading received'}"
                    )
                self._condition.wait(remaining)

    def get_calibration(self) -> dict:
        """Return the leader-owned raw→DFC contract from the WebSocket cache."""
        if self.sim:
            return {}
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            self._ensure_receiver_locked()
            while self._latest_diagnostics is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FactrError(
                        f"FACTR calibration {self.side} from {self.url} unavailable: "
                        f"{self._last_stream_error}"
                    )
                self._condition.wait(remaining)
            return dict(self._latest_diagnostics)

    def get_status(self) -> dict:
        """Return this leader's live grav-comp and force-feedback state."""
        if self.sim:
            return {
                "side": self.side,
                "grav_comp_gain": 0.0,
                "grav_comp_gain_target": 0.0,
                "force_feedback_gain": 0.0,
                "force_feedback_gain_target": 0.0,
                "grav_comp_enabled": False,
                "force_feedback_enabled": False,
            }
        path = f"/status_{self.side}"
        try:
            payload = self._get_http_json(path)
        except (OSError, http.client.HTTPException, ValueError, json.JSONDecodeError) as exc:
            self._reset_http_conn()
            raise FactrError(f"FACTR status {self.side} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise FactrError(f"FACTR status {self.side} returned non-object JSON")
        return payload

    def send_force_feedback(self, tau) -> None:
        """Push follower external joint torques ``(num_arm_joints,)`` to this leader.

        Fire-and-forget up the same WebSocket the receiver holds (``websockets``'
        sync connection allows one sender and one receiver thread concurrently).
        Joint-space today; the teleop applies each sample within its staleness
        window and decays to zero torque when the feed stops, so senders just skip
        failed sends. Raises :class:`FactrError` when the stream is not connected —
        the receiver reconnects on its own and feedback resumes with it.
        """
        if self.sim:
            return
        tau = np.asarray(tau, dtype=np.float64)
        if tau.ndim != 1 or tau.size == 0 or not np.all(np.isfinite(tau)):
            raise FactrError(
                f"FACTR force feedback {self.side}: tau must be a non-empty finite "
                f"1-D vector, got shape {tau.shape}"
            )
        frame = json.dumps({
            "type": "force_feedback",
            "side": self.side,
            "space": "joint",
            "tau": tau.tolist(),
        })
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            self._ensure_receiver_locked()
            while self._ws is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FactrError(
                        f"FACTR force feedback {self.side}: stream not connected: "
                        f"{self._last_stream_error}"
                    )
                self._condition.wait(remaining)
            ws = self._ws
        try:
            ws.send(frame)
        except (
            OSError,
            TimeoutError,
            ValueError,
            websockets.exceptions.WebSocketException,
        ) as exc:
            raise FactrError(
                f"FACTR force feedback {self.side} send failed: {exc}"
            ) from exc

    # -- WebSocket receiver ---------------------------------------------------

    def _ensure_receiver_locked(self) -> None:
        """Start the receiver exactly once; caller holds ``_condition``."""
        if self._stop.is_set():
            raise FactrError(f"FACTR client for {self.side} is closed")
        if self._receiver is not None and self._receiver.is_alive():
            return
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name=f"factr-ws-{self.side}",
            daemon=True,
        )
        self._receiver.start()

    def _receive_loop(self) -> None:
        """Keep the WebSocket drained and reconnect until :meth:`close` is called."""
        while not self._stop.is_set():
            ws = None
            try:
                ws = websockets.sync.client.connect(
                    self.url,
                    compression=None,
                    open_timeout=self.timeout_s,
                    close_timeout=self.timeout_s,
                    max_size=None,
                )
                with self._condition:
                    self._ws = ws
                    self._last_stream_error = "connected; waiting for a frame"
                    self._condition.notify_all()
                for message in ws:
                    if self._stop.is_set():
                        break
                    self._handle_stream_frame(message)
                if not self._stop.is_set():
                    raise FactrError("connection closed")
            except (
                OSError,
                TimeoutError,
                TypeError,
                UnicodeError,
                ValueError,
                FactrError,
                websockets.exceptions.WebSocketException,
            ) as exc:
                if not self._stop.is_set():
                    with self._condition:
                        self._last_stream_error = str(exc)
                        self._condition.notify_all()
            finally:
                with self._condition:
                    if self._ws is ws:
                        self._ws = None
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001 - already broken
                        pass
            self._stop.wait(min(0.1, self.timeout_s))

    def _handle_stream_frame(self, message) -> None:
        """Validate one typed JSON frame and atomically refresh its cache."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise FactrError(f"FACTR {self.side} sent a non-object WebSocket frame")
        frame_side = payload.get("side")
        if frame_side != self.side:
            raise FactrError(
                f"FACTR {self.side} stream received frame for side {frame_side!r}"
            )

        frame_type = payload.get("type")
        if frame_type == "reading":
            joint_pos = self._parse(payload)
            with self._condition:
                self._latest_joint_pos = joint_pos
                self._latest_joint_pos_at = time.monotonic()
                self._last_stream_error = ""
                self._condition.notify_all()
            return
        if frame_type == "diagnostics":
            diagnostics = dict(payload)
            diagnostics.pop("type", None)
            with self._condition:
                self._latest_diagnostics = diagnostics
                self._last_stream_error = ""
                self._condition.notify_all()
            return
        raise FactrError(f"FACTR {self.side} sent unknown frame type {frame_type!r}")

    # -- remaining HTTP status surface ---------------------------------------

    def _connect_http(self) -> None:
        self._http_conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout_s
        )

    def _reset_http_conn(self) -> None:
        if self._http_conn is not None:
            try:
                self._http_conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._http_conn = None

    def _get_http_json(self, path: str):
        if self._http_conn is None:
            self._connect_http()
        self._http_conn.request("GET", path)
        resp = self._http_conn.getresponse()
        body = resp.read()  # must fully read to reuse the keep-alive connection
        if resp.status != 200:
            raise http.client.HTTPException(
                f"HTTP {resp.status} from http://{self.host}:{self.port}{path}"
            )
        return json.loads(body)

    # -- parsing --------------------------------------------------------------

    def _parse(self, payload) -> np.ndarray:
        # Tolerate a payload keyed by this leader's side, else a flat/wrapped list.
        if isinstance(payload, dict) and self.side in payload:
            data = _extract_list(payload[self.side])
        else:
            data = _extract_list(payload)
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise FactrError(
                f"FACTR {self.side!r}@{self.url}: got {arr.shape}, expected ({self.dof},)"
            )
        return arr

    # -- sim ------------------------------------------------------------------

    def _sim_positions(self) -> np.ndarray:
        t = time.monotonic() - self._t0
        phase = 0.0 if self.side == "left" else math.pi / 2
        return 0.3 * np.sin(t + phase + np.arange(self.dof)).astype(np.float64)

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            ws = self._ws
            receiver = self._receiver
            self._condition.notify_all()
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=max(1.0, self.timeout_s + 0.5))
        self._reset_http_conn()


class FactrClient:
    """Group client: one :class:`FactrServerClient` per leader, sampled together."""

    def __init__(self, servers: dict[str, FactrServerClient]) -> None:
        self._servers = dict(servers)
        self.sides = list(self._servers)

    @classmethod
    def from_config(cls, cfg, *, sim: bool = False) -> "FactrClient":
        """Build one :class:`FactrServerClient` per entry in ``cfg.servers``."""
        servers = {
            side: FactrServerClient(
                side=side,
                host=s.host,
                port=s.port,
                endpoint=str(s.endpoint).format(side=side),
                dof=s.dof,
                timeout_s=s.request_timeout_s,
                sim=sim,
            )
            for side, s in cfg.servers.items()
        }
        return cls(servers)

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Read every leader's latest frame and return ``{side: joint_positions}``."""
        return {side: client.get_joint_positions() for side, client in self._servers.items()}

    def get_joint_positions_for(self, side: str) -> np.ndarray:
        """One leader's latest joint positions."""
        return self._servers[side].get_joint_positions()

    def get_calibration_for(self, side: str) -> dict:
        return self._servers[side].get_calibration()

    def wait_calibration_for(self, side: str, timeout_s: float = 30.0) -> dict:
        """Wait for FACTR startup to stream its leader-owned calibration contract.

        Transport failure and ``available=false`` mean the relay/teleop is still starting.
        The deadline remains strict: no contract raises :class:`FactrError` with the last
        observed condition.
        """
        deadline = time.monotonic() + float(timeout_s)
        last_error = "calibration not yet available"
        while time.monotonic() < deadline:
            try:
                data = self.get_calibration_for(side)
            except FactrError as exc:
                last_error = str(exc)
            else:
                if data.get("available") is True:
                    return data
                last_error = "stream returned available=false"
            time.sleep(0.1)
        raise FactrError(
            f"FACTR calibration {side} unavailable after {float(timeout_s):.1f}s: "
            f"{last_error}"
        )

    def get_status_for(self, side: str) -> dict:
        """One leader's live grav-comp state."""
        return self._servers[side].get_status()

    def send_force_feedback_for(self, side: str, tau) -> None:
        """Push one follower's external joint torques to its leader (joint-space)."""
        self._servers[side].send_force_feedback(tau)

    def send_force_feedback(self, taus: dict) -> None:
        """Push ``{side: external joint torques}`` to every listed leader.

        All sides are attempted; per-side failures are aggregated into one
        :class:`FactrError` so one dead leader cannot starve the other of feedback.
        """
        failures: list[str] = []
        for side, tau in taus.items():
            try:
                self._servers[side].send_force_feedback(tau)
            except FactrError as exc:
                failures.append(str(exc))
        if failures:
            raise FactrError("; ".join(failures))

    def preflight(self) -> None:
        """Probe every configured leader once; raise if any is unreachable.

        A launch-time check so a missing teleop leader fails fast with a clear,
        aggregated error (which the collection node re-raises as a non-zero exit, and
        the dashboard surfaces as an error popup) instead of the collection loop
        silently holding stale/zero actions for the whole recording. In ``sim`` mode
        every server fabricates positions, so this always passes (no hardware
        expected). Requires **all** configured servers to respond — a bimanual rig
        needs both leaders up; the shipped ``left_only`` default needs just the one.
        """
        failures: list[str] = []
        for side, client in self._servers.items():
            try:
                client.get_joint_positions()
            except FactrError as exc:
                failures.append(f"{side} @ {client.url}: {exc}")
        if failures:
            raise FactrError(
                "FACTR teleop preflight failed — leader(s) not reachable: "
                + "; ".join(failures)
            )

    def server(self, side: str) -> FactrServerClient:
        """The underlying per-leader client (e.g. for calibration/tests)."""
        return self._servers[side]

    def close(self) -> None:
        for client in self._servers.values():
            client.close()
