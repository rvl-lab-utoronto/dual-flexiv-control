"""Policy clients: the transport to a policy server + the policy facade.

The eval loop sees one interface — :class:`Policy`: ``infer(canonical_obs) ->
(horizon, action_dim)``. Behind it:

* :class:`RemotePolicy` = a wire :class:`~.schema.PolicySchema` (payload shape)
  composed with a transport (connection + serialization). The two vary
  independently across policy-server families.
* :class:`WebsocketTransport` speaks the openpi policy-server protocol —
  msgpack-numpy frames over a websocket, wire-compatible with
  ``openpi_client.WebsocketClientPolicy`` (metadata frame on connect; one
  packed request -> one packed response; a *text* frame is a server-side
  error traceback). Reimplemented here so the robot host needs only
  core ``websockets`` + ``msgpack`` (the ``[policy]`` extra), not the openpi repo.
* :class:`HoldPolicy` is the no-server stand-in: repeat the measured joint
  positions (stand still) — an end-to-end smoke test of the eval path
  (``policy.kind=hold``), also handy with ``runtime.sim=true``.

Failure semantics: :class:`PolicyUnavailable` is fatal (missing dependency, or
no server within the startup budget); :class:`PolicyError` is transient (one
inference failed / connection dropped) — the loop holds and retries, and the
transport reconnects on the next call.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class PolicyUnavailable(RuntimeError):
    """A policy client cannot be constructed (missing dep, server never came up)."""


class PolicyError(RuntimeError):
    """One inference failed (transient): the caller should hold and retry."""


# ---------------------------------------------------------------------------
# msgpack-numpy hooks — wire-compatible with openpi_client.msgpack_numpy
# ---------------------------------------------------------------------------


def pack_array(obj):
    """msgpack ``default`` hook: ndarray/scalar -> tagged dict (openpi's format)."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    """msgpack ``object_hook``: tagged dict -> ndarray/scalar (openpi's format)."""
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"])
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _import_wire():
    """The wire deps, or an actionable error for the optional msgpack codec."""
    try:
        import msgpack  # noqa: PLC0415
        import websockets.exceptions  # noqa: PLC0415
        import websockets.sync.client  # noqa: PLC0415
    except ImportError as exc:
        raise PolicyUnavailable(
            "policy eval requires the 'msgpack' package. "
            "Install it with:  pip install 'dual-flexiv-control[policy]'"
        ) from exc
    return msgpack, websockets.sync.client, websockets.exceptions


def _import_http():
    """The optional ACME transport deps, or an actionable error (``[acme]`` extra)."""
    try:
        import requests  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise PolicyUnavailable(
            "ACME policy eval requires the 'requests' and 'torch' packages. "
            "Install them with:  pip install 'dual-flexiv-control[acme]'"
        ) from exc
    return requests, torch


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class Transport(Protocol):
    """Delivers one schema-shaped request payload and returns the raw response."""

    def infer(self, payload: dict) -> dict: ...
    def close(self) -> None: ...


class WebsocketTransport:
    """The openpi websocket policy protocol (msgpack-numpy frames).

    Construction blocks until the server accepts a connection (retrying up to
    ``connect_timeout_s``; ``stop_event`` aborts the wait early). After a
    failed :meth:`infer` the connection is dropped and re-established on the
    next call, so a restarted server is picked up transparently.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        connect_timeout_s: float = 60.0,
        infer_timeout_s: float = 10.0,
        stop_event=None,
    ) -> None:
        self._msgpack, self._client, self._exc = _import_wire()
        self.uri = f"ws://{host}:{port}"
        self._headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._infer_timeout_s = infer_timeout_s
        self._ws = None
        #: msgpack metadata frame the server sends on connect (checkpoint info).
        self.server_metadata: dict = {}
        self._connect(deadline=time.monotonic() + connect_timeout_s, stop_event=stop_event)

    # -- connection -------------------------------------------------------------

    def _open(self) -> None:
        # compression off + unlimited frame size, matching openpi's client (image
        # payloads are large and latency-sensitive).
        self._ws = self._client.connect(
            self.uri,
            compression=None,
            max_size=None,
            additional_headers=self._headers,
            open_timeout=self._infer_timeout_s,
        )
        self.server_metadata = self._msgpack.unpackb(
            self._ws.recv(timeout=self._infer_timeout_s), object_hook=unpack_array
        )
        log.info("connected to policy server %s; metadata=%s", self.uri, self.server_metadata)

    def _connect(self, deadline: float, stop_event=None) -> None:
        attempt = 0
        while True:
            try:
                self._open()
                return
            except (OSError, TimeoutError, self._exc.WebSocketException) as exc:
                self._drop()
                if stop_event is not None and stop_event.is_set():
                    raise PolicyUnavailable(f"aborted waiting for {self.uri} (shutdown)") from exc
                if time.monotonic() >= deadline:
                    raise PolicyUnavailable(
                        f"no policy server at {self.uri} within the connect budget: {exc}"
                    ) from exc
                if attempt % 5 == 0:
                    log.info("waiting for policy server at %s ... (%s)", self.uri, exc)
                attempt += 1
                time.sleep(1.0)

    def _drop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - already broken; just release it
                pass
        self._ws = None

    # -- inference ----------------------------------------------------------------

    def infer(self, payload: dict) -> dict:
        if self._ws is None:  # reconnect after a dropped call (single attempt)
            try:
                self._open()
            except (OSError, TimeoutError, self._exc.WebSocketException) as exc:
                self._drop()
                raise PolicyError(f"policy server {self.uri} unreachable: {exc}") from exc
        try:
            self._ws.send(self._msgpack.packb(payload, default=pack_array))
            response = self._ws.recv(timeout=self._infer_timeout_s)
        except (OSError, TimeoutError, self._exc.WebSocketException) as exc:
            self._drop()
            raise PolicyError(f"policy inference over {self.uri} failed: {exc}") from exc
        if isinstance(response, str):  # server sends its traceback as a text frame
            self._drop()  # openpi servers close after an error; force a clean reconnect
            raise PolicyError(f"policy server error:\n{response}")
        return self._msgpack.unpackb(response, object_hook=unpack_array)

    def close(self) -> None:
        self._drop()


class AcmeHttpTransport:
    """The ACME HTTP policy protocol: multipart/form-data ``POST /predict``.

    Consumes the structured payload from
    :class:`~.schema.AcmeSchema` (``images`` / ``lowdim`` / ``form``) and encodes
    it on the wire: each image tensor ``torch.save``'d as ``(B, T, C, H, W)``
    uint8, the lowdim arrays bundled into one ``lowdim_data.npz`` as
    ``(B, T, D)`` (B = T = 1 — this client sends a single frame). The server
    replies with JSON; a non-2xx or ``success: false`` body is a transient
    :class:`PolicyError` (the loop holds and retries).

    Construction blocks on ``GET /`` until the server answers (retrying up to
    ``connect_timeout_s``; ``stop_event`` aborts the wait), mirroring the
    websocket transport's startup contract.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        connect_timeout_s: float = 60.0,
        infer_timeout_s: float = 10.0,
        stop_event=None,
    ) -> None:
        self._requests, self._torch = _import_http()
        self.base_url = f"http://{host}:{port}"
        self._headers = {"Authorization": f"Api-Key {api_key}"} if api_key else {}
        self._infer_timeout_s = infer_timeout_s
        #: server's ``GET /conventions`` contract, fetched on connect (best-effort).
        self.server_metadata: dict = {}
        self._wait_ready(deadline=time.monotonic() + connect_timeout_s, stop_event=stop_event)

    # -- connection -------------------------------------------------------------

    def _wait_ready(self, deadline: float, stop_event=None) -> None:
        attempt = 0
        while True:
            try:
                resp = self._requests.get(
                    f"{self.base_url}/", headers=self._headers, timeout=self._infer_timeout_s
                )
                if resp.ok:
                    self._fetch_conventions()
                    log.info(
                        "connected to ACME policy server %s; metadata=%s",
                        self.base_url, self.server_metadata,
                    )
                    return
                last = f"HTTP {resp.status_code}"
            except self._requests.RequestException as exc:
                last = str(exc)
            if stop_event is not None and stop_event.is_set():
                raise PolicyUnavailable(f"aborted waiting for {self.base_url} (shutdown)")
            if time.monotonic() >= deadline:
                raise PolicyUnavailable(
                    f"no ACME policy server at {self.base_url} within the connect budget: {last}"
                )
            if attempt % 5 == 0:
                log.info("waiting for ACME policy server at %s ... (%s)", self.base_url, last)
            attempt += 1
            time.sleep(1.0)

    def _fetch_conventions(self) -> None:
        try:
            resp = self._requests.get(
                f"{self.base_url}/conventions", headers=self._headers,
                timeout=self._infer_timeout_s,
            )
            self.server_metadata = resp.json() if resp.ok else {}
        except (self._requests.RequestException, ValueError):
            self.server_metadata = {}  # metadata is informational only

    # -- inference ----------------------------------------------------------------

    def infer(self, payload: dict) -> dict:
        files = self._encode_files(payload)
        form = {key: str(value) for key, value in payload["form"].items()}
        try:
            resp = self._requests.post(
                f"{self.base_url}/predict", files=files, data=form,
                headers=self._headers, timeout=self._infer_timeout_s,
            )
        except self._requests.RequestException as exc:
            raise PolicyError(f"ACME inference over {self.base_url} failed: {exc}") from exc
        if resp.status_code != 200:
            raise PolicyError(
                f"ACME server {self.base_url} returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise PolicyError(
                f"ACME server returned a non-JSON response: {resp.text[:500]}"
            ) from exc
        if not body.get("success", True):
            raise PolicyError(f"ACME inference unsuccessful: {body.get('message')}")
        return body

    def _encode_files(self, payload: dict) -> dict:
        """Structured payload -> multipart parts (torch-tensor images + one npz)."""
        torch = self._torch
        files: dict = {}
        for slot, image in payload["images"].items():
            # (H, W, 3) uint8 -> (B=1, T=1, C, H, W); copy so the tensor owns
            # writable memory (the source view may be read-only shared memory).
            arr = np.array(image, dtype=np.uint8)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()[None, None]
            buf = io.BytesIO()
            torch.save(tensor, buf)
            files[slot] = (f"{slot}.pt", buf.getvalue(), "application/octet-stream")
        lowdim = io.BytesIO()
        np.savez(
            lowdim,
            **{k: np.asarray(v, dtype=np.float32).reshape(1, 1, -1) for k, v in payload["lowdim"].items()},
        )
        files["lowdim_data"] = ("lowdim_data.npz", lowdim.getvalue(), "application/octet-stream")
        return files

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Policy facade
# ---------------------------------------------------------------------------


class Policy(Protocol):
    """What the eval loop needs: canonical observation in, action chunk out."""

    def infer(self, obs: dict) -> np.ndarray: ...
    def close(self) -> None: ...


#: Comm-event kinds emitted by :class:`RemotePolicy`'s ``on_comm`` hook (floats:
#: they travel as the first element of a float64 shared-memory sample).
COMM_SENT = 0.0     # request handed to the transport (packet leaving)
COMM_RECV = 1.0     # response received (round trip complete)
COMM_ERROR = 2.0    # request failed (timeout / refused / server error)


class RemotePolicy:
    """A remote policy server = wire schema (payload shape) + transport.

    ``on_comm`` is an optional viz hook, ``(kind, seq, elapsed_s) -> None``:
    :data:`COMM_SENT` as the encoded request goes to the transport (elapsed 0),
    then :data:`COMM_RECV` (round-trip seconds) or :data:`COMM_ERROR` (seconds
    until failure) for the same ``seq``. Purely observational — a hook failure
    never disturbs inference.
    """

    def __init__(self, schema, transport, on_comm=None) -> None:
        self._schema = schema
        self._transport = transport
        self._on_comm = on_comm
        self._seq = 0

    @property
    def server_metadata(self) -> dict:
        return getattr(self._transport, "server_metadata", {})

    def _emit(self, kind: float, seq: int, elapsed_s: float) -> None:
        if self._on_comm is None:
            return
        try:
            self._on_comm(kind, seq, elapsed_s)
        except Exception:  # noqa: BLE001 - viz hook only
            log.exception("policy comm hook failed (inference unaffected)")

    def infer(self, obs: dict) -> np.ndarray:
        payload = self._schema.request(obs)
        self._seq += 1
        seq = self._seq
        t0 = time.monotonic()
        self._emit(COMM_SENT, seq, 0.0)
        try:
            response = self._transport.infer(payload)
        except Exception:
            self._emit(COMM_ERROR, seq, time.monotonic() - t0)
            raise
        self._emit(COMM_RECV, seq, time.monotonic() - t0)
        return self._schema.actions(response)

    def close(self) -> None:
        self._transport.close()


class HoldPolicy:
    """Stand-still policy: hold the current configuration, gripper at 0.

    Exercises the whole eval path (observation assembly, chunking, the control
    channel) with no server and no network. ``q_slices`` maps each controlled
    side to where its measured ``q`` lands inside ``observation.state``.

    Stand-still per action space: for joint-position (``q_d``) sides it repeats the
    measured joint positions; for velocity kinds (``qvel``/``eef_vel``) a zero action
    already means "don't move", so those are left at zero.
    """

    def __init__(self, layout, q_slices: dict[str, slice], horizon: int = 10) -> None:
        self._layout = layout
        self._q_slices = dict(q_slices)
        self._horizon = max(1, int(horizon))

    def infer(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float64).ravel()
        action = np.zeros(self._layout.dim)
        for side in self._layout.sides:
            if self._layout.field(side) == "q_d":
                action[self._layout.primary_slice(side)] = state[self._q_slices[side]]
        return np.tile(action, (self._horizon, 1))

    def close(self) -> None:
        pass


def build_policy(cfg, layout, observer, stop_event=None, on_comm=None) -> Policy:
    """The configured :class:`Policy` for an eval run.

    ``cfg`` is a :class:`~dual_flexiv_control.configs.PolicyCfg`; ``layout`` an
    :class:`~.actions.ActionLayout`; ``observer`` an
    :class:`~.observation.ObservationBuilder` (the hold policy needs its state
    layout). Remote construction blocks until the server is reachable.
    ``on_comm`` (remote only) observes server round trips — see
    :class:`RemotePolicy`; the serverless hold policy has no comms to report.
    """
    from .schema import build_schema  # noqa: PLC0415 - avoid import cycle at module load

    if cfg.kind == "hold":
        try:
            q_slices = {side: observer.state_slice(side, "q") for side in layout.sides}
        except KeyError as exc:
            raise ValueError(
                "policy.kind=hold needs 'q' among task.state_signals"
            ) from exc
        return HoldPolicy(layout, q_slices)
    if cfg.kind == "remote":
        return RemotePolicy(
            build_schema(cfg), _build_transport(cfg, stop_event), on_comm=on_comm
        )
    raise ValueError(f"unknown policy.kind {cfg.kind!r} (expected 'remote' or 'hold')")


def _build_transport(cfg, stop_event=None) -> Transport:
    """The transport named by ``cfg.transport``, chosen independently of the schema."""
    kwargs = dict(
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
        connect_timeout_s=cfg.connect_timeout_s,
        infer_timeout_s=cfg.infer_timeout_s,
        stop_event=stop_event,
    )
    if cfg.transport == "websocket":
        return WebsocketTransport(**kwargs)
    if cfg.transport == "http":
        return AcmeHttpTransport(**kwargs)
    raise ValueError(
        f"unknown policy.transport {cfg.transport!r} (expected 'websocket' or 'http')"
    )
