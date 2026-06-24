"""On-request HTTP client(s) for the FACTR teleop servers.

FACTR (Force-Attending Curriculum Training, arXiv:2502.17432) runs a FastAPI
server per leader arm exposing that leader's current joint positions. There is
**one server per leader, each on its own port** (e.g. left on 5000, right on
5001). These are plain clients: call :meth:`get_joint_positions` whenever you
want the latest positions — no polling loop and no stream.

* :class:`FactrServerClient` talks to ONE server and returns ONE leader's joint
  positions, a ``(dof,)`` vector. A single ``GET http://{host}:{port}/{endpoint}``
  returns that leader's positions; tolerated response shapes (see ``_parse``):

      ``[...]``                              — bare list
      ``{"q": [...]}`` / ``{"positions": [...]}``  — wrapped
      ``{"<side>": [...]}`` / ``{"<side>": {"q": [...]}}`` — keyed by this side

* :class:`FactrClient` is the group the brain holds: one
  :class:`FactrServerClient` per side. :meth:`get_joint_positions` queries every
  server and returns ``{side: joint_positions}``.

Each keep-alive connection is created lazily on first request and reused.
``sim=True`` returns synthetic positions with no network (handy offline).
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import time

import numpy as np

log = logging.getLogger(__name__)

#: Keys under which a flat joint-position list may be nested in a JSON object.
_POSITION_KEYS = ("positions", "q", "joint_positions", "joints", "data")


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
    """Fetches ONE FACTR leader's joint positions from its own server/port."""

    def __init__(
        self,
        *,
        side: str,
        host: str = "localhost",
        port: int = 5000,
        endpoint: str = "get_joint_positions",
        dof: int = 7,
        timeout_s: float = 0.5,
        sim: bool = False,
    ) -> None:
        self.side = side
        self.host = host
        self.port = port
        self.path = "/" + endpoint.lstrip("/")
        self.dof = dof
        self.timeout_s = timeout_s
        self.sim = sim
        self._conn: http.client.HTTPConnection | None = None
        self._t0 = time.monotonic()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    # -- the on-request API ---------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Return this leader's joint positions ``(dof,)`` right now.

        Raises :class:`FactrError` if the server is unreachable or the response is
        malformed. The connection is reset on failure so the next call reconnects.
        """
        if self.sim:
            return self._sim_positions()
        try:
            payload = self._get_json()
        except (OSError, http.client.HTTPException, ValueError, json.JSONDecodeError) as exc:
            self._reset_conn()
            raise FactrError(f"FACTR request to {self.url} failed: {exc}") from exc
        return self._parse(payload)

    # -- HTTP -----------------------------------------------------------------

    def _connect(self) -> None:
        self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)

    def _reset_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._conn = None

    def _get_json(self):
        if self._conn is None:
            self._connect()
        self._conn.request("GET", self.path)
        resp = self._conn.getresponse()
        body = resp.read()  # must fully read to reuse the keep-alive connection
        if resp.status != 200:
            raise http.client.HTTPException(f"HTTP {resp.status} from {self.url}")
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
        self._reset_conn()


class FactrClient:
    """Group client: one :class:`FactrServerClient` per leader, queried together."""

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
                endpoint=s.endpoint,
                dof=s.dof,
                timeout_s=s.request_timeout_s,
                sim=sim,
            )
            for side, s in cfg.servers.items()
        }
        return cls(servers)

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Query every leader's server and return ``{side: joint_positions}``."""
        return {side: client.get_joint_positions() for side, client in self._servers.items()}

    def get_joint_positions_for(self, side: str) -> np.ndarray:
        """One leader's joint positions (queries only that leader's server)."""
        return self._servers[side].get_joint_positions()

    def server(self, side: str) -> FactrServerClient:
        """The underlying per-leader client (e.g. for diagnostics/tests)."""
        return self._servers[side]

    def close(self) -> None:
        for client in self._servers.values():
            client.close()
