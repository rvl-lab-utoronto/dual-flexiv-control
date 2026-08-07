"""Deprecated Rerun server implementation retained for ``dfc-rerun``.

The primary dashboard does not import this module; see
:mod:`dual_flexiv_control.viser`.

Two cooperating servers, both hosted inside the Streamlit process:

* a **gRPC server** (``serve_grpc``) that buffers the active recording's log data
  and hands out a ``rerun+http://…/proxy`` URI;
* a **web viewer** (``serve_web_viewer``) that serves the SDK-bundled viewer over
  HTTP and auto-connects it to that gRPC server.

The Streamlit page embeds :attr:`RerunServers.web_url` in an ``<iframe>``; the
viewer streams live from gRPC, so metrics update in the browser without any
Streamlit rerun. Both ``serve_*`` calls return immediately (non-blocking).

:func:`start_servers` must run **exactly once** per process — the ports bind on
first call. The Streamlit app guards it behind ``st.cache_resource``.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote

import rerun as rr

#: Default app id for the dashboard's Rerun recording.
DEFAULT_APP_ID = "dual-flexiv-experiments"
#: Default gRPC (data) and HTTP (viewer) ports.
DEFAULT_GRPC_PORT = 9876
DEFAULT_WEB_PORT = 9090
#: gRPC server memory cap; oldest non-static data is dropped past this (static
#: styling — series names/colours, README — is exempt). This is the ONLY bound on
#: what a freshly-loaded viewer must replay: a full page refresh reconnects to the
#: gRPC proxy, which streams the *entire retained store* from the start before it
#: catches up to live — so this cap, not the plots' visible-time window (which is
#: render-only; see :mod:`~.blueprints`), is what sets reload cost. The mirror logs
#: ~3.4 MB/min (measured). Keep enough headroom for the complete static robot-scene
#: initialization: an overly small cap can evict/reorder its geometry chunks before
#: a freshly-loaded viewer receives them. Override with ``DFC_DASHBOARD_MEMORY_LIMIT``.
DEFAULT_MEMORY_LIMIT = os.environ.get("DFC_DASHBOARD_MEMORY_LIMIT", "256MiB")

# Process-global singletons, with two deliberately-split lifetimes:
#
# * the **web-viewer HTTP host** binds exactly once per process and is NEVER torn
#   down — rerun 0.33 offers no way to stop or rebind it in-process
#   (``rerun_shutdown()`` releases the gRPC server but leaves the web port bound).
#   That is fine: the web viewer is stateless — it serves the viewer app, which
#   connects to whichever gRPC ``?url=`` an iframe asks for — so it never needs a
#   restart.
# * the **gRPC data server + recording** hold all the state and CAN be torn down
#   (:func:`teardown`) and re-served (a fresh :func:`start_servers`), which is what
#   the dashboard's *Reset services* action does without killing the process.
_LOCK = threading.Lock()
_SERVERS: "RerunServers | None" = None
#: The port the web viewer bound on (set once, on the first ``serve_web_viewer``);
#: never cleared, so teardown + restart reuses the same host instead of re-binding.
_WEB_VIEWER_PORT: int | None = None


def ports_from_env() -> tuple[int, int]:
    """``(grpc_port, web_port)`` honouring the ``DFC_DASHBOARD_*_PORT`` overrides."""
    return (
        int(os.environ.get("DFC_DASHBOARD_GRPC_PORT", DEFAULT_GRPC_PORT)),
        int(os.environ.get("DFC_DASHBOARD_WEB_PORT", DEFAULT_WEB_PORT)),
    )


@dataclass(frozen=True)
class RerunServers:
    """Handles to the running Rerun servers (for embedding + reconnection)."""

    app_id: str
    grpc_uri: str
    grpc_port: int
    web_port: int

    @property
    def web_base(self) -> str:
        """Bare viewer origin (no data source).

        Uses ``127.0.0.1`` to match the host in :attr:`grpc_uri`, so the page and
        the gRPC endpoint share a host and the browser's cross-origin request to
        the data server is as unsurprising as possible.
        """
        return f"http://127.0.0.1:{self.web_port}"

    @property
    def web_url(self) -> str:
        """Embeddable viewer URL pointed at the gRPC data server.

        The served viewer reads its data source from ``?url=`` (parsed with
        ``URLSearchParams``, which percent-decodes), so the gRPC URI must be
        percent-encoded — otherwise the bare viewer just shows its welcome page.
        ``persist=0`` keeps each load fresh so blueprint/phase switches aren't
        masked by cached viewer state. ``renderer=webgl`` forces the WebGL
        backend: when the browser exposes WebGPU via Dawn's GL-compatibility
        mode (Chrome on Linux/NVIDIA), the viewer's preferred WebGPU path runs
        ~50x slower than WebGL on the same GPU (measured 0.6 vs 32 fps).
        """
        return f"{self.web_base}/?url={quote(self.grpc_uri, safe='')}&persist=0&renderer=webgl"


#: How long to wait for a torn-down gRPC listener to release its port, and for a
#: freshly-served one to come up. Both are milliseconds in practice; the margin
#: covers a loaded machine.
_PORT_WAIT_S = 5.0


def _port_listening(port: int) -> bool:
    """True if something on this host accepts TCP connections on ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _await_port(port: int, *, listening: bool, timeout_s: float = _PORT_WAIT_S) -> bool:
    """Poll until ``port`` is (not) accepting connections; False on timeout."""
    deadline = time.monotonic() + timeout_s
    while _port_listening(port) != listening:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def serve_grpc_checked(serve: Callable[[], str], grpc_port: int, what: str) -> str:
    """Run a ``serve_grpc`` call and confirm a listener actually came up.

    ``serve_grpc`` binds in a background thread and **returns a URI regardless**:
    a bind failure (``Address already in use``) only kills that thread with a log
    line, leaving a handle whose port nobody serves — a silently-black viewer.
    The bind loses exactly when it races ``rerun_shutdown``, which releases the
    previous listener *asynchronously*; *Reset services* re-serves immediately
    after tearing down, so it lost that race routinely. So: wait for the previous
    listener to clear, serve, then require the new one to answer — raising on
    either timeout so the failure surfaces in the UI instead of as a dead iframe.
    """
    if not _await_port(grpc_port, listening=False):
        raise RuntimeError(
            f"port {grpc_port} is still in use, so the {what} gRPC server cannot "
            "bind — is another dashboard running? Stop it and press Reset services."
        )
    uri = serve()
    if not _await_port(grpc_port, listening=True):
        raise RuntimeError(
            f"the {what} gRPC server did not come up on port {grpc_port} — its "
            "server thread died (see the dashboard log). Press Reset services to retry."
        )
    return uri


def start_servers(
    app_id: str = DEFAULT_APP_ID,
    grpc_port: int = DEFAULT_GRPC_PORT,
    web_port: int = DEFAULT_WEB_PORT,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
) -> RerunServers:
    """Initialise the recording and bring up the gRPC data server + web viewer.

    Idempotent while up: repeated calls return the live handle. The gRPC data
    server + recording are (re)created on each fresh start — so after
    :func:`teardown` releases them, a new call re-serves the gRPC server (its port
    was freed) and re-installs the recording. The web-viewer HTTP host is bound
    **once** for the process lifetime and reused thereafter (it cannot be rebound
    in-process; see the module note). ``rr.init`` installs the process-global
    recording, so anything that later calls ``rr.log`` / ``rr.send_blueprint``
    (including the emitter thread in :mod:`~.runner`) feeds this same recording.

    A bind failure here (e.g. a stale dashboard still holding the port, or the
    just-torn-down listener not yet released) raises via
    :func:`serve_grpc_checked` — ``serve_grpc`` alone would not: its bind runs in
    a background thread that dies with only a log line — so the problem surfaces
    immediately instead of leaving a silently-black viewer.
    """
    global _SERVERS, _WEB_VIEWER_PORT
    with _LOCK:
        if _SERVERS is not None:
            return _SERVERS
        rr.init(app_id, spawn=False)
        # cors_allow_origin="*": the embedded viewer (served on the web port) makes
        # a cross-origin request to this gRPC server (a different port), so allow it.
        grpc_uri = serve_grpc_checked(
            lambda: rr.serve_grpc(
                grpc_port=grpc_port,
                server_memory_limit=memory_limit,
                cors_allow_origin=["*"],
            ),
            grpc_port,
            what="metrics",
        )
        if _WEB_VIEWER_PORT is None:
            rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=grpc_uri)
            _WEB_VIEWER_PORT = web_port
        _SERVERS = RerunServers(
            app_id=app_id, grpc_uri=grpc_uri, grpc_port=grpc_port, web_port=_WEB_VIEWER_PORT
        )
        return _SERVERS


def teardown() -> None:
    """Tear down the gRPC data server + every recording (releasing the gRPC port).

    ``rr.rerun_shutdown()`` is global: it stops **all** served gRPC servers (the
    metrics server here *and* the replay server in :mod:`~.replay`) and drops their
    in-memory recordings, freeing those ports for a clean re-serve. The web-viewer
    HTTP host is intentionally left
    running (it cannot be stopped in-process and is stateless), so a subsequent
    :func:`start_servers` reuses it.

    This only releases the servers. Callers that cached the now-dead recording — the
    robot scene (:func:`~.robot_view.reset`) and the replay viewer
    (:func:`~.replay.reset`) — must be reset alongside, and the metrics servers
    re-started, before logging resumes. The dashboard's *Reset services* action does
    exactly this.
    """
    global _SERVERS
    with _LOCK:
        rr.rerun_shutdown()
        _SERVERS = None
