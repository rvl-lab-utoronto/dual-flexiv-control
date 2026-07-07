"""Start (once) the Rerun servers the dashboard embeds.

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
import threading
from dataclasses import dataclass
from urllib.parse import quote

import rerun as rr

#: Default app id for the dashboard's Rerun recording.
DEFAULT_APP_ID = "dual-flexiv-experiments"
#: Default gRPC (data) and HTTP (viewer) ports.
DEFAULT_GRPC_PORT = 9876
DEFAULT_WEB_PORT = 9090
#: gRPC server memory cap; oldest non-static data is dropped past this.
DEFAULT_MEMORY_LIMIT = "2GiB"

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

    A bind failure here (e.g. a stale dashboard still holding the port) raises —
    it is not swallowed — so the problem surfaces immediately instead of leaving a
    silently-black viewer.
    """
    global _SERVERS, _WEB_VIEWER_PORT
    with _LOCK:
        if _SERVERS is not None:
            return _SERVERS
        rr.init(app_id, spawn=False)
        # cors_allow_origin="*": the embedded viewer (served on the web port) makes
        # a cross-origin request to this gRPC server (a different port), so allow it.
        grpc_uri = rr.serve_grpc(
            grpc_port=grpc_port,
            server_memory_limit=memory_limit,
            cors_allow_origin=["*"],
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
    HTTP host is intentionally left running (it cannot be stopped in-process and is
    stateless), so a subsequent :func:`start_servers` reuses it.

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
