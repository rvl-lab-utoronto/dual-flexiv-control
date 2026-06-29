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

# Process-global singleton: the servers bind exactly once per process, whether
# first touched by the launcher (eager, pre-Streamlit) or by the app's first
# session. Both paths funnel through start_servers().
_LOCK = threading.Lock()
_SERVERS: "RerunServers | None" = None


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
        masked by cached viewer state.
        """
        return f"{self.web_base}/?url={quote(self.grpc_uri, safe='')}&persist=0"


def start_servers(
    app_id: str = DEFAULT_APP_ID,
    grpc_port: int = DEFAULT_GRPC_PORT,
    web_port: int = DEFAULT_WEB_PORT,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
) -> RerunServers:
    """Initialise the recording and bring up the gRPC + web-viewer servers.

    Idempotent per process: the first call binds the ports; later calls (e.g. the
    Streamlit app reusing what the launcher already started) return the same
    handle without re-binding. ``rr.init`` installs the process-global recording,
    so anything that later calls ``rr.log`` / ``rr.send_blueprint`` (including the
    emitter thread in :mod:`~.runner`) feeds this same recording and viewer.
    """
    global _SERVERS
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
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=grpc_uri)
        _SERVERS = RerunServers(
            app_id=app_id, grpc_uri=grpc_uri, grpc_port=grpc_port, web_port=web_port
        )
        return _SERVERS
