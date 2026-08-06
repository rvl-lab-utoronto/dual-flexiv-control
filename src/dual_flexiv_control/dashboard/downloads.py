"""Stream large dashboard downloads without buffering them in Streamlit.

The collection videos are already finalized MP4 files, often hundreds of
megabytes each. ``st.download_button`` retains its payload in the Streamlit
process while the user is connected, so rendering one button per saved camera
would duplicate the whole dataset in RAM. This small, process-local HTTP server
instead publishes only explicitly registered files and streams them in chunks.

The server is intentionally read-only:

* callers register an exact existing file path and a safe download filename;
* public URLs contain an unguessable token, never a filesystem path; and
* the request handler can only resolve tokens already present in that registry.
"""

from __future__ import annotations

import atexit
import logging
import mimetypes
import os
import re
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_PORT = 9092
_CHUNK_BYTES = 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class _PublishedFile:
    path: str
    filename: str
    content_type: str


class _DownloadHTTPServer(ThreadingHTTPServer):
    """Thread-per-download server whose workers cannot hold process shutdown."""

    daemon_threads = True
    allow_reuse_address = True


class DownloadServer:
    """A localhost-addressed, externally-bindable registry of streamed files."""

    def __init__(self, port: int) -> None:
        self.port = int(port)
        self._lock = threading.Lock()
        self._by_token: dict[str, _PublishedFile] = {}
        self._token_by_file: dict[tuple[str, str], str] = {}

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._serve(send_body=True)

            def _serve(self, *, send_body: bool) -> None:
                parts = urlsplit(self.path).path.strip("/").split("/")
                item = owner._resolve(parts[1] if len(parts) >= 2 and parts[0] == "download" else "")
                if item is None:
                    self.send_error(404, "download not found")
                    return
                try:
                    size = os.path.getsize(item.path)
                    stream = open(item.path, "rb")
                except OSError:
                    self.send_error(404, "download no longer exists")
                    return

                with stream:
                    self.send_response(200)
                    self.send_header("Content-Type", item.content_type)
                    self.send_header("Content-Length", str(size))
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{item.filename}"',
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    if not send_body:
                        return
                    try:
                        while chunk := stream.read(_CHUNK_BYTES):
                            self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        # A cancelled browser download is routine, not an app error.
                        pass

            def log_message(self, fmt: str, *args) -> None:
                log.debug("download server: " + fmt, *args)

        # Bind all interfaces so browsers using the dashboard over LAN/Tailscale
        # can reach this port. URLs still use loopback and app._browser_url rewrites
        # that host to the one serving Streamlit.
        self._httpd = _DownloadHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="dfc-download-server",
            daemon=True,
        )
        self._thread.start()

    def register(
        self,
        path: str | os.PathLike[str],
        filename: str,
        *,
        content_type: str | None = None,
    ) -> str:
        """Publish one exact file and return its browser-facing loopback URL."""
        resolved = str(Path(path).resolve(strict=True))
        if not os.path.isfile(resolved):
            raise ValueError(f"download is not a regular file: {resolved}")
        safe_name = _safe_filename(filename)
        key = (resolved, safe_name)
        with self._lock:
            token = self._token_by_file.get(key)
            if token is None:
                token = secrets.token_urlsafe(24)
                item = _PublishedFile(
                    path=resolved,
                    filename=safe_name,
                    content_type=(
                        content_type
                        or mimetypes.guess_type(safe_name)[0]
                        or "application/octet-stream"
                    ),
                )
                self._token_by_file[key] = token
                self._by_token[token] = item
        return (
            f"http://127.0.0.1:{self.port}/download/{token}/"
            f"{quote(safe_name, safe='')}"
        )

    def _resolve(self, token: str) -> _PublishedFile | None:
        with self._lock:
            return self._by_token.get(token)

    def close(self) -> None:
        """Stop accepting downloads and release the port."""
        self._httpd.shutdown()
        self._httpd.server_close()


def _safe_filename(filename: str) -> str:
    """One portable ASCII filename suitable for Content-Disposition."""
    name = Path(str(filename)).name
    name = _SAFE_NAME_RE.sub("-", name).strip(".-")
    name = re.sub(r"-+\.", ".", name)
    return name or "download"


_LOCK = threading.Lock()
_SERVER: DownloadServer | None = None


def port_from_env() -> int:
    """Download port, honoring ``DFC_DOWNLOAD_PORT``."""
    return int(os.environ.get("DFC_DOWNLOAD_PORT", DEFAULT_DOWNLOAD_PORT))


def start_server(port: int | None = None) -> DownloadServer:
    """Start the process-global download server once and return it."""
    global _SERVER
    with _LOCK:
        wanted = int(port if port is not None else port_from_env())
        if _SERVER is not None:
            if _SERVER.port != wanted:
                raise RuntimeError(
                    f"download server already uses port {_SERVER.port}, not {wanted}"
                )
            return _SERVER
        _SERVER = DownloadServer(wanted)
        atexit.register(_SERVER.close)
        return _SERVER
