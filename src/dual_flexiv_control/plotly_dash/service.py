"""Dashboard-agnostic facade around the Plotly Dash stream consumer."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from ..session import compose_config
from ..visualization import schema
from .consumer import DEFAULT_PLOT_RATE_HZ
from .consumer import PlotlyDashConsumerHandle
from .consumer import start_consumer
from .consumer import stop_consumer
from .view import PLOT_VIEW_REVISION

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9094


def endpoint_from_env() -> tuple[str, int]:
    return (
        os.environ.get("DFC_PLOTLY_HOST", DEFAULT_HOST),
        int(os.environ.get("DFC_PLOTLY_PORT", DEFAULT_PORT)),
    )


@dataclass
class PlotlyDashService:
    host: str
    port: int
    runtime_dir: str
    stream_names: list[str]
    consumer: PlotlyDashConsumerHandle
    plot_rate_hz: float = DEFAULT_PLOT_RATE_HZ
    view_revision: int = PLOT_VIEW_REVISION

    @property
    def web_url(self) -> str:
        public = os.environ.get("DFC_PLOTLY_PUBLIC_URL")
        return public.rstrip("/") if public else f"http://127.0.0.1:{self.port}"

    def web_url_for_host(self, hostname: str | None) -> str:
        public = os.environ.get("DFC_PLOTLY_PUBLIC_URL")
        return public.rstrip("/") if public else f"http://{hostname or '127.0.0.1'}:{self.port}"


_LOCK = threading.Lock()
_SERVICE: PlotlyDashService | None = None


def start_service(
    runtime_dir: str | None = None,
    plot_rate_hz: float = DEFAULT_PLOT_RATE_HZ,
    host: str | None = None,
    port: int | None = None,
) -> PlotlyDashService:
    global _SERVICE
    with _LOCK:
        if _SERVICE is not None and _SERVICE.consumer.process.is_alive():
            return _SERVICE
        rig = os.environ.get("DFC_DASHBOARD_RIG", "").strip()
        cfg = compose_config([f"rig={rig}"] if rig else [])
        runtime_dir = os.path.abspath(
            runtime_dir or os.environ.get("DFC_RUNTIME_DIR") or cfg.runtime.runtime_dir
        )
        stream_names = schema.plot_stream_names()
        env_host, env_port = endpoint_from_env()
        host, port = host or env_host, int(port or env_port)
        consumer = start_consumer(
            runtime_dir, stream_names, host, port, plot_rate_hz
        )
        _SERVICE = PlotlyDashService(
            host, port, runtime_dir, stream_names, consumer, plot_rate_hz
        )
        return _SERVICE


def get_service() -> PlotlyDashService:
    if _SERVICE is None:
        raise RuntimeError("Plotly Dash service has not been started")
    return _SERVICE


def service_if_started() -> PlotlyDashService | None:
    return _SERVICE


def stop_service() -> None:
    global _SERVICE
    with _LOCK:
        stop_consumer()
        _SERVICE = None


def restart_service() -> PlotlyDashService:
    stop_service()
    return start_service()


def main() -> None:
    service = start_service()
    print(f"Plotly Dash stream service: {service.web_url}")
    try:
        while True:
            if not service.consumer.process.is_alive():
                raise RuntimeError("Plotly Dash consumer exited")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_service()
