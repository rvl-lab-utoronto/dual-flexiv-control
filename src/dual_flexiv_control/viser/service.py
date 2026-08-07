"""Dashboard-agnostic facade around the Viser stream-consumer process."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..session import compose_config
from ..visualization import schema
from .consumer import DEFAULT_VIEWER_RATE_HZ
from .consumer import ViserConsumerHandle
from .consumer import start_consumer
from .consumer import stop_consumer

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9090


def endpoint_from_env() -> tuple[str, int]:
    return (
        os.environ.get("DFC_VISER_HOST", DEFAULT_HOST),
        int(os.environ.get(
            "DFC_VISER_PORT",
            os.environ.get("DFC_DASHBOARD_WEB_PORT", DEFAULT_PORT),
        )),
    )


def _viewer_metadata():
    rig = os.environ.get("DFC_DASHBOARD_RIG", "").strip()
    cfg = compose_config([f"rig={rig}"] if rig else [])
    cameras = {
        name: cam for name, cam in cfg.cameras.items()
        if "left" in cam.views and "depth" in cam.views
    }
    factr_urdfs = {}
    try:
        workdir = Path(str(cfg.factr.launch.workdir)).expanduser().resolve()
        package = workdir / "src" / "factr_teleop" / "factr_teleop"
        for side in cfg.factr.servers:
            config_path = package / "configs" / f"factr_rizon_{side}.yaml"
            payload = yaml.safe_load(config_path.read_text()) or {}
            name = payload.get("arm_teleop", {}).get("leader_urdf")
            path = (package / "urdf" / str(name)).resolve()
            if name and path.is_file():
                factr_urdfs[side] = path
    except Exception:
        # The follower-shaped fallback remains useful when the external FACTR
        # checkout is absent (sim/test machines).
        pass
    return cfg, cameras, factr_urdfs


@dataclass
class ViserService:
    host: str
    port: int
    runtime_dir: str
    stream_names: list[str]
    consumer: ViserConsumerHandle
    viewer_rate_hz: float = DEFAULT_VIEWER_RATE_HZ

    @property
    def web_url(self) -> str:
        public = os.environ.get("DFC_VISER_PUBLIC_URL")
        return public.rstrip("/") if public else f"http://127.0.0.1:{self.port}"

    def web_url_for_host(self, hostname: str | None) -> str:
        public = os.environ.get("DFC_VISER_PUBLIC_URL")
        return public.rstrip("/") if public else f"http://{hostname or '127.0.0.1'}:{self.port}"

    def send(self, command: dict) -> None:
        self.consumer.send(command)

    def log_event(self, message: str) -> None:
        self.send({"kind": "event", "message": message})

    def show_calibration_target(self, side: str, q) -> None:
        self.send({"kind": "show_calibration", "side": side, "q": list(q)})

    def clear_calibration_targets(self) -> None:
        self.send({"kind": "clear_calibration"})


_LOCK = threading.Lock()
_SERVICE: ViserService | None = None


def start_service(
    runtime_dir: str | None = None,
    viewer_rate_hz: float = DEFAULT_VIEWER_RATE_HZ,
    host: str | None = None,
    port: int | None = None,
) -> ViserService:
    global _SERVICE
    with _LOCK:
        if _SERVICE is not None and _SERVICE.consumer.process.is_alive():
            return _SERVICE
        cfg, cameras, factr_urdfs = _viewer_metadata()
        runtime_dir = os.path.abspath(
            runtime_dir or os.environ.get("DFC_RUNTIME_DIR") or cfg.runtime.runtime_dir
        )
        stream_names = schema.scene_stream_names(camera_names=tuple(cameras))
        env_host, env_port = endpoint_from_env()
        host, port = host or env_host, int(port or env_port)
        consumer = start_consumer(
            runtime_dir, stream_names, host, port, viewer_rate_hz,
            cameras=cameras, factr_urdfs=factr_urdfs,
            factr_max_age_s=float(cfg.factr.max_age_s),
        )
        _SERVICE = ViserService(
            host, port, runtime_dir, stream_names, consumer, viewer_rate_hz
        )
        return _SERVICE


def get_service() -> ViserService:
    if _SERVICE is None:
        raise RuntimeError("Viser service has not been started")
    return _SERVICE


def service_if_started() -> ViserService | None:
    return _SERVICE


def stop_service() -> None:
    global _SERVICE
    with _LOCK:
        stop_consumer()
        _SERVICE = None


def restart_service() -> ViserService:
    stop_service()
    return start_service()


def main() -> None:
    service = start_service()
    print(f"Viser stream service: {service.web_url}")
    try:
        while True:
            if not service.consumer.process.is_alive():
                raise RuntimeError("Viser consumer exited")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_service()
