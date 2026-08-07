"""Detached compatibility service for the deprecated Rerun viewer."""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

from ..dashboard.session import SessionView
from ..session import read_state


class _StateReader:
    """SessionMirror's narrow manager contract, backed only by ``session.json``."""

    def __init__(self, runtime_dir: str) -> None:
        self.runtime_dir = runtime_dir

    def view(self) -> SessionView:
        raw = read_state(self.runtime_dir) or {}
        return SessionView(
            state=str(raw.get("state") or "down"),
            rig=raw.get("rig"), sim=raw.get("sim"), run_id=raw.get("run_id"),
            task=raw.get("task"), phase=raw.get("phase"),
            run_seq=int(raw.get("run_seq") or 0),
            run_started_ts=raw.get("run_started_ts"), message=raw.get("message"),
            last_outcome=raw.get("last_outcome"),
            cameras_down=tuple(raw.get("cameras_down") or ()),
            arms_down=tuple(raw.get("arms_down") or ()),
            pending=raw.get("pending"), factr_servers=raw.get("factr_servers"),
        )


@dataclass
class RerunService:
    """Legacy Rerun server with a 3 Hz shared-memory ``SessionMirror`` heart."""

    servers: object
    mirror: object

    @property
    def web_url(self) -> str:
        return self.servers.web_url

    def stop(self) -> None:
        from ..dashboard import robot_view
        from ..dashboard.viewer import teardown

        self.mirror.stop()
        teardown()
        robot_view.reset()


def start_service(runtime_dir: str | None = None) -> RerunService:
    warnings.warn(
        "Rerun visualization is deprecated; use dfc-viser or dfc-dashboard",
        DeprecationWarning,
        stacklevel=2,
    )
    import rerun as rr

    from ..dashboard import blueprints
    from ..dashboard import robot_view
    from ..dashboard import runner
    from ..dashboard.viewer import ports_from_env
    from ..dashboard.viewer import start_servers

    runtime_dir = os.path.abspath(runtime_dir or os.environ.get("DFC_RUNTIME_DIR", "runtime"))
    grpc_port, web_port = ports_from_env()
    servers = start_servers(grpc_port=grpc_port, web_port=web_port)
    robot_view.attach()
    rr.send_blueprint(blueprints.welcome_blueprint())
    runner.log_welcome()
    mirror = runner.SessionMirror(_StateReader(runtime_dir))
    mirror.start()
    return RerunService(servers, mirror)


def main() -> None:
    service = start_service()
    print(f"Deprecated Rerun stream viewer: {service.web_url}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
