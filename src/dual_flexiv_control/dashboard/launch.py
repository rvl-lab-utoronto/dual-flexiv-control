"""Console entry point for the dashboard: ``dfc-dashboard``.

Streamlit apps run under ``streamlit run <file>`` (not ``python <file>``), so this
shim hands :mod:`~.app` to Streamlit's CLI. Extra args pass through, e.g.::

    dfc-dashboard --server.port 8502
    DFC_DASHBOARD_WEB_PORT=9091 dfc-dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    _prebind_rerun_servers()

    app_path = str(Path(__file__).resolve().parent / "app.py")
    # --server.headless skips the first-run email prompt and browser auto-open;
    # --theme.base=dark forces the dark theme for everyone (overridable by the
    # caller, since user args are appended last). Streamlit still prints the URL.
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        "--theme.base=dark",
        # Hide the Deploy button and the three-dot toolbar menu.
        "--client.toolbarMode=minimal",
        *sys.argv[1:],
    ]
    raise SystemExit(stcli.main())


def _prebind_rerun_servers() -> None:
    """Start the Rerun gRPC + web-viewer servers before Streamlit boots.

    Streamlit only runs the app script once a browser session connects, so
    without this the metrics Rerun ports stay unbound until first page load.
    Binding them here means those (Streamlit + metrics 9090/9876) listen
    immediately — friendlier for port-forwarding, and the viewer shows its idle
    screen before anyone connects. Streamlit runs the app in **this** process, so
    its first session reuses these very servers (``start_servers`` is idempotent).

    The **replay** gRPC data server (9880) is intentionally NOT pre-bound: it starts
    lazily on the first ▶ click (``app._replay_viewer``) and is embedded in this same
    web viewer (no separate web port).

    Failures here are **not** swallowed: if the ports can't bind (e.g. a stale
    dashboard still holding :9090), this raises and aborts startup so the operator
    sees it immediately, instead of leaving a silently-black viewer that only a full
    restart clears.
    """
    import rerun as rr

    from dual_flexiv_control.dashboard import blueprints
    from dual_flexiv_control.dashboard import robot_view
    from dual_flexiv_control.dashboard import runner
    from dual_flexiv_control.dashboard.viewer import ports_from_env
    from dual_flexiv_control.dashboard.viewer import start_servers

    grpc_port, web_port = ports_from_env()
    start_servers(grpc_port=grpc_port, web_port=web_port)
    # Log the robot scene into the metrics recording (it now shares the metrics
    # viewer's 3D panel), then send the idle blueprint that shows it + the README.
    robot_view.attach()
    rr.send_blueprint(blueprints.welcome_blueprint())
    runner.log_welcome()


if __name__ == "__main__":
    main()
