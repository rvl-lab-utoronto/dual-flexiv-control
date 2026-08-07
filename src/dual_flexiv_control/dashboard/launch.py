"""Console entry point for the dashboard: ``dfc-dashboard``.

Streamlit apps run under ``streamlit run <file>`` (not ``python <file>``), so this
shim hands :mod:`~.app` to Streamlit's CLI. Extra args pass through, e.g.::

    dfc-dashboard --rig left_only
    dfc-dashboard --server.port 8502
    DFC_VISER_PORT=9094 dfc-dashboard

The rig is a **launch option** (``--rig <name>`` or ``DFC_DASHBOARD_RIG``), not a
dashboard control: changing rigs restarts the session daemon (arms + cameras),
which is too destructive to offer mid-session. Pick the rig here; the dashboard
pins every compose to it for the process lifetime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Env var carrying the launch-selected rig into the app (Streamlit runs the app
#: in this same process, but the env also survives a ``streamlit run app.py``).
RIG_ENV_VAR = "DFC_DASHBOARD_RIG"


def main() -> None:
    from streamlit.web import cli as stcli

    rig, passthrough = _pop_rig_arg(sys.argv[1:])
    if rig is not None:
        _validate_rig(rig)
        os.environ[RIG_ENV_VAR] = rig
    elif os.environ.get(RIG_ENV_VAR):
        _validate_rig(os.environ[RIG_ENV_VAR])

    _prebind_viser_service()

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
        *passthrough,
    ]
    raise SystemExit(stcli.main())


def _pop_rig_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract ``--rig <name>`` / ``--rig=<name>`` from ``argv``.

    Streamlit's CLI does not know ``--rig``, so it must be stripped before the
    remaining args pass through. Returns ``(rig_or_None, remaining_args)``; a
    trailing ``--rig`` with no value is a usage error.
    """
    rig: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--rig":
            if i + 1 >= len(argv):
                raise SystemExit("usage: dfc-dashboard --rig <name> (missing rig name)")
            rig = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--rig="):
            rig = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return rig, rest


def _validate_rig(rig: str) -> None:
    """Abort startup with the valid rig names if ``rig`` is not in ``conf/rig``."""
    from dual_flexiv_control.dashboard.tasks import discover_rigs

    names = [r.name for r in discover_rigs()]
    if rig not in names:
        raise SystemExit(
            f"unknown rig {rig!r} — pick one of: {', '.join(names) or '(none found)'} "
            "(conf/rig/<name>.yaml)"
        )


def _prebind_viser_service() -> None:
    """Start the isolated 3 Hz stream consumer before Streamlit boots."""
    from dual_flexiv_control.viser.service import start_service

    start_service()


# Kept for launch integrations that imported the old private hook. It no longer
# starts or imports Rerun.
_prebind_rerun_servers = _prebind_viser_service


if __name__ == "__main__":
    main()
