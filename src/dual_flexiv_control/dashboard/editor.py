"""Open a task's YAML in the workstation's VSCode from the dashboard.

The dashboard, the ``conf/task`` files, and VSCode all run on the same machine
(the workstation driving the brain), so opening a file is a local shell-out to
the VSCode CLI: ``code --reuse-window <file>`` hands the path to the running
editor over its IPC socket.

Finding that socket is the only fiddly part — VSCode leaves stale
``vscode-ipc-*.sock`` files around. We prefer ``VSCODE_IPC_HOOK_CLI`` (correct
when the dashboard was launched from a VSCode integrated terminal, which is the
recommended way) and fall back to the most recently modified socket.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenResult:
    """Outcome of an open-in-editor attempt, surfaced to the UI."""

    ok: bool
    message: str


def open_in_vscode(path: Path, timeout_s: float = 10.0) -> OpenResult:
    """Open ``path`` in the running VSCode (reusing its window).

    Returns an :class:`OpenResult` rather than raising, so the dashboard can show
    a toast/warning either way. ``path`` comes from a discovered
    :class:`~.tasks.TaskInfo`, never raw user input.
    """
    path = Path(path)
    if not path.exists():
        return OpenResult(False, f"File not found: {path}")

    code = _find_code_cli()
    if code is None:
        return OpenResult(
            False,
            "VSCode CLI ('code') not found. Launch the dashboard from a VSCode "
            "integrated terminal, or add 'code' to PATH.",
        )

    env = dict(os.environ)
    socket = _ipc_socket()
    if socket is not None:
        env["VSCODE_IPC_HOOK_CLI"] = socket
    elif "VSCODE_IPC_HOOK_CLI" not in env:
        return OpenResult(
            False,
            "No VSCode IPC socket found — is the editor open on this machine? "
            f"You can open it manually: {path}",
        )

    try:
        proc = subprocess.run(
            [code, "--reuse-window", str(path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return OpenResult(False, "Timed out talking to VSCode. Is the editor open here?")
    except OSError as exc:
        return OpenResult(False, f"Could not run the VSCode CLI: {exc}")

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = detail[-1] if detail else f"exit code {proc.returncode}"
        return OpenResult(False, f"VSCode CLI failed: {hint}")

    return OpenResult(True, f"Opened {path.name} in VSCode")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _find_code_cli() -> str | None:
    """Locate the ``code`` binary: PATH first, then known server-CLI layouts."""
    found = shutil.which("code")
    if found:
        return found
    patterns = (
        "~/.vscode-server/cli/servers/*/server/bin/remote-cli/code",
        "~/.vscode-server/bin/*/bin/remote-cli/code",
        "~/.vscode-server-insiders/cli/servers/*/server/bin/remote-cli/code",
        "~/.cursor-server/cli/servers/*/server/bin/remote-cli/code",
    )
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[-1]
    return None


def _ipc_socket() -> str | None:
    """The IPC socket for the live editor: inherited env var, else newest socket."""
    inherited = os.environ.get("VSCODE_IPC_HOOK_CLI")
    if inherited and os.path.exists(inherited):
        return inherited
    candidates: list[str] = []
    search_dirs = [
        os.environ.get("XDG_RUNTIME_DIR"),
        f"/run/user/{os.getuid()}" if hasattr(os, "getuid") else None,
        os.environ.get("TMPDIR"),
        "/tmp",
    ]
    for directory in search_dirs:
        if directory:
            candidates.extend(glob.glob(os.path.join(directory, "vscode-ipc-*.sock")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
