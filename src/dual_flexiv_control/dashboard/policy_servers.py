"""Inspect policy servers for the dashboard's Eval launcher.

The launcher already asks for the policy type, host, and port.  Once all three
are explicit, this module makes the same lightweight handshake the eval client
will make and returns the server-provided metadata for a hover-help summary.
Inspection is deliberately informational: launch preflight remains the
authority on whether an eval may start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from omegaconf import OmegaConf

from dual_flexiv_control.dashboard.tasks import POLICY_GROUP_DIR


@dataclass(frozen=True)
class PolicyServerInfo:
    """One policy-server inspection result, suitable for cached display."""

    policy_type: str
    adapter: str
    transport: str
    endpoint: str
    reachable: bool
    metadata: dict
    error: str | None = None

    @property
    def display_type(self) -> str:
        """Human-facing server family name."""
        return "OpenPI" if self.policy_type.lower() == "openpi" else self.policy_type.upper()

    @property
    def checkpoint(self) -> str | None:
        """Best checkpoint/model identifier advertised by the server."""
        return _find_checkpoint(self.metadata)


def inspect_policy_server(
    policy_type: str,
    host: str,
    port: int,
    *,
    timeout_s: float = 1.5,
    policy_dir: Path = POLICY_GROUP_DIR,
) -> PolicyServerInfo:
    """Probe a fully specified policy server and collect its handshake metadata."""
    adapter, transport = _policy_protocol(policy_type, policy_dir)
    scheme = "http" if transport == "http" else "ws"
    endpoint = f"{scheme}://{host}:{port}"
    try:
        if transport == "http":
            metadata = _probe_http(endpoint, timeout_s)
        elif transport == "websocket":
            metadata = _probe_websocket(host, port, timeout_s)
        else:
            raise ValueError(f"unsupported policy transport {transport!r}")
    except Exception as exc:  # noqa: BLE001 - inspection must never break the dashboard
        return PolicyServerInfo(
            policy_type=policy_type,
            adapter=adapter,
            transport=transport,
            endpoint=endpoint,
            reachable=False,
            metadata={},
            error=_short_error(exc),
        )
    return PolicyServerInfo(
        policy_type=policy_type,
        adapter=adapter,
        transport=transport,
        endpoint=endpoint,
        reachable=True,
        metadata=metadata,
    )


def policy_server_help(info: PolicyServerInfo) -> str:
    """Markdown shown by Streamlit's hover-help affordance."""
    state = "reachable" if info.reachable else "unreachable"
    lines = [
        "**Policy server details**",
        "",
        f"- Status: **{state}**",
        f"- Type: **{info.display_type}**",
        f"- Adapter: `{info.adapter}`",
        f"- Transport: `{info.transport}`",
        f"- Address: `{info.endpoint}`",
    ]
    if info.checkpoint:
        lines.append(f"- Checkpoint/model: `{info.checkpoint}`")
    if info.error:
        lines.extend(["", f"Inspection failed: {info.error}"])
    elif info.metadata:
        rendered = json.dumps(info.metadata, indent=2, default=_json_default, sort_keys=True)
        if len(rendered) > 3500:
            rendered = rendered[:3500] + "\n…"
        lines.extend(["", "**Server metadata**", "", f"```json\n{rendered}\n```"])
    else:
        lines.extend(["", "The server did not advertise checkpoint metadata."])
    return "\n".join(lines)


def _policy_protocol(policy_type: str, policy_dir: Path) -> tuple[str, str]:
    """Read adapter/transport overrides from one selectable policy group."""
    path = policy_dir / f"{policy_type}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown policy type {policy_type!r}")
    cfg = OmegaConf.load(path)
    return str(cfg.get("adapter", "openpi")), str(cfg.get("transport", "websocket"))


def _probe_websocket(host: str, port: int, timeout_s: float) -> dict:
    """Read the initial metadata frame from an OpenPI-compatible server."""
    from dual_flexiv_control.policy.client import WebsocketTransport

    transport = WebsocketTransport(
        host,
        port,
        connect_timeout_s=0.0,  # one inspection attempt; never enter the eval retry loop
        infer_timeout_s=timeout_s,
    )
    try:
        metadata = transport.server_metadata
        return metadata if isinstance(metadata, dict) else {"metadata": metadata}
    finally:
        transport.close()


def _probe_http(endpoint: str, timeout_s: float) -> dict:
    """Read ACME's root description and best-effort ``/conventions`` contract."""
    metadata: dict = {}
    root = _get_json(endpoint + "/", timeout_s, required=True)
    if root:
        metadata["server"] = root
    conventions = _get_json(endpoint + "/conventions", timeout_s, required=False)
    if conventions:
        metadata["conventions"] = conventions
    return metadata


def _get_json(url: str, timeout_s: float, *, required: bool) -> object | None:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - operator-entered host
            body = response.read()
    except HTTPError:
        if not required:
            return None
        raise
    except URLError:
        raise
    if not body:
        return None
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # A successful root response is enough to establish reachability even
        # when an older ACME server returns plain text.
        return None


def _find_checkpoint(value: object) -> str | None:
    """Find a compact checkpoint/model name in heterogeneous server metadata."""
    preferred = (
        "checkpoint_dir",
        "checkpoint_path",
        "checkpoint",
        "policy_name",
        "model_name",
        "model_type",
    )
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        for key in preferred:
            candidate = lowered.get(key)
            if isinstance(candidate, (str, int, float)) and str(candidate):
                return _compact_identifier(str(candidate))
        for item in value.values():
            candidate = _find_checkpoint(item)
            if candidate:
                return candidate
    elif isinstance(value, (list, tuple)):
        for item in value:
            candidate = _find_checkpoint(item)
            if candidate:
                return candidate
    return None


def _compact_identifier(value: str) -> str:
    """Keep the useful tail of long checkpoint paths in the one-line summary."""
    stripped = value.rstrip("/")
    if len(stripped) <= 72:
        return stripped
    parts = Path(stripped).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else stripped[-72:]


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:300] or exc.__class__.__name__


def _json_default(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else str(value)
