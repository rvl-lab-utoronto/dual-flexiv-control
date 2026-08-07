"""Live Viser client customizations that are not exposed by its Python API.

Viser 1.0 exposes floating, collapsible, and fixed control layouts, but no
canvas-only layout.  The dashboard's live view is intentionally display-only,
so serve the installed client from a content-addressed cache with a tiny script
that removes the control panel after React mounts it.  The Viser websocket and
scene protocol remain untouched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

HIDDEN_PANEL_LABEL = "__DFC_HIDE_VISER_CONTROL_PANEL__"
VISER_VIEW_REVISION = 1
_INJECTION_MARKER = "data-dfc-hide-viser-control-panel"
_PATCH_LOCK = threading.Lock()

_SIDEBAR_HIDER = f"""
<script {_INJECTION_MARKER}>
(() => {{
  const sentinel = {HIDDEN_PANEL_LABEL!r};
  const hideControlPanel = () => {{
    const label = Array.from(document.querySelectorAll("div")).find(
      (element) =>
        element.children.length === 0 &&
        element.textContent.trim() === sentinel,
    );
    if (!label) return;

    // Connection label -> handle -> fixed-width contents -> panel Paper.
    const panel = label.parentElement?.parentElement?.parentElement;
    if (!(panel instanceof HTMLElement)) return;
    panel.style.setProperty("display", "none", "important");

    // The preceding Paper reserves the sidebar width.  The element before
    // that is the collapsed-sidebar handle; remove both as well.
    const shadow = panel.previousElementSibling;
    if (shadow instanceof HTMLElement) {{
      shadow.style.setProperty("display", "none", "important");
      const collapsedHandle = shadow.previousElementSibling;
      if (collapsedHandle instanceof HTMLElement) {{
        collapsedHandle.style.setProperty("display", "none", "important");
      }}
    }}
    observer.disconnect();
  }};

  const observer = new MutationObserver(hideControlPanel);
  observer.observe(document.documentElement, {{
    childList: true,
    subtree: true,
    characterData: true,
  }});
  hideControlPanel();
}})();
</script>
""".strip()


def _inject_sidebar_hider(index_html: str) -> str:
    """Inject the canvas-only behavior once into a Viser client document."""
    if _INJECTION_MARKER in index_html:
        return index_html
    if "</head>" not in index_html:
        raise ValueError("Viser client index has no </head> element")
    return index_html.replace("</head>", f"{_SIDEBAR_HIDER}</head>", 1)


def _cache_root() -> Path:
    override = os.environ.get("DFC_VISER_CLIENT_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "dual-flexiv-control" / "viser-client"


def _hidden_client_root(source_root: Path, cache_root: Path | None = None) -> Path:
    """Build and return a content-addressed copy of Viser's web client."""
    source_root = Path(source_root).resolve()
    source_index = source_root / "index.html"
    index_html = source_index.read_text(encoding="utf-8")
    modified = _inject_sidebar_hider(index_html)
    digest = hashlib.sha256(modified.encode("utf-8")).hexdigest()[:20]
    target_root = Path(cache_root or _cache_root()) / digest
    target_index = target_root / "index.html"
    if target_index.is_file():
        return target_root

    target_root.mkdir(parents=True, exist_ok=True)
    # Current Viser wheels are single-file builds.  Preserve any future static
    # siblings as well so this adapter remains version-tolerant.
    for source in source_root.iterdir():
        if source.name == "index.html":
            continue
        target = target_root / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target_root,
            prefix=".index-", suffix=".html", delete=False,
        ) as handle:
            handle.write(modified)
            temporary = Path(handle.name)
        os.replace(temporary, target_index)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target_root


def create_live_server(viser_module, **kwargs):
    """Create a Viser server whose HTTP client omits the control sidebar.

    Viser hard-codes its wheel's client root inside :class:`ViserServer`.  Swap
    only the constructor used during this one server creation, then immediately
    restore it.  The resulting websocket server is otherwise the stock class.
    """
    from viser import infra

    installed_root = (
        Path(viser_module.__file__).resolve().parent / "client" / "build"
    )
    client_root = _hidden_client_root(installed_root)
    with _PATCH_LOCK:
        base_server = infra.WebsockServer

        class CanvasOnlyWebsockServer(base_server):
            def __init__(self, *args, **server_kwargs):
                server_kwargs["http_server_root"] = client_root
                super().__init__(*args, **server_kwargs)

        infra.WebsockServer = CanvasOnlyWebsockServer
        try:
            return viser_module.ViserServer(**kwargs)
        finally:
            infra.WebsockServer = base_server
