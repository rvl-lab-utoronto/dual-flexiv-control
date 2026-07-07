"""Discover policy-server host candidates from the operator's SSH config.

The GPU machines serving policies (borabora, jeju, …) are the same ones people
ssh into, so ``~/.ssh/config`` is the natural registry: one ``Host`` alias per
machine, with ``HostName`` carrying the real address (e.g. a tailscale IP that
plain DNS cannot resolve). The eval launcher shows the aliases in a dropdown
and sends the *resolved* address as the ``policy.host`` override.

Deliberately a minimal parser: ``Host``/``HostName`` keywords plus simple
``Include`` globs. Wildcard/negated patterns (``Host *``) are skipped — they
are match rules, not machines. ``Match`` blocks end any open ``Host`` block and
are otherwise ignored.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

#: Glob metacharacters marking a Host pattern (a match rule, not an alias).
_PATTERN_CHARS = ("*", "?", "!")

DEFAULT_SSH_CONFIG = Path.home() / ".ssh" / "config"

#: Guard against pathological Include recursion.
_MAX_INCLUDE_DEPTH = 8


@dataclass(frozen=True)
class SshHost:
    """One concrete ``Host`` alias and the address ssh would dial for it."""

    alias: str
    #: ``HostName`` if the block has one, else the alias itself.
    address: str


def _include_paths(pattern: str, base_dir: Path) -> list[str]:
    pattern = os.path.expanduser(pattern)
    if not os.path.isabs(pattern):
        pattern = str(base_dir / pattern)
    return sorted(glob.glob(pattern))


def _parse_file(path: Path, depth: int, out: list[SshHost], seen: set[str]) -> None:
    if depth > _MAX_INCLUDE_DEPTH:
        return
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return

    #: Aliases of the Host block being read, still waiting for a HostName.
    open_aliases: list[str] = []

    def close_block() -> None:
        # No HostName seen: the alias itself is the address ssh dials.
        for alias in open_aliases:
            if alias not in seen:
                seen.add(alias)
                out.append(SshHost(alias=alias, address=alias))
        open_aliases.clear()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # "Key Value" or "Key=Value"; keywords are case-insensitive.
        key, _, value = line.replace("=", " ", 1).partition(" ")
        key, value = key.lower(), value.strip()
        if key == "host":
            close_block()
            open_aliases.extend(
                a for a in value.split()
                if a and not any(c in a for c in _PATTERN_CHARS)
            )
        elif key == "match":
            close_block()
        elif key == "hostname" and open_aliases:
            for alias in open_aliases:
                if alias not in seen:
                    seen.add(alias)
                    out.append(SshHost(alias=alias, address=value))
            open_aliases.clear()
        elif key == "include":
            for inc in value.split():
                for p in _include_paths(inc, path.parent):
                    _parse_file(Path(p), depth + 1, out, seen)
    close_block()


def discover_ssh_hosts(config_path: Path | None = None) -> list[SshHost]:
    """Concrete hosts from the SSH config, in file order; [] if there is none.

    First ``Host`` block wins for a repeated alias (matching ssh's first-match
    semantics for ``HostName``).
    """
    path = config_path or DEFAULT_SSH_CONFIG
    hosts: list[SshHost] = []
    _parse_file(path, depth=0, out=hosts, seen=set())
    return hosts
