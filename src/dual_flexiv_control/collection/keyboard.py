"""Operator episode control over the terminal (tty), dependency-free.

Multi-episode collection is operator-paced: one process records demos back to
back, and the operator ends / re-records / stops with a keypress. This reads
single keystrokes from stdin in cbreak mode on a background thread and exposes
them as :class:`Event` values the loop drains each tick.

Keys::

    ENTER / SPACE   end the current episode and save it, start the next
    r               discard the in-progress episode and re-record it
    q / ESC         stop collecting

When stdin is not a tty (dashboard, systemd, a pipe) key control is unavailable:
:attr:`active` is False and the loop falls back to single-episode-per-run,
bounded by the shared stop event. Always use as a context manager so the
terminal mode is restored on exit.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque

log = logging.getLogger(__name__)

# Event kinds drained by the loop.
END = "end"
DISCARD = "discard"
STOP = "stop"

_KEYMAP = {
    "\n": END, "\r": END, " ": END,
    "r": DISCARD, "R": DISCARD,
    "q": STOP, "Q": STOP, "\x1b": STOP,
}


class EpisodeControl:
    """Background tty key reader; :meth:`poll` drains pending events."""

    def __init__(self) -> None:
        self._events: deque[str] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._restore = None
        self.active = _stdin_is_tty()

    def __enter__(self) -> "EpisodeControl":
        if self.active:
            try:
                self._enable_cbreak()
                self._thread = threading.Thread(
                    target=self._reader, name="dfc-episode-keys", daemon=True
                )
                self._thread.start()
                log.info("episode keys: ENTER/SPACE=save  r=re-record  q/ESC=stop")
            except Exception:  # noqa: BLE001 - no usable tty -> disable, single-episode
                log.warning("keyboard control unavailable; single episode per run")
                self.active = False
                self._disable_cbreak()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._disable_cbreak()

    def poll(self) -> list[str]:
        """Return and clear the events seen since the last poll (oldest first)."""
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    # -- internals ------------------------------------------------------------

    def _reader(self) -> None:
        while not self._stop.is_set():
            ch = sys.stdin.read(1)
            if ch == "":  # EOF
                break
            event = _KEYMAP.get(ch)
            if event is not None:
                with self._lock:
                    self._events.append(event)

    def _enable_cbreak(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._restore = (fd, termios.tcgetattr(fd))
        tty.setcbreak(fd)

    def _disable_cbreak(self) -> None:
        if self._restore is None:
            return
        import termios

        fd, saved = self._restore
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:  # noqa: BLE001 - best-effort restore
            pass
        self._restore = None


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, OSError):
        return False
