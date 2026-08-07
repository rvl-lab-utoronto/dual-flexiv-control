"""Plotly Dash live plots backed by an isolated stream consumer.

The package name avoids shadowing either the external :mod:`plotly` or
:mod:`dash` packages. Viser remains responsible for the live 3D scene.
"""

from __future__ import annotations

__all__ = ["consumer", "service", "view"]
