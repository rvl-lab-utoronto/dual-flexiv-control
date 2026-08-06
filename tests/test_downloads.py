"""Tests for the Storage-tab streaming download registry."""

from __future__ import annotations

import os
import threading

import pytest

from dual_flexiv_control.dashboard.downloads import DownloadServer
from dual_flexiv_control.dashboard.downloads import _safe_filename


def test_safe_filename_strips_paths_and_unsafe_characters():
    assert _safe_filename("../../my camera (left).mp4") == "my-camera-left.mp4"
    assert _safe_filename("...") == "download"


def test_register_is_stable_and_never_exposes_the_source_path(tmp_path, monkeypatch):
    video = tmp_path / "private source.mp4"
    video.write_bytes(b"mp4")

    server = object.__new__(DownloadServer)
    server.port = 19092
    server._lock = threading.Lock()
    server._by_token = {}
    server._token_by_file = {}
    monkeypatch.setattr("secrets.token_urlsafe", lambda _n: "opaque-token")

    first = server.register(video, "episode 1.mp4", content_type="video/mp4")
    second = server.register(video, "episode 1.mp4", content_type="video/mp4")

    assert first == second
    assert first.endswith("/download/opaque-token/episode-1.mp4")
    assert str(tmp_path) not in first
    item = server._resolve("opaque-token")
    assert item is not None
    assert item.path == os.path.realpath(video)
    assert item.content_type == "video/mp4"


def test_register_rejects_a_missing_file(tmp_path):
    server = object.__new__(DownloadServer)
    server.port = 19092
    server._lock = threading.Lock()
    server._by_token = {}
    server._token_by_file = {}
    with pytest.raises(FileNotFoundError):
        server.register(tmp_path / "missing.mp4", "missing.mp4")
