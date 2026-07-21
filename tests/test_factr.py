"""Tests for the FACTR HTTP clients, the stream producer (FactrInterface), and
the stream-side helpers (fresh_leader_positions / wait_leaders_fresh)."""

from __future__ import annotations

import http.server
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dual_flexiv_control.interfaces.factr import FactrClient
from dual_flexiv_control.interfaces.factr import FactrError
from dual_flexiv_control.interfaces.factr import FactrServerClient


def _server_client(side="left", host="localhost", port=5000, sim=False, timeout_s=1.0):
    return FactrServerClient(
        side=side, host=host, port=port, endpoint="get_joint_positions",
        dof=7, timeout_s=timeout_s, sim=sim,
    )


def _factr_cfg(left_addr, right_addr):
    """A fake FactrCfg (duck-typed) with one server entry per side."""
    def srv(addr):
        host, port = addr
        return SimpleNamespace(
            host=host, port=port, endpoint="get_joint_positions",
            request_timeout_s=1.0, dof=7,
        )
    return SimpleNamespace(
        servers={"left": srv(left_addr), "right": srv(right_addr)},
        rate_hz=100.0, max_age_s=0.5,
    )


# -- single-server response parsing (no network) -----------------------------


def test_parse_bare_list():
    c = _server_client(side="left")
    np.testing.assert_allclose(c._parse(list(range(7))), np.arange(7))


def test_parse_wrapped():
    c = _server_client(side="right")
    np.testing.assert_allclose(c._parse({"q": list(range(7))}), np.arange(7))
    np.testing.assert_allclose(c._parse({"positions": list(range(7))}), np.arange(7))


def test_parse_keyed_by_own_side():
    c = _server_client(side="left")
    np.testing.assert_allclose(c._parse({"left": list(range(7)), "right": [0] * 7}), np.arange(7))
    c2 = _server_client(side="right")
    np.testing.assert_allclose(c2._parse({"right": {"q": list(range(10, 17))}}), np.arange(10, 17))


def test_parse_wrong_length_raises():
    c = _server_client(side="left")
    with pytest.raises(FactrError):
        c._parse(list(range(3)))


# -- live HTTP (stdlib server stands in for one leader's FastAPI server) ------


def _serve(payload):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # enable keep-alive so the conn is reused

        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_server_client_get_and_reuse():
    server = _serve(list(range(7)))
    host, port = server.server_address
    c = _server_client(side="left", host=host, port=port)
    try:
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))  # keep-alive reuse
    finally:
        c.close()
        server.shutdown()
        server.server_close()


def test_group_queries_two_servers():
    """One server per leader, each on its own port; the group merges them."""
    left_srv = _serve({"left": list(range(7))})
    right_srv = _serve({"right": list(range(20, 27))})
    try:
        client = FactrClient.from_config(
            _factr_cfg(left_srv.server_address, right_srv.server_address)
        )
        assert left_srv.server_address[1] != right_srv.server_address[1]  # distinct ports
        out = client.get_joint_positions()
        np.testing.assert_allclose(out["left"], np.arange(7))
        np.testing.assert_allclose(out["right"], np.arange(20, 27))
        np.testing.assert_allclose(client.get_joint_positions_for("right"), np.arange(20, 27))
        client.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()
            s.server_close()


def test_unreachable_server_raises():
    # Bind then immediately shut down to get a port nothing is listening on.
    server = _serve([])
    host, port = server.server_address
    server.shutdown()
    server.server_close()

    c = _server_client(host=host, port=port, timeout_s=0.3)
    with pytest.raises(FactrError):
        c.get_joint_positions()
    c.close()


def test_sim_needs_no_server():
    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5001))
    client = FactrClient.from_config(cfg, sim=True)
    out = client.get_joint_positions()
    assert out["left"].shape == (7,)
    assert out["right"].shape == (7,)
    # distinct phase per side -> the two leaders are not identical
    assert not np.allclose(out["left"], out["right"])
    client.close()


def test_preflight_passes_when_all_leaders_reachable():
    left_srv = _serve({"left": list(range(7))})
    right_srv = _serve({"right": list(range(7))})
    try:
        client = FactrClient.from_config(
            _factr_cfg(left_srv.server_address, right_srv.server_address)
        )
        client.preflight()  # both up -> no raise
        client.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()
            s.server_close()


def test_preflight_raises_and_names_unreachable_leader():
    # Left is served; right is bound-then-closed so nothing listens on its port.
    left_srv = _serve({"left": list(range(7))})
    right_srv = _serve([])
    right_addr = right_srv.server_address
    right_srv.shutdown()
    right_srv.server_close()
    try:
        client = FactrClient.from_config(_factr_cfg(left_srv.server_address, right_addr))
        with pytest.raises(FactrError) as exc:
            client.preflight()
        msg = str(exc.value)
        assert "right @" in msg  # the failing leader is named
        assert "left @" not in msg  # the reachable leader is not reported as a failure
        client.close()
    finally:
        left_srv.shutdown()
        left_srv.server_close()


def test_preflight_passes_in_sim_without_servers():
    # sim fabricates positions -> preflight needs no live server.
    client = FactrClient.from_config(_factr_cfg(("localhost", 5000), ("localhost", 5001)), sim=True)
    client.preflight()  # no raise despite nothing listening
    client.close()


# -- the stream producer (FactrInterface) + stream-side helpers ---------------


def test_factr_interface_declares_one_stream_per_side():
    from dual_flexiv_control.interfaces.factr import FactrInterface
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import raw_factr_stream_name

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5000))
    node = FactrInterface(cfg, SimpleNamespace(runtime_dir="/tmp", sim=False), "rid")
    specs = {s.name: s for s in node.declare_streams()}
    assert set(specs) == {
        factr_stream_name("left"), factr_stream_name("right"),
        raw_factr_stream_name("left"), raw_factr_stream_name("right"),
    }
    assert specs[factr_stream_name("left")].dim == 7  # the fixture's server dof
    assert specs[factr_stream_name("left")].dtype == "float64"


def test_factr_interface_polls_live_sides_and_tolerates_a_dead_one():
    """poll() returns each reachable leader's payload under its stream name and
    simply omits an unreachable side (its stream goes stale; nothing raises)."""
    from dual_flexiv_control.interfaces.factr import FactrInterface
    from dual_flexiv_control.interfaces.factr import factr_stream_name

    left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    left_srv = _serve({"left": left})
    dead_srv = _serve([])
    dead_addr = dead_srv.server_address
    dead_srv.shutdown()
    dead_srv.server_close()
    try:
        cfg = _factr_cfg(left_srv.server_address, dead_addr)
        node = FactrInterface(cfg, SimpleNamespace(runtime_dir="/tmp", sim=False), "rid")
        node.open_source()
        sample = node.poll()
        assert set(sample) == {factr_stream_name("left")}  # right omitted, no raise
        np.testing.assert_allclose(sample[factr_stream_name("left")], left)
        node.close_source()
    finally:
        left_srv.shutdown()
        left_srv.server_close()


class _StubSamples:
    def __init__(self, vec, t_ns):
        arr = np.asarray(vec, dtype=np.float64)
        self.n = 0 if arr.size == 0 else 1
        self.newest = arr if self.n else None
        self.newest_t_ns = t_ns


class _StubSource:
    """Duck-typed Brain: latest(name) from a dict; KeyError when unsubscribed."""

    def __init__(self, samples: dict):
        self._samples = samples

    def latest(self, name):
        return self._samples[name]


def test_fresh_leader_positions_gates_on_sample_age():
    import time

    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import fresh_leader_positions

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5000))
    cfg.max_age_s = 0.5
    now = time.monotonic_ns()
    source = _StubSource({
        factr_stream_name("left"): _StubSamples(np.arange(7.0), now),           # fresh
        factr_stream_name("right"): _StubSamples(np.arange(7.0), now - int(60e9)),  # stale
    })
    out = fresh_leader_positions(source, cfg)
    assert set(out) == {"left"}  # stale right omitted, never fabricated
    np.testing.assert_allclose(out["left"], np.arange(7.0))

    # An unsubscribed stream (KeyError) is omitted the same way.
    out = fresh_leader_positions(_StubSource({}), cfg)
    assert out == {}


def test_wait_leaders_fresh_raises_naming_missing_sides():
    import time

    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import wait_leaders_fresh

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5000))
    cfg.max_age_s = 0.5
    source = _StubSource({
        factr_stream_name("left"): _StubSamples(np.arange(7.0), time.monotonic_ns()),
        factr_stream_name("right"): _StubSamples([], 0),  # never produced
    })
    with pytest.raises(FactrError) as exc:
        wait_leaders_fresh(source, cfg, timeout_s=0.2)
    assert "right" in str(exc.value)
    assert "left" not in str(exc.value).split("stream(s):")[1]


def test_wait_leaders_fresh_returns_when_all_fresh():
    import time

    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import wait_leaders_fresh

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5000))
    cfg.max_age_s = 0.5
    now = time.monotonic_ns()
    source = _StubSource({
        factr_stream_name("left"): _StubSamples(np.arange(7.0), now),
        factr_stream_name("right"): _StubSamples(np.arange(7.0), now),
    })
    wait_leaders_fresh(source, cfg, timeout_s=0.2)  # no raise
