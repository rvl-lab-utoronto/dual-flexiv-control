"""Tests for the on-request FACTR HTTP clients (one server per leader) and the
brain integration."""

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
    """A fake FactrCfg (duck-typed) with one server per side."""
    def srv(addr):
        host, port = addr
        return SimpleNamespace(
            host=host, port=port, endpoint="get_joint_positions",
            request_timeout_s=1.0, dof=7,
        )
    return SimpleNamespace(servers={"left": srv(left_addr), "right": srv(right_addr)})


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


def test_brain_queries_two_factr_servers(tmp_path):
    """The brain exposes factr_joint_positions(), backed by two live servers."""
    from dual_flexiv_control.brain import Brain
    from dual_flexiv_control.streams.registry import StreamRegistry

    left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    right = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    left_srv = _serve({"left": left})
    right_srv = _serve({"right": right})
    try:
        client = FactrClient.from_config(
            _factr_cfg(left_srv.server_address, right_srv.server_address)
        )
        brain = Brain(StreamRegistry(tmp_path, "rid"), stream_names=[], factr=client)
        out = brain.factr_joint_positions()
        np.testing.assert_allclose(out["left"], left)
        np.testing.assert_allclose(out["right"], right)
        brain.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()
            s.server_close()


def test_brain_without_factr_raises():
    from dual_flexiv_control.brain import Brain
    from dual_flexiv_control.streams.registry import StreamRegistry

    brain = Brain(StreamRegistry("/tmp", "rid"), stream_names=[], factr=None)
    with pytest.raises(RuntimeError):
        brain.factr_joint_positions()
