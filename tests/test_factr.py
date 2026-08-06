"""Tests for the FACTR WebSocket clients, the stream producer (FactrInterface), and
the stream-side helpers (fresh_leader_positions / wait_leaders_fresh)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from dual_flexiv_control.interfaces.factr import FactrClient
from dual_flexiv_control.interfaces.factr import FactrError
from dual_flexiv_control.interfaces.factr import FactrServerClient
from dual_flexiv_control.configs import FactrLeaderCfg
from dual_flexiv_control.configs import FactrTransformCfg
from dual_flexiv_control.configs import JointConventionCfg


def _server_client(side="left", host="localhost", port=5000, sim=False, timeout_s=1.0):
    return FactrServerClient(
        side=side, host=host, port=port, endpoint=f"ws/{side}",
        dof=7, timeout_s=timeout_s, sim=sim,
    )


def _factr_cfg(left_addr, right_addr):
    """A fake FactrCfg (duck-typed) with one server entry per side."""
    def srv(side, addr):
        host, port = addr
        return SimpleNamespace(
            host=host, port=port, endpoint=f"ws/{side}",
            request_timeout_s=1.0, dof=7,
        )
    convention = JointConventionCfg(
        offsets_deg=[0.0] * 6,
        sign_flip_joints=[],
        wrap_deg=True,
        drop_trailing=1,
        gripper_open=0.0,
        gripper_closed=1.0,
    )
    leader = FactrLeaderCfg(
        raw_to_dfc=convention,
        home_q_rad=[0.0] * 6,
        dfc_to_factr=FactrTransformCfg(
            signs=[1.0] * 6, offset_rad=[0.0] * 6
        ),
    )
    return SimpleNamespace(
        servers={"left": srv("left", left_addr), "right": srv("right", right_addr)},
        leaders={"left": leader, "right": leader},
        rate_hz=100.0, max_age_s=0.5,
    )


def _telemetry(side):
    fields = (
        "raw_q_rad", "model_q_rad", "model_dq_rad_s", "home_error_rad",
        "joint_offsets_rad", "model_signs", "limit_torque_nm", "null_torque_nm",
        "gravity_torque_nm", "friction_torque_nm",
        "force_feedback_torque_nm", "applied_torque_nm",
    )
    return {
        "type": "telemetry",
        "side": side,
        **{field: [0.1] * 6 for field in fields},
        "stamp_monotonic_ns": 123,
        "grav_comp_gain": 0.2,
        "grav_comp_gain_target": 1.0,
        "force_feedback_gain": 0.3,
        "force_feedback_gain_target": 0.0,
    }


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


# -- live WebSocket (sync server stands in for one leader's FACTR relay) ------


def _serve(side, payload, telemetry=None, close_after=None):
    telemetry = telemetry or _telemetry(side)
    state = SimpleNamespace(connections=0, received=[])

    def handler(websocket):
        state.connections += 1
        try:
            websocket.send(json.dumps(telemetry))
            sent = 0
            while True:
                websocket.send(json.dumps({
                    "type": "reading", "side": side, "joint_pos": payload,
                }))
                sent += 1
                if close_after is not None and sent >= close_after:
                    return
                # Drain any client->server frames (force feedback) between pushes.
                try:
                    while True:
                        state.received.append(json.loads(websocket.recv(timeout=0)))
                except TimeoutError:
                    pass
                time.sleep(0.005)
        except ConnectionClosed:
            pass

    server = serve(handler, "127.0.0.1", 0)
    server.test_state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _server_address(server):
    host, port = server.socket.getsockname()[:2]
    return host, port


def test_server_client_get_and_reuse_websocket():
    server = _serve("left", list(range(7)))
    host, port = _server_address(server)
    c = _server_client(side="left", host=host, port=port)
    try:
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
        assert c._receiver is not None
        receiver = c._receiver
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
        assert c._receiver is receiver  # one persistent connection/receiver
    finally:
        c.close()
        server.shutdown()


def test_server_client_reconnects_after_stream_closes():
    server = _serve("left", list(range(7)), close_after=1)
    host, port = _server_address(server)
    c = _server_client(side="left", host=host, port=port)
    try:
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
        deadline = time.monotonic() + 2.0
        while server.test_state.connections < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.test_state.connections >= 2
        np.testing.assert_allclose(c.get_joint_positions(), np.arange(7))
    finally:
        c.close()
        server.shutdown()


def test_server_client_live_telemetry_cache():
    """Live telemetry shares the WebSocket but remains separate from readings."""
    telemetry = _telemetry("left")
    telemetry["raw_q_rad"] = list(range(6))
    server = _serve("left", list(range(7)), telemetry=telemetry)
    host, port = _server_address(server)
    c = _server_client(side="left", host=host, port=port)
    try:
        c.get_joint_positions()  # starts/drains the receiver
        version, data = c.get_telemetry()
        assert version >= 1 and data["side"] == "left"
        assert data["raw_q_rad"] == list(range(6))
        assert c._http_conn is None
    finally:
        c.close()
        server.shutdown()


def test_group_queries_two_servers():
    """One server per leader, each on its own port; the group merges them."""
    left_srv = _serve("left", list(range(7)))
    right_srv = _serve("right", list(range(20, 27)))
    try:
        client = FactrClient.from_config(
            _factr_cfg(_server_address(left_srv), _server_address(right_srv))
        )
        assert _server_address(left_srv)[1] != _server_address(right_srv)[1]
        out = client.get_joint_positions()
        np.testing.assert_allclose(out["left"], np.arange(7))
        np.testing.assert_allclose(out["right"], np.arange(20, 27))
        np.testing.assert_allclose(client.get_joint_positions_for("right"), np.arange(20, 27))
        client.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()


def test_send_force_feedback_reaches_server():
    server = _serve("left", list(range(7)))
    host, port = _server_address(server)
    c = _server_client(side="left", host=host, port=port)
    try:
        tau = [0.5, -1.0, 0.0, 2.5, 0.0, 0.0, -0.25]
        c.send_force_feedback(tau)
        deadline = time.monotonic() + 2.0
        while not server.test_state.received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.test_state.received, "server never saw the force_feedback frame"
        assert server.test_state.received[0] == {
            "type": "force_feedback", "side": "left", "space": "joint", "tau": tau,
        }
    finally:
        c.close()
        server.shutdown()


def test_group_send_force_feedback_routes_by_side():
    left_srv = _serve("left", list(range(7)))
    right_srv = _serve("right", list(range(7)))
    try:
        client = FactrClient.from_config(
            _factr_cfg(_server_address(left_srv), _server_address(right_srv))
        )
        client.send_force_feedback({
            "left": np.full(7, 0.5), "right": np.full(7, -0.5),
        })
        deadline = time.monotonic() + 2.0
        while (
            not (left_srv.test_state.received and right_srv.test_state.received)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert left_srv.test_state.received[0]["tau"] == [0.5] * 7
        assert left_srv.test_state.received[0]["side"] == "left"
        assert right_srv.test_state.received[0]["tau"] == [-0.5] * 7
        assert right_srv.test_state.received[0]["side"] == "right"
        client.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()


def test_send_force_feedback_without_server_raises():
    server = _serve("left", [])
    host, port = _server_address(server)
    server.shutdown()

    c = _server_client(host=host, port=port, timeout_s=0.3)
    with pytest.raises(FactrError):
        c.send_force_feedback(np.zeros(7))
    c.close()


def test_send_force_feedback_rejects_bad_vectors():
    # Validation happens before any connection is attempted.
    c = _server_client()
    with pytest.raises(FactrError):
        c.send_force_feedback(np.zeros((2, 7)))     # not 1-D
    with pytest.raises(FactrError):
        c.send_force_feedback([float("nan")] * 7)   # non-finite
    with pytest.raises(FactrError):
        c.send_force_feedback([])                   # empty
    c.close()


def test_sim_send_force_feedback_is_noop():
    c = _server_client(sim=True)
    c.send_force_feedback(np.zeros(7))  # no server, no error
    c.close()


def test_unreachable_server_raises():
    # Bind then immediately shut down to get a port nothing is listening on.
    server = _serve("left", [])
    host, port = _server_address(server)
    server.shutdown()

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


def test_group_expands_side_placeholder_in_websocket_endpoint():
    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5001))
    cfg.servers["left"].endpoint = "ws/{side}"
    client = FactrClient.from_config(cfg, sim=True)
    try:
        assert client.server("left").url == "ws://localhost:5000/ws/left"
    finally:
        client.close()


def test_preflight_passes_when_all_leaders_reachable():
    left_srv = _serve("left", list(range(7)))
    right_srv = _serve("right", list(range(7)))
    try:
        client = FactrClient.from_config(
            _factr_cfg(_server_address(left_srv), _server_address(right_srv))
        )
        client.preflight()  # both up -> no raise
        client.close()
    finally:
        for s in (left_srv, right_srv):
            s.shutdown()


def test_preflight_raises_and_names_unreachable_leader():
    # Left is served; right is bound-then-closed so nothing listens on its port.
    left_srv = _serve("left", list(range(7)))
    right_srv = _serve("right", [])
    right_addr = _server_address(right_srv)
    right_srv.shutdown()
    try:
        client = FactrClient.from_config(_factr_cfg(_server_address(left_srv), right_addr))
        with pytest.raises(FactrError) as exc:
            client.preflight()
        msg = str(exc.value)
        assert "right @" in msg  # the failing leader is named
        assert "left @" not in msg  # the reachable leader is not reported as a failure
        client.close()
    finally:
        left_srv.shutdown()


def test_preflight_passes_in_sim_without_servers():
    # sim fabricates positions -> preflight needs no live server.
    client = FactrClient.from_config(_factr_cfg(("localhost", 5000), ("localhost", 5001)), sim=True)
    client.preflight()  # no raise despite nothing listening
    client.close()


# -- the stream producer (FactrInterface) + stream-side helpers ---------------


def test_factr_interface_declares_pose_and_explicit_telemetry_streams():
    from dual_flexiv_control.interfaces.factr import FactrInterface
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import raw_factr_stream_name
    from dual_flexiv_control.interfaces.factr import factr_telemetry_stream_name

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5000))
    node = FactrInterface(cfg, SimpleNamespace(runtime_dir="/tmp", sim=False), "rid")
    specs = {s.name: s for s in node.declare_streams()}
    assert {
        factr_stream_name("left"), factr_stream_name("right"),
        raw_factr_stream_name("left"), raw_factr_stream_name("right"),
    } <= set(specs)
    assert factr_telemetry_stream_name("left", "model_q_rad") in specs
    assert specs[factr_telemetry_stream_name("right", "grav_comp_gain")].dim == 1
    assert specs[factr_telemetry_stream_name("left", "gravity_torque_nm")].dim == 6
    assert specs[factr_stream_name("left")].dim == 7  # the fixture's server dof
    assert specs[factr_stream_name("left")].dtype == "float64"


def test_factr_interface_polls_live_sides_and_tolerates_a_dead_one():
    """poll() returns each reachable leader's payload under its stream names and
    simply omits a side that drops out (its stream goes stale; nothing raises).

    DFC calibration is already local; the right network side is unavailable.
    """
    from dual_flexiv_control.interfaces.factr import FactrInterface
    from dual_flexiv_control.interfaces.factr import factr_stream_name
    from dual_flexiv_control.interfaces.factr import raw_factr_stream_name

    left = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    left_srv = _serve("left", left)
    dead_srv = _serve("right", [])
    dead_addr = _server_address(dead_srv)
    dead_srv.shutdown()
    try:
        cfg = _factr_cfg(_server_address(left_srv), dead_addr)
        node = FactrInterface(cfg, SimpleNamespace(runtime_dir="/tmp", sim=False), "rid")
        # Bypass startup calibration here: this test targets a runtime outage,
        # after both leaders would already have supplied a valid contract.
        node._client = FactrClient.from_config(cfg)
        node._conventions = {
            side: JointConventionCfg(
                offsets_deg=[0.0] * 6,
                sign_flip_joints=[],
                drop_trailing=1,
                wrap_deg=True,
                gripper_open=0.0,
                gripper_closed=1.0,
            )
            for side in cfg.servers
        }
        sample = node.poll()
        assert factr_stream_name("left") in sample
        assert raw_factr_stream_name("left") in sample
        assert factr_stream_name("right") not in sample
        np.testing.assert_allclose(sample[raw_factr_stream_name("left")], left)
        # The injected contract is the identity mapping, so the converted stream
        # carries the same values (and the 0..1 gripper endpoints keep 0.7).
        np.testing.assert_allclose(sample[factr_stream_name("left")], left)
        node.close_source()
    finally:
        left_srv.shutdown()


def test_factr_interface_forwards_each_fresh_external_torque_sample_once(tmp_path):
    """The FACTR node converts RDK tau_ext into FACTR's opposite sign convention."""
    from dual_flexiv_control.interfaces.factr import FactrInterface
    from dual_flexiv_control.streams.registry import StreamRegistry
    from dual_flexiv_control.streams.spec import StreamSpec
    from dual_flexiv_control.streams.stream import StreamWriter

    cfg = _factr_cfg(("localhost", 5000), ("localhost", 5001))
    run_id = "feedback-test"
    node = FactrInterface(
        cfg, SimpleNamespace(runtime_dir=str(tmp_path), sim=False), run_id
    )

    class Client:
        def __init__(self):
            self.sent = []

        def send_force_feedback_for(self, side, tau):
            self.sent.append((side, np.asarray(tau).copy()))

        def close(self):
            pass

    client = Client()
    node._client = client
    registry = StreamRegistry(str(tmp_path), run_id)
    node._feedback_registry = registry
    writer = StreamWriter.create(
        StreamSpec("right/tau_ext", dim=7, capacity=8, dtype="float64"),
        run_id,
        registry,
    )
    try:
        tau = np.arange(7, dtype=np.float64) + 0.25
        writer.write(tau)
        node._forward_force_feedback("right")
        assert len(client.sent) == 1
        assert client.sent[0][0] == "right"
        # RDK reports environment-on-follower torque. FACTR's original feedback
        # equation expects follower-on-environment and negates that input itself,
        # so the transport must invert once for a same-direction net reflection.
        np.testing.assert_allclose(client.sent[0][1], -tau)

        node._forward_force_feedback("right")
        assert len(client.sent) == 1  # a sample is never replayed

        writer.write(tau + 1.0, time.monotonic_ns() - int(2e9))
        node._forward_force_feedback("right")
        assert len(client.sent) == 1  # stale contact is never forwarded
    finally:
        node.close_source()
        writer.close()
        writer.unlink()


def test_flexiv_state_mapping_exposes_external_joint_torque():
    from dual_flexiv_control.interfaces.flexiv.states import map_states

    tau_ext = np.linspace(-3.0, 3.0, 7)
    states = SimpleNamespace(
        q=np.zeros(7), dq=np.zeros(7), tau=np.ones(7), tau_ext=tau_ext,
        ext_wrench_in_tcp=np.zeros(6), ext_wrench_in_world=np.zeros(6),
        tcp_pose=np.zeros(7), tcp_vel=np.zeros(6),
    )
    mapped = map_states(states, "local")
    np.testing.assert_allclose(mapped["tau_ext"], tau_ext)
    assert not np.shares_memory(mapped["tau_ext"], mapped["tau"])


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
