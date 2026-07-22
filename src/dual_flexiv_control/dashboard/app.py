"""Streamlit page: experiment controls on the left, live Rerun viewer on the right.

Run via the ``dfc-dashboard`` console script (which wraps ``streamlit run`` — see
:mod:`~.launch`), not ``python app.py``.

Layout matches the spec: a left control column (task dropdown over ``conf/task``,
Collection / Eval launch buttons, run status) and a right area holding the
embedded Rerun web viewer where eval/collection metrics stream live.

Streamlit reruns this module top-to-bottom on every interaction, so the Rerun
servers and the run registry are created once behind ``st.cache_resource`` (one
instance per server process, shared across reruns and browser sessions). The
embedded viewer updates itself from the gRPC stream independently of these reruns.
"""

from __future__ import annotations

import os
import time

import streamlit as st
import streamlit.components.v1 as components

# Absolute imports: Streamlit executes this file as a top-level script (no package
# context), so relative imports would fail here. The package itself is installed,
# so its submodules resolve normally.
from dual_flexiv_control.dashboard import arms as _arms
from dual_flexiv_control.dashboard import blueprints
from dual_flexiv_control.dashboard import calibration as _calibration
from dual_flexiv_control.dashboard import cameras as _cameras
from dual_flexiv_control.dashboard import factr_servers as _factr_srv
from dual_flexiv_control.dashboard import logs as _logs
from dual_flexiv_control.dashboard import replay as _replay
from dual_flexiv_control.dashboard import robot_view as _robot
from dual_flexiv_control.dashboard import runner as _runner
from dual_flexiv_control.dashboard import skills as _skills
from dual_flexiv_control.dashboard import storage as _storage
from dual_flexiv_control.dashboard.arms import ArmStatus
from dual_flexiv_control.dashboard.arms import LeaderStatus
from dual_flexiv_control.dashboard.arms import configured_leader_sides
from dual_flexiv_control.dashboard.arms import discover_arms
from dual_flexiv_control.dashboard.arms import read_arm_status
from dual_flexiv_control.dashboard.arms import read_leader_status
from dual_flexiv_control.dashboard.cameras import CameraStatus
from dual_flexiv_control.dashboard.cameras import CameraView
from dual_flexiv_control.dashboard.cameras import discover_camera_views
from dual_flexiv_control.dashboard.cameras import get_frame
from dual_flexiv_control.dashboard.cameras import read_camera_statuses
from dual_flexiv_control.dashboard.editor import open_in_vscode
from dual_flexiv_control.dashboard.ssh_hosts import discover_ssh_hosts
from dual_flexiv_control.dashboard.tasks import RigInfo
from dual_flexiv_control.dashboard.tasks import TaskInfo
from dual_flexiv_control.dashboard.tasks import discover_policies
from dual_flexiv_control.dashboard.tasks import discover_rigs
from dual_flexiv_control.dashboard.tasks import discover_tasks
from dual_flexiv_control.dashboard.viewer import RerunServers
from dual_flexiv_control.dashboard.viewer import ports_from_env
from dual_flexiv_control.dashboard.viewer import start_servers
from dual_flexiv_control.dashboard.viewer import teardown as _teardown_servers

VIEWER_HEIGHT_PX = 1400
#: Camera-tab refresh cadence (live shm reads pace themselves; missing → error tile).
CAMERA_REFRESH = "0.15s"
#: Logs-tab tail cadence while "Follow" is on (a running system logs a beat every ~2s).
LOG_REFRESH = "2s"

#: Trim the default top padding, style the tab bar (its labels use monochrome
#: Material icons — forced white so the active tab's accent colour never tints
#: them), and enlarge the
#: per-episode action icons in the Storage tab (scoped via the
#: ``st-key-stor_rows`` container class).
_PAGE_CSS = """
<style>
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 1.5rem !important;
}
.stTabs [data-baseweb="tab-list"] button { padding: 1rem 2.2rem; }
.st-key-collection_launch button {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: white !important;
}
.st-key-collection_launch button:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
.st-key-eval_launch button,
.st-key-repeat_launch button {
    background-color: #16a34a !important;
    border-color: #16a34a !important;
    color: white !important;
}
.st-key-eval_launch button:hover,
.st-key-repeat_launch button:hover {
    background-color: #15803d !important;
    border-color: #15803d !important;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.35rem;
    font-weight: 600;
}
/* The markdown-rendered Material icons in the tab labels (span[role=img]):
   keep them white even on the active tab (whose text takes the accent color). */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] span[role="img"] {
    font-size: 1.5rem;
    color: #fafafa !important;
}
.st-key-stor_rows .stButton button p { font-size: 1.35rem; }
</style>
"""

#: Page background per session mode. VIEWING keeps the theme default (#0e1117,
#: hsv 220° 39% 9%); COLLECTION is the same hue one value step up (~15%); EVAL is
#: collection's value/saturation with the hue shifted purple (~280°); SKILL
#: (teach-and-repeat replay) shifts it green (~150°) instead.
_MODE_BG = {"collection": "#181d27", "eval": "#211627", "skill": "#16271d"}


def _apply_mode_background(view) -> None:
    """Tint the whole page by session mode so the active mode reads at a glance.

    ``saving`` keeps its run's tint (``view.phase`` stays set) so the color does
    not snap back to viewing while the episode finalizes. The header is made
    transparent so the tint runs edge to edge.

    ALWAYS emits exactly one markdown element (empty ``<style>`` when there is
    no tint): Streamlit identifies elements by their position in the tree, so a
    conditionally-present element here would shift everything below it on every
    viewing ↔ run transition — remounting the tabs and the embedded Rerun
    viewer iframe, i.e. a full viewer reload on every mode change.
    """
    color = _MODE_BG.get(view.phase if view.state == "saving" else view.state)
    css = (
        f".stApp {{ background-color: {color}; }} "
        '[data-testid="stHeader"] { background: transparent; }'
        if color
        else ""
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _browser_url(url: str) -> str:
    """Rewrite loopback viewer URLs to the host the browser used for this page.

    The Rerun handles report ``127.0.0.1`` (their own bind view), but the iframe
    is resolved by the *user's* browser — over Tailscale/LAN that loopback points
    at the user's machine and the viewers come up blank. All viewer ports bind
    ``0.0.0.0``, so the host serving Streamlit also serves them: reuse it. The
    replace also rewrites the percent-encoded ``?url=`` gRPC URI (host chars are
    not escaped by ``quote``), so the viewer fetches data from the right host too.
    """
    host = (st.context.headers.get("host") or "").split(":")[0]
    if host and host not in ("127.0.0.1", "localhost"):
        return url.replace("127.0.0.1", host)
    return url


@st.cache_resource
def _servers() -> RerunServers:
    """Reuse (or start) the Rerun servers; load the idle README on first start.

    The launcher usually pre-binds these (``dashboard.launch``); ``start_servers``
    is idempotent, so this returns the existing handle. When the app is run
    directly (``streamlit run app.py``) it starts them here instead. The robot scene
    is attached to this same recording so it fills the metrics viewer's 3D panel.
    """
    import rerun as rr

    grpc_port, web_port = ports_from_env()
    servers = start_servers(grpc_port=grpc_port, web_port=web_port)
    _robot.attach()  # log the robot scene into the metrics recording (shared 3D panel)
    rr.send_blueprint(blueprints.welcome_blueprint())
    _runner.log_welcome()
    return servers


@st.cache_resource
def _replay_viewer() -> _replay.ReplayViewer:
    """Start (once) the replay gRPC data server, embedded in the shared web viewer.

    Reuses the metrics web-viewer host (no second ``serve_web_viewer``), pointed at
    replay's own gRPC server so episode recordings stay isolated from live metrics.
    """
    return _replay.start_replay_viewer(web_port=_servers().web_port)


@st.cache_resource
def _calibration_viewer() -> _calibration.CalibViewer:
    """Start (once) the calibration gRPC recording, embedded in the shared web viewer.

    Its own recording (isolated from metrics/replay) so the Calibrate tab can pose an
    arm at the leader's commanded config without disturbing the live run scene.
    """
    return _calibration.start_calib_viewer(web_port=_servers().web_port)


@st.cache_resource
def _registry() -> _runner.RunRegistry:
    return _runner.RunRegistry()


def _reset_services(registry: _runner.RunRegistry) -> bool:
    """Restart the dashboard's live services: session daemon + viewers, from scratch.

    The dashboard equivalent of restarting the stack without killing the process:

    1. Refuse while a run is active — resetting would kill the in-flight episode;
       the operator stops the run first (returns False, surfaced as a warning).
    2. Shut the session daemon down gracefully (arms + cameras released) and stop
       the metrics mirror, so nothing logs into the recording while it is torn down.
    3. Drop the cached Hydra composes so arms + cameras re-read ``conf`` (and any
       changed ``runtime.sim``) on next use.
    4. Gracefully tear down the Rerun gRPC data servers (metrics + replay) and drop
       their recordings via :func:`~.viewer.teardown`, then reset the dependents that
       cached the now-dead recording (robot scene, replay viewer) and clear the
       ``st.cache_resource`` handles so they rebind.
    5. Re-serve the metrics gRPC server, re-attach the robot scene, and send the idle
       welcome layout; respawn the daemon (fresh robot connections) + the mirror.
       The web-viewer HTTP host is reused throughout (it cannot be rebound
       in-process); the replay server rebinds lazily on the next ▶.
    """
    if registry.session_view().run_active:
        return False
    registry.reset()
    registry.manager.shutdown()  # release the arms/cameras; respawned below
    _arms.reset()
    _cameras.reset()
    _storage.reset()
    _skills.reset()
    _teardown_servers()          # rerun_shutdown: releases metrics + replay + calib gRPC ports
    _robot.reset()               # forget the dead metrics recording
    _replay.reset()              # forget the (now released) replay server
    _calibration.reset()         # forget the (now released) calibration server + leader client
    _servers.clear()             # st.cache_resource: re-run start_servers on next call
    _replay_viewer.clear()
    _calibration_viewer.clear()
    _servers()                   # re-serve metrics gRPC + re-attach robot scene + welcome
    _runner.reset_viewer()       # idle welcome blueprint + README + reset event log
    # Fresh daemon on the (possibly re-read) rig/sim; ensure restarts the mirror too.
    registry.ensure_session(_arms.active_rig(), _arms.runtime_is_sim())
    return True


def _render_controls(
    tasks: list[TaskInfo], rig: RigInfo | None, registry: _runner.RunRegistry
) -> None:

    if not tasks:
        st.error(
            "No tasks found in `conf/task/`. Add one (copy `task/default.yaml`) "
            "and reload."
        )
        return
    if rig is None:
        st.error("No rigs found in `conf/rig/`. Add one (copy `rig/bimanual.yaml`).")
        return

    view = registry.session_view()

    # Two orthogonal axes: the rig (what hardware exists — conf/rig) and the task
    # (what is demonstrated/evaluated — conf/task). The rig is fixed at launch
    # (``dfc-dashboard --rig <name>``) — switching it means restarting the session
    # daemon, which is too destructive to offer as a live control.
    rig_cols = st.columns([4, 1], vertical_alignment="center")
    rig_cols[0].markdown(
        f"Rig: **`{rig.name}`**",
        help=(
            "Hardware setup from conf/rig — arms, cameras, FACTR leaders, serials. "
            "Fixed for this dashboard's lifetime; relaunch with "
            "`dfc-dashboard --rig <name>` (or the VSCode dashboard tasks) to switch."
        ),
    )
    if rig_cols[1].button(
        "✏️ Rig YAML", use_container_width=True,
        help=f"Open conf/rig/{rig.path.name} in VSCode on this machine.",
    ):
        result = open_in_vscode(rig.path)
        st.toast(result.message, icon="📝") if result.ok else st.warning(result.message)
    if rig.description:
        st.caption(rig.description)

    by_name = {t.name: t for t in tasks}
    task_cols = st.columns([4, 1], vertical_alignment="bottom")
    selected = task_cols[0].selectbox(
        "Task", list(by_name),
        help="Manipulation task from conf/task — instruction, dataset, episode counts.",
    )
    task = by_name[selected]
    if task_cols[1].button(
        "✏️ Task YAML", use_container_width=True,
        help=f"Open conf/task/{task.path.name} in VSCode on this machine.",
    ):
        result = open_in_vscode(task.path)
        st.toast(result.message, icon="📝") if result.ok else st.warning(result.message)

    # Launchable only from VIEWING with every rig camera streaming and every
    # arm publishing (the daemon refuses otherwise; disabling here just makes
    # that visible up front).
    launchable = (
        view.state == "viewing" and not view.cameras_down and not view.arms_down
    )
    # During a run the same buttons become atomic ⇄ switches: one click stops
    # the current run (its episode saves) and starts the new mode once idle.
    switching = view.run_active

    if st.button(
        "⇄ Collection" if switching else "▶ Collection [C]",
        key="collection_launch",
        use_container_width=True, disabled=not (launchable or switching),
        help=(
            "Teleoperated demonstration gathering (real recording run). While "
            "a run is active this switches to it: the current run stops (its "
            "episode saves), then collection starts — no manual retries."
        ),
    ):
        _launch(registry, task, "collection", rig.name, switch=switching)
    _collection_keybinds()

    # One compact row: labels collapsed (the column is narrow), meaning carried
    # by tooltips + the resolution caption underneath.
    eval_cols = st.columns([1.1, 1.1, 0.6, 0.85], vertical_alignment="center")
    policy_choice = eval_cols[0].selectbox(
        "Policy type", ["default", *discover_policies()], key="eval_policy_type",
        label_visibility="collapsed",
        help=(
            "Policy client/schema for this eval run (overrides the policy "
            "group, e.g. acme's multipart HTTP vs openpi's websocket). "
            "'default' uses the task's policy config."
        ),
    )
    policy = None if policy_choice == "default" else policy_choice
    by_alias = {h.alias: h.address for h in discover_ssh_hosts()}
    host_choice = eval_cols[1].selectbox(
        "Policy host", ["default", *by_alias], key="eval_policy_host",
        label_visibility="collapsed",
        help=(
            "Policy server for this eval run (overrides policy.host with the "
            "Host's real address). Options come from ~/.ssh/config; "
            "'default' uses the task's policy config."
        ),
    )
    host = by_alias.get(host_choice)
    port_raw = eval_cols[2].text_input(
        "Policy port", key="eval_policy_port", placeholder="port",
        label_visibility="collapsed",
        help=(
            "Policy server port for this eval run (overrides policy.port). "
            "Blank uses the task's policy config."
        ),
    )
    port, port_error = _parse_port(port_raw)
    if eval_cols[3].button(
        "⇄ Eval" if switching else "▶ Eval",
        key="eval_launch",
        type="primary", use_container_width=True,
        disabled=not (launchable or switching) or port_error is not None,
        help=(
            "Online policy rollout (needs the policy server reachable). While "
            "a run is active this switches to it: the current run stops (its "
            "episode saves), then eval starts — no manual retries."
        ),
    ):
        _launch(
            registry, task, "eval", rig.name,
            policy=policy, host=host, port=port, switch=switching,
        )
    if port_error:
        st.caption(f":red[{port_error}]")
    elif policy is not None or host is not None or port is not None:
        st.caption(
            f":gray[policy → {policy or 'config type'} @ "
            f"{host or 'config host'}:{port or 'config port'}]"
        )

    _reset_services_button(registry, key="reset_all_services", all_services=True)

    st.caption("Each launch runs a single episode.")

    if view.run_active:
        st.caption(
            "A run is active — ⇄ switches straight to a new run (the current "
            "episode saves first), or ■ Stop just ends it."
        )
    elif view.state == "starting":
        st.caption("Session starting — arms and cameras coming up…")
    elif view.state == "down":
        st.caption("Session daemon is not running — see the Logs tab or Reset services.")

    st.divider()
    _render_status(registry)


def _render_arm_row(s: ArmStatus) -> None:
    connected = s.source == "live"
    dot = "🟢" if connected else "⚫"
    operation = s.operational_status if connected else f":gray[{s.operational_status}]"
    if s.control_active:
        operation += " · :orange[controlling]"
    if s.estop_pressed:
        estop = ":red[🛑 **E-STOP PRESSED**]"
    elif s.estop_pressed is False:
        estop = ":green[clear]"
    else:
        estop = ":gray[—]"
    if s.servo_enabled is True:
        servo = ":green[on]"
    elif s.servo_enabled is False:
        servo = ":gray[off]"
    else:
        servo = ":gray[—]"
    mode = f"`{s.mode}`" if connected else ":gray[—]"
    st.markdown(
        f"{dot} **{s.info.name}** · {operation}  \n"
        f"E-stop: {estop} · Servo: {servo} · Mode: {mode}"
    )


@st.fragment(run_every="2s")
def _arm_status_rows(registry: _runner.RunRegistry) -> None:
    """Per-arm rows (operation mode + E-stop) with a ↻ reconnect button each."""
    try:
        arms = discover_arms()
    except Exception as exc:  # noqa: BLE001 - broken rig conf -> compact, visible note
        st.warning(f"Arm config failed to compose: {exc}", icon="🛠️")
        return
    run_active = registry.session_view().run_active
    for arm in arms:
        cols = st.columns([5, 1], vertical_alignment="center")
        with cols[0]:
            _render_arm_row(read_arm_status(arm))
        if cols[1].button(
            "↻", key=f"arm_reconnect::{arm.side}", disabled=run_active,
            help=f"Reconnect the {arm.side} arm now: replace its node for a "
                 "fresh RDK connection — back to IDLE telemetry, never "
                 "auto-controlled. Disabled while a run is active.",
        ):
            if registry.reconnect_arm(arm.side):
                st.toast(f"Reconnecting arm {arm.side}…", icon="🔁")
            else:
                st.warning("Session daemon not reachable.", icon="⚠️")


def _render_leader_row(s: LeaderStatus) -> None:
    dot = "🟢" if s.reachable else "⚫"
    if s.reachable:
        detail = ":orange[sim]" if s.sim else ":green[live]"
        if not s.sim:
            _, grav_detail = _arms.grav_comp_display(
                None if s.grav_comp_enabled is None else {
                    "grav_comp_enabled": s.grav_comp_enabled,
                    "force_gain": s.force_gain,
                    "force_gain_target": s.force_gain_target,
                }
            )
            detail += f" · {grav_detail}"
        detail += f" · {s.dof} joints"
        if s.gripper is not None:
            detail += f" · grip `{s.gripper:+.2f}`"
    else:
        detail = ":gray[no signal]"
    st.markdown(f"{dot} **{s.name}**  \n{detail}")


@st.fragment(run_every="2s")
def _leader_status_rows(registry: _runner.RunRegistry) -> None:
    """Per-leader live/grav-comp status with that leader's scrollable log."""
    try:
        sides = configured_leader_sides()
    except Exception as exc:  # noqa: BLE001 - broken factr conf -> compact, visible note
        st.warning(f"Leader config failed to compose: {exc}", icon="🛠️")
        return
    if not sides:
        st.caption(":gray[No teleop leaders configured.]")
        return
    info = registry.session_view().factr_servers or {}
    logs = info.get("logs") or {}
    # This fragment runs every 2 s. Convention discovery is deliberately a
    # non-blocking observer poll and cannot bring down the dashboard.
    _arms.discover_conventions()
    for side in sides:
        cols = st.columns([5, 1], vertical_alignment="center")
        with cols[0]:
            _render_leader_row(read_leader_status(side))
            poll = _arms.convention_poll_status(side)
            icon = {"live": "🟢", "stale": "🟠", "reset": "🔄"}.get(
                poll.get("state"), "⚪"
            )
            attempted = poll.get("attempted_at")
            age = f" · {max(0, int(time.time() - attempted))}s ago" if attempted else ""
            st.caption(f"{icon} calibration: {poll.get('message', 'polling')}{age}")
        with cols[1].popover("Logs", use_container_width=True):
            path = logs.get(f"teleop:{side}")
            tail = _factr_srv.tail_lines(path)[-100:] if path else []
            with st.container(height=320):
                if tail:
                    st.code("\n".join(tail), language="text")
                else:
                    st.caption(":gray[No log output available.]")


@st.fragment(run_every="2s")
def _render_factr_section(registry: _runner.RunRegistry) -> None:
    """The shared grav-comp controls; per-leader state is shown above."""
    st.markdown("##### Grav comp")
    view = registry.session_view()
    info = view.factr_servers
    if info is None:
        st.caption(
            ":gray[FACTR servers are not managed by this session (sim rig, or "
            "`factr.launch` disabled) — launch them externally if needed.]"
        )
        return
    state = info.get("state")

    if state in ("off", "countdown"):
        # The service auto-starts with the daemon; this is only seen briefly at
        # boot (or if it was stopped via the legacy start_factr/stop_factr path).
        st.caption(":gray[FACTR service starting…]")
    elif state == "stopping":
        st.caption(":gray[FACTR service stopping — the leaders are de-energizing…]")

    statuses = [
        _arms.read_leader_grav_comp_status(side)
        for side in configured_leader_sides()
    ]
    disable_is_action = any(
        status is not None and (
            status.get("grav_comp_enabled") is True
            or float(status.get("force_gain_target") or 0.0) >= 0.99
        )
        for status in statuses
    )
    states = [_arms.grav_comp_state(status) for status in statuses]
    if any(state in ("enabled", "enabling") for state in states):
        hotkey_action = "disable"
    elif any(state in ("disabling", "disabled") for state in states):
        hotkey_action = "enable"
    else:
        hotkey_action = "none"
    st.markdown(
        _grav_comp_button_css(disable_is_action)
        + f'<span data-factr-hotkey="{hotkey_action}" style="display:none"></span>',
        unsafe_allow_html=True,
    )

    grav_cols = st.columns(2)
    if grav_cols[0].button(
        "▶ Enable grav comp [F]", key="factr_enable", use_container_width=True,
        help=(
            "Ramp every leader's master output gain 0→1 over ~1s — the arms go "
            "from limp to gravity-compensated. The teleop processes stay up "
            "(they auto-start with the daemon)."
        ),
    ):
        if registry.enable_grav_comp():
            st.toast("Enabling grav comp — leaders energizing…", icon="🤖")
        else:
            st.warning("Session daemon not reachable.", icon="⚠️")
        st.rerun(scope="fragment")
    if grav_cols[1].button(
        "■ Disable grav comp (de-energize) [F]", key="factr_disable",
        use_container_width=True,
        help=(
            "Ramp every leader's master output gain 1→0 over ~1s — the arms go "
            "limp (energized but applying no torque). The processes stay up; "
            "use Enable to re-energize."
        ),
    ):
        if registry.disable_grav_comp():
            st.toast("Disabling grav comp — leaders de-energizing…", icon="🔻")
        else:
            st.warning("Session daemon not reachable.", icon="⚠️")
        st.rerun(scope="fragment")


def _grav_comp_button_css(disable_is_action: bool) -> str:
    """Solid orange marks the action opposite the leaders' current gain state."""
    solid = "factr_disable" if disable_is_action else "factr_enable"
    hollow = "factr_enable" if disable_is_action else "factr_disable"
    return f"""
    <style>
    .st-key-{solid} button {{
        background-color: #f97316 !important;
        border-color: #f97316 !important;
        color: white !important;
    }}
    .st-key-{solid} button:hover {{
        background-color: #ea580c !important;
        border-color: #ea580c !important;
    }}
    .st-key-{hollow} button {{
        background-color: transparent !important;
        border-color: #f97316 !important;
        color: #f97316 !important;
    }}
    .st-key-{hollow} button:hover {{
        background-color: rgba(249, 115, 22, 0.12) !important;
        border-color: #ea580c !important;
        color: #ea580c !important;
    }}
    </style>
    """
def _render_camera_row(s: CameraStatus) -> None:
    dot = "🟢" if s.detected else "⚫"
    if s.detected:
        detail = f":green[detected] · {', '.join(s.live_views)}"
    else:
        detail = ":gray[no signal]"
    st.markdown(f"{dot} **{s.camera}**  \n{detail}")


@st.fragment(run_every="2s")
def _camera_status_rows() -> None:
    """Per-camera detection rows (live shm streams), refreshed periodically."""
    try:
        statuses = read_camera_statuses()
    except Exception as exc:  # noqa: BLE001 - broken camera conf -> compact, visible note
        st.warning(f"Camera config failed to compose: {exc}", icon="🛠️")
        return
    if not statuses:
        st.caption(":gray[No cameras configured.]")
        return
    for s in statuses:
        _render_camera_row(s)


def _parse_port(raw: str) -> tuple[int | None, str | None]:
    """Parse the eval port box into ``(port, error)``.

    Blank means "use the task's policy config" — ``(None, None)``. A non-blank
    value must be a valid TCP port or the launch is blocked with the error.
    """
    raw = (raw or "").strip()
    if not raw:
        return None, None
    if not raw.isdigit() or not 0 < int(raw) < 65536:
        return None, f"port must be a number in 1–65535, got {raw!r}"
    return int(raw), None


def _launch(
    registry: _runner.RunRegistry, task: TaskInfo, phase: str, rig: str,
    policy: str | None = None, host: str | None = None, port: int | None = None,
    switch: bool = False,
) -> None:
    """Launch (or ⇄ switch to) a run, surfacing a refusal inline."""
    try:
        registry.launch(
            task, phase, rig, policy=policy, host=host, port=port, switch=switch
        )
    except RuntimeError as exc:
        st.error(str(exc), icon="⚠️")
        return
    st.rerun()


def _show_run_alert() -> None:
    """Surface a finished run once: a modal popup for errors, a toast for success.

    The status fragment stashes the registry's one-shot alert in session state and
    triggers a full rerun; this (main flow) pops and renders it. Popping at render
    time means the next interaction dismisses it naturally — and a dialog closed
    via ✕ does not reopen.
    """
    alert = st.session_state.pop("run_alert", None)
    if not alert:
        return
    if alert.get("kind") == "info":
        st.toast(alert["detail"], icon="✅")
        return

    @st.dialog("⚠️ Collection stopped abnormally", width="large")
    def _popup() -> None:
        st.error(alert["detail"])
        tail = alert.get("tail")
        if tail:
            st.caption("Last lines from the recording system's log:")
            st.code(tail, language="text")
        if st.button("OK", use_container_width=True):
            st.rerun()

    _popup()


@st.fragment(run_every="2s")
def _run_status_panel(registry: _runner.RunRegistry) -> None:
    """Live view of the active run: state, recording heartbeat, Stop button.

    A fragment so the heartbeat / saving-state refresh without user interaction.
    When the registry reports a run ended (its one-shot alert), stash it and
    rerun the whole app — that re-enables the launch buttons and pops the alert.
    """
    alert = registry.take_alert()
    if alert is not None:
        key = (
            "factr_control_alert"
            if alert.get("kind") == "error" and alert.get("phase") == "collection"
            else "run_alert"
        )
        st.session_state[key] = alert
        st.rerun(scope="app")
    view = registry.session_view()
    # The launch buttons/rig lock live OUTSIDE this fragment; when the session's
    # mode changes (starting→viewing, viewing→collection, …) rerun the whole app
    # so they follow without a user interaction.
    prev = st.session_state.get("_session_state_seen")
    if prev != view.state:
        st.session_state["_session_state_seen"] = view.state
        if prev is not None:
            st.rerun(scope="app")
    if view.message and _service_message_section(view.message) is None:
        st.warning(view.message, icon="⚠️")
    if view.pending:
        st.info(
            f"⇄ Switching to **{view.pending.get('phase') or '?'}** — the "
            "current run is stopping / the arms are winding down; the new run "
            "starts automatically.",
            icon="⏳",
        )
    active = registry.active()
    if active is None:
        if view.state == "viewing":
            st.info("👁 **Viewing** — arms live, read-only. Pick a task and launch.")
            if view.cameras_down:
                st.warning(
                    "Waiting on camera(s): **" + ", ".join(view.cameras_down) + "** — "
                    "retried automatically; launches are disabled until every rig "
                    "camera streams (replug/fix, or pick a rig without it).",
                    icon="📷",
                )
                for name in view.cameras_down:
                    if st.button(
                        f"↻ Retry {name} now", key=f"cam_retry::{name}",
                        help="Kill + respawn this camera node immediately (e.g. "
                             "right after a replug), instead of waiting out the "
                             "automatic retry pacing.",
                    ):
                        if registry.respawn_camera(name):
                            st.toast(f"Respawning {name}…", icon="🔁")
                        else:
                            st.warning("Session daemon not reachable.", icon="⚠️")
            if view.arms_down:
                st.warning(
                    "Waiting on arm(s): **" + ", ".join(view.arms_down) + "** — "
                    "reconnected automatically (back to read-only telemetry; "
                    "never auto-controlled); launches are disabled until the "
                    "arm publishes again.",
                    icon="🦾",
                )
        elif view.state == "starting":
            st.info("⏳ Session starting — arms and cameras coming up…")
        else:  # down (view.message above says why / whether it is retrying)
            st.error("Session daemon is not running.", icon="🛑")
            tail = registry.manager.log_tail()
            if tail:
                with st.expander("Session daemon log (tail)"):
                    st.code(tail, language="text")
        return
    st.success(f"**{active.phase.upper()}** · {active.task} · rig `{active.rig}`")
    st.caption(f"run `{active.run_id}` · started {active.started_wall}")
    stopping = active.status == "stopping"
    cs = registry.collection_status()
    if cs is not None:
        if cs.state == "running":
            st.caption(f"🔴 {cs.detail}")
        elif cs.state == "saving":
            st.warning(cs.detail, icon="💾")
        elif cs.state == "exited-ok":
            st.caption(f"✅ {cs.detail}")
        else:  # exited-error (transient — the popup carries the details)
            st.error(cs.detail, icon="⚠️")
            if cs.error_tail:
                with st.expander("Recording log (tail)"):
                    st.code(cs.error_tail, language="text")
    if st.button(
        (
            "💾 Saving episode…" if stopping else
            ("■ Stop [S]" if active.phase == "collection" else "■ Stop")
        ),
        key="collection_stop" if active.phase == "collection" else "run_stop",
        use_container_width=True,
        disabled=stopping,
        help="Stops the run; the in-progress episode is saved (video finalize can take a while).",
    ):
        registry.stop_active()
        st.rerun(scope="app")


def _collection_keybinds() -> None:
    """Bind C=start Collection and S=stop through the existing guarded buttons."""
    components.html(
        """
        <script>
        (() => {
          const host = window.parent;
          if (host.__dfcCollectionKeyHandler) {
            host.document.removeEventListener('keydown', host.__dfcCollectionKeyHandler);
          }
          const handler = (event) => {
            if (event.defaultPrevented || event.repeat || event.ctrlKey ||
                event.metaKey || event.altKey) return;
            const target = event.target;
            const tag = (target && target.tagName || '').toUpperCase();
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
                (target && target.isContentEditable)) return;

            const key = event.key.toLowerCase();
            let button = null;
            if (key === 'c') {
              // Start-only: never turn C into an implicit run switch.
              const stop = host.document.querySelector(
                '.st-key-collection_stop button, .st-key-run_stop button'
              );
              if (stop) return;
              button = host.document.querySelector('.st-key-collection_launch button');
            } else if (key === 's') {
              button = host.document.querySelector('.st-key-collection_stop button');
            } else if (key === 'f') {
              // The grav-comp section publishes the authoritative F action as
              // a hidden marker; "none" (state unknown) keeps F inert.
              const marker = host.document.querySelector('[data-factr-hotkey]');
              const action = marker && marker.getAttribute('data-factr-hotkey');
              if (action === 'disable') {
                button = host.document.querySelector('.st-key-factr_disable button');
              } else if (action === 'enable') {
                button = host.document.querySelector('.st-key-factr_enable button');
              }
            }
            if (!button || button.disabled) return;
            event.preventDefault();
            button.click();
          };
          host.__dfcCollectionKeyHandler = handler;
          host.document.addEventListener('keydown', handler);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _render_status(registry: _runner.RunRegistry) -> None:
    view = registry.session_view()
    with st.container(border=True):
        with st.container(
            horizontal=True, horizontal_alignment="left",
            vertical_alignment="center", gap="small",
        ):
            st.markdown("#### Follower Arms (Flexiv Rizon 4s)", width="content")
            _reset_services_button(registry, key="reset_follower_services")
        if view.message and _service_message_section(view.message) == "arms":
            st.info(view.message)
        _arm_status_rows(registry)
    with st.container(border=True, key="factr_status_section"):
        with st.container(
            horizontal=True, horizontal_alignment="left",
            vertical_alignment="center", gap="small",
        ):
            st.markdown("#### Leaders (FACTR)", width="content")
            _reset_services_button(registry, key="reset_factr_services")
        if view.message and _service_message_section(view.message) == "factr":
            st.info(view.message)
        alert = st.session_state.get("factr_control_alert")
        if alert:
            st.error(f"Collection control error: {alert['detail']}", icon="⚠️")
            if alert.get("tail"):
                with st.expander("Control log (tail)"):
                    st.code(alert["tail"], language="text")
            if st.button("Dismiss", key="dismiss_factr_control_alert"):
                st.session_state.pop("factr_control_alert", None)
                st.rerun()
        _leader_status_rows(registry)
        _render_factr_section(registry)
    with st.container(border=True):
        with st.container(
            horizontal=True, horizontal_alignment="left",
            vertical_alignment="center", gap="small",
        ):
            st.markdown("#### Cameras", width="content")
            _reset_services_button(registry, key="reset_camera_services")
        if view.message and _service_message_section(view.message) == "cameras":
            st.info(view.message)
        _camera_status_rows()
    st.divider()
    _run_status_panel(registry)

    history = registry.history()
    if history:
        with st.expander(f"History ({len(history)})"):
            for record in reversed(history):
                st.text(
                    f"{record.started_wall}  {record.phase:<10} "
                    f"{record.task:<14} {record.status}"
                )

def _service_message_section(message: str) -> str | None:
    """Route hardware-service messages to their owning status card."""
    text = message.lower()
    if "control error" in text or "dropped out of control" in text:
        return "factr"
    if "factr" in text or "grav comp" in text or "leader" in text:
        return "factr"
    if "camera" in text:
        return "cameras"
    if "arm" in text:
        return "arms"
    return None


def _reset_services_button(
    registry: _runner.RunRegistry, *, key: str, all_services: bool = False
) -> None:
    """Render a consistently safe reset control at the requested UI location."""
    if st.button(
        "Reset all services" if all_services else "↻",
        key=key,
        type="secondary",
        use_container_width=all_services,
        disabled=registry.session_view().run_active,
        help=(
            "Restart the session daemon (fresh arm/camera connections), reload "
            "config from conf, and return the viewers to idle. Stop any active "
            "run first."
        ),
    ):
        if _reset_services(registry):
            st.toast("Services reset — session restarted.", icon="🔄")
        else:
            st.warning("A run is active — stop it before resetting services.", icon="⚠️")
        st.rerun()


def _render_logs_tab(registry: _runner.RunRegistry) -> None:
    """Browse the flexiv control system's logs: the live session daemon and past runs.

    Dashboard-launched runs all share one log for the daemon's lifetime (it never
    runs a fresh Hydra job — see :func:`~.logs.live_daemon_log`); that entry is
    listed first (and selected by default) since it is the one still growing. Past
    runs launched straight from the CLI each wrote their own ``system.log`` under
    Hydra's ``outputs/<timestamp>/`` and are listed newest-first after it. Follow
    tails the selected one live; the picker and controls are rendered here, only
    :func:`_log_tail_view` reruns on the follow tick.
    """
    live = _logs.live_daemon_log(registry.session_view().log_path)
    # The live daemon's session.log now lives under the outputs root, so
    # discover_logs() also finds it — drop that copy so it appears once (as 🔴).
    past = [f for f in _logs.discover_logs() if live is None or f.path != live.path]
    files = ([live] if live is not None else []) + past
    top = st.columns([3, 1])
    if top[1].button("🔄 Refresh", use_container_width=True,
                     help="Re-scan for the live daemon log and past run logs."):
        st.rerun()
    if not files:
        st.info(
            f"No logs yet. Launch a run from the dashboard, or run "
            f"`dual-flexiv-control` from the CLI (writes under `{_logs.outputs_root()}`)."
        )
        return

    def _label(f: _logs.LogFile) -> str:
        prefix = "🔴 " if f is live else ""
        return f"{prefix}{f.name}  ·  {_logs.human_size(f.size_bytes)}"

    by_name = {_label(f): f for f in files}
    label = top[0].selectbox(
        "Run log", list(by_name), key="log_sel",
        help="🔴 = the live session daemon log (all dashboard runs); others are "
             "past dual-flexiv-control CLI runs (Hydra outputs/<timestamp>/system.log).",
    )
    selected = by_name[label]
    st.caption(f"`{selected.path}`")
    follow = st.checkbox(
        "Follow (auto-refresh tail)", value=True, key="log_follow",
        help="Re-read the tail every 2s so a running system's log streams live.",
    )
    _log_tail_view(selected.path, follow)


def _render_log_code(path) -> None:
    # Fixed-height container -> the full tail scrolls inside a capped box.
    with st.container(height=600):
        st.code(_logs.read_tail(path), language="text")


@st.fragment(run_every=LOG_REFRESH)
def _log_tail_view_following(path) -> None:
    _render_log_code(path)


def _log_tail_view(path, follow: bool) -> None:
    """Render the log tail, auto-refreshing only when ``follow`` is on."""
    if follow:
        _log_tail_view_following(path)
    else:
        _render_log_code(path)


def _render_camera_tab(views: list[CameraView]) -> None:
    if not views:
        st.info("No cameras configured (none found in `conf/camera`).")
        return
    by_key = {v.key: v for v in views}
    st.selectbox(
        "Camera", list(by_key), key="camera_key",
        help="Live view from the selected camera stream (cam/<camera>/<view>).",
    )
    _camera_feed(by_key)


@st.fragment(run_every=CAMERA_REFRESH)
def _camera_feed(by_key: dict[str, CameraView]) -> None:
    """Auto-refreshing image for the selected camera (only this fragment reruns).

    A camera that is not producing reads as **missing** (nothing is fabricated), so
    it is surfaced as an error rather than an animated placeholder.
    """
    key = st.session_state.get("camera_key") or next(iter(by_key))
    view = by_key.get(key)
    if view is None:
        return
    frame, source = get_frame(view)
    if source != "live" or frame is None:
        st.error(
            f"No live frame for `{view.key}` — this camera is not producing. "
            "The session streams cameras continuously; check the camera / ZED "
            "connection (or whether the session is up).",
            icon="⚠️",
        )
        st.caption(f"`{view.key}` · {view.width}×{view.height} · ⚫ no signal")
        return
    st.image(frame, width="stretch")
    st.caption(f"`{view.key}` · {view.width}×{view.height} · 🟢 live")


@st.fragment(run_every="0.5s")
def _depth_overlay_feed(camera: str) -> None:
    """Push the latest RGB-D cloud into the robot scene, whenever one is live.

    Reads the camera's ``left`` + ``depth`` shm streams, back-projects to a
    coloured point cloud (:func:`~.cameras.depth_point_cloud`), poses it with
    the camera extrinsics from ``conf/camera`` (:func:`~.robot_view.camera_world_pose`),
    and logs it into the robot recording. There is no on/off control here —
    show/hide the cloud via the entity's visibility in the Rerun sidebar.
    """
    cloud = _cameras.depth_point_cloud(camera)
    if cloud is None:
        return
    pts_cam, colors = cloud
    rot, t = _robot.camera_world_pose(_cameras.camera_cfg(camera))
    _robot.log_depth_points(pts_cam @ rot.T + t, colors)


def _render_skill_bar(registry: _runner.RunRegistry) -> None:
    """Teach-and-repeat bar: pick a taught skill and ▶ Repeat it on the live arms.

    Repeat launches a ``skill`` run through the session daemon — the same
    lifecycle as Collection/Eval (arms get the skill coefficients, the skill
    consumer streams the trajectory, Stop lives in the status panel), so all the
    usual guardrails apply. A skill run needs no cameras, so the only gate is an
    idle (viewing) session. Skills are taught from the **Storage** tab: each
    episode row has a 🎓 Teach button beside ▶ replay.
    """
    view = registry.session_view()
    infos = _skills.discover()
    if not infos:
        st.caption(
            "🎓 Teach & repeat: no skills yet — hit **🎓 Teach** on any episode "
            "row in the **Storage** tab; it becomes repeatable here."
        )
        return
    by_name = {s.name: s for s in infos}
    # A completed rename re-selects the new name; the widget's state can only be
    # written BEFORE the selectbox is instantiated, so it is staged on rerun.
    staged = st.session_state.pop("_skill_sel_next", None)
    if staged in by_name:
        st.session_state["skill_sel"] = staged
    cols = st.columns([4, 1.2, 0.5, 0.5], vertical_alignment="center")
    selected = cols[0].selectbox(
        "Skill", list(by_name), key="skill_sel", label_visibility="collapsed",
        help="Taught skills (files under the skills root, `skill.root`). Teach "
             "new ones from a replayed episode in the Storage tab.",
    )
    info = by_name[selected]
    launchable = view.state == "viewing"
    if cols[1].button(
        "▶ Repeat", key="repeat_launch", type="primary",
        use_container_width=True, disabled=not launchable,
        help="Retrace this skill on the live arm(s): move to its start pose, then "
             "replay the taught joint trajectory. Stop from the status panel.",
    ):
        try:
            registry.launch_skill(info.name, _arms.active_rig() or "")
        except RuntimeError as exc:
            st.error(str(exc), icon="⚠️")
        else:
            st.rerun()
    if cols[2].button(
        "✏️", key="skill_ren", use_container_width=True, help="Rename this skill."
    ):
        st.session_state["skill_rename_pending"] = info.name
        st.session_state.pop("skill_delete_pending", None)
    if cols[3].button(
        "🗑️", key="skill_del", use_container_width=True, help="Delete this skill."
    ):
        st.session_state["skill_delete_pending"] = info.name
        st.session_state.pop("skill_rename_pending", None)

    src = info.source or {}
    origin = ""
    if src.get("repo_id") and src.get("episode_index") is not None:
        origin = f" · from `{src['repo_id']}` #{src['episode_index']}"
    st.caption(
        f"🎓 `{info.name}` · {info.frames} frames · {info.duration_s:.1f}s @ "
        f"{info.fps:g} fps · arms: {', '.join(info.sides)}{origin}"
        + (f" · taught {info.created}" if info.created else "")
    )

    renaming = st.session_state.get("skill_rename_pending")
    if renaming and renaming in by_name:
        # Per-skill widget key: reopening the row for a different skill never
        # inherits a previous rename's half-typed text.
        rc = st.columns([1.2, 3, 1, 1], vertical_alignment="center")
        rc[0].markdown(f"Rename `{renaming}` to:")
        new_name = rc[1].text_input(
            "New skill name", value=renaming, key=f"skill_rename_to::{renaming}",
            label_visibility="collapsed",
            help="New file stem (letters, digits, '.', '_', '-'). Refuses a name "
                 "that already exists.",
        )
        if rc[2].button("✓ Rename", type="primary", use_container_width=True):
            try:
                _skills.rename(renaming, (new_name or "").strip())
            except Exception as exc:  # noqa: BLE001 - report, don't crash the page
                st.error(f"Rename failed: {exc}")
            else:
                st.session_state.pop("skill_rename_pending", None)
                st.session_state["_skill_sel_next"] = (new_name or "").strip()
                st.toast(f"Renamed `{renaming}` → `{(new_name or '').strip()}`.", icon="✏️")
                st.rerun()
        if rc[3].button("Cancel", use_container_width=True, key="skill_ren_cancel"):
            st.session_state.pop("skill_rename_pending", None)
            st.rerun()
    elif renaming:  # the pending skill vanished (renamed/deleted elsewhere)
        st.session_state.pop("skill_rename_pending", None)

    pending = st.session_state.get("skill_delete_pending")
    if pending and pending in by_name:
        st.error(f"Delete skill `{pending}`? This cannot be undone.")
        cc = st.columns([1, 1, 3])
        if cc[0].button("✓ Confirm delete", type="primary", use_container_width=True):
            _skills.delete(pending)
            st.session_state.pop("skill_delete_pending", None)
            st.toast(f"Deleted skill `{pending}`.", icon="🗑️")
            st.rerun()
        if cc[1].button("Cancel", use_container_width=True):
            st.session_state.pop("skill_delete_pending", None)
            st.rerun()
    elif pending:  # the pending skill vanished (deleted elsewhere) — drop the gate
        st.session_state.pop("skill_delete_pending", None)


@st.fragment(run_every="2s")
def _robot_data_status() -> None:
    """Metrics-tab banner: warn about arms with no live measured ``q``.

    Such arms are drawn red and held still in the 3D robot scene (see
    :func:`~.robot_view.update_poses`); this is the accompanying text message.
    """
    from dual_flexiv_control.dashboard.arms import read_live_joint_positions

    missing = [a.name for a in discover_arms() if read_live_joint_positions(a.side) is None]
    if missing:
        st.warning(
            "No live joint data for **" + ", ".join(missing) + "** — shown in red and "
            "held still (control boxes off, or the system isn't publishing `<side>/q`).",
            icon="⚠️",
        )
    else:
        st.caption("🟢 Live joint data streaming for all arms.")


def _sel_key(repo_id: str, index: int) -> str:
    return f"stor_sel::{repo_id}::{index}"


def _clear_selection(repo_id: str) -> None:
    prefix = f"stor_sel::{repo_id}::"
    for key in [k for k in st.session_state if k.startswith(prefix)]:
        del st.session_state[key]


def _selected_indices(repo_id: str, episodes) -> list[int]:
    return [e.index for e in episodes if st.session_state.get(_sel_key(repo_id, e.index))]


def _do_delete(ds, indices: list[int] | None) -> None:
    """Delete episodes (or the whole dataset if ``indices`` is None), then rerun."""
    try:
        if indices is None:
            _storage.delete_dataset(ds)
            msg = f"Removed dataset `{ds.repo_id}`."
        else:
            result = _storage.delete_episodes(ds, indices)
            msg = (
                f"Removed dataset `{ds.repo_id}`."
                if result == "dataset-removed"
                else f"Deleted {len(indices)} episode(s) from `{ds.repo_id}`; re-indexed."
            )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the page
        st.error(f"Delete failed: {exc}")
        return
    _clear_selection(ds.repo_id)
    st.session_state.pop("stor_pending", None)
    # Deleting from THIS dataset re-indexes its episodes, so an open replay of it now
    # points at a different (or missing) episode — drop it so it reloads on the next ▶.
    # A replay of a *different* dataset is unaffected, so leave it alone.
    open_replay = st.session_state.get("replay_target")
    if open_replay and open_replay[0] == ds.repo_id:
        st.session_state.pop("replay_target", None)
        st.session_state.pop("replay_logged", None)
    st.toast(msg, icon="🗑️")
    st.rerun()


def _render_delete_dataset_button(ds) -> None:
    """A 'delete entire dataset' control (used for empty/corrupt datasets)."""
    if st.button(
        "🗑️ Delete entire dataset", type="primary", use_container_width=True,
        key=f"stor_delds::{ds.repo_id}",
    ):
        st.session_state["stor_pending"] = {"repo_id": ds.repo_id, "whole": True}
        st.rerun()


def _calib_offsets(side: str) -> list[float]:
    """The in-progress offsets for ``side``, seeded from config on first use."""
    key = f"calib_offsets::{side}"
    if key not in st.session_state:
        st.session_state[key] = _calibration.initial_offsets(side)
    return st.session_state[key]


def _calib_flips(side: str) -> set[int]:
    """The in-progress sign-flip joint set for ``side``, seeded from config on first use."""
    key = f"calib_flips::{side}"
    if key not in st.session_state:
        st.session_state[key] = set(_calibration.current_convention(side).sign_flip_joints)
    return st.session_state[key]


def _calib_gripper(side: str):
    """(open, closed) raw gripper endpoints for ``side``, seeded from config; None until set."""
    conv = _calibration.current_convention(side)
    ok, ck = f"calib_grip_open::{side}", f"calib_grip_closed::{side}"
    if ok not in st.session_state:
        st.session_state[ok] = conv.gripper_open
    if ck not in st.session_state:
        st.session_state[ck] = conv.gripper_closed
    return st.session_state[ok], st.session_state[ck]


@st.fragment(run_every="0.3s")
def _calibration_view_feed(side: str) -> None:
    """Pose the live view at the leader's commanded config (in-progress convention).

    Only this fragment reruns on its cadence — it reads the leader and logs one arm
    frame into the calibration recording (using the working offsets **and** sign
    flips, so a flip visibly mirrors that link); the embedded viewer picks it up over
    gRPC. On a read failure it says so rather than fabricating motion.
    """
    offsets = st.session_state.get(f"calib_offsets::{side}")
    if offsets is None:
        return
    flips = sorted(st.session_state.get(f"calib_flips::{side}", set()))
    try:
        _calibration.render(side, offsets, flips)
        st.caption(f"🟢 live: `{side}` leader → follower (solid) vs straight/home (ghost)")
    except Exception as exc:  # noqa: BLE001 - leader down -> report, hold the last pose
        st.caption(f"⚪ no live `{side}` leader — {exc}")


def _render_calibration_tab() -> None:
    """Measure a leader's joint convention **one joint at a time**, with a live view.

    Per joint: straighten the link and **📸 Capture** to solve its ``offsets_deg``
    (``offset = wrap(-degrees(leader_joint))``), and **↔ Flip** to toggle its sign in
    ``sign_flip_joints`` — the 3D view mirrors the link live so you can confirm the
    direction. The view poses a Flexiv arm at the config the current leader maps to
    under the working convention (solid) against the straight/home target (ghost), so
    each link visibly snaps into alignment as it is calibrated. **💾 Sync to file**
    writes the result into the rig YAML; hit **Reset services** to apply it.
    """
    sides = _calibration.configured_leader_sides()
    if not sides:
        st.info("No FACTR leader is configured for the active rig — nothing to calibrate.")
        return
    if _arms.runtime_is_sim():
        st.warning(
            "`runtime.sim` is on — leader positions are a synthetic sinusoid, so "
            "captured offsets are meaningless (the view still animates so you can "
            "confirm the plumbing). Calibrate against real hardware.",
            icon="⚠️",
        )

    side = sides[0] if len(sides) == 1 else st.selectbox(
        "Leader to calibrate", sides, format_func=str.capitalize, key="calib_side"
    )
    offsets = _calib_offsets(side)
    flips = _calib_flips(side)
    grip_open, grip_closed = _calib_gripper(side)
    done = st.session_state.setdefault(f"calib_done::{side}", {})  # joint -> captured_deg

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.caption(
            f"Straighten one link at a time: **Capture** its offset, **Flip** its sign "
            f"if the view mirrors the wrong way. **{len(done)}/{len(offsets)}** joints captured."
        )
        for j in range(len(offsets)):
            c0, c1, c2 = st.columns([3, 1, 1])
            captured = j in done
            mark = "✅" if captured else "⬜"
            flipped = j in flips
            c0.markdown(
                f"{mark} **J{j}** · offset `{offsets[j]:+.2f}°`"
                + (" · ↔" if flipped else "")
                + (f" · straight@`{done[j]:+.2f}°`" if captured else "")
            )
            if c1.button("📸", key=f"calib_cap::{side}::{j}", help=f"Capture J{j} offset",
                         use_container_width=True):
                try:
                    cap = _calibration.capture_joint(side, j)
                    offsets[j] = cap.offset_deg
                    done[j] = cap.captured_deg
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
                    st.error(f"Could not read the {side} leader: {exc}", icon="🛑")
            if c2.button("↔", key=f"calib_flip::{side}::{j}",
                         type="primary" if flipped else "secondary",
                         help=f"Toggle sign flip for J{j}", use_container_width=True):
                flips.discard(j) if flipped else flips.add(j)
                st.rerun()

        st.divider()
        st.markdown("**Gripper** — record the raw trigger reading at each extreme:")
        g0, g1 = st.columns(2)
        if g0.button("📗 Record open", key=f"calib_gopen::{side}", use_container_width=True,
                     help="Capture the raw gripper value at the FULLY OPEN trigger (maps to 0.0)."):
            try:
                st.session_state[f"calib_grip_open::{side}"] = _calibration.read_gripper(side)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
                st.error(f"Could not read the {side} gripper: {exc}", icon="🛑")
        if g1.button("📕 Record closed", key=f"calib_gclosed::{side}", use_container_width=True,
                     help="Capture the raw gripper value at the FULLY CLOSED trigger (maps to 1.0)."):
            try:
                st.session_state[f"calib_grip_closed::{side}"] = _calibration.read_gripper(side)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surface, don't crash the page
                st.error(f"Could not read the {side} gripper: {exc}", icon="🛑")
        o_txt = f"`{grip_open:+.4f}`" if grip_open is not None else "—"
        c_txt = f"`{grip_closed:+.4f}`" if grip_closed is not None else "—"
        st.caption(f"open {o_txt} · closed {c_txt} rad (raw servo)")
        preview = _calibration.gripper_preview(grip_open, grip_closed)
        if preview:
            st.caption("normalizes: " + " · ".join(f"{lab}→{frac:.2f}" for lab, _raw, frac in preview))
        elif grip_open is not None and grip_closed is not None:
            st.caption("⚠️ open and closed are equal — move the trigger between captures.")

        st.divider()
        if st.button("↺ Reset to config", key=f"calib_reset::{side}"):
            conv = _calibration.current_convention(side)
            st.session_state[f"calib_offsets::{side}"] = _calibration.initial_offsets(side)
            st.session_state[f"calib_flips::{side}"] = set(conv.sign_flip_joints)
            st.session_state[f"calib_grip_open::{side}"] = conv.gripper_open
            st.session_state[f"calib_grip_closed::{side}"] = conv.gripper_closed
            st.session_state[f"calib_done::{side}"] = {}
            st.rerun()

        flips_sorted = sorted(flips)
        st.info("Calibration is leader-owned. Copy the values below into the FACTR arm YAML and restart FACTR.")
        with st.expander("Preview / copy the config"):
            st.code(_calibration.format_yaml(side, offsets, flips_sorted, grip_open, grip_closed),
                    language="yaml")

    with right:
        # OFF by default: st.tabs renders every tab body on each rerun and run_every
        # fragments fire on their own timer, so an always-on feed would poll the FACTR
        # leader (~3 Hz) and bind the viewer's gRPC port even when nobody is on this
        # tab — contending with a live collection/eval run. Gate it behind opt-in so
        # the leader is touched only while actively calibrating.
        live = st.checkbox(
            "🔴 Live 3D view",
            value=False,
            key=f"calib_live::{side}",
            help="Pose a Flexiv arm at the leader's commanded config, refreshing ~3 Hz. "
            "Off by default so it doesn't poll the leader while you're not calibrating.",
        )
        if live:
            viewer = _calibration_viewer()
            st.iframe(_browser_url(viewer.web_url), height=VIEWER_HEIGHT_PX)
            _calibration_view_feed(side)
        else:
            st.info(
                "Enable **Live 3D view** to pose the arm at the leader's commanded "
                "config and watch each link snap onto the straight/home ghost as you "
                "calibrate. Capture/Flip/Sync work without it."
            )


def _render_storage_tab(registry: _runner.RunRegistry) -> None:
    root = _storage.collection_root()
    if st.button("🔄 Refresh", help="Re-scan the collection root for datasets."):
        _storage.reset()
        st.rerun()

    datasets = _storage.discover_datasets(root)
    if not datasets:
        st.info(f"No datasets recorded yet under `{root}`.")
        return

    by_id = {d.repo_id: d for d in datasets}
    sel = st.selectbox("Dataset", list(by_id), key="storage_ds")
    ds = by_id[sel]
    st.caption(
        f"`{ds.path}` · {ds.num_episodes} episodes · {ds.num_frames} frames · "
        f"{_storage.human_size(ds.size_bytes)}"
    )

    # A collection run may be appending to a dataset right now; editing it then
    # would corrupt the in-flight write. Block deletion while a run is active.
    active = registry.active()
    locked = active is not None and active.phase == "collection"
    if locked:
        st.warning(
            "A collection run is active — stop it before deleting episodes "
            "(deleting mid-write would corrupt the dataset).",
            icon="⚠️",
        )

    # -- confirmation gate (destructive) --------------------------------------
    pending = st.session_state.get("stor_pending")
    if pending and pending.get("repo_id") == ds.repo_id:
        whole = pending.get("whole", False)
        idxs = pending.get("indices")
        if whole or (idxs is not None and len(idxs) >= ds.num_episodes):
            target = "the **entire dataset**"
        else:
            target = f"**{len(idxs)} episode(s)**: {idxs}"
        st.error(f"Delete {target}? This cannot be undone.")
        cc = st.columns(2)
        if cc[0].button("✓ Confirm delete", type="primary", use_container_width=True):
            _do_delete(ds, None if whole else idxs)
        if cc[1].button("Cancel", use_container_width=True):
            st.session_state.pop("stor_pending", None)
            st.rerun()

    # An empty (nothing committed) or unreadable (corrupt) dataset can't be listed
    # per-episode — offer a whole-dataset delete so it can be cleaned up.
    if ds.num_episodes <= 0:
        st.info("This dataset has no committed episodes (an empty or interrupted run).")
        if not locked:
            _render_delete_dataset_button(ds)
        return
    try:
        episodes = _storage.list_episodes(ds)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read episodes: {exc}")
        if not locked:
            _render_delete_dataset_button(ds)
        return

    selected = _selected_indices(ds.repo_id, episodes)

    # -- bulk action bar ------------------------------------------------------
    bar = st.columns([1, 1, 2])
    if bar[0].button("Select all", use_container_width=True, disabled=locked):
        for e in episodes:
            st.session_state[_sel_key(ds.repo_id, e.index)] = True
        st.rerun()
    if bar[1].button("Clear", use_container_width=True, disabled=locked or not selected):
        _clear_selection(ds.repo_id)
        st.rerun()
    if bar[2].button(
        f"🗑️ Delete selected ({len(selected)})",
        type="primary",
        use_container_width=True,
        disabled=locked or not selected,
    ):
        st.session_state["stor_pending"] = {"repo_id": ds.repo_id, "indices": selected}
        st.rerun()

    st.divider()

    # -- per-episode rows: checkbox · info · replay · delete ------------------
    header = st.columns([0.6, 5, 0.8, 0.8, 0.8])
    header[0].caption("Sel")
    header[1].caption("Episode")
    header[2].caption("Replay")
    header[3].caption("Teach")
    header[4].caption("Delete")
    # key -> a stable `st-key-stor_rows` class, the hook _PAGE_CSS uses to
    # enlarge these rows' icon buttons without touching buttons elsewhere.
    with st.container(height=420, key="stor_rows"):
        for e in episodes:
            row = st.columns([0.6, 5, 0.8, 0.8, 0.8])
            row[0].checkbox(
                "select",
                key=_sel_key(ds.repo_id, e.index),
                label_visibility="collapsed",
                disabled=locked,
            )
            task = (e.tasks[0] if e.tasks else "—")
            row[1].markdown(
                f"**#{e.index}** · {e.length} frames · {e.duration_s:.1f}s  \n"
                f":gray[{task}]"
            )
            if row[2].button("▶", key=f"stor_play::{ds.repo_id}::{e.index}",
                             help="Replay this episode (3D arms + cameras + plots)."):
                st.session_state["replay_target"] = (ds.repo_id, e.index)
                st.rerun()
            if row[3].button("🎓", key=f"stor_teach::{ds.repo_id}::{e.index}",
                             help="Teach: save this episode's measured trajectory as "
                                  "a skill — repeat it from the Viewer tab's ▶ Repeat."):
                _teach_episode(ds, e.index)
            if row[4].button("🗑️", key=f"stor_del::{ds.repo_id}::{e.index}", disabled=locked):
                st.session_state["stor_pending"] = {
                    "repo_id": ds.repo_id, "indices": [e.index]
                }
                st.rerun()

    _render_replay_panel(ds)


def _render_replay_panel(ds) -> None:
    """Embedded replay viewer for the episode picked via a row's ▶ button.

    Logs the episode into the dedicated replay recording only when the target
    changes (Streamlit reruns this on every interaction), then embeds the viewer.
    """
    target = st.session_state.get("replay_target")
    if not target or target[0] != ds.repo_id:
        return
    repo_id, index = target
    if index >= ds.num_episodes:                     # dataset shrank (deletions) — stale target
        st.session_state.pop("replay_target", None)
        st.session_state.pop("replay_logged", None)
        return

    st.divider()
    head = st.columns([5, 1])
    head[0].markdown(f"### ▶ Replaying episode **#{index}** of `{repo_id}`")
    if head[1].button("✕ Close", use_container_width=True):
        st.session_state.pop("replay_target", None)
        st.session_state.pop("replay_logged", None)
        st.rerun()

    viewer = _replay_viewer()
    if st.session_state.get("replay_logged") != target:
        with st.spinner(f"Loading episode #{index} (decoding video + posing arms)…"):
            try:
                n = _replay.log_episode(ds, index)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the page
                st.error(f"Replay failed: {exc}")
                return
        st.session_state["replay_logged"] = target
        st.toast(f"Loaded {n} frames — press play or scrub the timeline.", icon="▶")

    st.caption(
        "Solid arms = recorded joint state · translucent ghost = action target · "
        "press play (▶) or scrub the timeline in the viewer."
    )
    st.iframe(_browser_url(viewer.web_url), height=VIEWER_HEIGHT_PX)


def _teach_episode(ds, index: int) -> None:
    """🎓 Teach: save one episode's measured trajectory as a skill, one click.

    Named ``<dataset>-ep<index>`` (re-teaching the same episode overwrites its
    own skill); the skill lands under the skills root (``skill.root``) and shows
    up in the Viewer tab's teach-and-repeat bar, where ▶ Repeat retraces it on
    the live arms. Reads only the episode's proprio columns (no video decode),
    so this is quick even for long episodes.
    """
    name = _skills.default_name(ds.repo_id, index)
    with st.spinner(f"Teaching `{name}` from episode #{index}…"):
        try:
            info = _skills.teach_from_episode(ds, index, name)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the page
            st.error(f"Teach failed: {exc}")
            return
    st.toast(
        f"Taught skill `{info.name}` — {info.frames} frames · "
        f"{info.duration_s:.1f}s · {', '.join(info.sides)}. "
        "Repeat it from the Viewer tab.",
        icon="🎓",
    )


def _resolve_rig(rigs: list[RigInfo]) -> RigInfo | None:
    """Pin every dashboard compose (arms/cameras/storage) to the launch rig.

    The rig is a launch option (``dfc-dashboard --rig <name>``, carried in the
    ``DFC_DASHBOARD_RIG`` env var) — it never changes while the dashboard runs,
    so switching rigs means relaunching (rig changes restart the session daemon,
    which is too destructive for a live control). With no option set, prefers
    ``bimanual``, else the first rig. An unknown name (the launcher validates,
    but the env can be set directly) surfaces a banner and falls back.
    """
    if not rigs:
        return None
    names = [r.name for r in rigs]
    wanted = os.environ.get("DFC_DASHBOARD_RIG", "").strip()
    if wanted and wanted not in names:
        st.error(
            f"Unknown rig `{wanted}` (from `DFC_DASHBOARD_RIG` / `--rig`) — "
            f"expected one of: {', '.join(names)}. Falling back to the default.",
            icon="🛠️",
        )
        wanted = ""
    name = wanted or ("bimanual" if "bimanual" in names else names[0])
    # set_active_rig drops the compose cache, so only pin when it actually changes
    # (this runs on every Streamlit rerun).
    if name != _arms.active_rig():
        _arms.set_active_rig(name)
        _cameras.reset()
        _storage.reset()
    return next(r for r in rigs if r.name == name)


def main() -> None:
    st.set_page_config(
        page_title="dual-flexiv experiments", page_icon="🤖", layout="wide"
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    servers = _servers()
    registry = _registry()
    _show_run_alert()
    tasks = discover_tasks()
    # Pin the launch rig BEFORE the first session spawn, so the daemon starts on
    # the rig the whole dashboard composes against.
    rig = _resolve_rig(discover_rigs())
    # The session daemon runs for the dashboard's lifespan: spawn it now (no-op when
    # already matching), restart it when the rig or runtime.sim changed.
    try:
        registry.ensure_session(_arms.active_rig(), _arms.runtime_is_sim())
    except Exception as exc:  # noqa: BLE001 - a broken conf must not kill the page
        st.error(f"Session daemon failed to start: {exc}", icon="🛑")
    _apply_mode_background(registry.session_view())
    # A rig YAML edit can break composition (e.g. a camera left in `defaults` but
    # removed from the inline block -> MISSING placement). That must surface as a
    # banner the operator can act on — never a dead page with the error in a log.
    try:
        cameras = discover_camera_views()
        compose_error = None
    except Exception as exc:  # noqa: BLE001 - broken conf must stay visible + recoverable
        cameras = []
        rig_label = _arms.active_rig() or "(default)"
        compose_error = (
            f"Config for rig `{rig_label}` failed to compose — fix its YAML in "
            f"`conf/rig/` (or relaunch with another rig) and reload.\n\n```\n{exc}\n```"
        )
    if compose_error:
        st.error(compose_error, icon="🛠️")

    controls, panel = st.columns([1, 3], gap="large")
    with controls:
        _render_controls(tasks, rig, registry)
    with panel:
        tab_viewer, tab_camera, tab_calib, tab_storage, tab_logs = st.tabs(
            [
                ":material/monitoring: Viewer",
                ":material/videocam: Camera",
                ":material/adjust: Calibrate",
                ":material/database: Storage",
                ":material/terminal: Logs",
            ]
        )
        with tab_viewer:
            st.iframe(_browser_url(servers.web_url), height=VIEWER_HEIGHT_PX)
            # The robot 3D scene shares this viewer's left panel (it replaced the
            # old EEF trace); its controls + status sit below so the viewer leads.
            _render_skill_bar(registry)
            depth_cams = _cameras.depth_cameras()
            for name in depth_cams:
                # Cheap static re-log each rerun: keeps the frustum in sync with
                # conf/camera pose edits after a "Reset services".
                _robot.log_camera_frustum(name, _cameras.camera_cfg(name))
            if depth_cams:
                # Streams whenever live RGB-D exists; show/hide the cloud from the
                # Rerun sidebar (entity visibility) rather than a dashboard control.
                _depth_overlay_feed(depth_cams[0])
            st.caption(
                "Rizon 4s on the Vention pedestal beside the live run metrics — "
                "**solid** arms = measured joint state, **translucent ghost** = "
                "commanded teleop, **red** = no live joint data."
            )
            _robot_data_status()
        with tab_camera:
            _render_camera_tab(cameras)
        with tab_calib:
            _render_calibration_tab()
        with tab_storage:
            st.caption(
                "Recorded LeRobot episodes — select with checkboxes, delete "
                "individually or in bulk."
            )
            _render_storage_tab(registry)
        with tab_logs:
            st.caption(
                "The live session daemon log (all dashboard runs) plus any past "
                "`dual-flexiv-control` CLI runs — pick one, follow the tail live, "
                "or read a past run's output."
            )
            _render_logs_tab(registry)


main()
