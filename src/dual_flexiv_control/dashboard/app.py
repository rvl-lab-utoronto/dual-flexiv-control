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

import streamlit as st

# Absolute imports: Streamlit executes this file as a top-level script (no package
# context), so relative imports would fail here. The package itself is installed,
# so its submodules resolve normally.
from dual_flexiv_control.dashboard import arms as _arms
from dual_flexiv_control.dashboard import blueprints
from dual_flexiv_control.dashboard import cameras as _cameras
from dual_flexiv_control.dashboard import logs as _logs
from dual_flexiv_control.dashboard import replay as _replay
from dual_flexiv_control.dashboard import robot_view as _robot
from dual_flexiv_control.dashboard import runner as _runner
from dual_flexiv_control.dashboard import storage as _storage
from dual_flexiv_control.dashboard.arms import ArmStatus
from dual_flexiv_control.dashboard.arms import discover_arms
from dual_flexiv_control.dashboard.arms import read_arm_status
from dual_flexiv_control.dashboard.cameras import CameraStatus
from dual_flexiv_control.dashboard.cameras import CameraView
from dual_flexiv_control.dashboard.cameras import discover_camera_views
from dual_flexiv_control.dashboard.cameras import get_frame
from dual_flexiv_control.dashboard.cameras import read_camera_statuses
from dual_flexiv_control.dashboard.editor import open_in_vscode
from dual_flexiv_control.dashboard.ssh_hosts import discover_ssh_hosts
from dual_flexiv_control.dashboard.tasks import RigInfo
from dual_flexiv_control.dashboard.tasks import TaskInfo
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

#: Trim the default top padding and enlarge the tab buttons.
_PAGE_CSS = """
<style>
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 1.5rem !important;
}
.stTabs [data-baseweb="tab-list"] button { padding: 0.6rem 1.4rem; }
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.25rem;
    font-weight: 600;
}
</style>
"""

#: Page background per session mode. VIEWING keeps the theme default (#0e1117,
#: hsv 220° 39% 9%); COLLECTION is the same hue one value step up (~15%); EVAL is
#: collection's value/saturation with the hue shifted purple (~280°).
_MODE_BG = {"collection": "#181d27", "eval": "#211627"}


def _apply_mode_background(view) -> None:
    """Tint the whole page by session mode so the active mode reads at a glance.

    ``saving`` keeps its run's tint (``view.phase`` stays set) so the color does
    not snap back to viewing while the episode finalizes. The header is made
    transparent so the tint runs edge to edge.
    """
    color = _MODE_BG.get(view.phase if view.state == "saving" else view.state)
    if color is None:
        return
    st.markdown(
        f"<style>.stApp {{ background-color: {color}; }} "
        f'[data-testid="stHeader"] {{ background: transparent; }}</style>',
        unsafe_allow_html=True,
    )


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
    _teardown_servers()          # rerun_shutdown: releases metrics + replay gRPC ports
    _robot.reset()               # forget the dead metrics recording
    _replay.reset()              # forget the (now released) replay server
    _servers.clear()             # st.cache_resource: re-run start_servers on next call
    _replay_viewer.clear()
    _servers()                   # re-serve metrics gRPC + re-attach robot scene + welcome
    _runner.reset_viewer()       # idle welcome blueprint + README + reset event log
    # Fresh daemon on the (possibly re-read) rig/sim; ensure restarts the mirror too.
    registry.ensure_session(_arms.active_rig(), _arms.runtime_is_sim())
    return True


def _render_controls(
    tasks: list[TaskInfo], rig: RigInfo | None, registry: _runner.RunRegistry
) -> None:

    st.subheader("Experiment")
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
    st.markdown(
        f"Rig: **`{rig.name}`**",
        help=(
            "Hardware setup from conf/rig — arms, cameras, FACTR leaders, serials. "
            "Fixed for this dashboard's lifetime; relaunch with "
            "`dfc-dashboard --rig <name>` (or the VSCode dashboard tasks) to switch."
        ),
    )
    if rig.description:
        st.caption(rig.description)

    by_name = {t.name: t for t in tasks}
    selected = st.selectbox(
        "Task", list(by_name),
        help="Manipulation task from conf/task — instruction, dataset, episode counts.",
    )
    task = by_name[selected]

    edit_cols = st.columns(2)
    if edit_cols[0].button(
        "✏️ Task YAML", use_container_width=True,
        help=f"Open conf/task/{task.path.name} in VSCode on this machine.",
    ):
        result = open_in_vscode(task.path)
        st.toast(result.message, icon="📝") if result.ok else st.warning(result.message)
    if edit_cols[1].button(
        "✏️ Rig YAML", use_container_width=True,
        help=f"Open conf/rig/{rig.path.name} in VSCode on this machine.",
    ):
        result = open_in_vscode(rig.path)
        st.toast(result.message, icon="📝") if result.ok else st.warning(result.message)

    # Launchable only from VIEWING with every rig camera streaming (the daemon
    # refuses otherwise; disabling here just makes that visible up front).
    launchable = view.state == "viewing" and not view.cameras_down

    if st.button(
        "▶ Collection", use_container_width=True, disabled=not launchable,
        help="Teleoperated demonstration gathering (real recording run).",
    ):
        _launch(registry, task, "collection", rig.name)

    # One compact row: labels collapsed (the column is narrow), meaning carried
    # by tooltips + the resolution caption underneath.
    eval_cols = st.columns([1.4, 1.2, 0.8], vertical_alignment="center")
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
    if eval_cols[0].button(
        "▶ Eval", type="primary", use_container_width=True,
        disabled=not launchable or port_error is not None,
        help="Online policy rollout (needs the policy server reachable).",
    ):
        _launch(registry, task, "eval", rig.name, host=host, port=port)
    if port_error:
        st.caption(f":red[{port_error}]")
    elif host is not None or port is not None:
        st.caption(
            f":gray[policy → {host or 'config host'}:{port or 'config port'}]"
        )

    st.caption("Each launch runs a single episode.")

    if view.run_active:
        st.caption("A run is active — stop it before launching another.")
    elif view.state == "starting":
        st.caption("Session starting — arms and cameras coming up…")
    elif view.state == "down":
        st.caption("Session daemon is not running — see the Logs tab or Reset services.")

    st.divider()
    _render_status(registry)


def _render_arm_row(s: ArmStatus) -> None:
    connected = s.source == "live"
    dot = "🟢" if connected else "⚫"
    mode = s.mode if connected else f":gray[{s.mode}]"
    if s.control_active:
        mode += " · :orange[controlling]"
    if s.estop_pressed:
        estop = ":red[🛑 **E-STOP PRESSED**]"
    elif s.estop_pressed is False:
        estop = ":green[clear]"
    else:
        estop = ":gray[—]"
    st.markdown(f"{dot} **{s.info.name}** · {mode}  \nE-stop: {estop}")


@st.fragment(run_every="2s")
def _arm_status_rows() -> None:
    """Read-only per-arm rows (operation mode + E-stop), refreshed periodically."""
    try:
        arms = discover_arms()
    except Exception as exc:  # noqa: BLE001 - broken rig conf -> compact, visible note
        st.warning(f"Arm config failed to compose: {exc}", icon="🛠️")
        return
    for arm in arms:
        _render_arm_row(read_arm_status(arm))


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
    host: str | None = None, port: int | None = None,
) -> None:
    """Launch a run, surfacing a refused launch (e.g. one is still saving) inline."""
    try:
        registry.launch(task, phase, rig, host=host, port=port)
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
        st.session_state["run_alert"] = alert
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
    if view.message:
        st.warning(view.message, icon="⚠️")
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
        "💾 Saving episode…" if stopping else "■ Stop",
        use_container_width=True,
        disabled=stopping,
        help="Stops the run; the in-progress episode is saved (video finalize can take a while).",
    ):
        registry.stop_active()
        st.rerun(scope="app")


def _render_status(registry: _runner.RunRegistry) -> None:
    st.subheader("Status")
    st.caption("Arms")
    _arm_status_rows()
    st.caption("Cameras")
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

    st.divider()
    if st.button(
        "🔄 Reset services",
        use_container_width=True,
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


def _render_logs_tab() -> None:
    """Browse the flexiv control system's per-run logs (Hydra ``outputs/*/system.log``).

    Lists every run's log newest-first, tails the selected one (bounded read), and
    — while *Follow* is on — auto-refreshes so a live run streams in. The picker and
    controls are rendered here; only :func:`_log_tail_view` reruns on the follow tick.
    """
    files = _logs.discover_logs()
    top = st.columns([3, 1])
    if top[1].button("🔄 Refresh", use_container_width=True,
                     help="Re-scan the outputs directory for run logs."):
        st.rerun()
    if not files:
        st.info(
            f"No run logs found under `{_logs.outputs_root()}`. Launch a run "
            "(or run `dual-flexiv-control` from the CLI) and they'll appear here."
        )
        return

    by_name = {f"{f.name}  ·  {_logs.human_size(f.size_bytes)}": f for f in files}
    label = top[0].selectbox(
        "Run log", list(by_name), key="log_sel",
        help="One log per dual-flexiv-control run (Hydra outputs/<timestamp>/system.log).",
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
    """While the depth checkbox is on, push the latest RGB-D cloud into the robot scene.

    Reads the camera's ``left`` + ``depth`` shm streams, back-projects to a
    coloured point cloud (:func:`~.cameras.depth_point_cloud`), poses it with
    the camera extrinsics from ``conf/camera`` (:func:`~.robot_view.camera_world_pose`),
    and logs it into the robot recording. On checkbox off, clears the cloud once.
    """
    enabled = bool(st.session_state.get("robot_depth", False))
    was_on = bool(st.session_state.get("_robot_depth_was_on", False))
    st.session_state["_robot_depth_was_on"] = enabled
    if not enabled:
        if was_on:
            _robot.clear_depth_points()
        return
    cloud = _cameras.depth_point_cloud(camera)
    if cloud is None:
        st.caption(
            f"⚪ no live RGB-D from `{camera}` — needs the system running with "
            "`depth` in that camera's views."
        )
        return
    pts_cam, colors = cloud
    rot, t = _robot.camera_world_pose(_cameras.camera_cfg(camera))
    _robot.log_depth_points(pts_cam @ rot.T + t, colors)
    st.caption(f"🟢 depth overlay: {len(pts_cam):,} points from `{camera}`")


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
    header = st.columns([0.6, 5, 0.8, 0.8])
    header[0].caption("Sel")
    header[1].caption("Episode")
    header[2].caption("Replay")
    header[3].caption("Delete")
    with st.container(height=420):
        for e in episodes:
            row = st.columns([0.6, 5, 0.8, 0.8])
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
            if row[3].button("🗑️", key=f"stor_del::{ds.repo_id}::{e.index}", disabled=locked):
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
        tab_metrics, tab_camera, tab_storage, tab_logs = st.tabs(
            ["📊 Metrics", "📷 Camera", "💾 Storage", "📜 Logs"]
        )
        with tab_metrics:
            # The robot 3D scene now shares this viewer's left panel (it replaced the
            # old EEF trace), so its controls + status live alongside the metrics.
            st.caption(
                "Rizon 4s on the Vention pedestal beside the live run metrics — "
                "**solid** arms = measured joint state, **translucent ghost** = "
                "commanded teleop, **red** = no live joint data."
            )
            _robot_data_status()
            depth_cams = _cameras.depth_cameras()
            for name in depth_cams:
                # Cheap static re-log each rerun: keeps the frustum in sync with
                # conf/camera pose edits after a "Reset services".
                _robot.log_camera_frustum(name, _cameras.camera_cfg(name))
            if depth_cams:
                st.checkbox(
                    f"Depth overlay (`{depth_cams[0]}`)",
                    key="robot_depth",
                    help=(
                        "Project the camera's RGB-D as coloured points into the 3D "
                        "scene, posed via pose_frame/pose_xyz/pose_rpy in conf/camera."
                    ),
                )
                _depth_overlay_feed(depth_cams[0])
            st.iframe(_browser_url(servers.web_url), height=VIEWER_HEIGHT_PX)
        with tab_camera:
            _render_camera_tab(cameras)
        with tab_storage:
            st.caption(
                "Recorded LeRobot episodes — select with checkboxes, delete "
                "individually or in bulk."
            )
            _render_storage_tab(registry)
        with tab_logs:
            st.caption(
                "Per-run flexiv control logs (Hydra `outputs/<timestamp>/system.log`) "
                "— pick a run, follow the tail live, or read a past run's output."
            )
            _render_logs_tab()


main()
