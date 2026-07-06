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

import streamlit as st

# Absolute imports: Streamlit executes this file as a top-level script (no package
# context), so relative imports would fail here. The package itself is installed,
# so its submodules resolve normally.
from dual_flexiv_control.dashboard import arms as _arms
from dual_flexiv_control.dashboard import blueprints
from dual_flexiv_control.dashboard import cameras as _cameras
from dual_flexiv_control.dashboard import replay as _replay
from dual_flexiv_control.dashboard import robot_view as _robot
from dual_flexiv_control.dashboard import runner as _runner
from dual_flexiv_control.dashboard import storage as _storage
from dual_flexiv_control.dashboard.arms import ArmStatus
from dual_flexiv_control.dashboard.arms import discover_arms
from dual_flexiv_control.dashboard.arms import read_arm_status
from dual_flexiv_control.dashboard.cameras import CameraView
from dual_flexiv_control.dashboard.cameras import discover_camera_views
from dual_flexiv_control.dashboard.cameras import get_frame
from dual_flexiv_control.dashboard.editor import open_in_vscode
from dual_flexiv_control.dashboard.tasks import TaskInfo
from dual_flexiv_control.dashboard.tasks import discover_launchables
from dual_flexiv_control.dashboard.viewer import RerunServers
from dual_flexiv_control.dashboard.viewer import ports_from_env
from dual_flexiv_control.dashboard.viewer import start_servers

VIEWER_HEIGHT_PX = 1400
#: Camera-tab refresh cadence (placeholder feed; real shm reads pace themselves).
CAMERA_REFRESH = "0.15s"

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
    """Start (once) the dedicated episode-replay viewer on its own ports."""
    return _replay.start_replay_viewer()


@st.cache_resource
def _registry() -> _runner.RunRegistry:
    return _runner.RunRegistry()


def _reset_services(registry: _runner.RunRegistry) -> None:
    """Restart the dashboard's live services: stop the run, drop caches, reset viewers.

    The dashboard equivalent of restarting the stack, without killing the process
    (the Rerun servers keep their bound ports):

    1. Stop the active run — joins its emitter thread, closing the live
       ``FlexivSource`` / FACTR-leader connections it holds — and clear history.
    2. Drop the cached Hydra composes so arms + cameras re-read ``conf`` (and any
       changed ``runtime.sim``) on next use.
    3. Return the metrics viewer to idle and snap the robot scene back to home.
    """
    registry.reset()
    _arms.reset()
    _cameras.reset()
    _storage.reset()
    _runner.reset_viewer()
    _robot.relog_scene()


def _render_controls(tasks: list[TaskInfo], registry: _runner.RunRegistry) -> None:

    st.subheader("Experiment")
    if not tasks:
        st.error(
            "No tasks found in `conf/task/`. Add one (copy `task/default.yaml`) "
            "and reload."
        )
        return

    # Tasks (conf/task, run on the default config) plus run profiles (whole-config,
    # e.g. `test` = dummy arm + one real camera) — profiles are marked with ⚙.
    by_label = {(f"⚙ {t.name} (profile)" if t.config_name else t.name): t for t in tasks}
    selected = st.selectbox(
        "Task", list(by_label),
        help="Tasks from conf/task, plus ⚙ run profiles (whole-config, e.g. `test`).",
    )
    task = by_label[selected]
    if task.config_name:
        st.caption(f"⚙ Profile — launches the whole `{task.config_name}` config (its own arms/cameras).")

    if st.button(
        "✏️ Edit YAML",
        use_container_width=True,
        help=f"Open conf/{task.path.name} in VSCode on this machine.",
    ):
        result = open_in_vscode(task.path)
        if result.ok:
            st.toast(result.message, icon="📝")
        else:
            st.warning(result.message)

    active = registry.active()
    running = active is not None

    launch_cols = st.columns(2)
    if launch_cols[0].button(
        "▶ Collection", use_container_width=True, disabled=running,
        help="Teleoperated demonstration gathering.",
    ):
        _launch(registry, task, "collection")
    if launch_cols[1].button(
        "▶ Eval", type="primary", use_container_width=True, disabled=running,
        help="Online policy rollouts.",
    ):
        _launch(registry, task, "eval")

    st.caption("Each launch runs a single episode.")

    if running:
        st.caption("A run is active — stop it before launching another.")

    st.divider()
    _render_status(registry)


def _render_arm_row(s: ArmStatus) -> None:
    connected = s.source == "live"
    dot = "🟢" if connected else "⚫"
    mode = s.mode if connected else f":gray[{s.mode}]"
    if s.estop_pressed:
        estop = ":red[🛑 **E-STOP PRESSED**]"
    elif s.estop_pressed is False:
        estop = ":green[clear]"
    else:
        estop = ":gray[—]"
    st.markdown(f"{dot} **{s.info.name}** · {mode}  \nE-stop: {estop}")


@st.fragment(run_every="2s")
def _arm_status_rows() -> None:
    """Two read-only per-arm rows (operation mode + E-stop), refreshed periodically."""
    for arm in discover_arms():
        _render_arm_row(read_arm_status(arm))


def _launch(registry: _runner.RunRegistry, task: TaskInfo, phase: str) -> None:
    """Launch a run, surfacing a refused launch (e.g. one is still saving) inline."""
    try:
        registry.launch(task, phase)
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
    active = registry.active()
    if active is None:
        st.info("No run active. Pick a task and launch.")
        return
    st.success(f"**{active.phase.upper()}** · {active.task}")
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
    _arm_status_rows()
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
            "Stop any active run and restart connections: closes the live arm / "
            "FACTR links, reloads config from conf, and returns the viewers to idle."
        ),
    ):
        _reset_services(registry)
        st.toast("Services reset — connections restarted.", icon="🔄")
        st.rerun()


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
    """Auto-refreshing image for the selected camera (only this fragment reruns)."""
    key = st.session_state.get("camera_key") or next(iter(by_key))
    view = by_key.get(key)
    if view is None:
        return
    frame, source = get_frame(view)
    st.image(frame, width="stretch")
    badge = "🟢 live" if source == "live" else "⚪ placeholder"
    st.caption(f"`{view.key}` · {view.width}×{view.height} · {badge}")


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


def main() -> None:
    st.set_page_config(
        page_title="dual-flexiv experiments", page_icon="🤖", layout="wide"
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    servers = _servers()
    registry = _registry()
    _show_run_alert()
    tasks = discover_launchables()
    cameras = discover_camera_views()

    controls, panel = st.columns([1, 3], gap="large")
    with controls:
        _render_controls(tasks, registry)
    with panel:
        tab_metrics, tab_camera, tab_storage = st.tabs(
            ["📊 Metrics", "📷 Camera", "💾 Storage"]
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


main()
