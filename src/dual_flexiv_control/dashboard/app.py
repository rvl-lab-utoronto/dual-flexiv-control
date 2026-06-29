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
from dual_flexiv_control.dashboard import blueprints
from dual_flexiv_control.dashboard import runner as _runner
from dual_flexiv_control.dashboard.editor import open_in_vscode
from dual_flexiv_control.dashboard.tasks import TaskInfo
from dual_flexiv_control.dashboard.tasks import discover_tasks
from dual_flexiv_control.dashboard.viewer import RerunServers
from dual_flexiv_control.dashboard.viewer import ports_from_env
from dual_flexiv_control.dashboard.viewer import start_servers

VIEWER_HEIGHT_PX = 860


@st.cache_resource
def _servers() -> RerunServers:
    """Reuse (or start) the Rerun servers; load the idle README on first start.

    The launcher usually pre-binds these (``dashboard.launch``); ``start_servers``
    is idempotent, so this returns the existing handle. When the app is run
    directly (``streamlit run app.py``) it starts them here instead.
    """
    import rerun as rr

    grpc_port, web_port = ports_from_env()
    servers = start_servers(grpc_port=grpc_port, web_port=web_port)
    rr.send_blueprint(blueprints.welcome_blueprint())
    _runner.log_welcome()
    return servers


@st.cache_resource
def _registry() -> _runner.RunRegistry:
    return _runner.RunRegistry()


def _render_controls(tasks: list[TaskInfo], registry: _runner.RunRegistry) -> None:
    st.title("Run a task")

    if not tasks:
        st.error(
            "No tasks found in `conf/task/`. Add one (copy `task/default.yaml`) "
            "and reload."
        )
        return

    by_name = {t.name: t for t in tasks}
    selected = st.selectbox("Task", list(by_name), help="Pulled from the conf/task group.")
    task = by_name[selected]

    st.markdown(f"**Instruction**\n\n{task.language_instruction}")
    cols = st.columns(2)
    cols[0].metric("Collection episodes", task.num_episodes if task.num_episodes is not None else "—")
    cols[1].metric("Eval timesteps", task.num_timesteps if task.num_timesteps is not None else "—")

    if st.button(
        "✏️ Edit task YAML",
        use_container_width=True,
        help=f"Open conf/task/{task.path.name} in VSCode on this machine.",
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
        registry.launch(task, "collection")
        st.rerun()
    if launch_cols[1].button(
        "▶ Eval", type="primary", use_container_width=True, disabled=running,
        help="Online policy rollouts.",
    ):
        registry.launch(task, "eval")
        st.rerun()

    if running:
        st.caption("A run is active — stop it before launching another.")

    st.divider()
    _render_status(registry)


def _render_status(registry: _runner.RunRegistry) -> None:
    st.subheader("Running")
    active = registry.active()
    if active is None:
        st.info("No run active. Pick a task and launch.")
    else:
        st.success(f"**{active.phase.upper()}** · {active.task}")
        st.caption(f"run `{active.run_id}` · started {active.started_wall}")
        if st.button("■ Stop", use_container_width=True):
            registry.stop_active()
            st.rerun()

    history = registry.history()
    if history:
        with st.expander(f"History ({len(history)})"):
            for record in reversed(history):
                st.text(
                    f"{record.started_wall}  {record.phase:<10} "
                    f"{record.task:<14} {record.status}"
                )


def main() -> None:
    st.set_page_config(
        page_title="dual-flexiv experiments", page_icon="🤖", layout="wide"
    )
    servers = _servers()
    registry = _registry()
    tasks = discover_tasks()

    controls, metrics = st.columns([1, 3], gap="large")
    with controls:
        _render_controls(tasks, registry)
    with metrics:
        st.caption(f"Metrics — live Rerun viewer · {servers.web_base}")
        st.iframe(servers.web_url, height=VIEWER_HEIGHT_PX)


main()
