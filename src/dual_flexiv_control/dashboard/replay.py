"""Replay a recorded LeRobot episode into a dedicated Rerun viewer (Storage tab).

"Show everything" for one demo on a single scrubbable timeline:

* both arms posed in 3D at the recorded joint state (solid), with the recorded
  *action* joint target as a translucent ghost (joint-position kinds only),
* every camera view as video, and
* the ``observation.state`` / ``action`` channels as time series.

It reuses :mod:`~dual_flexiv_control.dashboard.robot_view` for the arm geometry +
FK posing and logs everything on that module's ``"elapsed"`` timeline so the 3D
scene, the images, and the plots scrub together.

**Each replay is its own Rerun recording** (fresh ``recording_id``, timeline from
0), streamed to a persistent gRPC data server. A fresh recording — rather than
clearing and re-logging a reused one — is what actually isolates episodes: a
time-scoped ``rr.Clear`` cannot retroactively purge a previous (longer) episode's
per-frame rows, so replaying a shorter episode after a longer one would otherwise
show the previous episode's tail. The per-arm ``q`` / action split is read from the
dataset's **stored column names**, so replay reflects exactly what was recorded
(robust to later config/control-kind changes), not the current config.

Replay has its **own gRPC data server** (so its recordings stay isolated from the
live metrics stream) but **no web viewer of its own** — it is embedded in the single
shared web-viewer host that :mod:`~.viewer` binds for the metrics tab, just pointed
at this server via the iframe's ``?url=``. That deliberately avoids a second
``serve_web_viewer``: rerun's web viewer cannot be stopped or rebound in-process, and
binding a second one is what used to wedge on a rapid restart (black viewer needing a
full process restart). The gRPC data server, by contrast, is torn down cleanly by
:func:`~.viewer.teardown` and re-served on demand.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from . import robot_view
from .robot_view import _POSE_TIMELINE  # shared timeline so arms + images + plots scrub together
from .storage import _hub_offline

log = logging.getLogger(__name__)

REPLAY_APP_ID = "dual-flexiv-replay"
DEFAULT_REPLAY_GRPC_PORT = 9880

#: Canonical camera-image key prefix in a LeRobot frame.
_IMAGE_PREFIX = "observation.images."

_LOCK = threading.Lock()
_VIEWER: "ReplayViewer | None" = None
#: The persistent server host recording + its proxy URI (fresh per-episode recordings
#: connect to this). Kept module-scoped so the server outlives every call.
_HOST = None
_SERVER_URI: str | None = None
#: Hold the most recent episode recording so it isn't GC'd before its data drains.
_LAST_REC = None
#: Monotonic per-replay counter → a unique recording_id each replay (so re-replaying
#: the same episode also yields a fresh, isolated recording). Date/random are avoided.
_GEN = 0


# --------------------------------------------------------------------------- #
# Recorded-layout parsing (from the dataset's stored column names)
# --------------------------------------------------------------------------- #


def _state_q_index(state_names: list[str]) -> dict[str, list[int]]:
    """``{side: [indices of that side's q]}`` from ``observation.state`` column names."""
    q_index: dict[str, list[int]] = {}
    for i, name in enumerate(state_names):
        parts = str(name).split(".")
        if len(parts) >= 3 and parts[1] == "q":     # e.g. "left.q.3"
            q_index.setdefault(parts[0], []).append(i)
    return q_index


def _action_layout(action_names: list[str]) -> dict[str, dict]:
    """``{side: {"field", "cmd": [idx], "gripper": idx|None}}`` from action column names.

    ``field`` is the recorded command field (``q_d``, ``dq_d``, …). Only ``q_d``
    yields a joint-position ghost; the block is still plotted for any field.
    """
    blocks: dict[str, dict] = {}
    for i, name in enumerate(action_names):
        parts = str(name).split(".")
        side = parts[0]
        blk = blocks.setdefault(side, {"field": None, "cmd": [], "gripper": None})
        if len(parts) == 2 and parts[1] == "gripper":   # "left.gripper"
            blk["gripper"] = i
        elif len(parts) >= 3:                            # "left.q_d.4"
            blk["field"] = parts[1]
            blk["cmd"].append(i)
    return blocks


def _names(feature) -> list[str]:
    """The flat column-name list for a feature (``[]`` if absent/non-list)."""
    names = (feature or {}).get("names")
    return list(names) if isinstance(names, (list, tuple)) else []


# --------------------------------------------------------------------------- #
# Episode reading
# --------------------------------------------------------------------------- #


def _to_hwc_uint8(img) -> np.ndarray:
    """A LeRobot image tensor (CHW float[0,1] or HWC uint8) → HWC uint8 for rr.Image."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))            # CHW -> HWC
    if a.dtype != np.uint8:
        if float(a.max(initial=0.0)) <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def read_episode(ds_info, episode_index: int):
    """Read one episode into per-frame dicts + the camera-view names.

    Returns ``(frames, cam_names)`` where each frame is
    ``{"t", "real_q": {side: (dof,)}, "ghost_q": {side: (dof,)}, "images": {name: HWC uint8},
       "action": {side: {"cmd": [...], "gripper": float}}}``.
    The per-arm ``q`` / action split comes from the dataset's stored column names, so
    it reflects exactly what was recorded. Purely local (never contacts the Hub).
    """
    with _hub_offline():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(ds_info.repo_id, root=ds_info.path)
        row = ds.meta.episodes[int(episode_index)]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])

        feats = ds.meta.features
        q_index = _state_q_index(_names(feats.get("observation.state")))
        action_blk = _action_layout(_names(feats.get("action")))
        cam_keys = [k for k in feats if k.startswith(_IMAGE_PREFIX)]
        cam_names = [k[len(_IMAGE_PREFIX):] for k in cam_keys]

        state_dim = int(np.asarray(ds[lo]["observation.state"]).size)
        pose_sides = [s for s, idx in q_index.items() if idx and max(idx) < state_dim]

        frames = []
        for i in range(lo, hi):
            item = ds[i]
            state = np.asarray(item["observation.state"], dtype=np.float64).ravel()
            action = np.asarray(item["action"], dtype=np.float64).ravel()
            real_q, ghost_q, blocks = {}, {}, {}
            for side in pose_sides:
                real_q[side] = state[q_index[side]]
            for side, blk in action_blk.items():
                if blk["cmd"] and max(blk["cmd"]) < action.size:
                    cmd = action[blk["cmd"]]
                    grip = (float(action[blk["gripper"]])
                            if blk["gripper"] is not None and blk["gripper"] < action.size else 0.0)
                    blocks[side] = {"cmd": cmd.tolist(), "gripper": grip}
                    if blk["field"] == "q_d" and side in real_q:   # joint-position -> ghost arm
                        ghost_q[side] = cmd
            images = {name: _to_hwc_uint8(item[key]) for name, key in zip(cam_names, cam_keys)}
            frames.append({
                "t": float(np.asarray(item["timestamp"]).item()),
                "real_q": real_q, "ghost_q": ghost_q, "images": images, "action": blocks,
            })
        return frames, cam_names


# --------------------------------------------------------------------------- #
# Logging into a fresh per-episode recording
# --------------------------------------------------------------------------- #


def log_episode(ds_info, episode_index: int) -> int:
    """Stream one episode as its OWN Rerun recording to the replay server; frame count.

    A fresh ``recording_id`` per call means each episode has an isolated timeline
    (from 0) — no leftover frames from a previously-replayed (longer) episode.
    """
    global _LAST_REC, _GEN
    frames, cam_names = read_episode(ds_info, episode_index)

    with _LOCK:
        if _SERVER_URI is None:
            raise RuntimeError("replay viewer not started; call start_replay_viewer() first")
        _GEN += 1
        rec = rr.RecordingStream(REPLAY_APP_ID, recording_id=f"ep-{ds_info.repo_id}-{episode_index}-{_GEN}")
        rec.connect_grpc(_SERVER_URI)
        _LAST_REC = rec  # keep alive until the next replay so its data fully drains

    _log_frames(rec, frames, cam_names)
    rec.flush()
    return len(frames)


def _log_frames(rec, frames, cam_names) -> None:
    """Log the static scene + every frame (arms, images, plots) onto ``rec``.

    Pure Rerun logging against whatever stream is passed — a served per-episode
    recording in production, or a plain buffered ``RecordingStream`` under test.
    """
    try:
        robot_view.log_scene(rec)              # static pedestal + arm geometry
    except Exception:                          # noqa: BLE001 - missing URDF must not kill replay
        log.exception("replay: arm scene unavailable; showing cameras + plots only")

    for fr in frames:
        t = fr["t"]
        try:
            robot_view.update_poses(rec, fr["real_q"], fr["ghost_q"], t)
        except Exception:                      # noqa: BLE001
            log.debug("replay: arm pose update failed at t=%s", t, exc_info=True)
        rec.set_time(_POSE_TIMELINE, duration=t)
        for name, img in fr["images"].items():
            rec.log(f"replay/cam/{name}", rr.Image(img))
        for side, q in fr["real_q"].items():
            rec.log(f"replay/state/{side}/q", rr.Scalars(np.asarray(q).tolist()))
        for side, blk in fr["action"].items():
            rec.log(f"replay/action/{side}/cmd", rr.Scalars(blk["cmd"]))
            rec.log(f"replay/action/{side}/gripper", rr.Scalars([blk["gripper"]]))

    try:
        rec.send_blueprint(replay_blueprint(cam_names))
    except Exception:                          # noqa: BLE001 - blueprint is cosmetic
        log.exception("replay: could not send tailored blueprint")


def replay_blueprint(cam_names) -> rrb.Blueprint:
    """3D arms + a grid of camera views + state/action time series."""
    cam_views = [rrb.Spatial2DView(origin=f"replay/cam/{n}", name=n) for n in cam_names]
    right = rrb.Vertical(
        rrb.Grid(*cam_views) if cam_views else rrb.Spatial2DView(origin="replay/cam", name="cameras"),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="replay/state", name="state (q)"),
            rrb.TimeSeriesView(origin="replay/action", name="action"),
        ),
        row_shares=[3, 2],
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/robot",
                name="Replay — recorded state (solid) · action target (ghost)",
            ),
            right,
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


# --------------------------------------------------------------------------- #
# Dedicated recording server + web viewer (singleton per process)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReplayViewer:
    #: The SHARED metrics web-viewer port (:mod:`~.viewer`); replay has no web host
    #: of its own, it just embeds that viewer pointed at ``grpc_uri`` below.
    web_port: int
    #: Replay's own gRPC data server, keeping its recordings isolated from metrics.
    grpc_uri: str

    @property
    def web_url(self) -> str:
        # renderer=webgl for the same reason as the metrics viewer's web_url:
        # the WebGPU path is ~50x slower under Dawn's GL-compatibility mode.
        base = f"http://127.0.0.1:{self.web_port}"
        return f"{base}/?url={quote(self.grpc_uri, safe='')}&persist=0&renderer=webgl"


def grpc_port_from_env() -> int:
    """Replay gRPC-server port, honouring the ``DFC_REPLAY_GRPC_PORT`` override."""
    return int(os.environ.get("DFC_REPLAY_GRPC_PORT", DEFAULT_REPLAY_GRPC_PORT))


def start_replay_viewer(web_port: int, grpc_port: int | None = None) -> ReplayViewer:
    """Bring up the replay gRPC data server, embedded in the SHARED web viewer.

    ``web_port`` is the metrics web-viewer port (from :func:`~.viewer.start_servers`)
    — replay reuses that single HTTP viewer host and serves only its own gRPC data
    server here, so episodes stay isolated on their own recordings without a second
    ``serve_web_viewer``. Idempotent while up; after :func:`reset` (paired with
    :func:`~.viewer.teardown`) the next call re-serves a fresh gRPC server. A host
    recording owns the server; each :func:`log_episode` streams a fresh per-episode
    recording to it via ``connect_grpc``.
    """
    global _VIEWER, _HOST, _SERVER_URI
    with _LOCK:
        if _VIEWER is not None:
            return _VIEWER
        gp = grpc_port or grpc_port_from_env()
        _HOST = rr.RecordingStream(REPLAY_APP_ID, recording_id="replay-host")
        _SERVER_URI = _HOST.serve_grpc(
            grpc_port=gp,
            default_blueprint=replay_blueprint([]),
            cors_allow_origin=["*"],
        )
        _VIEWER = ReplayViewer(web_port=web_port, grpc_uri=_SERVER_URI)
        return _VIEWER


def reset() -> None:
    """Drop the replay viewer singletons so the next replay re-serves its gRPC.

    Paired with :func:`~.viewer.teardown` — that global ``rerun_shutdown`` releases
    this gRPC port too, so after a *Reset services* the next ▶ rebinds a fresh replay
    server (reusing the same shared web viewer).
    """
    global _VIEWER, _HOST, _SERVER_URI, _LAST_REC
    with _LOCK:
        _VIEWER = None
        _HOST = None
        _SERVER_URI = None
        _LAST_REC = None
