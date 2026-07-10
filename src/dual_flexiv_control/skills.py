"""Teach-and-repeat skills: save a demonstrated joint trajectory, replay it later.

A **skill** is a joint-position trajectory (per side, at a fixed fps) saved as a
single JSON file under the skills root (``skill.root``, default ``skills/`` in
the launch cwd — beside ``datasets/``). Teaching **from a replayed episode** is
the first (implemented) teach type: the dashboard's 🎓 Teach button extracts the
*measured* joint trajectory (the ``<side>.q.*`` columns of ``observation.state``)
plus the recorded gripper command from a LeRobot episode and saves it as a skill.
The format is teach-source-agnostic, so a future live/freedrive teach writes the
same files.

Repeating a skill is a run phase (``runtime.phase=skill``) hosted exactly like
collection/eval: the session daemon hands each driven arm an ``EnterControl``
with the ``skill.coeffs`` and spawns :class:`SkillNode` — the run's consumer —
which streams the trajectory as joint-position setpoints on the normal control
channel. Every existing guardrail applies unchanged: the arm-side MoveJ
bootstrap to the first target, the L-inf safety gate, the deadman, the dropout
watchdog, E-stop. The loop holds the trajectory's first target until the arm has
actually converged onto it (the bootstrap MoveJ can take seconds), then steps
through the frames on the wall clock at the skill's fps.

Why not the RDK's own teach tools: flexivrdk 1.8 is NRT-only and offers plan /
primitive execution (plans are authored in Flexiv Elements), but no
record-and-replay API. Streaming the recorded trajectory through the existing
``qpos`` control path reuses all of the guardrails above, so it is both simpler
and safer than a parallel RDK-side implementation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np

from .brain import Brain
from .configs import ArmCfg
from .configs import BrainCfg
from .configs import RuntimeCfg
from .configs import SkillCfg
from .control import control_specs
from .control import pack_action
from .process import ProcessNode
from .process import RateLimiter
from .streams.registry import AttachAborted
from .streams.registry import StreamRegistry
from .streams.spec import StreamSpec
from .streams.stream import StreamWriter

log = logging.getLogger(__name__)

#: On-disk format version (bump on breaking changes; load refuses newer files).
FORMAT_VERSION = 1

#: Skill names double as file stems and Hydra override values (``skill.name=…``),
#: so they are restricted to the same safe charset the daemon validates.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Setpoints are re-posted at least this fast regardless of the skill's fps, so a
#: low-fps skill never trips the arm's staleness deadman (default soft 100 ms)
#: between frames — the same target is simply posted again until the next frame.
_MIN_POST_HZ = 20.0

#: The dashboard's 3D scene poses a purple target ghost from this stream (see
#: ``dashboard.arms.HORIZON_STREAM``). The eval node publishes its chunk-end
#: prediction here; a skill replay publishes the skill's END configuration, so
#: the viewer shows where the arm is headed. Kept in sync with
#: ``policy.loop.horizon_stream_name`` (not imported: the spawned skill process
#: should not drag the policy client stack in).
_HORIZON_STREAM = "eval/{side}/q_horizon"


@contextlib.contextmanager
def _hub_offline():
    """Force lerobot/HF fully local for the block (same rationale as
    :func:`dashboard.storage._hub_offline`: local repo_ids must never fall back
    to a Hub download that 401s and hides the real, local error)."""
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# The skill file
# --------------------------------------------------------------------------- #


@dataclass
class Skill:
    """One taught trajectory: per-side joint positions (+ gripper) at ``fps``."""

    name: str
    fps: float
    #: ``{side: (frames, dof) float64}`` — the joint trajectory to retrace.
    q: dict[str, np.ndarray]
    #: ``{side: (frames,) float64}`` — recorded gripper command (kept for when
    #: gripper actuation lands; not actuated yet, same as collection/eval).
    gripper: dict[str, np.ndarray] = field(default_factory=dict)
    #: Where this skill came from (teach provenance, e.g. dataset + episode).
    source: dict = field(default_factory=dict)
    created: str = ""

    @property
    def sides(self) -> list[str]:
        return list(self.q)

    @property
    def frames(self) -> int:
        return min((arr.shape[0] for arr in self.q.values()), default=0)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0

    def validate(self) -> None:
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(
                f"invalid skill name {self.name!r} (use letters, digits, '.', '_', '-')"
            )
        if not self.q:
            raise ValueError("skill has no joint trajectory for any side")
        if self.fps <= 0:
            raise ValueError(f"skill fps must be positive, got {self.fps}")
        for side, arr in self.q.items():
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
                raise ValueError(
                    f"skill {self.name!r}: side {side!r} trajectory must be "
                    f"(frames, dof), got shape {arr.shape}"
                )


@dataclass(frozen=True)
class SkillInfo:
    """One saved skill, as listed in the dashboard's skill bar."""

    name: str
    path: str
    sides: tuple[str, ...]
    frames: int
    fps: float
    duration_s: float
    created: str
    source: dict


def resolve_root(root: str | os.PathLike[str]) -> Path:
    """The skills directory as an absolute path (resolved against the cwd)."""
    return Path(root).absolute()


def skill_file(root: str | os.PathLike[str], name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise ValueError(
            f"invalid skill name {name!r} (use letters, digits, '.', '_', '-')"
        )
    return resolve_root(root) / f"{name}.json"


def save_skill(skill: Skill, root: str | os.PathLike[str]) -> Path:
    """Write ``skill`` under ``root`` (atomically); returns the file path.

    Saving an existing name overwrites it — teaching a skill again replaces the
    old trajectory.
    """
    skill.validate()
    path = skill_file(root, skill.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "name": skill.name,
        "created": skill.created or time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": float(skill.fps),
        "sides": skill.sides,
        "frames": skill.frames,
        "source": skill.source,
        "q": {s: np.asarray(a, dtype=np.float64).tolist() for s, a in skill.q.items()},
        "gripper": {
            s: np.asarray(a, dtype=np.float64).ravel().tolist()
            for s, a in skill.gripper.items()
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    log.info(
        "saved skill %r: %d frames at %.1f fps, sides=%s -> %s",
        skill.name, skill.frames, skill.fps, skill.sides, path,
    )
    return path


def load_skill(root: str | os.PathLike[str], name: str | None) -> Skill:
    """Load one skill by name; raises ``ValueError`` with a clear cause."""
    if not name:
        raise ValueError("no skill selected (skill.name is unset)")
    path = skill_file(root, name)
    if not path.is_file():
        raise ValueError(f"skill {name!r} not found under {resolve_root(root)}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"skill file {path} is unreadable: {exc}") from exc
    version = int(payload.get("version") or 0)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"skill {name!r} has format version {version}; this build reads up to "
            f"{FORMAT_VERSION}"
        )
    skill = Skill(
        name=str(payload.get("name") or name),
        fps=float(payload.get("fps") or 0.0),
        q={s: np.asarray(a, dtype=np.float64) for s, a in (payload.get("q") or {}).items()},
        gripper={
            s: np.asarray(a, dtype=np.float64).ravel()
            for s, a in (payload.get("gripper") or {}).items()
        },
        source=dict(payload.get("source") or {}),
        created=str(payload.get("created") or ""),
    )
    skill.validate()
    return skill


def discover_skills(root: str | os.PathLike[str]) -> list[SkillInfo]:
    """Every readable skill under ``root``, sorted by name. Corrupt files are
    skipped (logged) — one bad file must not hide the rest."""
    directory = resolve_root(root)
    if not directory.is_dir():
        return []
    out: list[SkillInfo] = []
    for path in sorted(directory.glob("*.json")):
        try:
            skill = load_skill(directory, path.stem)
        except ValueError:
            log.exception("skipping unreadable skill at %s", path)
            continue
        out.append(
            SkillInfo(
                name=skill.name,
                path=str(path),
                sides=tuple(skill.sides),
                frames=skill.frames,
                fps=skill.fps,
                duration_s=skill.duration_s,
                created=skill.created,
                source=skill.source,
            )
        )
    return out


def delete_skill(root: str | os.PathLike[str], name: str) -> None:
    skill_file(root, name).unlink(missing_ok=True)
    log.info("deleted skill %r under %s", name, resolve_root(root))


def rename_skill(root: str | os.PathLike[str], old: str, new: str) -> Path:
    """Rename a saved skill — the file and its embedded name; returns the new path.

    Refuses to clobber an existing skill of the target name (unlike teach, which
    deliberately overwrites its own). Provenance (source/created) is preserved.
    """
    src = skill_file(root, old)
    dst = skill_file(root, new)      # validates the new name
    skill = load_skill(root, old)    # raises if missing/corrupt
    if dst == src:
        return src
    if dst.exists():
        raise ValueError(f"a skill named {new!r} already exists")
    skill.name = new
    path = save_skill(skill, root)
    src.unlink(missing_ok=True)
    log.info("renamed skill %r -> %r under %s", old, new, resolve_root(root))
    return path


# --------------------------------------------------------------------------- #
# Teach type 1: from a recorded LeRobot episode (the Storage tab's replay)
# --------------------------------------------------------------------------- #


def _state_q_index(state_names: list[str]) -> dict[str, list[int]]:
    """``{side: [indices of that side's q]}`` from ``observation.state`` names
    (same stored-column parsing as ``dashboard.replay``)."""
    q_index: dict[str, list[int]] = {}
    for i, name in enumerate(state_names):
        parts = str(name).split(".")
        if len(parts) >= 3 and parts[1] == "q":     # e.g. "left.q.3"
            q_index.setdefault(parts[0], []).append(i)
    return q_index


def _gripper_index(action_names: list[str]) -> dict[str, int]:
    """``{side: index of "<side>.gripper"}`` from the action column names."""
    out: dict[str, int] = {}
    for i, name in enumerate(action_names):
        parts = str(name).split(".")
        if len(parts) == 2 and parts[1] == "gripper":
            out[parts[0]] = i
    return out


def _read_episode_arrays(ds_root: str, repo_id: str, episode_index: int):
    """One episode's non-video columns: ``(state_names, action_names, states,
    actions, fps, tasks)``. Reads the underlying parquet directly (never decodes
    video — teaching needs only the proprio columns). Purely local."""
    with _hub_offline():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id, root=ds_root)
        row = ds.meta.episodes[int(episode_index)]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        feats = ds.meta.features
        state_names = [str(n) for n in (feats.get("observation.state") or {}).get("names") or []]
        action_names = [str(n) for n in (feats.get("action") or {}).get("names") or []]
        sel = ds.hf_dataset.select(range(lo, hi))
        states = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in sel["observation.state"]])
        actions = np.stack([np.asarray(x, dtype=np.float64).ravel() for x in sel["action"]])
        tasks = [str(t) for t in (row.get("tasks") or [])]
        return state_names, action_names, states, actions, float(ds.meta.fps), tasks


def skill_from_episode(
    ds_root: str, repo_id: str, episode_index: int, name: str
) -> Skill:
    """Teach type 1: a skill from one recorded episode's MEASURED trajectory.

    The joint trajectory is the ``<side>.q.*`` slice of ``observation.state`` —
    what the follower actually did (smoother and truer to the demonstrated motion
    than the raw teleop command ``q_d``); the gripper channel is the recorded
    ``<side>.gripper`` action. Which sides exist comes from the dataset's stored
    column names, so a single-arm episode teaches a single-arm skill.
    """
    state_names, action_names, states, actions, fps, tasks = _read_episode_arrays(
        ds_root, repo_id, episode_index
    )
    q_index = _state_q_index(state_names)
    if not q_index:
        raise ValueError(
            f"episode {episode_index} of {repo_id!r} has no per-side q columns in "
            "observation.state — nothing to teach from"
        )
    grip_index = _gripper_index(action_names)
    q = {side: states[:, idx].astype(np.float64) for side, idx in q_index.items()}
    gripper = {
        side: actions[:, i].astype(np.float64)
        for side, i in grip_index.items()
        if side in q and i < actions.shape[1]
    }
    skill = Skill(
        name=name,
        fps=fps,
        q=q,
        gripper=gripper,
        source={
            "type": "episode",
            "repo_id": repo_id,
            "episode_index": int(episode_index),
            "task": tasks[0] if tasks else "",
        },
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    skill.validate()
    return skill


# --------------------------------------------------------------------------- #
# Repeat: the skill-replay loop + its process node
# --------------------------------------------------------------------------- #


class SkillLoop:
    """Streams one skill's trajectory as qpos setpoints through a Brain.

    The reusable core (takes an already-attached :class:`Brain`), so it runs
    in-process in tests with fakes; :class:`SkillNode` wraps it in the standard
    spawned-process shape. Two stages:

    1. **Converge** — post the trajectory's FIRST target repeatedly (keeping the
       deadman fed) until every driven side's measured ``q`` is within
       ``start_tolerance`` of it. The arm-side bootstrap MoveJs to that first
       setpoint, so this stage simply outlasts the (multi-second) approach;
       ``start_timeout_s`` bounds it (an E-stopped or halted arm raises, which
       surfaces as a crashed run rather than a silent freeze).
    2. **Replay** — step through the frames on the wall clock at the skill's fps
       (elapsed-based indexing, so a late tick skips ahead instead of drifting),
       re-posting at ≥ ``_MIN_POST_HZ`` so low-fps skills never starve the
       deadman. After the last frame the final target is held for ``settle_s``.
    """

    def __init__(
        self,
        brain: Brain,
        skill: Skill,
        control_arms: dict[str, ArmCfg],
        frequency_hz: float | None = None,
        start_tolerance: float = 0.1,
        start_timeout_s: float = 60.0,
        settle_s: float = 1.0,
    ) -> None:
        self.brain = brain
        self.skill = skill
        self.control_arms = {s: a for s, a in control_arms.items() if s in skill.q}
        if not self.control_arms:
            raise ValueError(
                f"skill {skill.name!r} (sides {skill.sides}) matches no driveable arm"
            )
        self.fps = float(frequency_hz or skill.fps)
        if self.fps <= 0:
            raise ValueError(f"replay rate must be positive, got {self.fps}")
        self.start_tolerance = float(start_tolerance)
        self.start_timeout_s = float(start_timeout_s)
        self.settle_s = float(settle_s)
        self.frames_posted = 0

    def _post(self, index: int) -> None:
        for side, arm in self.control_arms.items():
            traj = self.skill.q[side]
            q_d = traj[min(index, traj.shape[0] - 1)]
            self.brain.command(side, pack_action(arm.control, q_d))

    def _far_sides(self, index: int) -> list[str]:
        """Driven sides whose measured ``q`` is missing or > tolerance from frame
        ``index``'s target (L-inf, over the target's dof)."""
        far = []
        for side in self.control_arms:
            target = self.skill.q[side][index]
            s = self.brain.latest(f"{side}/q")
            if s.n == 0:
                far.append(side)
                continue
            measured = np.asarray(s.newest, dtype=np.float64)[: target.shape[0]]
            if float(np.max(np.abs(measured - target))) > self.start_tolerance:
                far.append(side)
        return far

    def run(self, stop_event) -> bool:
        """Replay the skill; True if it ran to completion, False if stopped."""
        post_hz = max(self.fps, _MIN_POST_HZ)
        n = self.skill.frames
        log.info(
            "skill %r: driving %s to the start pose (tol %.3f rad, timeout %.0fs)…",
            self.skill.name, list(self.control_arms), self.start_tolerance,
            self.start_timeout_s,
        )
        rate = RateLimiter(post_hz)
        rate.reset()
        deadline = time.monotonic() + self.start_timeout_s
        while not stop_event.is_set():
            self._post(0)
            far = self._far_sides(0)
            if not far:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"arm(s) {far} did not reach the skill's start pose within "
                    f"{self.start_timeout_s:.0f}s (within {self.start_tolerance} rad) "
                    "— check for an E-stop / fault / safety halt"
                )
            rate.sleep()
        if stop_event.is_set():
            log.info("skill %r: stopped before the start pose was reached", self.skill.name)
            return False

        log.info(
            "skill %r: start pose reached; replaying %d frames at %.1f fps (%.1fs)",
            self.skill.name, n, self.fps, n / self.fps,
        )
        end_t = (n - 1) / self.fps + self.settle_s
        t0 = time.monotonic()
        last_index = -1
        while not stop_event.is_set():
            elapsed = time.monotonic() - t0
            index = min(int(elapsed * self.fps), n - 1)
            self._post(index)
            if index != last_index:
                self.frames_posted += index - last_index
                last_index = index
            if elapsed >= end_t:
                log.info("skill %r: replay finished (%d frames)", self.skill.name, n)
                return True
            rate.sleep()
        log.info(
            "skill %r: stopped at frame %d/%d", self.skill.name, max(last_index, 0), n
        )
        return False


class SkillNode(ProcessNode):
    """Live-system consumer for ``runtime.phase=skill``: teach-and-repeat replay.

    Drives the intersection of the skill's sides and the rig's control-enabled
    ``qpos``-kind arms (exposed as :attr:`drive_sides` so the session daemon can
    hand ``EnterControl`` only to arms this node will actually feed). Also
    publishes each driven side's END configuration on the horizon stream, so the
    dashboard's 3D scene shows a purple ghost at the skill's destination for the
    whole run.
    """

    def __init__(
        self,
        skill_cfg: SkillCfg,
        runtime: RuntimeCfg,
        brain_cfg: BrainCfg,
        run_id: str,
        arms: dict[str, ArmCfg],
        skill: Skill,
    ) -> None:
        self.name = "skill"
        self.skill_cfg = skill_cfg
        self.runtime = runtime
        self.brain_cfg = brain_cfg
        self.run_id = run_id
        self.arms = arms
        self.skill = skill
        self.drive_sides = [
            side for side in skill.sides
            if side in arms and arms[side].control_enabled
            and arms[side].control.kind == "qpos"
        ]

    def run(self, stop_event) -> None:
        skipped = [s for s in self.skill.sides if s not in self.drive_sides]
        if skipped:
            log.warning(
                "skill %r: side(s) %s not driven (need a control-enabled arm with a "
                "qpos control kind on this rig)", self.skill.name, skipped,
            )
        if not self.drive_sides:
            # One-shot CLI path; the session daemon refuses such a start up front.
            raise ValueError(
                f"skill {self.skill.name!r} has no driveable side on this rig"
            )

        registry = StreamRegistry(self.runtime.runtime_dir, self.run_id)
        brain = Brain(
            registry,
            [f"{side}/q" for side in self.drive_sides],
            self.brain_cfg.attach_timeout_s,
        )
        try:
            brain.attach(stop_event=stop_event)
        except AttachAborted:
            log.info("skill attach aborted by shutdown")
            brain.close()
            return

        control_registry = StreamRegistry(self.runtime.runtime_dir, self.run_id, sub="control")
        brain.open_control(
            control_registry,
            {side: control_specs(side, self.arms[side].control) for side in self.drive_sides},
        )

        # Destination ghost: the skill's end configuration per driven side, on the
        # same stream the eval horizon uses (the dashboard mirror already poses a
        # purple ghost from it in every mode).
        horizon_writers: dict[str, StreamWriter] = {}
        for side in self.drive_sides:
            name = _HORIZON_STREAM.format(side=side)
            writer = StreamWriter.create(
                StreamSpec(
                    name=name, dim=int(self.skill.q[side].shape[1]),
                    capacity=16, dtype="float64", rate_hz=1.0,
                ),
                self.run_id,
                registry,
            )
            writer.write(np.ascontiguousarray(self.skill.q[side][-1], dtype=np.float64))
            horizon_writers[side] = writer

        try:
            loop = SkillLoop(
                brain,
                self.skill,
                {side: self.arms[side] for side in self.drive_sides},
                frequency_hz=self.skill_cfg.frequency_hz,
                start_tolerance=self.skill_cfg.start_tolerance,
                start_timeout_s=self.skill_cfg.start_timeout_s,
                settle_s=self.skill_cfg.settle_s,
            )
            loop.run(stop_event)
        finally:
            for side, writer in horizon_writers.items():
                try:
                    writer.close()
                    writer.unlink()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("error releasing horizon stream for %s", side)
                registry.remove(_HORIZON_STREAM.format(side=side))
            # Hand control back cleanly: STOP the arms (immediate exit from their
            # control session instead of riding the deadman), then unlink channels.
            brain.stop_arms()
            brain.close()
