"""The collection loop: teleoperate, command the arms, and record LeRobot frames.

:class:`CollectionLoop` is the reusable core (takes an already-attached
:class:`~dual_flexiv_control.brain.Brain`, a :class:`FrameBuilder`, and a
:class:`Recorder`) so it runs in-process in tests with fakes. :class:`CollectionNode`
wraps it in the standard spawned-process shape for the live system, building the
real Brain, the real :class:`LeRobotRecorder`, and the tty episode control.

Each tick at :attr:`CollectionCfg.frequency_hz` (default 15 Hz):

1. read the FACTR leaders (on request) + convert to Rizon joint targets,
2. post them on the setpoint channel — commanding every control-enabled arm,
3. snapshot every subscribed stream (proprio + all cameras) at one instant
   (software synchronisation — newest frame of each camera at this tick),
4. record one LeRobot frame (state + action + images).

Episodes are operator-paced (see :mod:`.keyboard`); without a tty the run
records a single episode bounded by the shared stop event.
"""

from __future__ import annotations

import logging

import numpy as np

from ..brain import Brain
from ..configs import ArmCfg
from ..configs import BrainCfg
from ..configs import CameraCfg
from ..configs import FactrCfg
from ..configs import RecordingCfg
from ..configs import RuntimeCfg
from ..configs import TaskCfg
from ..control import control_specs
from ..control import pack_streamed
from ..interfaces.factr import FactrClient
from ..interfaces.factr.client import FactrError
from ..process import ProcessNode
from ..process import RateLimiter
from ..streams.registry import AttachAborted
from ..streams.registry import StreamRegistry
from . import keyboard
from .features import FrameBuilder
from .recorder import LeRobotRecorder
from .recorder import Recorder

log = logging.getLogger(__name__)


class CollectionLoop:
    """Drives one attached Brain through teleop + recording at a fixed rate."""

    def __init__(
        self,
        brain: Brain,
        frame_builder: FrameBuilder,
        recorder: Recorder,
        command_arms: dict[str, ArmCfg],
        conventions: dict,
        frequency_hz: float = 15.0,
        num_episodes: int = 1,
        events=None,
    ) -> None:
        self.brain = brain
        self.frames = frame_builder
        self.recorder = recorder
        #: arms we actuate (post setpoints to) — the control-enabled subset. The
        #: SET of sides whose action is *recorded* is ``frame_builder.action_sides``,
        #: which is independent (we log the teleop command even for arms we don't drive).
        self.command_arms = command_arms
        self.conventions = conventions
        self.frequency_hz = frequency_hz
        self.num_episodes = max(1, int(num_episodes))
        #: callable -> list[event]; defaults to no operator events (single episode).
        self._poll = events if events is not None else (lambda: [])
        #: operator-paced (keyboard) vs single-episode fallback bounded by stop.
        self._operator_paced = events is not None
        #: last-known target per side, so a dropped leader read holds rather than
        #: stalling the whole recording (a fixed-width action must always be filled).
        self._last_qd: dict = {}
        self._last_grip: dict = {}
        #: streams that blocked the most recent skipped tick (diagnostics).
        self._last_missing: list[str] = []
        self.episodes_done = 0

    def run(self, stop_event) -> None:
        rate = RateLimiter(self.frequency_hz)
        rate.reset()
        frames_in_ep = 0
        log.info(
            "collection: %d episode(s) target at %.1f Hz; recording actions for %s, "
            "commanding %s",
            self.num_episodes, self.frequency_hz,
            self.frames.action_sides, list(self.command_arms),
        )
        ticks = 0
        recorded = 0
        recorded_at_last_beat = 0
        beat_every = max(1, int(self.frequency_hz * 2))  # ~2 s
        try:
            while not stop_event.is_set() and self.episodes_done < self.num_episodes:
                if self.tick():
                    frames_in_ep += 1
                    recorded += 1
                ticks += 1
                if ticks % beat_every == 0:
                    if recorded > recorded_at_last_beat:
                        log.info(
                            "collection: recording (%d frames this episode, %d total)",
                            frames_in_ep, recorded,
                        )
                    else:
                        # No frame recorded in the last interval — say why, loudly,
                        # so a stuck source (e.g. a camera that never grabs) is obvious.
                        log.warning(
                            "collection: NO frames recorded in the last ~2s — waiting "
                            "on stream(s) with no data yet: %s",
                            self._last_missing or "(unknown)",
                        )
                    recorded_at_last_beat = recorded
                for event in self._poll():
                    if event == keyboard.END:
                        frames_in_ep = self._save(frames_in_ep)
                    elif event == keyboard.DISCARD:
                        if frames_in_ep:
                            log.info("discarding episode (%d frames)", frames_in_ep)
                        self.recorder.discard_episode()
                        frames_in_ep = 0
                    elif event == keyboard.STOP:
                        log.info("stop requested by operator")
                        stop_event.set()
                rate.sleep()
        finally:
            # Single-episode fallback (no keyboard): the run IS one episode, so
            # save whatever accrued when it stops (duration / Ctrl-C). Operator-
            # paced runs only save on an explicit END keypress — an in-progress
            # demo interrupted by STOP / shutdown is dropped, not half-saved.
            if not self._operator_paced:
                self._save(frames_in_ep)
            elif frames_in_ep:
                log.info("dropping %d unsaved frames (no END before stop)", frames_in_ep)
                self.recorder.discard_episode()
            self.recorder.finalize()
            log.info("collection finished: %d episode(s) recorded", self.episodes_done)

    def tick(self) -> bool:
        """Read + command + record one frame. Returns True if a frame was recorded."""
        try:
            leaders = self.brain.factr_joint_positions()
        except FactrError as exc:
            log.warning("FACTR read failed; holding last targets this tick: %s", exc)
            leaders = {}
        acts = self.frames.actions_from_leaders(leaders, self.conventions)
        q_d, gripper = acts["q_d"], acts["gripper"]

        # Fill every recorded action side: fresh reading if we got one, else hold the
        # last known (or zeros before the first). A fixed-width action must always be
        # complete, so a transiently-unreachable leader can't drop the whole frame.
        for side in self.frames.action_sides:
            if side in q_d:
                self._last_qd[side] = q_d[side]
                self._last_grip[side] = gripper[side]
            else:
                q_d[side] = self._last_qd.get(side, np.zeros(self.frames.action_dof(side)))
                gripper[side] = self._last_grip.get(side, 0.0)

        # Command every control-enabled arm we have a target for (joint-position only).
        for side, arm in self.command_arms.items():
            if side in q_d and arm.control.kind == "qpos":
                setpoint = pack_streamed(
                    arm.control, {"q_d": q_d[side], "dq_d": np.zeros_like(q_d[side])}
                )
                self.brain.command(side, setpoint)

        observation = self.brain.observe()
        frame = self.frames.build(observation, q_d, gripper)
        if frame is None:
            self._last_missing = self.frames.missing_streams(observation)
            return False
        self.recorder.add_frame(frame)
        return True

    def _save(self, frames_in_ep: int) -> int:
        if frames_in_ep <= 0:
            return 0
        # save_episode() drains the video encoder — for a long episode that is
        # thousands of frames and can take many seconds. Announce it so a Ctrl-C /
        # dashboard-Stop save that is legitimately busy is not mistaken for a hang
        # (the shutdown path grants a generous window for exactly this — see
        # RuntimeCfg.save_grace_s).
        log.info("saving episode (%d frames) — finalizing video encode, please wait…",
                 frames_in_ep)
        self.recorder.save_episode()
        self.episodes_done += 1
        log.info(
            "saved episode %d/%d (%d frames)",
            self.episodes_done, self.num_episodes, frames_in_ep,
        )
        return 0


class CollectionNode(ProcessNode):
    """Live-system process node: attach, teleop + command, record to LeRobot.

    Takes the whole :class:`TaskCfg` (instruction + shared ``state_signals`` +
    the collection template) plus the run-wide :class:`RecordingCfg` (dataset
    export machinery — destination root, encoders, writer threading).
    """

    def __init__(
        self,
        task: TaskCfg,
        runtime: RuntimeCfg,
        factr: FactrCfg,
        brain: BrainCfg,
        recording: RecordingCfg,
        run_id: str,
        arms: dict[str, ArmCfg],
        cameras: dict[str, CameraCfg],
    ) -> None:
        self.name = "collection"
        self.task = task
        self.cfg = task.collection
        self.runtime = runtime
        self.factr_cfg = factr
        self.brain_cfg = brain
        self.recording = recording
        self.run_id = run_id
        self.arms = arms
        self.cameras = cameras

    def run(self, stop_event) -> None:
        # Record the teleop action for EVERY arm (each has a FACTR leader + a
        # convention); this is independent of actuation. Only control-enabled arms
        # are commanded — the rest are recorded but not driven (a warning notes it).
        record_sides = list(self.arms)
        command_sides = [s for s, arm in self.arms.items() if arm.control_enabled]
        if not command_sides:
            log.warning(
                "collection: no arm has control_enabled=true — recording teleop "
                "state + action + images, but NOT driving the arms. Enable control "
                "to actuate, e.g. arms.left.control_enabled=true.",
            )
        builder = FrameBuilder(
            self.arms, record_sides, self.cameras, self.task.language_instruction,
            self.task.state_signals, video=self.recording.video,
        )
        registry = StreamRegistry(self.runtime.runtime_dir, self.run_id)
        factr = FactrClient.from_config(self.factr_cfg, sim=self.runtime.sim)
        brain = Brain(
            registry, builder.stream_names, self.brain_cfg.attach_timeout_s, factr=factr
        )
        try:
            brain.attach(stop_event=stop_event)
        except AttachAborted:
            log.info("collection attach aborted by shutdown")
            brain.close()
            return

        control_registry = StreamRegistry(
            self.runtime.runtime_dir, self.run_id, sub="control"
        )
        specs_by_side = {
            side: control_specs(side, self.arms[side].control) for side in command_sides
        }
        brain.open_control(control_registry, specs_by_side)
        command_arms = {side: self.arms[side] for side in command_sides}
        conventions = {side: self.arms[side].convention for side in record_sides}

        # LeRobot recorder — raises RecorderUnavailable if lerobot is not installed
        # (run_node catches, logs, and brings the system down with a clear message).
        recorder = LeRobotRecorder(self.cfg, self.recording, builder.features())

        try:
            with keyboard.EpisodeControl() as keys:
                events = keys.poll if (keys.active and self.recording.keyboard) else None
                loop = CollectionLoop(
                    brain, builder, recorder, command_arms, conventions,
                    frequency_hz=self.cfg.frequency_hz,
                    num_episodes=self.cfg.num_episodes,
                    events=events,
                )
                loop.run(stop_event)
        finally:
            brain.close()
