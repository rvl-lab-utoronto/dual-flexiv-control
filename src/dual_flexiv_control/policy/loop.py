"""The eval loop: observe, query the policy, and drive the arms.

:class:`EvalLoop` is the reusable core (takes an already-attached
:class:`~dual_flexiv_control.brain.Brain`, an :class:`ObservationBuilder`, a
:class:`~.client.Policy`, and an :class:`~.actions.ActionLayout`) so it runs
in-process in tests with fakes. :class:`EvalNode` wraps it in the standard
spawned-process shape for the live system.

Each tick at :attr:`EvalCfg.frequency_hz` (one policy *timestep*):

1. snapshot every subscribed stream (proprio + all cameras) at one instant
   (software synchronisation, same as collection),
2. if the pending action chunk is empty, build the canonical observation and
   query the policy for a fresh chunk (keeping the first ``replan_steps``
   actions — receding horizon; 0 keeps the whole chunk),
3. pop one action, split it per side, and post joint-position setpoints on the
   control channel.

A rollout has no operator, so it is bounded by :attr:`EvalCfg.num_timesteps`
(plus the shared stop event). Incomplete observations and transient policy
failures *hold* (no setpoint posted, timestep not counted) — the arms' deadman
then parks them at the last target, exactly as during a teleop dropout.

For visualization, the node also publishes ``eval/<side>/q_horizon`` — each
controlled arm's joint target at the END of the policy's returned chunk,
refreshed once per inference. The dashboard's metrics viewer reads it (read-only
shared memory, like ``<side>/q``) to pose a purple horizon-target ghost and
trace current-EEF → horizon-EEF in the 3D robot scene.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ..brain import Brain
from ..configs import ArmCfg
from ..configs import BrainCfg
from ..configs import CameraCfg
from ..configs import PolicyCfg
from ..configs import RuntimeCfg
from ..configs import TaskCfg
from ..control import action_hold_fields
from ..control import control_specs
from ..control import pack_action
from ..process import ProcessNode
from ..process import RateLimiter
from ..streams.registry import AttachAborted
from ..streams.registry import StreamRegistry
from ..streams.spec import StreamSpec
from ..streams.stream import StreamWriter
from .actions import ActionLayout
from .client import Policy
from .client import PolicyError
from .client import build_policy
from .observation import ObservationBuilder

log = logging.getLogger(__name__)

#: Proprio signal that supplies each held (absolute, non-primary) control field —
#: e.g. under ``force`` control the motion-axis ``pose_d`` is held at the measured
#: TCP pose (the arm's ``eef`` stream). Only fields in
#: :data:`~dual_flexiv_control.control.channel._ABSOLUTE_FIELDS` appear here.
_HOLD_SIGNAL = {"pose_d": "eef", "q_d": "q"}


def hold_stream_names(side: str, ctrl_cfg) -> list[str]:
    """Proprio streams the eval loop must read to hold ``side``'s absolute fields."""
    return [f"{side}/{_HOLD_SIGNAL[f]}" for f in action_hold_fields(ctrl_cfg)]


def horizon_stream_name(side: str) -> str:
    """Stream carrying ``side``'s joint target at the end of the policy horizon."""
    return f"eval/{side}/q_horizon"


class EvalLoop:
    """Drives one attached Brain through a fixed-horizon policy rollout."""

    def __init__(
        self,
        brain: Brain,
        observer: ObservationBuilder,
        policy: Policy,
        layout: ActionLayout,
        control_arms: dict[str, ArmCfg],
        frequency_hz: float = 15.0,
        num_timesteps: int = 1,
        replan_steps: int = 0,
        on_chunk=None,
    ) -> None:
        self.brain = brain
        self.observer = observer
        self.policy = policy
        self.layout = layout
        self.control_arms = control_arms
        self.frequency_hz = frequency_hz
        self.num_timesteps = max(1, int(num_timesteps))
        self.replan_steps = int(replan_steps)
        #: callable ``{side: q_d at the END of the returned chunk} -> None``, invoked
        #: once per inference with the policy's horizon-end joint targets (viz hook —
        #: EvalNode publishes them on ``eval/<side>/q_horizon``). Never fatal.
        self._on_chunk = on_chunk
        self.timesteps_done = 0
        self.inferences = 0
        self._pending: deque[np.ndarray] = deque()
        self._warned_kinds: set[str] = set()
        self._holds = 0  # consecutive held ticks, for ~1 Hz hold diagnostics
        #: streamed absolute fields to hold at the measured value, per side (e.g.
        #: force's pose_d), and the latest measured values captured at inference.
        self._hold = {s: action_hold_fields(a.control) for s, a in control_arms.items()}
        self._held: dict[str, dict] = {}

    def run(self, stop_event) -> None:
        rate = RateLimiter(self.frequency_hz)
        rate.reset()
        log.info(
            "eval rollout: %d timesteps at %.1f Hz; control sides=%s; replan_steps=%s",
            self.num_timesteps, self.frequency_hz, list(self.control_arms),
            self.replan_steps or "full chunk",
        )
        while not stop_event.is_set() and self.timesteps_done < self.num_timesteps:
            if self.tick():
                self.timesteps_done += 1
            rate.sleep()
        log.info(
            "eval finished: %d/%d timesteps executed (%d inference(s))",
            self.timesteps_done, self.num_timesteps, self.inferences,
        )

    def tick(self) -> bool:
        """One timestep: (re)infer if needed, execute one action. False = held."""
        if not self._pending:
            snapshot = self.brain.observe()
            observation = self.observer.build(snapshot)
            if observation is None:
                # A stream is not warm yet (or a camera stalled): hold. Say which
                # streams are dry roughly once a second so a stall is diagnosable.
                self._holds += 1
                if self._holds % max(1, int(self.frequency_hz)) == 1:
                    log.warning(
                        "observation incomplete; holding (missing: %s)",
                        self.observer.missing(snapshot),
                    )
                return False
            self._capture_holds(snapshot)
            try:
                chunk = self.policy.infer(observation)
            except PolicyError as exc:
                log.warning("policy inference failed; holding this tick: %s", exc)
                return False
            self.inferences += 1
            chunk = self._validate_chunk(chunk)
            self._announce_horizon(chunk)
            # Receding horizon: execute only the chunk prefix before re-inferring
            # (0 = the whole chunk, open-loop within it).
            self._pending.extend(chunk[: self.replan_steps] if self.replan_steps > 0 else chunk)
        self._holds = 0
        self._execute(self._pending.popleft())
        return True

    def _validate_chunk(self, chunk) -> np.ndarray:
        """Coerce a returned chunk to ``(horizon, action_dim)``.

        A wrong ``action_dim`` is a config/checkpoint mismatch, not a transient
        fault — raise (bringing the run down) instead of holding forever.
        """
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != self.layout.dim:
            raise ValueError(
                f"policy returned actions of shape {arr.shape}; expected "
                f"(horizon, {self.layout.dim}) for layout {self.layout.names}"
            )
        return arr

    def _announce_horizon(self, chunk: np.ndarray) -> None:
        """Hand the horizon-END joint targets (the FULL chunk's last action — the
        policy's intent, even when ``replan_steps`` re-infers earlier) to the viz
        hook. Only sides whose action space is joint-position (``q_d``) yield a joint
        horizon target; other kinds (velocity/cartesian) have none, so they're
        skipped. Purely observational: a failure must never disturb the rollout."""
        if self._on_chunk is None:
            return
        try:
            targets = self.layout.split(chunk[-1])
            q_targets = {side: t["q_d"] for side, t in targets.items() if "q_d" in t}
            if q_targets:
                self._on_chunk(q_targets)
        except Exception:  # noqa: BLE001 - viz hook only
            log.exception("horizon viz hook failed (rollout unaffected)")

    def _capture_holds(self, snapshot: dict) -> None:
        """Latch measured values for any absolute non-primary fields (e.g. force's
        ``pose_d``). Held across the chunk; if a source stream is momentarily dry we
        keep the last latch (``_execute`` holds the command if none was ever taken)."""
        for side, fields in self._hold.items():
            for f in fields:
                samples = snapshot.get(f"{side}/{_HOLD_SIGNAL[f]}")
                newest = samples.newest if samples is not None else None
                if newest is not None:
                    self._held.setdefault(side, {})[f] = np.asarray(newest, dtype=np.float64)

    def _execute(self, action: np.ndarray) -> None:
        """Post one action's setpoints for each arm's control kind.

        The policy's per-side primary field (``q_d``/``dq_d``/``pose_d``/``twist_d``/
        ``wrench_d``, per the arm's kind) is packed into a full setpoint by
        :func:`~dual_flexiv_control.control.pack_action` — feedforward fields zeroed,
        absolute non-primary fields held at the latched measured value. Gripper is
        decoded in the layout but not yet actuated.
        """
        targets = self.layout.split(action)
        for side, arm in self.control_arms.items():
            ctrl = arm.control
            primary = targets[side][self.layout.field(side)]
            try:
                setpoint = pack_action(ctrl, primary, self._held.get(side))
            except ValueError as exc:
                if side not in self._warned_kinds:
                    self._warned_kinds.add(side)
                    log.warning("[%s] cannot build %s setpoint (%s); holding",
                                side, ctrl.kind, exc)
                continue
            self.brain.command(side, setpoint)


class EvalNode(ProcessNode):
    """Live-system process node for ``runtime.phase=eval``: policy-driven rollout.

    Takes the whole :class:`TaskCfg` because the eval observation reuses the
    task's shared schema (``state_signals`` + instruction): what the policy sees
    online must match what it was trained on.
    """

    def __init__(
        self,
        task: TaskCfg,
        runtime: RuntimeCfg,
        policy: PolicyCfg,
        brain: BrainCfg,
        run_id: str,
        arms: dict[str, ArmCfg],
        cameras: dict[str, CameraCfg],
    ) -> None:
        self.name = "eval"
        self.task = task
        self.runtime = runtime
        self.policy_cfg = policy
        self.brain_cfg = brain
        self.run_id = run_id
        self.arms = arms
        self.cameras = cameras

    def run(self, stop_event) -> None:
        control_sides = [s for s, arm in self.arms.items() if arm.control_enabled]
        if not control_sides:
            log.warning(
                "eval: no arm has control_enabled=true — the policy will be queried "
                "but its actions will not drive any arm. Enable control, e.g. "
                "arms.left.control_enabled=true."
            )
        observer = ObservationBuilder(
            self.arms, self.cameras,
            self.task.language_instruction, self.task.state_signals,
        )
        layout = ActionLayout(self.arms, control_sides)

        # Some control kinds hold an absolute field at the measured value (force ->
        # pose_d from the arm's eef). Subscribe those proprio streams too, so the
        # loop can latch them; the observation vector itself is unchanged (the
        # observer only reads its own state_signals).
        hold_streams = [
            name for side in control_sides
            for name in hold_stream_names(side, self.arms[side].control)
        ]
        subscribe = list(dict.fromkeys([*observer.stream_names, *hold_streams]))

        registry = StreamRegistry(self.runtime.runtime_dir, self.run_id)
        brain = Brain(registry, subscribe, self.brain_cfg.attach_timeout_s)
        try:
            brain.attach(stop_event=stop_event)
        except AttachAborted:
            log.info("eval attach aborted by shutdown")
            brain.close()
            return

        control_registry = StreamRegistry(
            self.runtime.runtime_dir, self.run_id, sub="control"
        )
        specs_by_side = {
            side: control_specs(side, self.arms[side].control) for side in control_sides
        }
        brain.open_control(control_registry, specs_by_side)
        control_arms = {side: self.arms[side] for side in control_sides}

        # Horizon-target streams (viz): one per controlled arm, refreshed once per
        # inference with the chunk-end joint target. Owned by this node (created +
        # unlinked here), read by the dashboard's 3D robot scene.
        horizon_writers = {
            side: StreamWriter.create(
                StreamSpec(
                    name=horizon_stream_name(side),
                    dim=int(self.arms[side].dof),
                    capacity=64,
                    dtype="float64",
                    rate_hz=self.task.eval.frequency_hz,
                ),
                self.run_id,
                registry,
            )
            for side in control_sides
        }

        def publish_horizon(targets: dict[str, np.ndarray]) -> None:
            for side, q_d in targets.items():
                writer = horizon_writers.get(side)
                if writer is not None:
                    writer.write(np.ascontiguousarray(q_d, dtype=np.float64))

        policy = None
        try:
            # May block up to policy.connect_timeout_s waiting for the server;
            # raises PolicyUnavailable (bringing the system down with a clear
            # message) if it never appears.
            policy = build_policy(self.policy_cfg, layout, observer, stop_event=stop_event)
            loop = EvalLoop(
                brain, observer, policy, layout, control_arms,
                frequency_hz=self.task.eval.frequency_hz,
                num_timesteps=self.task.eval.num_timesteps,
                replan_steps=self.policy_cfg.replan_steps,
                on_chunk=publish_horizon if horizon_writers else None,
            )
            loop.run(stop_event)
        finally:
            if policy is not None:
                policy.close()
            for side, writer in horizon_writers.items():
                try:
                    writer.close()
                    writer.unlink()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("error releasing horizon stream for %s", side)
                registry.remove(horizon_stream_name(side))
            brain.close()
