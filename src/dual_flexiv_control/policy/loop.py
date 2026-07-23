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

For visualization, the node also publishes each arm's *estimated chunk-end
state* — where the policy's returned chunk is predicted to land the arm —
refreshed once per inference (see
:func:`~dual_flexiv_control.control.estimate_chunk_end` for the per-kind
estimate). Joint kinds (``qpos``/``qvel``) yield a joint config on
``eval/<side>/q_horizon``; cartesian kinds (``end_effector``/``eef_vel``) yield
a base-frame TCP position on ``eval/<side>/eef_horizon``; ``force`` predicts no
motion. The dashboard's metrics viewer reads these (read-only shared memory,
like ``<side>/q``) to pose a purple horizon-target ghost (joint kinds) and
trace current-EEF → predicted-EEF in the 3D robot scene. Visualization does not
require actuation: with no control-enabled arm the rollout is a *dry run* — the
policy is still queried over the full action space and its predictions are
published, but no setpoints are posted.
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
from ..control import estimate_chunk_end
from ..control import horizon_kind
from ..control import horizon_signals
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


def eef_horizon_stream_name(side: str) -> str:
    """Stream carrying ``side``'s estimated base-frame TCP position ``[x y z]`` at
    the end of the policy horizon (cartesian control kinds, which have no joint
    target to pose a ghost from)."""
    return f"eval/{side}/eef_horizon"


def comm_stream_name() -> str:
    """Stream of policy-server comm events: ``[kind, seq, elapsed_s]`` per event.

    ``kind`` is a :mod:`~.client` comm constant (``COMM_SENT`` / ``COMM_RECV`` /
    ``COMM_ERROR``), ``seq`` the 1-based inference request it belongs to, and
    ``elapsed_s`` the round-trip (or time-to-failure) — 0 on the SENT event. The
    ring's own per-sample timestamps carry *when* each packet left / arrived;
    the dashboard mirror reads them to plot send/receive activity live.
    """
    return "eval/policy_comm"


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
        max_consecutive_errors: int = 3,
        on_chunk=None,
        horizon_arms: dict[str, ArmCfg] | None = None,
    ) -> None:
        self.brain = brain
        self.observer = observer
        self.policy = policy
        self.layout = layout
        self.control_arms = control_arms
        self.frequency_hz = frequency_hz
        self.num_timesteps = max(1, int(num_timesteps))
        self.replan_steps = int(replan_steps)
        self.max_consecutive_errors = max(1, int(max_consecutive_errors))
        #: callable ``{side: ("q"|"eef", vector)} -> None``, invoked once per
        #: inference with each arm's estimated chunk-end state (viz hook — EvalNode
        #: publishes them on ``eval/<side>/{q,eef}_horizon``). Never fatal.
        self._on_chunk = on_chunk
        #: arms the horizon estimate covers (viz): every layout side, not just the
        #: driven ones — a dry run still predicts. Defaults to ``control_arms``.
        self.horizon_arms = horizon_arms if horizon_arms is not None else control_arms
        self.timesteps_done = 0
        self.inferences = 0
        self._pending: deque[np.ndarray] = deque()
        self._warned_kinds: set[str] = set()
        self._holds = 0  # consecutive held ticks, for ~1 Hz hold diagnostics
        self._consecutive_policy_errors = 0
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
                self._consecutive_policy_errors += 1
                if self._consecutive_policy_errors >= self.max_consecutive_errors:
                    raise PolicyError(
                        "policy inference failed "
                        f"{self._consecutive_policy_errors} consecutive times; "
                        f"aborting eval. Last error:\n{exc}"
                    ) from exc
                log.warning(
                    "policy inference failed (%d/%d); holding this tick: %s",
                    self._consecutive_policy_errors,
                    self.max_consecutive_errors,
                    exc,
                )
                return False
            self._consecutive_policy_errors = 0
            self.inferences += 1
            chunk = self._validate_chunk(chunk)
            self._announce_horizon(chunk, snapshot)
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

    def _announce_horizon(self, chunk: np.ndarray, snapshot: dict) -> None:
        """Hand each arm's estimated chunk-END state to the viz hook.

        The estimate covers the FULL chunk (the policy's intent, even when
        ``replan_steps`` re-infers earlier), per that side's control kind (see
        :func:`~dual_flexiv_control.control.estimate_chunk_end`): joint kinds yield
        ``("q", joint config)``, cartesian kinds ``("eef", TCP position)``; velocity
        kinds integrate from the measured baseline in ``snapshot`` (side skipped if
        that stream is dry), ``force`` predicts nothing. Purely observational: a
        failure must never disturb the rollout."""
        if self._on_chunk is None:
            return
        try:
            dt = 1.0 / self.frequency_hz
            targets: dict[str, tuple[str, np.ndarray]] = {}
            for side in self.layout.sides:
                arm = self.horizon_arms.get(side)
                if arm is None:
                    continue
                measured = {}
                for sig in horizon_signals(arm.control):
                    samples = snapshot.get(f"{side}/{sig}")
                    newest = samples.newest if samples is not None else None
                    if newest is not None:
                        measured[sig] = np.asarray(newest, dtype=np.float64)
                est = estimate_chunk_end(
                    arm.control, chunk[:, self.layout.primary_slice(side)], dt, measured
                )
                if est is not None:
                    targets[side] = est
            if targets:
                self._on_chunk(targets)
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
        # The action layout (what the policy emits) is decoupled from actuation:
        # with no control-enabled arm the rollout is a dry run over ALL arms —
        # the policy is queried and its horizon predictions visualized, but no
        # setpoint is posted.
        action_sides = control_sides or list(self.arms)
        if not control_sides:
            log.warning(
                "eval: no arm has control_enabled=true — dry run: the policy is "
                "queried over all arms' action space (%s) and its predictions are "
                "visualized, but no arm is driven. Enable control, e.g. "
                "arms.left.control_enabled=true.",
                sorted(self.arms),
            )
        observer = ObservationBuilder(
            self.arms, self.cameras,
            self.task.language_instruction, self.task.state_signals,
        )
        layout = ActionLayout(self.arms, action_sides)
        horizon_arms = {side: self.arms[side] for side in layout.sides}

        # Some control kinds hold an absolute field at the measured value (force ->
        # pose_d from the arm's eef). Subscribe those proprio streams too, so the
        # loop can latch them; the observation vector itself is unchanged (the
        # observer only reads its own state_signals). Likewise the measured
        # baselines the horizon estimate integrates velocity kinds from (viz).
        hold_streams = [
            name for side in control_sides
            for name in hold_stream_names(side, self.arms[side].control)
        ]
        viz_streams = [
            f"{side}/{sig}" for side in layout.sides
            for sig in horizon_signals(self.arms[side].control)
        ]
        subscribe = list(dict.fromkeys([*observer.stream_names, *hold_streams, *viz_streams]))

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

        # Horizon-target streams (viz): one per layout arm, refreshed once per
        # inference with the estimated chunk-end state — a joint config
        # (``q_horizon``, joint kinds) or a base-frame TCP position
        # (``eef_horizon``, cartesian kinds); ``force`` predicts nothing. Owned by
        # this node (created + unlinked here), read by the dashboard's 3D scene.
        horizon_writers: dict[str, tuple[str, StreamWriter, str]] = {}
        for side in layout.sides:
            hk = horizon_kind(self.arms[side].control)
            if hk is None:
                continue
            name = horizon_stream_name(side) if hk == "q" else eef_horizon_stream_name(side)
            writer = StreamWriter.create(
                StreamSpec(
                    name=name,
                    dim=int(self.arms[side].dof) if hk == "q" else 3,
                    capacity=64,
                    dtype="float64",
                    rate_hz=self.task.eval.frequency_hz,
                ),
                self.run_id,
                registry,
            )
            horizon_writers[side] = (hk, writer, name)

        def publish_horizon(targets: dict[str, tuple[str, np.ndarray]]) -> None:
            for side, (hk, vec) in targets.items():
                entry = horizon_writers.get(side)
                if entry is not None and entry[0] == hk:
                    entry[1].write(np.ascontiguousarray(vec, dtype=np.float64))

        # Policy-server comm events (viz): one sample per packet sent / received /
        # failed, written by RemotePolicy's on_comm hook. Owned by this node like
        # the horizon streams; the dashboard mirror plots them below the robot
        # metrics. Capacity is generous — events come at most a few per second.
        comm_writer = StreamWriter.create(
            StreamSpec(
                name=comm_stream_name(), dim=3, capacity=512, dtype="float64",
                rate_hz=self.task.eval.frequency_hz,
            ),
            self.run_id,
            registry,
        )

        def publish_comm(kind: float, seq: int, elapsed_s: float) -> None:
            comm_writer.write(np.array([kind, float(seq), elapsed_s], dtype=np.float64))

        policy = None
        try:
            # May block up to policy.connect_timeout_s waiting for the server;
            # raises PolicyUnavailable (bringing the system down with a clear
            # message) if it never appears.
            policy = build_policy(
                self.policy_cfg, layout, observer, stop_event=stop_event,
                on_comm=publish_comm,
            )
            loop = EvalLoop(
                brain, observer, policy, layout, control_arms,
                frequency_hz=self.task.eval.frequency_hz,
                num_timesteps=self.task.eval.num_timesteps,
                replan_steps=self.policy_cfg.replan_steps,
                max_consecutive_errors=self.policy_cfg.max_consecutive_errors,
                on_chunk=publish_horizon if horizon_writers else None,
                horizon_arms=horizon_arms,
            )
            loop.run(stop_event)
        finally:
            if policy is not None:
                policy.close()
            try:
                comm_writer.close()
                comm_writer.unlink()
            except Exception:  # noqa: BLE001 - teardown must not raise
                log.exception("error releasing the policy comm stream")
            registry.remove(comm_stream_name())
            for side, (_hk, writer, name) in horizon_writers.items():
                try:
                    writer.close()
                    writer.unlink()
                except Exception:  # noqa: BLE001 - teardown must not raise
                    log.exception("error releasing horizon stream for %s", side)
                registry.remove(name)
            # Hand control back cleanly: STOP the arms (they exit their control
            # session at once instead of riding the deadman), then unlink channels.
            brain.stop_arms()
            brain.close()
