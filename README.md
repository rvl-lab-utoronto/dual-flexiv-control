# dual-flexiv-control

Control, visualization, data collection, and online policy evaluation for the RVL **bimanual Flexiv**. Four
kinds of component, decoupled by zero-copy shared-memory data streams so each runs in its
own process at its own rate:

1. **The brain** (`brain/`) — the main processing pipeline. Subscribes to data
   streams, observes them, and pulls the last `k` elements of any signal.
2. **The Flexiv interface** (`interfaces/flexiv/`) — wraps **real `flexivrdk`**
   (RDK 1.8), one process per arm. Read-only by default (proprio streams only);
   with `arm.control_enabled=true` the same process also **consumes the brain's
   control channel and actuates** (the single robot connection per arm forces the
   control loop to live here).
4. **The control channel** (`control/`) — a **second IPC category**, brain→arm
   (the inverse of the telemetry streams). The brain posts targets to a latest-wins
   **setpoint** mailbox and discrete events to a reliable **command** queue; the arm
   consumes them. Reuses the same shared-memory ring under `<run>/control/`.
3. **The ZED camera interface** (`interfaces/zed/`) — wraps the **ZED SDK
   (`pyzed`)**, one process per camera, publishing image frames as streams: two
   **ZED X Nano** wrist cameras (one per arm) and one static external **ZED 2**
   stereo camera.
4. **The FACTR interface** (`interfaces/factr/`) — consumes each leader's WebSocket and
   publishes its raw Dynamixel reading, converted DFC/Rizon pose, and explicit
   per-field live telemetry streams. The
   socket is duplex: `FactrClient.send_force_feedback` pushes follower external
   joint torques back up it (joint-space `force_feedback` frames) for the leader's
   force-feedback term. At that boundary, RDK's environment-on-follower `tau_ext`
   is negated into FACTR's follower-on-environment convention; FACTR's original
   feedback equation negates it again, preserving the reflected torque direction.

## Architecture

```
 FlexivInterface(LEFT/RIGHT)  process ─┐  shared-memory ring buffers  ┌─ BrainNode  process
 ZedInterface(wrist_l/wrist_r/static)  ─┤      (one per signal)        └─  reads last k
 FactrInterface(leader WebSockets)      ─┘
```

* **Streams** are single-producer / multi-consumer ring buffers in POSIX shared
  memory (`/dev/shm`). One producer (the owning interface) appends; any number of
  consumers attach by name and read the newest `k` samples — no copies, no IPC
  round-trips. Coordination is a lock-free **seqlock** (a published global write
  count + per-slot sequence stamps); readers re-validate each slot so a buffer
  that laps a slow reader degrades to "freshest valid suffix", never corruption.
* **One stream per signal, per arm.** Each arm publishes `q`, `dq`, `tau`,
  `tau_ext`, `wrench`, `eef`, `eef_vel` as separate streams named `left/…`
  and `right/…`.
* **One stream per camera view.** Each ZED camera publishes one stream per
  *view* — `left`/`right` RGB (`uint8`, `H×W×3`) and optional `depth` (`float32`,
  `H×W`, metres) — named `cam/<camera>/<view>` (e.g. `cam/wrist_left/left`,
  `cam/static/left`, `cam/static/right`). A frame is stored **flattened** to a
  fixed-dimension `(H*W*C,)` vector (the ring carries fixed-dimension vectors);
  consumers reshape it back with `cameras.reshape_frame`. The image dtypes
  (`uint8`/`uint16`) are the only addition to the otherwise float-only ring.
* **Multiprocess, spawned (never forked).** flexivrdk runs live threads/services;
  forking would corrupt them, so every node is a fresh interpreter that opens its
  own connection. Discovery is a directory of tiny JSON manifests per run.

### Proprio → Flexiv RDK 1.8 mapping

| Stream            | dim | RDK 1.8 `RobotStates` field            |
|-------------------|-----|----------------------------------------|
| `<side>/q`        | 7   | `q` (link-side joint positions)        |
| `<side>/dq`       | 7   | `dq` (link-side joint velocities)      |
| `<side>/tau`      | 7   | `tau` (measured joint torques)         |
| `<side>/tau_ext`  | 7   | `tau_ext` (estimated external joint torques) |
| `<side>/wrench`   | 6   | `ext_wrench_in_tcp` (TCP frame) — or `ext_wrench_in_world` (world) via `--wrench-frame` |
| `<side>/eef`      | 7   | `tcp_pose` `[x,y,z,qw,qx,qy,qz]`       |
| `<side>/eef_vel`  | 6   | `tcp_vel` `[v(3), ω(3)]`               |

### Cameras → ZED SDK mapping

One `ZedInterface` process per camera; `dim` is **derived** from the configured
resolution (`width*height*channels`), never hand-set. The default rig:

| Camera        | Model      | Streams (views)                  | view → ZED source            | dtype / shape         |
|---------------|------------|----------------------------------|------------------------------|-----------------------|
| `wrist_left`  | ZED X Nano | `cam/wrist_left/left`            | `VIEW.LEFT` (RGB)            | `uint8` `(H,W,3)`     |
| `wrist_right` | ZED X Nano | `cam/wrist_right/left`           | `VIEW.LEFT` (RGB)            | `uint8` `(H,W,3)`     |
| `static`      | ZED 2      | `cam/static/left`, `…/right`     | `VIEW.LEFT` / `VIEW.RIGHT`   | `uint8` `(H,W,3)`     |
| *(optional)*  | —          | `cam/<cam>/depth`               | `MEASURE.DEPTH` (metres)     | `float32` `(H,W)`     |

Add `right`/`depth` to any camera's `views` to publish more (depth also needs a
`depth_mode`). The **ZED SDK 4.x / `pyzed`** is required only for real cameras
(installed separately via Stereolabs' `get_python_api.py`, not from PyPI — hence
not a project dependency); with `runtime.sim=true` a `FakeZedSource` synthesises
animated frames so the whole pipeline runs hardware-free, exactly like the Flexiv
sim source.

## Environment

Python is pinned to **3.12** (newest CPython with a published `flexivrdk`
manylinux x86_64 wheel and the most mature ecosystem). A dedicated conda env:

```bash
conda env create -f environment.yml          # creates env "dual-flexiv-control"
conda activate dual-flexiv-control
pip install -e ".[dev]"                       # editable install + pytest
```

## Configuration (Hydra)

Configuration is composed by [Hydra](https://hydra.cc) from
[`conf/`](src/dual_flexiv_control/conf) and validated against the structured schema
in [configs.py](src/dual_flexiv_control/configs.py). Three orthogonal axes select a
run — **rig** (what hardware exists), **task** (what is demonstrated/evaluated),
and **phase** (`runtime.phase=collection|eval`):

```
conf/
  config.yaml              # tiny: composes the groups below (rig=bimanual, task=default, …)
  rig/                     # WHAT HARDWARE EXISTS (select with `rig=<name>`):
    bimanual.yaml          #   both arms + 3 ZEDs + both FACTR leaders (default)
    left_only.yaml         #   left arm + static ZED + left leader
    bench.yaml             #   dummy left arm + the one real ZED (bring-up/testing)
  task/                    # WHAT IS DEMONSTRATED (select with `task=<name>`):
    default.yaml  handover.yaml    # instruction + dataset (repo_id) + phase counts
  recording/default.yaml   # dataset-export machinery (root, encoders, writer threads)
  runtime/default.yaml     # sim, runtime_dir, duration_s, phase, save_grace_s
  brain/default.yaml       # rate, attach timeout, subscriptions
  factr/                   # FACTR server sets (selected by the rig): bimanual / left
  arm/                     # arm templates: flexiv (real), flexiv_dummy (fabricated)
  camera/                  # camera templates: zedx_wrist, zed2_static
  control/                 # control-type library (command schemas, SDK-aligned):
    qpos.yaml  qvel.yaml  end_effector.yaml  eef_vel.yaml  force.yaml
```

A **rig** file carries the hardware composition — which `arm@arms.<side>` /
`camera@cameras.<name>` / `factr` entries exist, their serials and display names —
via Hydra's [experiment pattern](https://hydra.cc/docs/patterns/configuring_experiments/)
(`# @package _global_` + absolute package-directed defaults). Bring-up on a bench
with no robots is one override away:

```bash
dual-flexiv-control rig=bench runtime.phase=collection   # dummy arm + real ZED,
                                                         # records to datasets/bench/
```

Cameras compose just like arms: `camera@cameras.<name>: <template>` places a
template at `cameras.<name>`, with per-camera `serial`/`placement` set in the rig
file. Override resolution/fps/views from the CLI, e.g.
`cameras.static.resolution=HD1080 cameras.static.width=1920 cameras.static.height=1080`
or `'+cameras.static.views=[left,right,depth]' cameras.static.depth_mode=NEURAL`.
Camera streams are produced unconditionally but are **not** in the brain's default
subscription (proprio only); subscribe to them explicitly, e.g.
`'brain.subscribe=[left/q, cam/static/left]'`.

Each **task** carries the `language_instruction` and `state_signals` shared by
both phases, its LeRobot dataset (`collection.repo_id` — required, so tasks never
silently share one), plus the counts unique to each phase —
`collection.num_episodes` (demos to teleoperate) and `eval.num_timesteps` (rollout
horizon). Add a task by copying `task/default.yaml`; select it with `task=<name>`
and tune fields inline, e.g. `task=handover task.eval.num_timesteps=800`.

Each stream's schema (`dim`, `dtype`, `capacity`, `rate_hz`) lives under its path,
e.g. `arms.left.streams.tau` → stream `left/tau`. The **control configs** lay out
each control kind's command schema; `streamed` lists which fields the brain posts
per tick (the rest are static limits from the coeffs). All paths are **NRT**
(verified against flexivrdk 1.8) — the brain posts setpoints over IPC at
~50-200 Hz (RDK 1.8 is NRT-only — no hard-1 kHz RT modes, no `Stream*` methods):

| Controller | RDK mode | send fn | streamed (per-tick) |
|---|---|---|---|
| `qpos` | `NRT_JOINT_POSITION` | `SendJointPosition` | `q_d`, `dq_d` |
| `qpos_impedance` | `NRT_JOINT_IMPEDANCE` | `SendJointPosition` | `q_d`, `dq_d` |
| `qpos_overdamped` | `NRT_JOINT_IMPEDANCE` | `SendJointPosition` | `q_d`, `dq_d` (overdamped) |
| `qvel` | `NRT_JOINT_POSITION` | `SendJointPosition` | `dq_d` (arm integrates `q_d`) |
| `end_effector` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `pose_d`, `twist_d` |
| `eef_vel` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `twist_d` (arm integrates `pose_d`) |
| `force` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `wrench_d`, `pose_d` |

Each policy imports its action/controller semantics under `policy.control`, and
each arm in the selected rig uses that controller for the run. Tasks contain the
instruction, observation schema, dataset identity, and phase counts only. Collection
still records its complete configured frame and teleop action regardless of the
selected policy controller. ``qpos_impedance`` is the neutral absolute-joint preset;
plain ``qpos`` leaves stiffness to Flexiv's SDK, while ``qpos_overdamped`` and
Cartesian controllers carry their own impedance blocks.

**Policy boundary.** DFC owns the canonical observation and action vectors.
`layout.py` derives state/action names, dimensions, and per-arm slices from the
selected rig and controller; collection and evaluation both reuse it. An
endpoint adapter only implements an endpoint family's protocol. For OpenPI this
means request key placement, image nesting/layout,
prompt placement, and response parsing. Any checkpoint-specific embodiment
projection is declared in its policy YAML as named `source` → `target` index
copies, constants, and measured-state holds. See
`conf/policy/pi05_aloha.yaml`: the omitted Flexiv joints, ALOHA gripper slots,
camera mapping, and the complete 14-D → 16-D return mapping are readable there;
there is no ALOHA-specific Python adapter.

**Controller coefficients** (motion/contact limits plus a joint-stiffness scale)
default per phase from the schema. The named presets
(`compliant`/`stiff`/`default`/`very_compliant`) are registered in
[configs.py](src/dual_flexiv_control/configs.py) (single source of truth — no YAML
files); swap one per phase with an appended group override, e.g.
`'+control_coeffs@task.eval.coeffs=compliant'`, or tune fields directly
(`task.eval.coeffs.max_joint_vel=2.0`). `runtime.phase` (`collection`|`eval`)
selects which set the arms apply. The arm applies, after `SwitchMode`, only the
limits its mode accepts.

### Teleoperation (FACTR → follower)

The convention boundary is intentionally at the FACTR interface. `factr/raw/<side>`
is the untouched hardware payload; it is retained for diagnostics and datasets as
`observation.factr_raw.<side>`. `factr/<side>` is converted once into the canonical
DFC/Rizon convention and is what the viewer, brain, and
`observation.factr.<side>` use. All calibration lives in DFC under
`conf/factr/<group>.yaml` → `leaders.<side>`: `raw_to_dfc`, canonical
`home_q_rad`, and `dfc_to_factr`. The Calibration tab loads that YAML object,
updates `raw_to_dfc`, and atomically saves it. For managed FACTR processes, the
DFC supervisor injects the leader object at launch; FACTR derives its private
inverse-dynamics model calibration rather than storing a second copy. Only the
DFC→FACTR axis signs are persisted; the affine zero offset is derived from
`home_q_rad` and FACTR's authoritative `model_home_q_rad` at every launch.

Each leader WebSocket carries two outbound frame types: `reading` (raw encoder
packet) and `telemetry` (live FACTR model/control state). DFC publishes telemetry
as individual `factr/telemetry/<side>/<field>` streams: raw/model position,
model velocity, home error, model offsets/signs, limit/null/gravity/friction/
force-feedback/applied torque, and the gravity/feedback gain values and targets.

With `arm.control_enabled=true`, the brain posts the already-converted FACTR pose as
the Rizon qpos setpoint. A hardware-free run:

```bash
dual-flexiv-control runtime.sim=true runtime.duration_s=3 \
    arms.left.control_enabled=true arms.right.control_enabled=true
```

## Run

Hardware-free smoke run — spawns all four processes against simulated sources.
Override anything from the CLI, Hydra-style:

```bash
python scripts/run_system.py runtime.sim=true runtime.duration_s=10
# or: dual-flexiv-control runtime.sim=true runtime.duration_s=10
```

Against two real arms:

```bash
dual-flexiv-control arms.left.serial=Rizon4-XXXXXX arms.right.serial=Rizon4-YYYYYY
```

More overrides:

```bash
dual-flexiv-control control@policy.control=force arms.left.wrench_frame=world \
                  brain.rate_hz=200 arms.right.streams.tau.capacity=8192
dual-flexiv-control factr.host=192.168.1.50 factr.port=8080   # FACTR server location
dual-flexiv-control --cfg job        # print the fully composed config and exit
```

## Dashboard

A dark-mode [Rerun](https://rerun.io)-backed experiment dashboard, coupled to a
long-lived **session daemon** (`dfc-session`, `session.py`) that holds the rig
for the dashboard's whole lifespan and runs a three-mode state machine:

* **VIEWING** (default while stopped) — arms connected **read-only**, cameras
  streaming; live status (operation mode, E-stop), proprio plots, and the 3D
  scene all work with **no control of any kind**.
* **COLLECTION** — FACTR teleop + LeRobot recording (the real `CollectionNode`).
* **EVAL** — a real policy rollout (the real `EvalNode`).

The dashboard spawns the daemon on startup and talks to it over JSON lines on
stdin (`start`/`stop`/`shutdown`; stdin EOF = shutdown, so a dead dashboard can
never orphan robot connections) and reads its state from
`<runtime_dir>/session.json`. Per run, the daemon hands each control-enabled arm
that phase's controller coefficients and spawns the consumer; when it exits the
arms `Stop()` and drop back to read-only VIEWING — connections, streams, and
cameras persist across runs, so runs start fast and status is always live.
Supervision: an **arm** node dying is session-fatal; a **camera** node dying is
not — it reads as down (launches are gated until every rig camera streams) and
is respawned periodically, so a replugged/recovered camera rejoins on its own.

The **left column** drives experiments — pick a **rig** (`conf/rig`) and a
**task** (`conf/task`), ✏️ open either YAML in VSCode, then launch **Collection**
or **Eval**. Switching the rig restarts the session onto that hardware set (and
re-points the arm-status rows, camera tab, and storage root). The **right area**
is tabbed: **📊 Viewer** embeds a live Rerun web viewer (live in every mode,
including VIEWING) topped by the teach-and-repeat bar (pick a taught skill,
**▶ Repeat**, ✏️ rename, 🗑 delete); **📷 Camera** shows a live view of any camera stream
(`cam/<camera>/<view>`, streaming continuously while the session is up);
**💾 Storage** lists recorded episodes with direct per-camera MP4 downloads,
replay, teach, and (bulk) delete. Downloads stream the finalized files from disk
instead of buffering copies in the dashboard process.

```bash
pip install -e ".[dashboard]"     # adds rerun-sdk + streamlit
dfc-dashboard                      # streamlit run; open the URL it prints
# ports configurable: DFC_DASHBOARD_GRPC_PORT / DFC_DASHBOARD_WEB_PORT
# MP4 downloads: DFC_DOWNLOAD_PORT (default 9092)
```

Rerun's viewer is a visualization layer and can't host the dropdown/launch
buttons itself, so Streamlit hosts the controls and serves the version-matched
Rerun web viewer (`serve_grpc` + `serve_web_viewer`) to embed alongside. The
viewer streams live over gRPC, so metrics update in the browser without a
Streamlit rerun.

```
 ┌──────────────┬─[ 📊 Viewer ]──[ 📷 Camera ]─[ 💾 Storage ]─┐
 │  Rig:  [▼]   │  Skill: [▼]  ▶ Repeat  🗑                   │
 │  Task: [▼]   │   ┌────────────┬───────────────┐            │
 │  ✏️ YAML  ✏️  │   │  3D robot  │ proprio series│            │  Viewer  → skills bar + live 3D scene/plots
 │  ▶ Collection│   │  scene     │ FACTR leaders │            │  Camera  → live cam/<cam>/<view>
 │  ▶ Eval      │   └────────────┴───────────────┘            │  Storage → MP4 ⬇, ▶ replay, 🎓 teach, 🗑 delete
 │  Running ▣   │                                             │
 └──────────────┴─────────────────────────────────────────────┘
```

**Collection** and **Eval** run the real consumers inside the session; the
Viewer tab mirrors live proprio/FACTR/3D-scene in every mode, and run outcomes
surface as popups (episode saved / crash with log tail). Stop is asynchronous —
the consumer gets `runtime.save_grace_s` to finalize the episode video while the
arms return to VIEWING. The one-shot CLI (`dual-flexiv-control
runtime.phase=collection|eval|skill`) still works standalone, spawning and
tearing down its own hardware nodes; the headless session daemon does too
(`dfc-session rig=<r>`, commands on stdin).

### Teach and repeat (skills)

A **skill** is a taught joint trajectory saved as one JSON file under
`skills/` (`skill.root`). Teach type 1 works **from a recorded episode**: every
episode row in the Storage tab has a **🎓 Teach** button (beside ▶ replay) — one
click extracts the episode's *measured* joint trajectory (plus the recorded
gripper channel) straight from the dataset's proprio columns (no video decode)
and saves it as `<dataset>-ep<index>`. The Viewer tab's bar then
repeats it: **▶ Repeat** launches a `skill` run through the session daemon — the
arm gets the `skill.coeffs` controller coefficients, MoveJ-bootstraps to the
skill's start pose, and retraces the trajectory as `qpos` setpoints at the
taught fps through the normal control channel (deadman, L-inf safety gate,
dropout watchdog, and E-stop all apply; purple ghost = the skill's end pose).
flexivrdk 1.8 has no native record-and-replay API (plans are authored in Flexiv
Elements), so replaying through the existing control path — with all of its
guardrails — is deliberate. Also usable headless:
`dual-flexiv-control runtime.phase=skill skill.name=<name>`.

## Use the brain API directly

```python
from dual_flexiv_control.streams import StreamRegistry, StreamReader

reg = StreamRegistry(runtime_dir, run_id)
reader = StreamReader.attach(reg.get("right/tau"))
window = reader.last(50)        # (n<=50, 7) oldest -> newest, with .t_ns and .seq
latest = reader.latest().newest # (7,) most recent torque vector
```

## Tests

```bash
pytest                # ring-buffer correctness, stream stack, cross-process flow
pytest -m "not slow"  # skip the full-system spawn test
```

## Layout

```
src/dual_flexiv_control/
  streams/      ring.py (shm SPMC ring) · stream.py · registry.py · spec.py
  interfaces/
    flexiv/     source.py (real flexivrdk + sim) · states.py (mapping) · interface.py
    zed/        source.py (real pyzed + sim) · interface.py (ZedInterface)
    factr/      backend.py (black-box skeleton) · interface.py
  brain/        brain.py (Brain + BrainNode)
  dashboard/    Streamlit control panel + embedded Rerun viewer:
                app.py · tasks.py · blueprints.py · viewer.py · runner.py ·
                session.py (daemon client) · launch.py
  process.py    RateLimiter · ProcessNode · StreamProducerNode · run_node
  configs.py    structured config schema (StreamCfg, ArmCfg, CameraCfg, ControlCfg, …)
  conf/         Hydra YAML tree (rig/task/recording/runtime/brain/factr/arm/camera/control)
  proprio.py    Side · canonical signals · config→StreamSpec builder
  cameras.py    canonical camera views · config→StreamSpec builder · reshape_frame
  system.py     Hydra @main orchestrator / console entry point (one-shot runs)
  session.py    session daemon (dfc-session): persistent hardware + the
                VIEWING ↔ COLLECTION ↔ EVAL state machine the dashboard drives
```

## Status / TODO

* FACTR streams raw **joint positions** and live telemetry as typed JSON frames on one
  persistent WebSocket per leader, and accepts `force_feedback` frames back on
  the same socket. Calibration is DFC-owned and not part of the wire protocol.
* **Control is implemented** over the control channel (`control/`). `qpos` FACTR
  teleop is verified end-to-end in sim; `qvel`/`end_effector`/`eef_vel`/`force` send
  paths are wired and verified against the flexivrdk 1.8 docs but **not yet
  hardware-tested**. The brain's `process()` runs FACTR→follower teleop by default;
  override it to post policy setpoints (`pack_streamed` + `Brain.command`).
* Both leaders have independent DFC-owned calibration; do not assume symmetry.
