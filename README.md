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
4. **The FACTR client** (`interfaces/factr/`) — an **on-request** HTTP client
   (no polling, no stream). The brain holds a `FactrClient`; calling
   `get_joint_positions()` GETs the FACTR server's `get_joint_positions` endpoint
   and returns both leaders' joint positions split per side (host/port configurable).

## Architecture

```
 FlexivInterface(LEFT/RIGHT)  process ─┐  shared-memory ring buffers  ┌─ BrainNode  process
 ZedInterface(wrist_l/wrist_r/static)  ─┤      (one per signal)        └─  reads last k
                                        └                              └─ FactrClient.get_joint_positions()  (on request, HTTP)
```

* **Streams** are single-producer / multi-consumer ring buffers in POSIX shared
  memory (`/dev/shm`). One producer (the owning interface) appends; any number of
  consumers attach by name and read the newest `k` samples — no copies, no IPC
  round-trips. Coordination is a lock-free **seqlock** (a published global write
  count + per-slot sequence stamps); readers re-validate each slot so a buffer
  that laps a slow reader degrades to "freshest valid suffix", never corruption.
* **One stream per signal, per arm.** Each arm publishes `q`, `dq`, `tau`,
  `wrench`, `eef`, `eef_vel` as separate streams named `left/…` and `right/…`.
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
in [configs.py](src/dual_flexiv_control/configs.py). The tree is hierarchical along
the stream paths:

```
conf/
  config.yaml              # composes the groups below; per-arm serials
  runtime/default.yaml     # sim, runtime_dir, duration_s
  brain/default.yaml       # rate, attach timeout, subscriptions
  factr/bimanual.yaml      # FACTR server host/port/endpoint + factr/left,factr/right schemas
  arm/flexiv.yaml          # per-arm: dof, wrench_frame, the 6 proprio stream schemas
  camera/                  # per-camera templates (resolution, fps, views, capacity):
    zedx_wrist.yaml  zed2_static.yaml
  control/                 # control-type library (command schemas, SDK-aligned):
    qpos.yaml  qvel.yaml  end_effector.yaml  force.yaml
  task/                    # one file per manipulation task (select with `task=<name>`):
    default.yaml  handover.yaml
```

Cameras compose just like arms: `camera@cameras.<name>: <template>` places a
template at `cameras.<name>`, with per-camera `serial`/`placement` set in
`config.yaml`. Override resolution/fps/views from the CLI, e.g.
`cameras.static.resolution=HD1080 cameras.static.width=1920 cameras.static.height=1080`
or `'+cameras.static.views=[left,right,depth]' cameras.static.depth_mode=NEURAL`.
Camera streams are produced unconditionally but are **not** in the brain's default
subscription (proprio only); subscribe to them explicitly, e.g.
`'brain.subscribe=[left/q, cam/static/left]'`.

Each **task** carries one `language_instruction` shared by both phases, plus the
counts unique to each phase — `collection.num_episodes` (demos to teleoperate) and
`eval.num_timesteps` (rollout horizon). Add a task by copying `task/default.yaml`;
select it with `task=<name>` and tune fields inline, e.g.
`task=handover task.eval.num_timesteps=800`.

Each stream's schema (`dim`, `dtype`, `capacity`, `rate_hz`) lives under its path,
e.g. `arms.left.streams.tau` → stream `left/tau`. The **control configs** lay out
each control kind's command schema; `streamed` lists which fields the brain posts
per tick (the rest are static limits from the coeffs). All paths are **NRT**
(verified against flexivrdk 1.8) — the brain posts setpoints over IPC at
~50-200 Hz (RDK 1.8 is NRT-only — no hard-1 kHz RT modes, no `Stream*` methods):

| Controller | RDK mode | send fn | streamed (per-tick) |
|---|---|---|---|
| `qpos` | `NRT_JOINT_POSITION` | `SendJointPosition` | `q_d`, `dq_d` |
| `qvel` | `NRT_JOINT_POSITION` | `SendJointPosition` | `dq_d` (arm integrates `q_d`) |
| `end_effector` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `pose_d`, `twist_d` |
| `eef_vel` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `twist_d` (arm integrates `pose_d`) |
| `force` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` | `wrench_d`, `pose_d` |

**Controller coefficients** (impedances + motion limits) are a *separate* importable
group, `conf/control_coeffs/` (`default`/`compliant`/`stiff`), pulled into each task
**per phase** — `task.collection.coeffs` (training) and `task.eval.coeffs` (eval) —
so a task runs compliant during teleop collection and stiff during policy eval.
`runtime.phase` (`collection`|`eval`) selects which set the arms apply. The arm
applies, after `SwitchMode`, only the coeffs its mode accepts (e.g. cartesian
impedance for the Cartesian kinds; `dq_max`/`ddq_max` for the joint kinds).

### Teleoperation (FACTR → follower)

With `arm.control_enabled=true`, the brain's `process()` reads the FACTR leaders,
maps them to Rizon joint targets via the per-arm `JointConventionCfg` (offsets,
sign-flips, wrap, gripper drop — captured from the hardware test), and posts qpos
setpoints. A hardware-free run:

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
dual-flexiv-control control@arms.left.control=force arms.left.wrench_frame=world \
                  brain.rate_hz=200 arms.right.streams.tau.capacity=8192
dual-flexiv-control factr.host=192.168.1.50 factr.port=8080   # FACTR server location
dual-flexiv-control --cfg job        # print the fully composed config and exit
```

## Dashboard

A dark-mode [Rerun](https://rerun.io)-backed experiment dashboard. The **left
column** drives experiments — pick a task from the `conf/task` group, ✏️ open its
YAML in VSCode, then launch **Collection** (teleop demos) or **Eval** (policy
rollouts). The **right area** is tabbed: **📊 Metrics** embeds a live Rerun web
viewer holding the run's metrics; **📷 Camera** shows a live view of any camera
stream (`cam/<camera>/<view>`, selected from a dropdown), reading frames from
shared memory when the system is running and falling back to an animated
placeholder otherwise.

```bash
pip install -e ".[dashboard]"     # adds rerun-sdk + streamlit
dfc-dashboard                      # streamlit run; open the URL it prints
# ports configurable: DFC_DASHBOARD_GRPC_PORT / DFC_DASHBOARD_WEB_PORT
```

Rerun's viewer is a visualization layer and can't host the dropdown/launch
buttons itself, so Streamlit hosts the controls and serves the version-matched
Rerun web viewer (`serve_grpc` + `serve_web_viewer`) to embed alongside. The
viewer streams live over gRPC, so metrics update in the browser without a
Streamlit rerun.

```
 ┌──────────────┬─[ 📊 Metrics ]─[ 📷 Camera ]──────┐
 │  Task: [▼]   │   ┌────────────┬───────────────┐  │
 │  ✏️ Edit YAML│   │  EEF 3D    │ x/y/z series  │  │   Metrics → eval: 3D EEF
 │  ▶ Collection│   │  tracking  │ reward/success│  │             collection: progress
 │  ▶ Eval      │   └────────────┴───────────────┘  │   Camera  → live cam/<cam>/<view>
 │  Running ▣   │   (cam tab: dropdown + live image) │
 └──────────────┴──────────────────────────────────┘
```

**Status:** launching is a **stub** — it switches the viewer's blueprint and
streams *synthetic placeholder* metrics (eval traces 3D end-effector paths for
both arms; collection shows demo progress). The seam for the real run is marked
in [`dashboard/runner.py`](src/dual_flexiv_control/dashboard/runner.py): the eval/
collection process will spawn and `rr.connect_grpc` back to the dashboard's
server, logging real telemetry to the same entity paths
([`dashboard/blueprints.py`](src/dual_flexiv_control/dashboard/blueprints.py)) the
placeholder uses — so the viewer needs no change when control lands.

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
                app.py · tasks.py · blueprints.py · viewer.py · runner.py · launch.py
  process.py    RateLimiter · ProcessNode · StreamProducerNode · run_node
  configs.py    structured config schema (StreamCfg, ArmCfg, CameraCfg, ControlCfg, …)
  conf/         Hydra YAML tree (runtime/brain/factr/task/arm/camera/control groups)
  proprio.py    Side · canonical signals · config→StreamSpec builder
  cameras.py    canonical camera views · config→StreamSpec builder · reshape_frame
  system.py     Hydra @main orchestrator / console entry point
```

## Status / TODO

* FACTR reads **joint positions** from the server's `get_joint_positions`
  endpoint. Other FACTR signals (e.g. force feedback) can be added as more
  endpoints/streams when needed (`interfaces/factr/backend.py`).
* **Control is implemented** over the control channel (`control/`). `qpos` FACTR
  teleop is verified end-to-end in sim; `qvel`/`end_effector`/`eef_vel`/`force` send
  paths are wired and verified against the flexivrdk 1.8 docs but **not yet
  hardware-tested**. The brain's `process()` runs FACTR→follower teleop by default;
  override it to post policy setpoints (`pack_streamed` + `Brain.command`).
* Only the **left** arm's `JointConventionCfg` is known (from the test); the right
  arm's offsets/sign-flips must be measured — do not assume symmetry.
