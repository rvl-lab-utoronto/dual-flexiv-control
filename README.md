# dual-flexiv-control

Control, visualization, data collection, and online policy evaluation for the RVL **bimanual Flexiv**. Three
components, decoupled by zero-copy shared-memory data streams so each runs in its
own process at its own rate:

1. **The brain** (`brain/`) — the main processing pipeline. Subscribes to data
   streams, observes them, and pulls the last `k` elements of any signal.
2. **The Flexiv interface** (`interfaces/flexiv/`) — wraps **real `flexivrdk`**
   (RDK 2.x), one process per arm, exposing proprioception as streams. Read-only
   for now (it never enables the robot or takes control).
3. **The FACTR client** (`interfaces/factr/`) — an **on-request** HTTP client
   (no polling, no stream). The brain holds a `FactrClient`; calling
   `get_joint_positions()` GETs the FACTR server's `get_joint_positions` endpoint
   and returns both leaders' joint positions split per side (host/port configurable).

## Architecture

```
 FlexivInterface(LEFT)   process ─┐  shared-memory ring buffers  ┌─ BrainNode  process
 FlexivInterface(RIGHT)  process ─┘      (one per signal)        └─  reads last k
                                                                    └─ FactrClient.get_joint_positions()  (on request, HTTP)
```

* **Streams** are single-producer / multi-consumer ring buffers in POSIX shared
  memory (`/dev/shm`). One producer (the owning interface) appends; any number of
  consumers attach by name and read the newest `k` samples — no copies, no IPC
  round-trips. Coordination is a lock-free **seqlock** (a published global write
  count + per-slot sequence stamps); readers re-validate each slot so a buffer
  that laps a slow reader degrades to "freshest valid suffix", never corruption.
* **One stream per signal, per arm.** Each arm publishes `q`, `dq`, `tau`,
  `wrench`, `eef`, `eef_vel` as separate streams named `left/…` and `right/…`.
* **Multiprocess, spawned (never forked).** flexivrdk runs live threads/services;
  forking would corrupt them, so every node is a fresh interpreter that opens its
  own connection. Discovery is a directory of tiny JSON manifests per run.

### Proprio → Flexiv RDK 2.x mapping

| Stream            | dim | RDK 2.x `RobotStates` field            |
|-------------------|-----|----------------------------------------|
| `<side>/q`        | 7   | `q` (link-side joint positions)        |
| `<side>/dq`       | 7   | `dq` (link-side joint velocities)      |
| `<side>/tau`      | 7   | `tau` (measured joint torques)         |
| `<side>/wrench`   | 6   | `tcp_wrench_local` (TCP frame) — or `tcp_wrench` (world) via `--wrench-frame` |
| `<side>/eef`      | 7   | `tcp_pose` `[x,y,z,qw,qx,qy,qz]`       |
| `<side>/eef_vel`  | 6   | `tcp_twist` `[v(3), ω(3)]`             |

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
  control/                 # control-type library (command schemas, SDK-aligned):
    qpos.yaml  qvel.yaml  end_effector.yaml  force.yaml
```

Each stream's schema (`dim`, `dtype`, `capacity`, `rate_hz`) lives under its path,
e.g. `arms.left.streams.tau` → stream `left/tau`. The four **control configs** lay
out the command schema for the future control implementation, aligned with the
flexivrdk 2.x command structs:

| Controller | RDK mode | stream fn | command (field → dim) |
|---|---|---|---|
| `qpos` | `RT_JOINT_POSITION` | `StreamJointPosition` | `RtJointPositionCmd`: q_d 7, dq_d 7, ddq_d 7 |
| `qvel` | `RT_JOINT_POSITION` | `StreamJointPosition` | dq_d 7 (primary) + q_d 7, ddq_d 7 |
| `end_effector` | `RT_CARTESIAN_MOTION_FORCE` | `StreamCartesianMotionForce` | `RtCartesianCmd`: pose_d 7, twist_d 6, acc_d 6, wrench_d 6 |
| `force` | `RT_CARTESIAN_MOTION_FORCE` | `StreamCartesianMotionForce` | wrench_d 6 (primary) + axes 6, frame, max_linear_vel 3, max_contact_wrench 6 |

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
    factr/      backend.py (black-box skeleton) · interface.py
  brain/        brain.py (Brain + BrainNode)
  process.py    RateLimiter · ProcessNode · StreamProducerNode · run_node
  configs.py    structured config schema (StreamCfg, ArmCfg, ControlCfg, …)
  conf/         Hydra YAML tree (runtime/brain/factr/arm/control groups)
  proprio.py    Side · canonical signals · config→StreamSpec builder
  system.py     Hydra @main orchestrator / console entry point
```

## Status / TODO

* FACTR reads **joint positions** from the server's `get_joint_positions`
  endpoint. Other FACTR signals (e.g. force feedback) can be added as more
  endpoints/streams when needed (`interfaces/factr/backend.py`).
* The brain's `process(observation)` hook is a no-op placeholder for the
  downstream control/policy pipeline (more to come).
