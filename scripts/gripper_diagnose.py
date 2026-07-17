#!/usr/bin/env python3
"""Read-only diagnostic: what does Flexiv Elements say about this arm's gripper?

Answers two questions without enabling the arm or moving anything:

  1. Is the gripper actually initialized/registered correctly as a Flexiv device?
     -> ``Device(robot).list()`` returns every device configured in Elements Studio
        with its enabled flag; ``Device.exist(name)`` / ``Device.params(name)``
        read the specific gripper's config.
  2. Can we read the Elements config through the RDK at all?
     -> Yes: the Device and Tool APIs below are exactly that.

Connects only (never calls Enable/Move/Init), so it is safe to run any time —
the brakes stay engaged and the gripper is untouched.

    python scripts/gripper_diagnose.py            # default left serial + gripper name
    python scripts/gripper_diagnose.py --name GripperDahuanModbus
"""

from __future__ import annotations

import argparse
import sys
import time


DEFAULT_SERIAL = "Rizon4s-062841"
DEFAULT_GRIPPER = "GripperDahuanModbus"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read the gripper/device/tool config Elements exposes via RDK.")
    p.add_argument("--serial", default=DEFAULT_SERIAL)
    p.add_argument("--name", default=DEFAULT_GRIPPER, help="gripper device name to inspect")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--timeout", type=float, default=15.0)
    return p.parse_args()


def hr(title: str) -> None:
    print(f"\n----- {title} -----")


def main() -> int:
    args = parse_args()
    import flexivrdk

    print(f"[diag] connecting to {args.serial} (read-only) ...")
    robot = flexivrdk.Robot(args.serial, verbose=args.verbose)
    deadline = time.monotonic() + args.timeout
    while not robot.connected():
        if time.monotonic() > deadline:
            print("[diag] ERROR: did not connect", file=sys.stderr)
            return 1
        time.sleep(0.01)
    print("[diag] connected")

    hr("robot")
    info = robot.info()
    print(f"  model      : {info.model_name}")
    print(f"  serial     : {info.serial_num}")
    print(f"  software   : {info.software_ver}")
    print(f"  DoF        : {info.DoF}  (motion {info.DoF_m}, external {info.DoF_e})")
    print(f"  FT sensor  : {info.has_FT_sensor}")
    print(f"  mode       : {robot.mode()}")
    print(f"  op status  : {robot.operational_status()}")
    print(f"  operational: {robot.operational()}   fault: {robot.fault()}   "
          f"estop_released: {robot.estop_released()}   busy: {robot.busy()}")

    # --- Devices configured in Elements Studio ---------------------------------
    hr("Device.list()  (everything Elements has configured on this arm)")
    dev = flexivrdk.Device(robot)
    try:
        devices = dev.list()  # dict[name -> enabled]
    except Exception as exc:  # noqa: BLE001
        print(f"  Device.list() failed: {exc}")
        devices = {}
    if not devices:
        print("  (no devices reported)")
    for name, enabled in sorted(devices.items()):
        mark = "ENABLED " if enabled else "disabled"
        print(f"  [{mark}] {name}")

    # --- The specific gripper we're targeting ----------------------------------
    hr(f"gripper device {args.name!r}")
    try:
        exists = dev.exist(args.name)
    except Exception as exc:  # noqa: BLE001
        exists = None
        print(f"  Device.exist() failed: {exc}")
    print(f"  exists in Elements config: {exists}")
    if exists:
        print(f"  enabled: {devices.get(args.name)}")
        try:
            params = dev.params(args.name)
            print("  Device.params():")
            for k, v in params.items():
                print(f"      {k}: {v}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Device.params() failed: {exc}")
    else:
        print("  >>> NOT a configured device name. The gripper is likely registered")
        print("  >>> under a different name — see Device.list() above and update")
        print("  >>> arms.left.gripper.name in conf/rig/left_only.yaml to match.")

    # --- Tool / TCP config ------------------------------------------------------
    hr("Tool (TCP config from Elements)")
    try:
        tool = flexivrdk.Tool(robot)
        print(f"  current tool: {tool.name()}")
        print(f"  all tools   : {tool.list()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Tool query failed: {exc}")

    # --- Recent controller events (gripper faults show up here) -----------------
    hr("recent event_log (last 10)")
    try:
        for ev in robot.event_log()[-10:]:
            print(f"  {ev}")
    except Exception as exc:  # noqa: BLE001
        print(f"  event_log failed: {exc}")

    print("\n[diag] done (read-only; nothing was enabled or moved)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
