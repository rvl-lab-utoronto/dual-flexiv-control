#!/usr/bin/env python3
"""Standalone smoke test: open the LEFT arm's gripper via the Flexiv RDK.

A minimal, hydra-free diagnostic to answer one question — "does the follower
gripper physically open when commanded?" — without spinning up the whole
producer/consumer stack. It mirrors the verified RDK 1.8.0 call sequence the real
interface uses (see
``dual_flexiv_control.interfaces.flexiv.source.FlexivSource.setup_gripper`` /
``send_gripper``): connect the ``Robot``, clear any fault, ``Enable`` and wait
until operational, then ``Gripper.Enable(name)`` → ``Init`` (home) → ``Move`` to
the gripper's max width (fully open).

DEVICE NAME MATTERS. The left arm (Rizon4s-062841) physically carries the
**Flexiv-GN01** gripper, so that is the default here. Commanding a
configured-but-absent device (e.g. ``GripperDahuanModbus``, which is still what
``conf/rig/left_only.yaml`` names) is accepted by the RDK without error but moves
nothing. Confirm the mounted device with ``scripts/gripper_diagnose.py``.

FEEDBACK CAVEAT. The Flexiv-GN01 does not stream position back through
``gripper.states()`` — ``width`` reads 0.0 and ``is_moving`` stays False even
while it physically moves. So this script reports whether the command was
*accepted*, prints whatever feedback exists, and asks you to confirm the motion
visually. It does not (cannot) gate success on a width readout for this gripper.

    conda activate dual-flexiv-control   # the miniforge 'dual-flexiv-control' env
    python scripts/test_gripper_open.py                 # open the Flexiv-GN01
    python scripts/test_gripper_open.py --close         # close it instead
    python scripts/test_gripper_open.py --no-init       # skip homing (faster re-test)
    python scripts/test_gripper_open.py --name GripperDahuanModbus  # a different device

SAFETY: ``Enable()`` releases the arm's brakes to reach an operational state
(required before the gripper will actuate). The arm holds position and is not
commanded to move, but be clear of it and have the E-stop within reach before
running. Ctrl-C or the E-stop stops everything.
"""

from __future__ import annotations

import argparse
import sys
import time


# Defaults for the left arm. name = the physically-mounted gripper device
# (verify with scripts/gripper_diagnose.py -> Device.list()).
DEFAULT_SERIAL = "Rizon4s-062841"
DEFAULT_GRIPPER = "Flexiv-GN01"
DEFAULT_VELOCITY = 0.1   # m/s (clamped to the gripper's params)
DEFAULT_FORCE = 20.0     # N   (clamped to the gripper's params)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open (or close) the left arm's gripper via flexivrdk.")
    p.add_argument("--serial", default=DEFAULT_SERIAL, help=f"arm serial (default: {DEFAULT_SERIAL})")
    p.add_argument("--name", default=DEFAULT_GRIPPER, help=f"gripper device name (default: {DEFAULT_GRIPPER})")
    p.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY, help="Move velocity [m/s]")
    p.add_argument("--force", type=float, default=DEFAULT_FORCE, help="Move force [N]")
    p.add_argument("--close", action="store_true", help="close (min width) instead of open (max width)")
    p.add_argument("--no-init", dest="init", action="store_false", help="skip Init() homing")
    p.add_argument("--init-wait", type=float, default=6.0, help="seconds to let homing settle")
    p.add_argument("--watch", type=float, default=3.0, help="seconds to observe after Move")
    p.add_argument("--verbose", action="store_true", help="verbose RDK logging")
    p.add_argument("--timeout", type=float, default=30.0, help="connect / operational timeout [s]")
    return p.parse_args()


def wait_until(predicate, timeout_s: float, what: str) -> None:
    """Poll ``predicate`` until true or raise ``TimeoutError`` after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout_s:.0f}s waiting for {what}")
        time.sleep(0.01)


def watch_states(gripper, seconds: float, tag: str) -> None:
    """Print gripper state samples for ``seconds``, de-duplicated.

    Feedback may be static (the GN01 doesn't report width), so identical lines are
    printed once instead of flooding — no ``\\r`` (which piles up when stdout isn't
    a TTY). Exits early once motion is reported to have stopped after starting.
    """
    end = time.monotonic() + seconds
    last = None
    moved = False
    while time.monotonic() < end:
        st = gripper.states()
        line = f"  [{tag}] width={st.width:.4f} m  force={st.force:6.2f} N  is_moving={st.is_moving}"
        if line != last:
            print(line, flush=True)
            last = line
        if st.is_moving:
            moved = True
        elif moved:
            return  # started then stopped -> done
        time.sleep(0.25)


def main() -> int:
    args = parse_args()

    try:
        import flexivrdk
    except ModuleNotFoundError:
        print(
            "flexivrdk not importable. Activate the project env first, e.g.\n"
            "  conda activate dual-flexiv-control\n"
            "(installed at ~/miniforge3/envs/dual-flexiv-control)",
            file=sys.stderr,
        )
        return 2

    print(f"[gripper-test] connecting to {args.serial} ...")
    robot = flexivrdk.Robot(args.serial, verbose=args.verbose)

    # 1. Clear any standing fault so the connection is clean.
    if robot.fault():
        print("[gripper-test] robot has a fault; ClearFault() ...")
        if not robot.ClearFault():
            print("[gripper-test] ERROR: failed to clear fault", file=sys.stderr)
            return 1

    wait_until(robot.connected, args.timeout, "connection")
    print("[gripper-test] connected")

    # 2. Enable (releases brakes) and wait until operational — required before the
    #    gripper will actuate. The arm is NOT commanded to move.
    if not robot.operational():
        print("[gripper-test] Enable() -> waiting for operational (brakes releasing) ...")
        robot.Enable()
        wait_until(robot.operational, args.timeout, "operational state")
    print("[gripper-test] operational")

    # 3. Confirm the target device is actually configured, so a wrong --name fails
    #    loud instead of silently no-op'ing (the whole reason the first attempts
    #    "did nothing": GripperDahuanModbus is configured but not mounted).
    dev = flexivrdk.Device(robot)
    if not dev.exist(args.name):
        print(
            f"[gripper-test] ERROR: {args.name!r} is not a configured device. "
            f"Configured: {sorted(dev.list())}",
            file=sys.stderr,
        )
        return 1

    # 4. Bring up the gripper: Enable(device_name) then optionally Init() to home.
    gripper = flexivrdk.Gripper(robot)
    print(f"[gripper-test] Gripper.Enable({args.name!r})")
    gripper.Enable(args.name)
    time.sleep(0.5)
    if args.init:
        print(f"[gripper-test] Init() (homing); settling {args.init_wait:.0f}s ...")
        gripper.Init()
        watch_states(gripper, args.init_wait, "home")

    params = gripper.params()
    print(
        f"[gripper-test] params: width [{params.min_width:.4f}, {params.max_width:.4f}] m, "
        f"vel [{params.min_vel:.3f}, {params.max_vel:.3f}] m/s, "
        f"force [{params.min_force:.1f}, {params.max_force:.1f}] N"
    )

    # 5. Move: open == max width, close == min width. Velocity/force are clamped to
    #    the device's valid range (matches FlexivSource.setup_gripper).
    target = params.min_width if args.close else params.max_width
    vel = min(max(args.velocity, params.min_vel), params.max_vel)
    force = min(max(args.force, params.min_force), params.max_force)
    action = "CLOSE" if args.close else "OPEN"
    print(f"[gripper-test] {action}: Move(width={target:.4f} m, vel={vel:.3f} m/s, force={force:.1f} N)")
    gripper.Move(target, vel, force)

    # 6. Observe. This gripper may not report width back, so this is informational —
    #    the real confirmation is watching the hardware.
    watch_states(gripper, args.watch, action.lower())
    st = gripper.states()
    print(
        f"[gripper-test] Move command accepted. states(): "
        f"width={st.width:.4f} m, force={st.force:.2f} N, is_moving={st.is_moving}"
    )
    if st.width == 0.0 and not st.is_moving:
        print(
            "[gripper-test] NOTE: no position feedback from this gripper "
            "(Flexiv-GN01 doesn't stream width via states()). "
            f"Confirm the {action} visually — the command was accepted without error."
        )
    return 0  # command sent successfully; physical result confirmed by eye


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[gripper-test] interrupted", file=sys.stderr)
        raise SystemExit(130)
