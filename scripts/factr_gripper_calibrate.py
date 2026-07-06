#!/usr/bin/env python3
"""Calibrate a FACTR leader's gripper open/closed endpoints for 0..1 normalization.

The FACTR server serves the trailing gripper value as an **un-normalized servo
angle in radians** (it stores no open/closed endpoints). To record the gripper
action as a clean 0..1 fraction, ``JointConventionCfg.gripper_open`` /
``gripper_closed`` must be set to the raw leader readings at the fully-open and
fully-closed trigger, and :func:`~dual_flexiv_control.control.normalize_gripper`
maps between them. This helper reads that live value so you can capture the two
endpoints and paste them into ``conf``.

Usage:
    python scripts/factr_gripper_calibrate.py [--side left] [--samples 15] [--sim]

Move the trigger to each endpoint when prompted (a live reading is shown), then
press Enter to capture. The captured value is the median of the most recent
samples (robust to jitter). At the end it prints ready-to-paste CLI overrides and
a YAML snippet, plus a normalization sanity check.

IMPORTANT: the deployed read-only FACTR publisher re-zeroes the gripper at startup,
so these endpoints are relative to the startup pose — calibrate against a
*consistent* startup pose (and re-calibrate if you change how FACTR is launched).
"""

from __future__ import annotations

import argparse
import select
import sys
from collections import deque

import numpy as np


def compose_factr(sim_override: bool | None):
    """Compose the project config and return ``(factr_cfg, sim)``.

    ``sim_override`` forces sim on/off; ``None`` takes ``runtime.sim`` from config.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from dual_flexiv_control.configs import register_configs

    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="dual_flexiv_control.conf"):
        cfg = compose(config_name="config")
    sim = cfg.runtime.sim if sim_override is None else sim_override
    return cfg.factr, bool(sim)


def read_gripper_once(client, side: str) -> float:
    """The current raw FACTR gripper value (trailing element of the DoF+1 vector)."""
    return float(np.asarray(client.get_joint_positions_for(side), dtype=np.float64).ravel()[-1])


def format_overrides(side: str, open_v: float, closed_v: float) -> str:
    """Hydra CLI overrides for one arm's gripper calibration."""
    return (
        f"arms.{side}.convention.gripper_open={open_v:.4f} "
        f"arms.{side}.convention.gripper_closed={closed_v:.4f}"
    )


def format_yaml(side: str, open_v: float, closed_v: float) -> str:
    """A conf/config.yaml snippet for one arm's gripper calibration."""
    return (
        "arms:\n"
        f"  {side}:\n"
        f"    convention: {{ gripper_open: {open_v:.4f}, gripper_closed: {closed_v:.4f} }}"
    )


def preview_normalization(open_v: float, closed_v: float) -> str:
    """A few raw→normalized samples using the real normalize_gripper, for a sanity check."""
    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control import normalize_gripper

    conv = JointConventionCfg(gripper_open=open_v, gripper_closed=closed_v)
    pts = [open_v, (open_v + closed_v) / 2.0, closed_v]
    labels = ["open", "mid", "closed"]
    rows = [f"    {lab:<6} {raw:+.4f} rad -> {normalize_gripper(raw, conv):.3f}"
            for lab, raw in zip(labels, pts)]
    return "\n".join(rows)


def capture_endpoint(client, side: str, label: str, samples: int, poll_hz: float) -> float:
    """Stream the live gripper reading and capture the median of recent samples on Enter."""
    print(f"\nMove the {side} gripper to the *{label}* position.")
    interactive = sys.stdin.isatty()
    if not interactive:
        print("  (stdin is not a TTY — capturing a single window without a live view)")
    recent: deque[float] = deque(maxlen=max(1, samples))
    period = 1.0 / max(1.0, poll_hz)
    while True:
        try:
            g = read_gripper_once(client, side)
            recent.append(g)
            sys.stdout.write(f"\r  {label:<6}: {g:+.4f} rad   (press Enter to capture) ")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 - a transient read failure shouldn't abort
            sys.stdout.write(f"\r  read failed: {exc}                         ")
            sys.stdout.flush()
        # Wait up to `period` for Enter; capture when it arrives (or immediately if not a TTY
        # once we have a full window).
        ready, _, _ = select.select([sys.stdin], [], [], period)
        if ready:
            sys.stdin.readline()
            break
        if not interactive and len(recent) >= recent.maxlen:
            break
    if not recent:
        raise SystemExit(f"no readings captured for {side} gripper — is the FACTR server up?")
    value = float(np.median(recent))
    print(f"\n  captured {label}: {value:+.4f} rad "
          f"(median of {len(recent)} samples, spread {max(recent) - min(recent):.4f})")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a FACTR leader gripper's 0..1 endpoints.")
    parser.add_argument("--side", default="left", help="leader side to calibrate (default: left)")
    parser.add_argument("--samples", type=int, default=15, help="samples to median per capture")
    parser.add_argument("--poll-hz", type=float, default=10.0, help="live-read rate while positioning")
    parser.add_argument("--sim", action="store_true", help="use the synthetic FACTR source (no hardware)")
    args = parser.parse_args()

    from dual_flexiv_control.interfaces.factr import FactrClient

    factr_cfg, sim = compose_factr(sim_override=True if args.sim else None)
    if args.side not in factr_cfg.servers:
        raise SystemExit(f"unknown side {args.side!r}; configured: {list(factr_cfg.servers)}")

    client = FactrClient.from_config(factr_cfg, sim=sim)
    print(f"FACTR gripper calibration — side={args.side} "
          f"({'SIM' if sim else client.server(args.side).url})")
    if sim:
        print("  NOTE: sim gripper is a synthetic sinusoid — for mechanics testing only.")

    # Fail fast with a clear message if the server/side is unreachable.
    try:
        read_gripper_once(client, args.side)
    except Exception as exc:  # noqa: BLE001
        client.close()
        raise SystemExit(
            f"cannot read the {args.side} FACTR gripper: {exc}\n"
            f"  Is the FACTR server for {args.side} running? (the right leader may have no "
            f"producer yet — see conf/factr/bimanual.yaml)"
        )

    try:
        open_v = capture_endpoint(client, args.side, "open", args.samples, args.poll_hz)
        closed_v = capture_endpoint(client, args.side, "closed", args.samples, args.poll_hz)
    finally:
        client.close()

    if abs(closed_v - open_v) < 1e-3:
        print("\n⚠️  open and closed readings are nearly identical — the trigger may not have "
              "moved, or the wrong side was calibrated. Normalization needs a real span.")

    print("\n" + "=" * 68)
    print(f"Gripper calibration for {args.side}:  open={open_v:+.4f}  closed={closed_v:+.4f} rad")
    print("-" * 68)
    print("CLI override:\n  " + format_overrides(args.side, open_v, closed_v))
    print("\nconf/config.yaml:\n" + format_yaml(args.side, open_v, closed_v))
    print("\nnormalization sanity check:")
    print(preview_normalization(open_v, closed_v))
    print("=" * 68)
    print("Reminder: the FACTR publisher re-zeroes at startup — recalibrate if you "
          "relaunch it, and always calibrate from a consistent startup pose.")


if __name__ == "__main__":
    main()
