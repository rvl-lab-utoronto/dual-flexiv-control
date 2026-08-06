#!/usr/bin/env python3
"""Serve π0.5 base with the Dual-Flexiv observation/action contract."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import socket

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config

from dfc_policy import ACTION_DIM
from dfc_policy import OpenPIActionDeltas
from dfc_policy import OpenPIInputs
from dfc_policy import OpenPIOutputs
from dfc_policy import STATE_DIM


@dataclasses.dataclass(frozen=True)
class DFCDataConfig(training_config.DataConfigFactory):
    """OpenPI data pipeline for 14 joint observations and 16 DFC actions."""

    repo_id: str = "dfc/handover_bimanual_v2"

    def create(self, assets_dirs: Path, model_config) -> training_config.DataConfig:
        del assets_dirs
        return training_config.DataConfig(
            repo_id=self.repo_id,
            asset_id="dual_flexiv",
            data_transforms=transforms.Group(
                inputs=[OpenPIInputs(), OpenPIActionDeltas()],
                outputs=[OpenPIActionDeltas(inverse=True), OpenPIOutputs()],
            ),
            model_transforms=training_config.ModelTransformFactory()(model_config),
            use_quantile_norm=True,
        )


def _load_stats(path: Path) -> dict:
    directory = path if path.is_dir() else path.parent
    stats = normalize.load(directory)
    expected = {"state": STATE_DIM, "actions": ACTION_DIM}
    for key, dim in expected.items():
        if key not in stats:
            raise ValueError(f"normalization stats have no {key!r} entry")
        if stats[key].mean.shape != (dim,):
            raise ValueError(
                f"normalization stats {key!r} has shape "
                f"{stats[key].mean.shape}; expected ({dim},)"
            )
    return stats


def _policy(args: argparse.Namespace):
    config = training_config.TrainConfig(
        name="pi05_dfc_base",
        model=pi0_config.Pi0Config(pi05=True),
        data=DFCDataConfig(),
        policy_metadata={
            "checkpoint": args.checkpoint,
            "model_type": "pi05_base",
            "robot": "dual_flexiv",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "camera_mapping": {
                "base_0_rgb": "observation/images/static_left",
                "left_wrist_0_rgb": None,
                "right_wrist_0_rgb": None,
            },
            "normalization": str(args.norm_stats),
            "base_model_warning": (
                "Base weights are not DFC-fine-tuned; validate predictions in "
                "dry-run before enabling follower control."
            ),
        },
    )
    return policy_config.create_trained_policy(
        config,
        args.checkpoint,
        default_prompt=args.default_prompt,
        norm_stats=_load_stats(args.norm_stats),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="π0.5 checkpoint directory (the public base checkpoint by default)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path(__file__).with_name("norm_stats.json"),
        help="DFC norm_stats.json file or its containing directory",
    )
    parser.add_argument("--default-prompt")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy = _policy(args)
    metadata = policy.metadata
    if args.record:
        from openpi.policies import policy as policy_module

        policy = policy_module.PolicyRecorder(policy, "policy_records")
    hostname = socket.gethostname()
    logging.info("Serving π0.5 DFC policy from %s on %s:%d", args.checkpoint, hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
