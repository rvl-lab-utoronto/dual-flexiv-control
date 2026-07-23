#!/usr/bin/env python3
"""Compute π0.5 quantile stats for DFC's relative-joint action pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from dfc_policy import ACTION_DIM
from dfc_policy import STATE_DIM


def _vectors(column) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float64)


def _load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet episodes below {root / 'data'}")
    states, actions, episodes = [], [], []
    for path in files:
        table = parquet.read_table(
            path, columns=["observation.state", "action", "episode_index"]
        )
        states.append(_vectors(table["observation.state"]))
        actions.append(_vectors(table["action"]))
        episodes.append(np.asarray(table["episode_index"], dtype=np.int64))
    state = np.concatenate(states)
    action = np.concatenate(actions)
    episode = np.concatenate(episodes)
    if state.shape != (episode.size, STATE_DIM):
        raise ValueError(f"expected state shape (*, {STATE_DIM}), got {state.shape}")
    if action.shape != (episode.size, ACTION_DIM):
        raise ValueError(f"expected action shape (*, {ACTION_DIM}), got {action.shape}")
    return state, action, episode, files


def _relative_action_chunks(
    state: np.ndarray,
    action: np.ndarray,
    episode: np.ndarray,
    horizon: int,
) -> np.ndarray:
    chunks = []
    for episode_id in np.unique(episode):
        indices = np.flatnonzero(episode == episode_id)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"episode {episode_id} is not contiguous")
        local_actions = action[indices]
        future = np.minimum(
            np.arange(indices.size)[:, None] + np.arange(horizon)[None, :],
            indices.size - 1,
        )
        chunk = local_actions[future].copy()
        current_state = state[indices]
        chunk[..., 0:7] -= current_state[:, None, 0:7]
        chunk[..., 8:15] -= current_state[:, None, 7:14]
        chunks.append(chunk)
    return np.concatenate(chunks)


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    flat = values.reshape(-1, values.shape[-1])
    return {
        "mean": np.mean(flat, axis=0).tolist(),
        "std": np.std(flat, axis=0).tolist(),
        "q01": np.quantile(flat, 0.01, axis=0).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).tolist(),
    }


def _digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")

    state, action, episode, files = _load_dataset(args.dataset)
    relative_actions = _relative_action_chunks(
        state, action, episode, args.horizon
    )
    payload = {
        "norm_stats": {
            "state": _stats(state),
            "actions": _stats(relative_actions),
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "dataset": str(args.dataset),
        "parquet_sha256": _digest(files, args.dataset),
        "episodes": int(np.unique(episode).size),
        "frames": int(episode.size),
        "action_horizon": args.horizon,
        "state_layout": "left.q[7], right.q[7]",
        "action_layout": "left.q[7], left.gripper, right.q[7], right.gripper",
        "joint_action_space": "delta from current observation.state",
        "gripper_action_space": "absolute",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
