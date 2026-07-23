"""Pure shape/coordinate tests for the OpenPI DFC deployment overlay."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from deploy.openpi_dfc.dfc_policy import ACTION_DIM
from deploy.openpi_dfc.dfc_policy import DFCInputs
from deploy.openpi_dfc.dfc_policy import DFCJointDeltas
from deploy.openpi_dfc.dfc_policy import DFCOutputs
from deploy.openpi_dfc.dfc_policy import STATE_DIM


def test_dfc_inputs_use_base_view_and_mask_nonexistent_wrist_cameras():
    left = np.full((3, 8, 12), 0.5, dtype=np.float32)
    right = np.full((8, 12, 3), 255, dtype=np.uint8)
    result = DFCInputs()({
        "observation/state": np.arange(STATE_DIM),
        "observation/images/static_left": left,
        "observation/images/static_right": right,
        "prompt": "handover",
    })

    assert result["state"].shape == (STATE_DIM,)
    assert result["image"]["base_0_rgb"].shape == (8, 12, 3)
    assert np.all(result["image"]["base_0_rgb"] == 127)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert not np.any(result["image"]["left_wrist_0_rgb"])
    assert result["prompt"] == "handover"


def test_dfc_joint_delta_transform_round_trips_interleaved_grippers():
    state = np.arange(STATE_DIM, dtype=np.float64)
    actions = np.tile(np.arange(ACTION_DIM, dtype=np.float64), (4, 1))
    original = {"state": state, "actions": actions}

    delta = DFCJointDeltas()(original)
    restored = DFCJointDeltas(inverse=True)(delta)

    np.testing.assert_allclose(restored["actions"], actions)
    np.testing.assert_allclose(delta["actions"][:, 0:7], actions[:, 0:7] - state[0:7])
    np.testing.assert_allclose(delta["actions"][:, 8:15], actions[:, 8:15] - state[7:14])
    np.testing.assert_allclose(delta["actions"][:, [7, 15]], actions[:, [7, 15]])
    np.testing.assert_allclose(original["actions"], actions)  # input was not mutated


def test_dfc_contract_rejects_aloha_sized_actions():
    with pytest.raises(ValueError, match="16"):
        DFCInputs()({
            "observation/state": np.zeros(STATE_DIM),
            "observation/images/static_left": np.zeros((8, 12, 3)),
            "actions": np.zeros((50, 14)),
        })


def test_dfc_outputs_crop_openpi_padding():
    actions = np.arange(50 * 32).reshape(50, 32)
    result = DFCOutputs()({"actions": actions})
    assert result["actions"].shape == (50, ACTION_DIM)
    np.testing.assert_array_equal(result["actions"], actions[:, :ACTION_DIM])


def test_checked_in_norm_stats_match_dfc_dimensions_and_provenance():
    root = Path(__file__).parents[1] / "deploy" / "openpi_dfc"
    stats = json.loads((root / "norm_stats.json").read_text())["norm_stats"]
    metadata = json.loads((root / "norm_stats.metadata.json").read_text())

    for field in ("mean", "std", "q01", "q99"):
        assert len(stats["state"][field]) == STATE_DIM
        assert len(stats["actions"][field]) == ACTION_DIM
    assert metadata["frames"] == 936
    assert metadata["action_horizon"] == 50
