"""Tests for the FACTR gripper calibration helper's pure logic (no hardware/TTY).

The interactive capture loop needs a TTY + a live leader, so it isn't exercised
here; the reusable pieces (config read, formatting, normalization preview, and a
single sim read) are.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "factr_gripper_calibrate.py"


def _load():
    spec = importlib.util.spec_from_file_location("factr_gripper_calibrate", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gcal = _load()


def test_format_overrides_and_yaml():
    assert gcal.format_overrides("left", 0.1234, 1.2) == (
        "arms.left.convention.gripper_open=0.1234 arms.left.convention.gripper_closed=1.2000"
    )
    y = gcal.format_yaml("right", 0.0, 1.0)
    assert "right:" in y and "gripper_open: 0.0000" in y and "gripper_closed: 1.0000" in y


def test_preview_matches_normalize_gripper():
    from dual_flexiv_control.configs import JointConventionCfg
    from dual_flexiv_control.control import normalize_gripper

    out = gcal.preview_normalization(0.2, 1.2)
    conv = JointConventionCfg(gripper_open=0.2, gripper_closed=1.2)
    # the midpoint row must reflect the real normalizer (0.5)
    assert "-> 0.500" in out
    assert normalize_gripper(0.7, conv) == pytest.approx(0.5)
    assert "-> 0.000" in out and "-> 1.000" in out


def test_read_gripper_once_uses_trailing_value():
    from dual_flexiv_control.interfaces.factr import FactrClient

    factr_cfg, _sim = gcal.compose_factr(sim_override=True)
    client = FactrClient.from_config(factr_cfg, sim=True)
    try:
        value = gcal.read_gripper_once(client, "left")
        # sim returns dof (8) values; the reading is the trailing one, a finite float
        import numpy as np
        assert np.isfinite(value)
        assert value == pytest.approx(
            float(np.asarray(client.get_joint_positions_for("left")).ravel()[-1]), abs=0.5
        )
    finally:
        client.close()
