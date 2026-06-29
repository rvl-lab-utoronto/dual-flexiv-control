"""Tests for the FACTR→Rizon joint convention (pure math, mirrors the hardware test)."""

from __future__ import annotations

import numpy as np
import pytest

from dual_flexiv_control.configs import JointConventionCfg
from dual_flexiv_control.control import convert_factr_to_rizon


def _reference(q_leader_rad, offsets, flips, drop=1, wrap=True):
    """Independent re-implementation of the rizon_tests conversion."""
    q = np.asarray(q_leader_rad, dtype=float)
    if drop:
        q = q[: len(q) - drop]
    deg = np.degrees(q) + np.asarray(offsets[: len(q)], dtype=float)
    for j in flips:
        deg[j] = -deg[j]
    if wrap:
        deg = (deg + 180.0) % 360.0 - 180.0
    return np.radians(deg)


def test_matches_reference_math():
    conv = JointConventionCfg()
    q8 = np.deg2rad([10, 20, 30, 40, 50, 60, 70, 5.0])  # 7 joints + trailing gripper
    np.testing.assert_allclose(
        convert_factr_to_rizon(q8, conv),
        _reference(q8, conv.offsets_deg, conv.sign_flip_joints),
    )


def test_drops_trailing_gripper_to_dof():
    conv = JointConventionCfg()  # drop_trailing=1
    out = convert_factr_to_rizon(np.zeros(8), conv)
    assert out.shape == (7,)


def test_sign_flip_is_joint_index_3_only():
    conv = JointConventionCfg(
        offsets_deg=[0.0] * 7, sign_flip_joints=[3], drop_trailing=0, wrap_deg=False
    )
    out = np.degrees(convert_factr_to_rizon(np.deg2rad([5.0] * 7), conv))
    assert out[3] == pytest.approx(-5.0)
    assert out[0] == pytest.approx(5.0)
    assert out[6] == pytest.approx(5.0)


def test_wraps_to_plus_minus_180():
    conv = JointConventionCfg(offsets_deg=[179.0] * 7, sign_flip_joints=[], drop_trailing=0)
    out = np.degrees(convert_factr_to_rizon(np.deg2rad([10.0] * 7), conv))
    np.testing.assert_allclose(out, [-171.0] * 7, atol=1e-9)  # 10+179=189 -> -171
