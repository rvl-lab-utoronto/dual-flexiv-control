"""Tests for the FACTR→Rizon joint convention (pure math, mirrors the hardware test)."""

from __future__ import annotations

import numpy as np
import pytest

from dual_flexiv_control.configs import JointConventionCfg
from dual_flexiv_control.control import convert_factr_to_rizon
from dual_flexiv_control.control import offsets_from_straight_pose


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


def test_default_joints_1_2_mirrored_with_home_plus_90():
    """Joint indices 1 and 2 on hardware: home maps to +90 deg, motion mirrored.

    The flip applies after the offset, so a flipped joint stores the negated
    offset: out = -(theta - 90) = -theta + 90.
    """
    conv = JointConventionCfg()
    assert set(conv.sign_flip_joints) == {1, 2, 3}
    home = np.degrees(convert_factr_to_rizon(np.zeros(8), conv))
    moved = np.degrees(convert_factr_to_rizon(np.deg2rad([0, 10, 10, 0, 0, 0, 0, 0]), conv))
    for j in (1, 2):
        assert home[j] == pytest.approx(90.0)
        assert moved[j] - home[j] == pytest.approx(-10.0)  # leader +10 -> follower -10


def test_wraps_to_plus_minus_180():
    conv = JointConventionCfg(offsets_deg=[179.0] * 7, sign_flip_joints=[], drop_trailing=0)
    out = np.degrees(convert_factr_to_rizon(np.deg2rad([10.0] * 7), conv))
    np.testing.assert_allclose(out, [-171.0] * 7, atol=1e-9)  # 10+179=189 -> -171


# -- straight-pose offset calibration (inverse of convert_factr_to_rizon) --------


def test_offsets_from_straight_pose_maps_capture_to_zero():
    """Solved offsets must send the captured leader pose to an all-zero follower."""
    conv = JointConventionCfg()  # keep its sign flips / drop / wrap
    q8 = np.deg2rad([12, -34, 56, -78, 90, -11, 22, 5.0])  # arbitrary straight pose + gripper
    offsets = offsets_from_straight_pose(q8, conv)
    assert len(offsets) == 7  # one per follower joint, gripper dropped
    new_conv = JointConventionCfg(
        offsets_deg=offsets,
        sign_flip_joints=conv.sign_flip_joints,
        drop_trailing=conv.drop_trailing,
        wrap_deg=conv.wrap_deg,
    )
    np.testing.assert_allclose(convert_factr_to_rizon(q8, new_conv), np.zeros(7), atol=1e-9)


def test_offsets_from_straight_pose_are_negated_and_wrapped():
    conv = JointConventionCfg()
    # A joint past ±180 after negation must wrap back into range.
    q8 = np.deg2rad([170, 0, 0, 0, 0, 0, 0, 0.0])
    offsets = offsets_from_straight_pose(q8, conv)
    assert offsets[0] == pytest.approx(-170.0)  # -170 already in range
    assert all(-180.0 <= o <= 180.0 for o in offsets)


def test_offsets_from_straight_pose_ignores_sign_flips():
    """The flip is irrelevant at the zero target, so flipped/unflipped agree."""
    q8 = np.deg2rad([12, -34, 56, -78, 90, -11, 22, 5.0])
    flipped = offsets_from_straight_pose(q8, JointConventionCfg(sign_flip_joints=[1, 2, 3]))
    unflipped = offsets_from_straight_pose(q8, JointConventionCfg(sign_flip_joints=[]))
    np.testing.assert_allclose(flipped, unflipped, atol=1e-12)
