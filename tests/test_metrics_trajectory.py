"""Trajectory quality metric invariants (PLAN.md 3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from edr.metrics.trajectory import ade, fde, minade
from edr.schema import TRAJ_DIMS, TRAJ_HORIZON


def _straight(dx: float = 0.0, dy: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """A stationary trajectory offset by (dx, dy) at a fixed yaw."""
    traj = np.zeros((TRAJ_HORIZON, TRAJ_DIMS))
    traj[:, 0] = dx
    traj[:, 1] = dy
    traj[:, 2] = yaw
    return traj


def test_ade_of_identical_trajectories_is_zero():
    ref = _straight()
    assert ade(ref, ref) == pytest.approx(0.0)


def test_ade_is_position_only_and_ignores_yaw():
    """TRAJ_DIMS carries yaw, but displacement error is metres, not radians."""
    ref = _straight()
    predicted = _straight(yaw=np.pi / 2)
    assert ade(predicted, ref) == pytest.approx(0.0)


def test_ade_measures_a_constant_offset():
    ref = _straight()
    predicted = _straight(dx=3.0, dy=4.0)
    assert ade(predicted, ref) == pytest.approx(5.0)


def test_fde_only_looks_at_the_final_waypoint():
    ref = _straight()
    predicted = ref.copy()
    predicted[-1, 0] = 3.0
    predicted[-1, 1] = 4.0
    assert fde(predicted, ref) == pytest.approx(5.0)
    assert ade(predicted, ref) < fde(predicted, ref)


@pytest.mark.parametrize(
    "shape", [(TRAJ_HORIZON, 2), (TRAJ_HORIZON - 1, TRAJ_DIMS), (1, TRAJ_HORIZON, TRAJ_DIMS)]
)
def test_ade_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        ade(np.zeros(shape), _straight())


def test_minade_picks_the_best_of_k():
    ref = _straight()
    samples = np.stack([_straight(dx=10.0), ref, _straight(dy=5.0)])
    assert minade(samples, ref, k=3) == pytest.approx(0.0)


def test_minade_with_k1_is_ade_of_the_lone_sample():
    """With K=1 this is just `ade` -- see the open issue on minADE-6."""
    ref = _straight()
    predicted = _straight(dx=3.0, dy=4.0)
    samples = predicted[np.newaxis]
    assert minade(samples, ref, k=1) == ade(predicted, ref)


def test_minade_requires_at_least_k_samples():
    ref = _straight()
    samples = np.stack([ref, ref])
    with pytest.raises(ValueError, match="K >= 6"):
        minade(samples, ref, k=6)
