"""Trajectory quality metrics (PLAN.md 3.2).

**These are weak proxies.** Open-loop displacement error is not a safety measure,
and PLAN.md 3.2 requires that label wherever these appear -- in the paper, in the
scorecard, and in plot axes. The framing PLAN.md 6 commits to is a cheap
reproducible screening method, not a closed-loop safety claim.
"""

from __future__ import annotations

import numpy as np

from edr.schema import TRAJ_DIMS, TRAJ_HORIZON

_SINGLE_SHAPE = (TRAJ_HORIZON, TRAJ_DIMS)


def _positions(trajectory: np.ndarray) -> np.ndarray:
    """First two of the three (x_m, y_m, yaw_rad) columns -- displacement error
    is metres, not radians, so yaw never enters the distance."""
    return trajectory[..., :2]


def _check_single(trajectory: np.ndarray, reference: np.ndarray) -> None:
    if trajectory.shape != _SINGLE_SHAPE:
        raise ValueError(f"trajectory must be {_SINGLE_SHAPE}, got {trajectory.shape}")
    if reference.shape != _SINGLE_SHAPE:
        raise ValueError(f"reference must be {_SINGLE_SHAPE}, got {reference.shape}")


def ade(trajectory: np.ndarray, reference: np.ndarray) -> float:
    """Average displacement error of a single trajectory, in metres."""
    _check_single(trajectory, reference)
    diff = _positions(trajectory) - _positions(reference)
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def minade(trajectory: np.ndarray, reference: np.ndarray, k: int = 6) -> float:
    """Minimum ADE over K sampled trajectories.

    Requires `trajectory` to carry K >= k samples on its leading axis. With K=1
    this is just `ade`, and calling the result minADE-6 -- or comparing it to
    NVIDIA's published 1.22 m -- would be wrong. See the open issue.
    """
    if trajectory.ndim != 3 or trajectory.shape[1:] != _SINGLE_SHAPE:
        raise ValueError(
            f"trajectory must be (K, {TRAJ_HORIZON}, {TRAJ_DIMS}), got {trajectory.shape}"
        )
    n = trajectory.shape[0]
    if n < k:
        raise ValueError(f"minade(k={k}) requires K >= {k} samples, got K={n}")
    # No per-sample ranking exists yet (the resampling question is still open on
    # the tracker), so the first k samples are the only well-defined,
    # deterministic choice when K > k.
    return min(ade(trajectory[i], reference) for i in range(k))


def fde(trajectory: np.ndarray, reference: np.ndarray) -> float:
    """Final displacement error, in metres."""
    _check_single(trajectory, reference)
    diff = _positions(trajectory)[-1] - _positions(reference)[-1]
    return float(np.linalg.norm(diff))
