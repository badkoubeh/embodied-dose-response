"""Trajectory quality metrics (PLAN.md 3.2).

STUB.

**These are weak proxies.** Open-loop displacement error is not a safety measure,
and PLAN.md 3.2 requires that label wherever these appear -- in the paper, in the
scorecard, and in plot axes. The framing PLAN.md 6 commits to is a cheap
reproducible screening method, not a closed-loop safety claim.
"""

from __future__ import annotations

import numpy as np


def ade(trajectory: np.ndarray, reference: np.ndarray) -> float:
    """Average displacement error of a single trajectory, in metres."""
    raise NotImplementedError


def minade(trajectory: np.ndarray, reference: np.ndarray, k: int = 6) -> float:
    """Minimum ADE over K sampled trajectories.

    Requires `trajectory` to carry K >= k samples on its leading axis. With K=1
    this is just `ade`, and calling the result minADE-6 -- or comparing it to
    NVIDIA's published 1.22 m -- would be wrong. See the open issue.
    """
    raise NotImplementedError


def fde(trajectory: np.ndarray, reference: np.ndarray) -> float:
    """Final displacement error, in metres."""
    raise NotImplementedError
