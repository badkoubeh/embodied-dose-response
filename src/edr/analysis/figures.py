"""Figures (PLAN.md 4).

STUB. The hero-figure plotter itself lives in `doseresponse-scorecard`; this is
the Alpamayo-side call into it.
"""

from __future__ import annotations

from pathlib import Path


def hero_figure(consistency_fit: object, trajectory_fit: object, out: Path) -> None:
    """Both dose-response curves on one severity axis: CI bands, both thresholds
    marked, ratio CI annotated. Label the trajectory axis as a weak proxy."""
    raise NotImplementedError
