"""Threshold extraction (PLAN.md 4).

STUB.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    """Which way "worse" runs for a metric.

    This exists because "sigma*_p = severity at p x clean baseline" is only well
    defined for HIGHER_IS_BETTER. For a metric where lower is better -- a
    displacement error, say -- p x baseline is an *improvement* that degradation
    never reaches, leaving the threshold and any ratio built on it undefined.
    Making the direction explicit is the fix.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    """A fully specified threshold: which metric, which direction, which p.

    For HIGHER_IS_BETTER, p is a retention fraction (p=0.5 -> half the clean
    rate). For LOWER_IS_BETTER, p is an inflation factor (p=0.5 -> the response
    has degraded to baseline/0.5). Both must be stated in the paper.
    """

    metric: str
    direction: Direction
    p: float
    baseline: float


def ed_p(fit: object, spec: ThresholdSpec) -> float:
    """The severity at which `fit` crosses the threshold `spec` defines."""
    raise NotImplementedError


def censored_lower_bound(fit: object, spec: ThresholdSpec, max_severity: float) -> float:
    """One-sided lower bound when the response never crosses within the grid.

    Report this, never the last grid point. `lifelines` handles the machinery.
    """
    raise NotImplementedError
