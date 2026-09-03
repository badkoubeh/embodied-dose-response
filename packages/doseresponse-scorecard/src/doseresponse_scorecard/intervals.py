"""Confidence intervals (PLAN.md 4).

STUB.

Profile likelihood for single thresholds; **Fieller for the ratio**, which is the
headline; delta method as a cross-check; parametric bootstrap if the link is
suspect.

Clopper-Pearson belongs to per-cell rates and does NOT transfer to an
interpolated threshold. PLAN.md 4 asks for that to be stated explicitly to
pre-empt the reviewer note.
"""

from __future__ import annotations


def profile_likelihood_ci(
    fit: object, threshold: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Profile-likelihood interval for a single threshold."""
    raise NotImplementedError


def fieller_ratio_ci(
    numerator_fit: object, denominator_fit: object, alpha: float = 0.05
) -> tuple[float, float]:
    """Fieller interval for a ratio of thresholds.

    Fieller's interval is unbounded, or a pair of disjoint rays, when the
    denominator is not clearly separated from zero -- which is exactly what a
    censored denominator arm produces. Handle that case explicitly and return a
    one-sided bound rather than a nonsensical pair.
    """
    raise NotImplementedError


def delta_method_ci(
    numerator_fit: object, denominator_fit: object, alpha: float = 0.05
) -> tuple[float, float]:
    """Delta-method interval. Cross-check on Fieller, not a replacement."""
    raise NotImplementedError


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for ONE cell's rate. Not for thresholds."""
    raise NotImplementedError
