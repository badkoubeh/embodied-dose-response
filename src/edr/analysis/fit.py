"""Fitting entrypoints (PLAN.md 4).

STUB. Wraps `doseresponse-scorecard`; contains no statistics of its own.

Three things to settle before this is written, all on the tracker:

* **Cluster by `scenario_id`.** A pooled binomial GLM treats repeated measures on
  the same scenarios as independent, understating SEs, narrowing the threshold
  CI, and narrowing the Fieller interval -- which is the headline object. A
  spurious "CI below 1" is precisely the failure mode.
* **Thresholds are directional.** "p x clean baseline" is well defined for
  consistency, where higher is better, but inverts for minADE, where lower is
  better: 0.8 x 1.22 m is an improvement that degradation never reaches. Use an
  explicit direction per metric.
* **Pre-specify the censored case.** If the trajectory arm never breaks in range,
  its threshold is a one-sided bound and the Fieller interval on the ratio is
  unbounded or exclusive. Fix the fallback rule before seeing the data.
"""

from __future__ import annotations

from collections.abc import Iterable

from edr.schema import ScoredRecord


def fit_binary(scored: Iterable[ScoredRecord], axis: str, link: str = "probit") -> object:
    """GLM of a binary predicate on severity, SEs clustered by `scenario_id`."""
    raise NotImplementedError


def fit_continuous(scored: Iterable[ScoredRecord], axis: str) -> object:
    """Monotone-constrained log-logistic or isotonic fit of a continuous metric."""
    raise NotImplementedError


def threshold_ratio(consistency_fit: object, trajectory_fit: object, p: float) -> object:
    """The headline: sigma*(consistency) / sigma*(trajectory) with a Fieller interval.

    Report this as a curve over p rather than a single number. The two arms use
    incommensurable definitions of "broken", so the value -- and possibly which
    side of 1 it falls on -- depends on p, and a reviewer will ask.
    """
    raise NotImplementedError
