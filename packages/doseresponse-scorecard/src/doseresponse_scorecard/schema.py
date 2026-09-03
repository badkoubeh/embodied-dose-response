"""The score-card JSON schema (PLAN.md 3.2).

STUB.

The portable artifact: a fitted dose-response result serialized so that two
different domains can be compared without either knowing anything about the
other. Must carry, alongside the numbers, the provenance needed to read them --
the threshold spec including its direction and p, the link function, whether the
severity axis was logged, whether the estimate is censored, and how standard
errors were clustered.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "0.0.1"


def to_scorecard(fits: dict[str, object], meta: dict[str, Any]) -> dict[str, Any]:
    """Serialize fitted results into the portable score-card dict."""
    raise NotImplementedError


def validate(scorecard: dict[str, Any]) -> None:
    """Raise if `scorecard` does not conform to `SCHEMA_VERSION`."""
    raise NotImplementedError
