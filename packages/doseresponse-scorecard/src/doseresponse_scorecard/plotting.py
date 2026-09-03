"""The hero-figure plotter (PLAN.md 4).

STUB. Domain-neutral: it takes fits and labels, never a model name.
"""

from __future__ import annotations

from typing import Any


def hero_figure(
    fits: dict[str, object], severity_label: str, annotations: dict[str, Any] | None = None
) -> Any:
    """Both dose-response curves on one severity axis, CI bands, both thresholds
    marked, ratio CI annotated. Returns the matplotlib Figure."""
    raise NotImplementedError
