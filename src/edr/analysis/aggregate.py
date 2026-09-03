"""Scoring and aggregation over stored raw artifacts (PLAN.md 3.3).

STUB.

`to_cells` exists for reporting and plotting. **The fit does not consume it.**
The same scenarios recur at every severity level, so observations are paired and
clustered by `scenario_id`; collapsing to cell means discards that structure and
mis-weights the fit. `fit` takes scored records for exactly this reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from edr.schema import Cell, RawRecord, ScoredRecord


def load_raw(records_dir: Path, trajectories_dir: Path) -> Iterator[RawRecord]:
    """Rejoin `records/*.jsonl` with `trajectories/*.npz` on `trajectory_key`."""
    raise NotImplementedError


def score(records: Iterator[RawRecord]) -> Iterator[ScoredRecord]:
    """Apply the metrics. Re-runnable at no GPU cost -- the point of PLAN.md 2.1."""
    raise NotImplementedError


def to_cells(scored: Iterator[ScoredRecord]) -> list[Cell]:
    """Aggregate to per-cell rates with Clopper-Pearson intervals. Reporting only."""
    raise NotImplementedError
