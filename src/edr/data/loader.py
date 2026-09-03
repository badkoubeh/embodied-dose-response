"""Scenario loading from the 937-sample challenging open-loop split (PLAN.md 4).

STUB -- the dataset is gated.

One constraint to honour when implementing: `select_scenarios` must return a
SINGLE fixed subset reused at every severity level of the sweep, not a fresh draw
per level. PLAN.md 2.3 requires a fixed common grid, and a per-level redraw would
confound between-scenario composition with severity while destroying the pairing
that the clustered fit depends on. With 937 available and at most 500 needed,
this costs nothing. See the open issue on repeated measures.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from edr.schema import Sample


def load_split(root: Path, split: str = "challenging") -> list[str]:
    """Return the scenario ids in `split`, in a stable order."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")


def select_scenarios(scenario_ids: list[str], n: int, seed: int) -> list[str]:
    """Pick the fixed scenario subset used at EVERY level. See module docstring."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")


def iter_samples(root: Path, scenario_ids: list[str]) -> Iterator[Sample]:
    """Yield one `Sample` per scenario id, ego-state loaded, video not decoded."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")
