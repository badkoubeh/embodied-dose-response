"""Cell enumeration and shard assignment (PLAN.md 4).

STUB.

Sharding must stay a pure partition of the cell list: PLAN.md 2.4 requires results
to be shard-order independent, which holds only if no cell's content depends on
which shard drew it. Seeds come from `edr.seeding`, never from a stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One unit of GPU work: a scenario at a severity level under one seed."""

    scenario_id: str
    axis: str
    level_index: int
    severity: float
    unit: str
    seed_index: int


def enumerate_cells(
    scenario_ids: list[str], axis: str, severities: list[float], unit: str, n_seeds: int
) -> list[CellSpec]:
    """Full cross product for one axis.

    Two economies worth taking (see the open issue on budget):
    the severity-0 cell is the same condition on every axis, so it should be run
    once rather than once per axis; and a deterministic axis such as `staleness`
    should take n_seeds=1, since a second seed is a bit-identical duplicate.
    """
    raise NotImplementedError


def assign_shard(cells: list[CellSpec], shard_index: int, n_shards: int) -> Iterator[CellSpec]:
    """Deterministic partition. Every cell in exactly one shard."""
    raise NotImplementedError
