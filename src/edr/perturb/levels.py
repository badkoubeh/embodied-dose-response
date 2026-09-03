"""Severity grids, loaded from config rather than hard-coded (PLAN.md 2.2, 2.3).

A level is a physical quantity with a name attached. The name is not decoration:
PLAN.md 3.2 anchors each axis in named hardware and field studies, and a severity
that cannot be pointed at a real-world referent is exactly the "arbitrary
multiple" PLAN.md 2.2 forbids.

Grids are a *fixed common* set (PLAN.md 2.3) -- both measures, both seeds, and
any second model see identical conditions. Adaptive dosing is permitted only in
the throwaway pilot, which is why `pilot` and `full` are separate named sets and
only `full` is reportable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Level:
    """One grid point: its index, its physical value, and what it corresponds to."""

    index: int
    value: float
    unit: str
    label: str

    @property
    def is_clean(self) -> bool:
        return self.value == 0.0


def load_levels(axis_cfg: Any, grid: str) -> list[Level]:
    """Read the named grid out of an `configs/axis/*.yaml` node.

    Validates the invariants the sweep and the fits rely on:

    * the grid is sorted ascending and has no duplicates, so `level_index` is a
      meaningful ordinal and monotone fits are well posed;
    * it starts at 0.0, the shared clean baseline every threshold in PLAN.md 4
      is defined relative to;
    * every level carries a label, per PLAN.md 2.2.
    """
    grids = axis_cfg["levels"]
    if grid not in grids:
        raise KeyError(f"axis {axis_cfg['name']!r} has no grid {grid!r}; has: {sorted(grids)}")

    unit = axis_cfg["unit"]
    raw = grids[grid]
    if not raw:
        raise ValueError(f"axis {axis_cfg['name']!r} grid {grid!r} is empty")

    levels = [
        Level(index=i, value=float(e["value"]), unit=unit, label=str(e["label"]))
        for i, e in enumerate(raw)
    ]

    values = [lv.value for lv in levels]
    if values[0] != 0.0:
        raise ValueError(
            f"axis {axis_cfg['name']!r} grid {grid!r} must start at the clean "
            f"baseline 0.0, got {values[0]}"
        )
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(
            f"axis {axis_cfg['name']!r} grid {grid!r} must be strictly ascending, got {values}"
        )
    if any(not lv.label for lv in levels):
        raise ValueError(f"axis {axis_cfg['name']!r} grid {grid!r} has an unlabelled level")

    return levels
