"""Ego-state read/write in the model's native 16-waypoint format (PLAN.md 4).

STUB -- the dataset is gated.

Implementing this settles the two unknowns flagged in `edr.schema`: the actual
waypoint spacing (PLAN.md gives 16 waypoints and a 100 ms *frame* period without
saying they match) and whether the layout really is planar (x, y, yaw). Every
physical severity value in configs/axis/ is provisional until it does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ego_history(path: Path) -> np.ndarray:
    """Load one scenario's ego history as (EGO_HISTORY_LEN, EGO_DIMS)."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")


def to_model_input(ego_history: np.ndarray) -> object:
    """Convert canonical ego-state into whatever the model wrapper expects.

    PLAN.md 2.2: conversion to model-space happens here, and the physical value
    is what gets logged. Nothing downstream of this may see model-space units.
    """
    raise NotImplementedError("blocked on model access (PLAN.md 5, Stage 0)")
